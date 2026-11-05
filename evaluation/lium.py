"""Journaled, non-authoritative development rounds on ephemeral Lium pods."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from evaluation.datasets.sampling import select_round, validate_plan
from evaluation.profile import EvaluationMode
from evaluation.runner import LocalEvaluationReport, summarize_report
from shared.checkpoints import validate_checkpoint
from shared.config import Settings, TargetConfig
from shared.contracts import Decision, FileDigest
from shared.datasets import RoundPlan
from shared.lium import LiumClient
from shared.ssh import SshTransport

PYTHON = "/opt/prexon/.venv/bin/python"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@contextmanager
def locked(state_path: Path) -> Iterator[None]:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.with_suffix(state_path.suffix + ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process owns this evaluation journal") from exc
        yield


def inputs(
    target: TargetConfig,
    checkpoint: Path,
    public_checkpoint: Path | None,
    incumbent_checkpoint: Path | None,
    seed: str,
    port: int,
    remote_target_path: Path | None,
    mode: EvaluationMode,
    *,
    dataset_cache: Path | None = None,
    prepared_round: RoundPlan | None = None,
) -> tuple[dict[str, Any], dict[str, Path]]:
    if not 1 <= port <= 65535:
        raise ValueError("Evaluation port must be between 1 and 65535")
    if remote_target_path is not None and not remote_target_path.is_absolute():
        raise ValueError("--remote-target-path must be absolute inside the pod")
    if mode not in {"head", "full"}:
        raise ValueError("Unknown evaluation mode")
    paths = {"candidate": checkpoint}
    if mode == "full" and public_checkpoint is not None:
        paths["public"] = public_checkpoint
    if mode == "full" and incumbent_checkpoint is not None:
        paths["incumbent"] = incumbent_checkpoint
    checkpoints = {
        arm: [f.model_dump() for f in validate_checkpoint(path, target.policy.max_checkpoint_bytes)]
        for arm, path in paths.items()
    }
    dataset: dict[str, Any] = {}
    if target.datasets is not None:
        plan = prepared_round or select_round(
            target, dataset_cache or Path("runtime/datasets"), seed
        )
        validate_plan(target, plan, seed)
        dataset["dataset_plan"] = plan.model_dump(mode="json")
    elif prepared_round is not None:
        raise ValueError("Unexpected saved dataset plan")
    return {
        "mode": mode,
        "target": target.model_dump(mode="json"),
        "checkpoints": checkpoints,
        "seed": seed,
        "port": port,
        "target_path": str(remote_target_path) if remote_target_path else None,
        **dataset,
    }, paths


def quote_evaluation(
    settings: Settings,
    target: TargetConfig,
    checkpoint: Path,
    *,
    api_key: str,
    public_checkpoint: Path | None = None,
    incumbent_checkpoint: Path | None = None,
    seed: str = "42",
    port: int = 30000,
    remote_target_path: Path | None = None,
    mode: EvaluationMode = "full",
) -> dict[str, Any]:
    inputs(
        target,
        checkpoint,
        public_checkpoint,
        incumbent_checkpoint,
        seed,
        port,
        remote_target_path,
        mode,
        dataset_cache=settings.dataset_cache,
    )
    provider = LiumClient(settings, api_key)
    now = time.time()
    request = provider.build_request(
        {"id": uuid.uuid4().hex, "started_at": now, "deadline": now + settings.session_seconds},
        target,
    )
    return provider.quote(request)


def _cleanup(provider: LiumClient, state: dict[str, Any], state_path: Path) -> bool:
    state["phase"] = "stopping"
    write_json(state_path, state)
    try:
        if not state.get("pod_id") and state.get("rent_attempted"):
            state["pod_id"] = provider.recover(state["id"])
            write_json(state_path, state)
        if state.get("pod_id"):
            provider.delete(state["pod_id"])
            state["cleanup"] = "deleted"
        elif state.get("rent_attempted"):
            request = state.get("request")
            hours = request.get("termination_hours") if isinstance(request, dict) else None
            deadline = state.get("deadline")
            # Retain ownership through the latest create time plus the saved provider lifetime.
            expired = (
                type(hours) is int
                and 1 <= hours <= 720
                and isinstance(deadline, (int, float))
                and not isinstance(deadline, bool)
                and math.isfinite(deadline)
                and deadline > 0
                and time.time() >= deadline + hours * 3600 + 60
            )
            state["cleanup"] = "expiry_unconfirmed" if expired else "pending"
        else:
            state["cleanup"] = "not_applicable"
    except (Exception, KeyboardInterrupt):
        state["cleanup"] = "pending"
    state["phase"] = "finished"
    write_json(state_path, state)
    return state["cleanup"] != "pending"


def cleanup_evaluation(settings: Settings, state_path: Path, *, api_key: str) -> bool:
    with locked(state_path):
        state = json.loads(state_path.read_bytes())
        provider = LiumClient(settings, api_key)
        success = _cleanup(provider, state, state_path)
        output = Path(state["output"])
        if output.exists():
            report = LocalEvaluationReport.model_validate_json(output.read_bytes())
            if report.execution_id != state["id"]:
                raise ValueError("Report and rental journal identities differ")
            write_json(
                output,
                report.model_copy(
                    update={
                        "backend": "lium",
                        "pod_id": state["pod_id"],
                        "rental_cleanup": state["cleanup"],
                    }
                ).model_dump(mode="json"),
            )
        return success


def _validate_report(report: LocalEvaluationReport, state: dict[str, Any]) -> None:
    request = state["inputs"]
    target = TargetConfig.model_validate(request["target"])
    expected = {
        arm: tuple(FileDigest.model_validate(f) for f in files)
        for arm, files in request["checkpoints"].items()
    }
    if (
        report.mode != request["mode"]
        or report.execution_id != state["id"]
        or report.competition_id != target.identity
        or report.target_id != target.model_id
        or report.target_revision != target.revision
        or report.seed != request["seed"]
        or report.checkpoints != expected
        or report.local_target_path != request["target_path"]
        or report.target_source != ("local_override" if request["target_path"] else "pinned_cache")
    ):
        raise ValueError("Returned report does not match this evaluation's inputs")
    if request.get("dataset_plan") is not None:
        expected_selection = RoundPlan.model_validate(request["dataset_plan"]).evidence()
        if (
            report.status == "complete" or report.dataset_selection is not None
        ) and report.dataset_selection != expected_selection:
            raise ValueError("Returned report dataset selection mismatch")
    derived = summarize_report(target, report)
    for field in ("status", "assessment", "decision", "head_metrics"):
        if getattr(derived, field) != getattr(report, field):
            raise ValueError("Returned report summary is inconsistent with its evidence")
    for field in ("score", "threshold", "required_score", "lower_gain"):
        actual, expected_value = getattr(report, field), getattr(derived, field)
        if (
            (actual is None) != (expected_value is None)
            or actual is not None
            and not math.isclose(actual, expected_value, rel_tol=1e-10)
        ):
            raise ValueError("Returned scores are inconsistent with measured evidence")
    if report.scores.keys() != derived.scores.keys() or any(
        not math.isclose(v, derived.scores[k], rel_tol=1e-10) for k, v in report.scores.items()
    ):
        raise ValueError("Returned scores are inconsistent with measured evidence")


def evaluate_on_lium(
    settings: Settings,
    target: TargetConfig,
    checkpoint: Path,
    output: Path,
    *,
    api_key: str,
    public_checkpoint: Path | None = None,
    incumbent_checkpoint: Path | None = None,
    seed: str = "42",
    port: int = 30000,
    remote_target_path: Path | None = None,
    mode: EvaluationMode = "full",
) -> LocalEvaluationReport:
    if not settings.ssh_private_key.expanduser().is_file() or not shutil.which("ssh"):
        raise ValueError("Configure an existing SSH private key and install OpenSSH before renting")
    state_path = output.with_suffix(".lium.json")
    provider = LiumClient(settings, api_key)
    with locked(state_path):
        saved = json.loads(state_path.read_bytes()) if state_path.exists() else None
        frozen_plan = None
        if saved is not None and saved["inputs"].get("dataset_plan") is not None:
            frozen_plan = RoundPlan.model_validate(saved["inputs"]["dataset_plan"])
        request, paths = inputs(
            target,
            checkpoint,
            public_checkpoint,
            incumbent_checkpoint,
            seed,
            port,
            remote_target_path,
            mode,
            dataset_cache=settings.dataset_cache,
            prepared_round=frozen_plan,
        )
        signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if saved is not None:
            state = saved
            if state["signature"] != signature or state["output"] != str(output.resolve()):
                raise ValueError("Evaluation inputs changed; use another output path for a new run")
            if state["phase"] in {"stopping", "finished"} or output.exists():
                _cleanup(provider, state, state_path)
                if output.exists():
                    previous = LocalEvaluationReport.model_validate_json(output.read_bytes())
                    _validate_report(previous, state)
                    previous = previous.model_copy(
                        update={
                            "backend": "lium",
                            "pod_id": state["pod_id"],
                            "rental_cleanup": state["cleanup"],
                        }
                    )
                    write_json(output, previous.model_dump(mode="json"))
                    return previous
                raise ValueError("Previous run stopped without evidence; use a new output path")
        else:
            now = time.time()
            state = {
                "id": uuid.uuid4().hex,
                "signature": signature,
                "inputs": request,
                "output": str(output.resolve()),
                "started_at": now,
                "deadline": now + settings.session_seconds,
                "max_cost": settings.session_max_cost,
                "phase": "active",
                "rent_attempted": False,
                "pod_id": None,
                "cleanup": "not_applicable",
            }
            state["request"] = provider.build_request(state, target)
            write_json(state_path, state)
        report: LocalEvaluationReport | None = None
        next_poll = 0.0
        pod: dict[str, Any] = {}
        transport: SshTransport | None = None
        root = f"/tmp/prexon-development-eval/{state['id']}"

        def command(action: str) -> list[str]:
            return [PYTHON, "-m", "evaluation.remote", action, "--root", root]

        def tick() -> None:
            nonlocal next_poll, pod
            now = time.time()
            if now >= state["deadline"]:
                raise TimeoutError("Development evaluation rental deadline reached")
            if state["pod_id"] and now >= next_poll:
                next_poll = now + 10
                pod = provider.status(state["pod_id"])
                status = str(pod.get("status", "")).lower()
                if status in {"deleted", "terminated", "failed", "stopped", "error"}:
                    raise RuntimeError("Evaluation pod stopped")
                for field in ("price", "pending_price"):
                    price = pod.get(field)
                    if price is not None and (
                        not math.isfinite(float(price))
                        or float(price) < 0
                        or float(price)
                        > state["request"].get(
                            "max_price_per_gpu_hour", target.hardware.max_price_per_gpu_hour
                        )
                        * target.hardware.gpu_count
                    ):
                        raise RuntimeError("Evaluation pod price exceeds the configured cap")
                spend = float(pod.get("spend_to_date") or 0)
                if not math.isfinite(spend) or spend < 0 or spend >= state["max_cost"]:
                    raise RuntimeError("Development evaluation rental spend limit reached")

        try:
            tick()
            # Snapshot before renting, and recheck after copying to avoid ongoing-training races.
            with tempfile.TemporaryDirectory(prefix="prexon-lium-") as temporary:
                staging = Path(temporary)
                for arm, path in paths.items():
                    directory = staging / arm
                    directory.mkdir()
                    for file in request["checkpoints"][arm]:
                        shutil.copyfile(path / file["path"], directory / file["path"])
                    actual = [
                        f.model_dump()
                        for f in validate_checkpoint(directory, target.policy.max_checkpoint_bytes)
                    ]
                    if actual != request["checkpoints"][arm]:
                        raise ValueError("Checkpoint changed while freezing evaluation inputs")
                remote_request = {**request, "execution_id": state["id"]}
                payload = json.dumps(remote_request, sort_keys=True).encode()
                (staging / "request.json").write_bytes(payload)
                (staging / "target.json").write_text(target.model_dump_json())
                archive_path = staging / "upload.tar"
                with tarfile.open(archive_path, "w") as archive:
                    for path in sorted(staging.rglob("*")):
                        if path.is_file() and path != archive_path:
                            archive.add(
                                path, arcname=str(path.relative_to(staging)), recursive=False
                            )
                tick()
                if not state["pod_id"]:
                    if state["rent_attempted"]:
                        state["pod_id"] = provider.recover(state["id"])
                        if not state["pod_id"]:
                            # An accepted create may not be visible yet; never create a second pod.
                            raise RuntimeError("Previous rental identity is not yet recoverable")
                    else:
                        state["rent_attempted"] = True
                        write_json(state_path, state)
                        state["pod_id"] = provider.rent(state["id"], state["request"])
                    write_json(state_path, state)
                next_poll = 0
                while True:
                    tick()
                    if str(pod.get("status", "")).lower() == "running" and pod.get(
                        "ssh_connect_cmd"
                    ):
                        break
                    time.sleep(1)
                transport = SshTransport(settings, pod["ssh_connect_cmd"], tick)

                code = transport.run(
                    [*command("upload"), "--digest", hashlib.sha256(payload).hexdigest()],
                    source=archive_path,
                )
                if code:
                    raise RuntimeError("Could not transfer evaluation inputs")
                # Detached execution and a remote lock allow host restart without duplicate engines.
                if transport.run(command("start")):
                    raise RuntimeError("Could not start remote evaluation")
                returned = staging / "returned.json"
                while True:
                    tick()
                    code = transport.run(command("report"), destination=returned)
                    if code == 0:
                        returned_report = LocalEvaluationReport.model_validate_json(
                            returned.read_bytes()
                        )
                        _validate_report(returned_report, state)
                        report = returned_report
                        break
                    if code not in {44, 255}:
                        raise RuntimeError("Remote evaluation ended without a valid report")
                    time.sleep(2)
                # Save evidence before optional log transfer or rental cleanup can fail.
                write_json(output, report.model_dump(mode="json"))
        except (Exception, KeyboardInterrupt) as exc:
            if report is None:
                report = LocalEvaluationReport(
                    status="inconclusive",
                    mode=mode,
                    decision=None
                    if mode == "head"
                    else Decision(outcome="inconclusive", reason="incomplete_or_expired_evidence"),
                    competition_id=target.identity,
                    target_id=target.model_id,
                    started_at=state["started_at"],
                    finished_at=time.time(),
                    target_revision=target.revision,
                    seed=seed,
                    execution_id=state["id"],
                    target_source="local_override" if remote_target_path else "pinned_cache",
                    local_target_path=str(remote_target_path) if remote_target_path else None,
                    checkpoints={
                        a: tuple(FileDigest.model_validate(f) for f in fs)
                        for a, fs in request["checkpoints"].items()
                    },
                    assessment="inconclusive",
                    error_code="lium_evaluation_failed",
                    notes=(f"Rental evaluation interrupted: {type(exc).__name__}.",),
                )
        finally:
            try:
                if transport is not None:
                    logs = output.with_suffix(".logs")
                    logs.mkdir(parents=True, exist_ok=True)
                    transport.run(
                        command("logs"),
                        destination=logs / "remote.tar.gz",
                        limit=512 * 1024**2,
                        timeout_seconds=30,
                    )
            except (Exception, KeyboardInterrupt):
                pass  # Diagnostics must not prevent rental cleanup or discard valid evidence.
            finally:
                _cleanup(provider, state, state_path)
        assert report is not None
        report = report.model_copy(
            update={
                "backend": "lium",
                "pod_id": state["pod_id"],
                "rental_cleanup": state["cleanup"],
            }
        )
        write_json(output, report.model_dump(mode="json"))
        return report
