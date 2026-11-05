"""Evaluate frozen local heads in a GPU pod."""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from pathlib import Path
from statistics import mean
from typing import Literal

from evaluation.datasets.sampling import validate_selection
from evaluation.measurement import hardware_matches, measure_configurations
from evaluation.metrics import throughput_summary
from evaluation.profile import EvaluationMode
from evaluation.scoring import score_benchmark
from evaluation.sglang import Arm
from shared.checkpoints import validate_checkpoint
from shared.config import Digest, FrozenModel, Positive, TargetConfig
from shared.contracts import BenchmarkEvidence, Decision, FileDigest
from shared.datasets import RoundPlan


class HeadMetrics(FrozenModel):
    throughput_tps: Positive
    p95_latency_seconds: Positive
    output_tokens: int
    elapsed_seconds: Positive
    acceptance_rate: float | None
    average_acceptance_length: float | None


class LocalEvaluationReport(BenchmarkEvidence):
    authoritative: Literal[False] = False
    purpose: Literal["development"] = "development"
    submission_count: Literal[1] = 1
    mode: EvaluationMode = "full"
    decision: Decision | None = Decision(
        outcome="inconclusive", reason="incomplete_or_expired_evidence"
    )
    head_metrics: dict[str, HeadMetrics] = {}
    execution_id: str | None = None
    backend: Literal["pod", "lium"] = "pod"
    pod_id: str | None = None
    rental_cleanup: Literal["not_applicable", "deleted", "pending", "expiry_unconfirmed"] = (
        "not_applicable"
    )
    competition_id: Digest
    target_id: str
    target_revision: str
    target_source: Literal["pinned_cache", "local_override"]
    local_target_path: str | None = None
    resolved_target_path: str | None = None
    seed: str
    checkpoints: dict[str, tuple[FileDigest, ...]]
    assessment: Literal[
        "head_measured", "passes_local_checks", "fails_local_checks", "inconclusive"
    ]
    score: Positive | None = None
    scores: dict[str, float] = {}
    threshold: Positive | None = None
    required_score: Positive | None = None
    lower_gain: float | None = None
    notes: tuple[str, ...] = ()


async def evaluate(
    target: TargetConfig,
    checkpoint: Path,
    log_dir: Path,
    *,
    public_checkpoint: Path | None = None,
    incumbent_checkpoint: Path | None = None,
    target_path: Path | None = None,
    seed: str = "42",
    port: int = 30000,
    mode: EvaluationMode = "full",
    dataset_cache: Path | None = None,
    prepared_round: RoundPlan | None = None,
) -> LocalEvaluationReport:
    paths: dict[Arm, Path | None] = {"candidate": checkpoint.resolve()}
    if mode == "full":
        paths = {"target": None, **paths}
    if mode == "full" and public_checkpoint is not None:
        paths["public"] = public_checkpoint.resolve()
    if mode == "full" and incumbent_checkpoint is not None:
        paths["incumbent"] = incumbent_checkpoint.resolve()
    files = {
        arm: validate_checkpoint(path, target.policy.max_checkpoint_bytes)
        for arm, path in paths.items()
        if path is not None
    }
    requested_target_path = str(target_path) if target_path else None
    if target_path is not None:
        target_path = target_path.resolve()
        if not target_path.is_dir() or not (target_path / "config.json").is_file():
            raise ValueError("Local target snapshot needs a config.json")
    staging_wall = time.time()
    staging_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="prexon-local-evaluation-") as temp:
        frozen: dict[Arm, Path | None] = {"target": None} if mode == "full" else {}
        for arm, path in paths.items():
            if path is None:
                continue
            directory = Path(temp) / arm
            directory.mkdir()
            for file in files[arm]:
                shutil.copyfile(path / file.path, directory / file.path)
            if validate_checkpoint(directory, target.policy.max_checkpoint_bytes) != files[arm]:
                raise ValueError("Checkpoint changed while preparing the evaluation snapshot")
            frozen[arm] = directory
        staging_seconds = time.perf_counter() - staging_started
        evidence = await measure_configurations(
            target,
            frozen,
            log_dir,
            time.time() + target.policy.max_job_seconds,
            seed,
            target_path=target_path,
            port=port,
            mode=mode,
            dataset_cache=dataset_cache,
            prepared_round=prepared_round,
        )
    if evidence.timings is not None:
        evidence = evidence.model_copy(
            update={
                "timings": evidence.timings.model_copy(
                    update={
                        "artifact_staging_seconds": staging_seconds,
                        "total_seconds": time.perf_counter() - staging_started,
                    }
                ),
                "started_at": staging_wall,
                "finished_at": time.time(),
            }
        )
    base = dict(
        **evidence.model_dump(),
        mode=mode,
        competition_id=target.identity,
        target_id=target.model_id,
        target_revision=target.revision,
        target_source="local_override" if target_path else "pinned_cache",
        local_target_path=requested_target_path,
        resolved_target_path=str(target_path) if target_path else None,
        seed=seed,
        checkpoints=files,
        notes=(
            "Local measurements and checks do not establish validator qualification or awards.",
            (
                "Head-only mode does not measure references, compare correctness, "
                "or determine speedup or a winner."
                if mode == "head"
                else "Threshold comparisons cover only the reference heads provided to this run."
            ),
        ),
    )
    return summarize_report(target, LocalEvaluationReport(**base, assessment="inconclusive"))


def summarize_report(target: TargetConfig, report: LocalEvaluationReport) -> LocalEvaluationReport:
    """Derive local checks from raw evidence, also when reading a transported report."""
    report = report.model_copy(
        update={
            "assessment": "inconclusive",
            "score": None,
            "scores": {},
            "threshold": None,
            "required_score": None,
            "lower_gain": None,
            "head_metrics": {},
        }
    )
    if report.mode == "head":
        report = report.model_copy(update={"decision": None})
    if report.status == "complete":
        try:
            validate_selection(target, report.dataset_selection, seed=report.seed)
            metrics = head_metrics(target, report)
        except ValueError:
            return report.model_copy(
                update={"status": "inconclusive", "error_code": "incomplete_measurements"}
            )
        report = report.model_copy(update={"head_metrics": metrics})
    if report.mode == "head":
        if (
            set(report.checkpoints) != {"candidate"}
            or any(m.arm != "candidate" for m in report.measurements)
            or report.reference_before
            or report.reference_after
            or report.correctness_passed
            or report.reference_correctness_passed
            or report.qualification_passed
        ):
            return report.model_copy(
                update={"status": "inconclusive", "error_code": "invalid_head_only_evidence"}
            )
        if report.status == "complete":
            return report.model_copy(update={"assessment": "head_measured"})
        return report
    arms = ("target", *report.checkpoints)
    decision = score_benchmark(target, report, arms)
    report = report.model_copy(update={"decision": decision})
    if report.status != "complete":
        return report
    try:
        summary = throughput_summary(target, report.measurements, arms)
    except ValueError:
        return report.model_copy(
            update={"status": "inconclusive", "error_code": "incomplete_measurements"}
        )
    return report.model_copy(
        update={
            "assessment": {
                "won": "passes_local_checks",
                "lost": "fails_local_checks",
                "inconclusive": "inconclusive",
            }[decision.outcome],
            "score": summary.scores["candidate"],
            "scores": summary.scores,
            "threshold": summary.threshold,
            "required_score": (1 + target.policy.improvement_margin) * summary.threshold,
            "lower_gain": summary.lower_gain,
        }
    )


def head_metrics(target: TargetConfig, report: LocalEvaluationReport) -> dict[str, HeadMetrics]:
    if (
        report.hardware is None
        or not hardware_matches(target, report.hardware)
        or report.finished_at <= report.started_at
    ):
        raise ValueError("Hardware/runtime or timing mismatch")
    rows = {(m.workload, m.repeat): m for m in report.measurements if m.arm == "candidate"}
    expected = {
        (w.name, repeat) for w in target.workloads for repeat in range(target.policy.repeats)
    }
    if (
        len(rows) != sum(m.arm == "candidate" for m in report.measurements)
        or rows.keys() != expected
    ):
        raise ValueError("Incomplete or duplicate candidate measurements")
    summaries = {}
    for workload in target.workloads:
        samples = [rows[workload.name, repeat] for repeat in range(target.policy.repeats)]
        budget = workload.requests * workload.output_tokens
        if any(m.output_tokens != budget or m.expected_tokens != budget for m in samples):
            raise ValueError("Incomplete candidate output budget")
        proposed = sum(m.drafted_tokens or 0 for m in samples)
        steps = sum(m.verify_steps or 0 for m in samples)
        summaries[workload.name] = HeadMetrics(
            throughput_tps=math.exp(mean(math.log(m.throughput) for m in samples)),
            p95_latency_seconds=max(m.p95_latency_seconds for m in samples),
            output_tokens=sum(m.output_tokens for m in samples),
            elapsed_seconds=sum(m.elapsed_seconds for m in samples),
            acceptance_rate=(
                sum(m.accepted_tokens or 0 for m in samples) / proposed
                if proposed
                and all(
                    m.accepted_tokens is not None and m.drafted_tokens is not None for m in samples
                )
                else None
            ),
            average_acceptance_length=(
                sum(m.output_tokens for m in samples) / steps
                if steps and all(m.verify_steps is not None for m in samples)
                else None
            ),
        )
    return summaries
