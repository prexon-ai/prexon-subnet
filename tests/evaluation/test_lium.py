import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation.runner import evaluate
from shared.config import Settings
from tests.factories import target, write_profile


@pytest.fixture
def rental(tmp_path, monkeypatch, pod):
    from evaluation import lium

    head, snapshot, _ = pod
    config = target().model_copy(update={"allocation": 20, "slot": 2})
    private_key = tmp_path / "key"
    private_key.write_text("test-key")
    settings = Settings(
        role="validator", targets=(config,), session_seconds=3600, ssh_private_key=private_key
    )
    output = tmp_path / "report.json"
    calls = []
    remote = tmp_path / "remote"

    class Provider:
        def __init__(self, *args, **kwargs):
            pass

        def build_request(self, session, profile):
            assert profile.identity == config.identity
            return {"pod_name": "prexon-" + session["id"], "termination_hours": 1}

        def quote(self, request):
            calls.append("quote")
            return {"dry_run": True, "price_per_hour": 1}

        def rent(self, session, request):
            calls.append("rent")
            journals = [json.loads(p.read_bytes()) for p in tmp_path.rglob("*.lium.json")]
            assert next(j for j in journals if j["id"] == session)["request"] == request
            return "pod-1"

        def recover(self, session):
            calls.append("recover")
            return "pod-1"

        def status(self, pod_id):
            return {"status": "RUNNING", "ssh_connect_cmd": "ssh root@example"}

        def delete(self, pod_id):
            calls.append("delete")

    class Transport:
        def __init__(self, settings, descriptor, tick):
            self.tick = tick

        def run(self, argv, *, source=None, destination=None, limit=0, timeout_seconds=None):
            from evaluation.remote import install

            self.tick()
            action = argv[3]
            calls.append(action)
            if action == "upload":
                with source.open("rb") as archive:
                    install(remote, archive, argv[-1])
            elif action == "start":
                request = json.loads((remote / "input/request.json").read_text())
                # These are the exact options passed to the real GPU entrypoint.
                from shared.config import TargetConfig

                transported = TargetConfig.model_validate(request["target"])
                assert transported.identity == config.identity
                report = asyncio.run(
                    evaluate(
                        transported,
                        remote / "input/candidate",
                        tmp_path / "logs",
                        target_path=Path(request["target_path"])
                        if request["target_path"]
                        else snapshot,
                        seed=request["seed"],
                        mode=request.get("mode", "full"),
                        public_checkpoint=remote / "input/public"
                        if "public" in request["checkpoints"]
                        else None,
                        incumbent_checkpoint=remote / "input/incumbent"
                        if "incumbent" in request["checkpoints"]
                        else None,
                    )
                )
                report = report.model_copy(
                    update={
                        "execution_id": request["execution_id"],
                        "target_source": "local_override"
                        if request["target_path"]
                        else "pinned_cache",
                        "local_target_path": report.local_target_path
                        if request["target_path"]
                        else None,
                    }
                )
                (remote / "report.json").write_text(report.model_dump_json())
            elif action == "report":
                destination.write_bytes((remote / "report.json").read_bytes())
            elif action == "logs":
                destination.write_bytes(b"logs archive")
            return 0

    monkeypatch.setattr(lium, "LiumClient", Provider)
    monkeypatch.setattr(lium, "SshTransport", Transport)
    return lium, settings, config, head, output, calls, Provider, remote


@pytest.fixture
def uncertain_cleanup(rental, monkeypatch):
    lium, settings, config, head, output, calls, provider, remote = rental
    output = output.parent / "reports/head/original.json"
    lium.evaluate_on_lium(settings, config, head, output, api_key="secret", mode="head")
    state_path = output.with_suffix(".lium.json")
    state = json.loads(state_path.read_bytes())
    state.update(deadline=1000, pod_id=None, cleanup="pending")
    state_path.write_text(json.dumps(state))
    calls.clear()
    monkeypatch.setattr(provider, "recover", lambda *args: None)
    monkeypatch.setattr(lium, "time", SimpleNamespace(time=lambda: 4660))
    return lium, settings, config, head, output, calls, provider, remote


def test_rental_transfers_same_profile_returns_evidence_and_deletes_pod(rental):
    lium, settings, config, head, output, calls, _, _ = rental
    report = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert report.score == pytest.approx(2)
    assert report.authoritative is False
    assert report.backend == "lium"
    assert report.rental_cleanup == "deleted"
    assert report.competition_id == config.identity
    assert calls.count("rent") == 1
    assert calls[-1] == "delete"
    journal = json.loads(output.with_suffix(".lium.json").read_text())
    assert '"api_key"' not in json.dumps(journal)
    assert journal["phase"] == "finished"
    again = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert again.execution_id == report.execution_id
    assert calls.count("rent") == 1


def test_rental_failure_and_cleanup_retry_preserve_report(rental, monkeypatch):
    lium, settings, config, head, output, calls, provider, _ = rental

    def fail_delete(self, pod_id):
        raise RuntimeError("temporary provider failure")

    monkeypatch.setattr(provider, "delete", fail_delete)
    report = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert report.score == pytest.approx(2)
    assert report.rental_cleanup == "pending"
    assert json.loads(output.read_text())["score"] == pytest.approx(2)
    monkeypatch.setattr(provider, "delete", lambda *args: calls.append("delete"))
    assert lium.cleanup_evaluation(settings, output.with_suffix(".lium.json"), api_key="secret")
    assert calls.count("rent") == 1


def test_changed_inputs_are_rejected_on_resume(rental):
    lium, settings, config, head, output, calls, _, _ = rental
    lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    with pytest.raises(ValueError, match="inputs"):
        lium.evaluate_on_lium(settings, config, head, output, api_key="secret", seed="changed")
    assert calls.count("rent") == 1


def test_invalid_remote_report_is_inconclusive_and_pod_deleted(rental, monkeypatch):
    lium, settings, config, head, output, calls, _, remote = rental
    original = lium.SshTransport.run

    def stale(self, argv, **kwargs):
        result = original(self, argv, **kwargs)
        if argv[3] == "report":
            data = json.loads(kwargs["destination"].read_bytes())
            data["execution_id"] = "stale"
            kwargs["destination"].write_text(json.dumps(data))
        return result

    monkeypatch.setattr(lium.SshTransport, "run", stale)
    report = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert report.assessment == "inconclusive"
    assert report.score is None
    assert calls[-1] == "delete"


def test_keyboard_interrupt_still_deletes_owned_pod(rental, monkeypatch):
    lium, settings, config, head, output, calls, _, _ = rental

    def cancel(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(lium.SshTransport, "run", cancel)
    result = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert result.assessment == "inconclusive"
    assert calls[-1] == "delete"


@pytest.mark.parametrize(
    "billing",
    [{"price": 100}, {"pending_price": 100}, {"spend_to_date": 100}, {"spend_to_date": -1}],
)
def test_provider_price_and_spend_bounds_stop_evaluation(rental, monkeypatch, billing):
    lium, settings, config, head, output, calls, provider, _ = rental
    monkeypatch.setattr(
        provider,
        "status",
        lambda *args: {"status": "RUNNING", "ssh_connect_cmd": "ssh root@example", **billing},
    )
    report = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert report.assessment == "inconclusive"
    assert "start" not in calls
    assert calls[-1] == "delete"


def test_ambiguous_rent_recovers_identity_and_deletes_without_renting_again(rental, monkeypatch):
    lium, settings, config, head, output, calls, provider, _ = rental

    def lost_response(self, session, request):
        calls.append("rent")
        raise TimeoutError("response lost after provider created rental")

    monkeypatch.setattr(provider, "rent", lost_response)
    report = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert report.assessment == "inconclusive"
    assert report.pod_id == "pod-1"
    assert calls == ["rent", "recover", "delete"]
    lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert calls.count("rent") == 1


def test_crashed_active_run_resumes_same_pod_without_second_rental(rental):
    lium, settings, config, head, output, calls, _, _ = rental
    report = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    state_path = output.with_suffix(".lium.json")
    state = json.loads(state_path.read_bytes())
    state.update(phase="active", cleanup="not_applicable", pod_id=None)
    state_path.write_text(json.dumps(state))
    output.unlink()
    resumed = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert resumed.execution_id == report.execution_id
    assert calls.count("rent") == 1
    assert "recover" in calls


def test_uncertain_active_rental_stays_pending_when_provider_cannot_find_it(rental, monkeypatch):
    lium, settings, config, head, output, calls, provider, _ = rental
    lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    state_path = output.with_suffix(".lium.json")
    state = json.loads(state_path.read_bytes())
    state.update(phase="active", cleanup="not_applicable", pod_id=None)
    state_path.write_text(json.dumps(state))
    output.unlink()
    calls.clear()
    monkeypatch.setattr(provider, "recover", lambda *args: None)

    report = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")

    assert report.assessment == "inconclusive"
    assert report.rental_cleanup == "pending"
    assert report.pod_id is None
    assert "rent" not in calls
    assert "start" not in calls
    assert json.loads(state_path.read_bytes())["cleanup"] == "pending"


@pytest.mark.parametrize(
    ("now", "expected", "success"),
    [(4659, "pending", False), (4660, "expiry_unconfirmed", True)],
)
def test_uncertain_cleanup_waits_for_saved_provider_termination_bound(
    uncertain_cleanup, monkeypatch, now, expected, success
):
    lium, settings, _, _, output, _, _, _ = uncertain_cleanup
    monkeypatch.setattr(lium.time, "time", lambda: now)
    state_path = output.with_suffix(".lium.json")

    assert lium.cleanup_evaluation(settings, state_path, api_key="secret") is success
    assert json.loads(state_path.read_bytes())["cleanup"] == expected
    assert json.loads(output.read_bytes())["rental_cleanup"] == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [("request", None), ("request", {"termination_hours": 0}), ("deadline", None)],
)
def test_uncertain_cleanup_requires_a_usable_saved_termination_bound(
    uncertain_cleanup, field, value
):
    lium, settings, _, _, output, _, _, _ = uncertain_cleanup
    state_path = output.with_suffix(".lium.json")
    state = json.loads(state_path.read_bytes())
    state[field] = value
    state_path.write_text(json.dumps(state))

    assert not lium.cleanup_evaluation(settings, state_path, api_key="secret")
    assert json.loads(state_path.read_bytes())["cleanup"] == "pending"


@pytest.mark.parametrize("operation", ["recover", "delete"])
def test_uncertain_cleanup_keeps_provider_errors_pending_after_termination_bound(
    uncertain_cleanup, monkeypatch, operation
):
    lium, settings, _, _, output, _, provider, _ = uncertain_cleanup
    monkeypatch.setattr(provider, "recover", lambda *args: "pod-1")

    def unavailable(*args):
        raise RuntimeError("provider temporarily unavailable")

    monkeypatch.setattr(provider, operation, unavailable)
    state_path = output.with_suffix(".lium.json")
    assert not lium.cleanup_evaluation(settings, state_path, api_key="secret")
    assert json.loads(output.read_bytes())["rental_cleanup"] == "pending"


def test_uncertain_cleanup_deletes_a_pod_found_on_a_later_attempt(uncertain_cleanup, monkeypatch):
    lium, settings, _, _, output, calls, provider, _ = uncertain_cleanup
    monkeypatch.setattr(lium.time, "time", lambda: 4659)
    state_path = output.with_suffix(".lium.json")
    assert not lium.cleanup_evaluation(settings, state_path, api_key="secret")

    monkeypatch.setattr(provider, "recover", lambda *args: "pod-1")
    assert lium.cleanup_evaluation(settings, state_path, api_key="secret")
    assert calls == ["delete"]
    report = json.loads(output.read_bytes())
    assert report["rental_cleanup"] == "deleted"
    assert report["pod_id"] == "pod-1"


def test_cli_releases_uncertain_output_directory_only_after_termination_bound(
    uncertain_cleanup, monkeypatch, capsys
):
    import shutil
    import sys

    from evaluation.cli import main, unfinished

    lium, settings, config, head, output, calls, _, remote = uncertain_cleanup
    monkeypatch.setattr("evaluation.cli.load_settings", lambda *args: settings)
    monkeypatch.setenv("LIUM_API_KEY", "secret")
    profile = write_profile(
        output.parents[2] / "local.json",
        head,
        backend="lium",
        target=config.model_dump(mode="json"),
    )
    argv = ["prexon-evaluate", "--profile", str(profile)]
    monkeypatch.setattr(lium.time, "time", lambda: 4659)
    monkeypatch.setattr(sys, "argv", [*argv, "--cleanup-only"])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    assert unfinished(output.parents[1]) == [output.with_suffix(".lium.json")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    assert "rent" not in calls

    monkeypatch.setattr(lium.time, "time", lambda: 4660)
    monkeypatch.setattr(sys, "argv", [*argv, "--cleanup-only"])
    main()
    assert "reconciliation complete" in capsys.readouterr().out.lower()
    assert unfinished(output.parents[1]) == []
    shutil.rmtree(remote)  # A new rental starts with a fresh remote filesystem.
    monkeypatch.setattr(sys, "argv", argv)
    main()
    assert calls.count("rent") == 1
    assert len(list(output.parent.glob("*.lium.json"))) == 2


def test_expired_run_cleans_up_without_starting_gpu_work(rental, monkeypatch):
    lium, settings, config, head, output, calls, _, _ = rental
    lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    state_path = output.with_suffix(".lium.json")
    state = json.loads(state_path.read_bytes())
    state.update(phase="active", deadline=1)
    state_path.write_text(json.dumps(state))
    output.unlink()
    calls.clear()
    result = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert result.assessment == "inconclusive"
    assert calls == ["delete"]


@pytest.mark.parametrize("mode", ["head", "full"])
def test_saved_evidence_survives_expired_active_rental_recovery(rental, mode):
    lium, settings, config, head, output, calls, _, _ = rental
    original = lium.evaluate_on_lium(settings, config, head, output, api_key="secret", mode=mode)
    state_path = output.with_suffix(".lium.json")
    state = json.loads(state_path.read_bytes())
    state.update(phase="active", cleanup="not_applicable", deadline=1)
    state_path.write_text(json.dumps(state))
    calls.clear()
    resumed = lium.evaluate_on_lium(settings, config, head, output, api_key="secret", mode=mode)
    assert resumed.assessment == original.assessment
    assert resumed.measurements == original.measurements
    assert resumed.execution_id == original.execution_id
    assert resumed.score == original.score
    assert resumed.head_metrics == original.head_metrics
    assert resumed.rental_cleanup == "deleted"
    assert calls == ["delete"]


def test_lium_transports_head_mode_without_reference_heads(rental):
    lium, settings, config, head, output, calls, _, remote = rental
    result = lium.evaluate_on_lium(
        settings,
        config,
        head,
        output,
        api_key="secret",
        mode="head",
        public_checkpoint=output.parent / "missing-baseline",
    )
    assert result.mode == "head"
    assert result.assessment == "head_measured"
    assert result.score is None and result.decision is None
    request = json.loads((remote / "input/request.json").read_bytes())
    assert request["mode"] == "head"
    assert set(request["checkpoints"]) == {"candidate"}
    assert calls[-1] == "delete"


def test_missing_ssh_key_is_rejected_before_rental(rental):
    lium, settings, config, head, output, calls, _, _ = rental
    settings = settings.model_copy(update={"ssh_private_key": output.parent / "missing"})
    with pytest.raises(ValueError, match="SSH"):
        lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert "rent" not in calls


def test_remote_success_claim_without_supporting_evidence_is_rejected(rental, monkeypatch):
    lium, settings, config, head, output, calls, _, _ = rental
    original = lium.SshTransport.run

    def forged(self, argv, **kwargs):
        code = original(self, argv, **kwargs)
        if argv[3] == "report":
            data = json.loads(kwargs["destination"].read_bytes())
            data.update(status="inconclusive", measurements=[], correctness_passed=False)
            kwargs["destination"].write_text(json.dumps(data))
        return code

    monkeypatch.setattr(lium.SshTransport, "run", forged)
    result = lium.evaluate_on_lium(settings, config, head, output, api_key="secret")
    assert result.assessment == "inconclusive"
    assert result.score is None
    assert calls[-1] == "delete"


@pytest.mark.parametrize("expired", [True, False])
def test_dataset_rental_recovery_uses_frozen_plan_without_original_corpus(
    rental, tmp_path, monkeypatch, expired
):
    import hashlib
    import time

    from shared.datasets import RoundPlan
    from tests.evaluation.test_datasets import dataset_target, prepared_pool

    lium, settings, _, head, output, calls, _, remote = rental
    _, manifest = prepared_pool(tmp_path / "datasets")
    config = dataset_target(manifest)
    request, _ = lium.inputs(
        config, head, None, None, "42", 30000, None, "head", dataset_cache=tmp_path / "datasets"
    )
    state = {
        "id": "e" * 32,
        "signature": hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest(),
        "inputs": request,
        "output": str(output.resolve()),
        "started_at": 1,
        "deadline": 1 if expired else time.time() + 3600,
        "max_cost": 100,
        "phase": "active",
        "rent_attempted": True,
        "pod_id": "pod-1",
        "cleanup": "not_applicable",
        "request": {"termination_hours": 1},
    }
    output.with_suffix(".lium.json").write_text(json.dumps(state))
    settings = settings.model_copy(update={"dataset_cache": tmp_path / "missing-corpus"})
    original = lium.SshTransport.run

    def run(self, argv, **kwargs):
        if argv[3] != "start":
            return original(self, argv, **kwargs)
        calls.append("start")
        frozen = json.loads((remote / "input/request.json").read_bytes())
        report = asyncio.run(
            evaluate(
                config,
                remote / "input/candidate",
                tmp_path / "logs",
                seed=frozen["seed"],
                mode="head",
                prepared_round=RoundPlan.model_validate(frozen["dataset_plan"]),
            )
        ).model_copy(update={"execution_id": frozen["execution_id"]})
        (remote / "report.json").write_text(report.model_dump_json())
        return 0

    monkeypatch.setattr(lium.SshTransport, "run", run)
    resumed = lium.evaluate_on_lium(settings, config, head, output, api_key="secret", mode="head")
    assert resumed.execution_id == "e" * 32
    if expired:
        assert resumed.assessment == "inconclusive"
        assert calls == ["delete"]
    else:
        assert resumed.assessment == "head_measured"
        assert (
            resumed.dataset_selection
            == RoundPlan.model_validate(request["dataset_plan"]).evidence()
        )
        assert "rent" not in calls
        assert calls[-1] == "delete"
