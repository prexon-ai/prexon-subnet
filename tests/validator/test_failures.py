import pytest

from shared.checkpoints import checkpoint_schema
from shared.contracts import Commitment, Qualification
from tests.factories import evidence, job, target
from tests.integration.test_controller import setup_pipeline
from tests.validator.test_state import admit
from validator.competition import score_evidence
from validator.controller import Controller


def test_second_target_can_start_while_first_is_running(tmp_path):
    controller, state, store, chain, provider, first = setup_pipeline(tmp_path)
    first = first.model_copy(update={"allocation": 20})
    second = target(2, 80)
    for t in (first, second):
        q = Qualification(
            profile_id=t.qualification_identity,
            engine_version=t.engine_version,
            approved_by="operator",
            evidence_description="GPU qualification",
            target_preserving_verification_reviewed=True,
            checkpoint_schemas=(checkpoint_schema(tmp_path / "head"),),
        )
        digest = store.put_bytes(q.model_dump_json().encode())
        t = t.model_copy(update={"qualification_artifact": digest})
        if t.slot == 1:
            first = t
        else:
            second = t
    settings = controller.settings.model_copy(update={"targets": (first, second)})
    controller = Controller(settings, state, store, chain, provider)
    for t in (first, second):
        admit(state, t, t.slot)
    controller.tick(21600)
    assert len(state.active_sessions()) == 2
    assert len(provider.rents) == 2
    assert {j.target.slot for j in provider.jobs} == {1, 2}


def test_failed_pod_deletion_retains_session_authority(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    deleted = provider.delete

    def fail(pod_id):
        raise TimeoutError("provider unavailable")

    provider.delete = fail
    controller.tick(40000)
    assert state.active_sessions()[0]["state"] == "stopping"
    stopping_id = state.active_sessions()[0]["id"]
    assert len(provider.rents) == 1
    provider.delete = deleted
    controller.tick(40001)
    assert len(state.active_sessions()) == 1
    assert state.active_sessions()[0]["id"] != stopping_id
    assert len(provider.rents) == 2
    assert state.submissions()[0]["submitted_at"] == 0


def test_repackaged_identical_weights_do_not_create_a_second_claim(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(1000)
    row = state.submissions()[0]
    from shared.contracts import SubmissionManifest

    original = SubmissionManifest.model_validate_json(row["manifest"])
    changed = original.model_copy(
        update={
            "provenance": original.provenance.model_copy(
                update={"training_description": "changed metadata"}
            )
        }
    )
    c = Commitment(
        netuid=1, block=100, block_hash="another", hotkey="copy", uid=2, artifact="f" * 64
    )
    fingerprint = next(iter(state.fingerprint_bank().values()))
    state.admit(c, changed, t, 1001, fingerprint=fingerprint)
    assert state.ready_count(t.identity) == 1
    assert state.submissions()[-1]["status"] == "rejected"


def test_missing_public_evidence_and_bad_slo_do_not_win():
    from shared.config import PublicReference
    from tests.factories import job
    from validator.competition import score_evidence

    t = target().model_copy(update={"public_reference": PublicReference(artifact="e" * 64)})
    j = job(t)
    assert score_evidence(j, evidence(j)).outcome == "inconclusive"
    t = target()
    w = t.workloads[0].model_copy(update={"max_p95_latency_seconds": 0.1})
    t = t.model_copy(update={"workloads": (w, t.workloads[1])})
    j = job(t)
    assert score_evidence(j, evidence(j)).reason == "service_requirement_failed"


def test_catching_up_chain_history_uses_fresh_ownership_for_emissions(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    j = provider.jobs[0]
    result = evidence(j).model_copy(update={"started_at": 21601, "finished_at": 21602})
    store.write_bytes(f"jobs/{j.job_id}/result.json", result.model_dump_json().encode())
    controller.tick(21603)
    chain.current_owners = lambda: {1: "replacement", 0: "burn"}
    # Replay snapshots still identify the old miner, but the UID has changed now.
    controller.tick(22201)
    assert chain.published[-1] == {0: 1.0}


def test_pending_provider_price_raise_stops_rental(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    provider.status = lambda _: {"status": "RUNNING", "spend_to_date": 0, "pending_price": 1000}
    controller.tick(21601)
    assert provider.deleted
    assert not state.active_sessions()


@pytest.mark.parametrize("field", ["price", "pending_price", "spend_to_date"])
@pytest.mark.parametrize("value", [float("nan"), -1])
def test_invalid_provider_billing_stops_rental_before_launching_more_work(tmp_path, field, value):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    first_job = provider.jobs[0]
    provider.status = lambda _: {"status": "RUNNING", "spend_to_date": 0, field: value}
    controller.tick(21601)
    assert provider.deleted
    assert not state.active_sessions()
    assert provider.jobs == [first_job]


def test_invalid_billing_retains_rental_authority_until_deletion_succeeds(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    original = state.active_sessions()[0]
    delete = provider.delete

    def unavailable(pod_id):
        raise TimeoutError("provider unavailable")

    provider.status = lambda _: {"status": "RUNNING", "price": float("nan")}
    provider.delete = unavailable
    controller.tick(21601)
    retained = state.active_sessions()[0]
    assert retained["state"] == "stopping"
    assert retained["id"] == original["id"]
    assert retained["pod_id"] == original["pod_id"]
    assert len(provider.rents) == len(provider.jobs) == 1
    provider.delete = delete
    controller.stop_session(retained, 21602)
    assert provider.deleted == [original["pod_id"]]
    assert not state.active_sessions()


def test_disabling_all_models_clears_previously_initialized_weights(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    j = provider.jobs[0]
    result = evidence(j).model_copy(update={"started_at": 21601, "finished_at": 21602})
    store.write_bytes(f"jobs/{j.job_id}/result.json", result.model_dump_json().encode())
    controller.tick(21603)
    disabled = Controller(
        controller.settings.model_copy(update={"targets": ()}), state, store, chain, provider
    )
    controller.tick(22201)
    assert chain.published[-1][1] == 0.2
    disabled.tick(22802)
    assert chain.published[-1] == {0: 1.0}


def test_reference_failure_is_inconclusive_instead_of_candidate_loss():
    j = job()
    result = evidence(j).model_copy(update={"reference_correctness_passed": False})
    assert score_evidence(j, result).outcome == "inconclusive"


def test_crash_between_job_creation_and_upload_is_recovered(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    j = provider.jobs[0]
    blob = tmp_path / "artifacts" / "jobs" / j.job_id / "job.json"
    blob.unlink()
    restarted = Controller(controller.settings, state, store, chain, provider)
    restarted.tick(21601)
    assert blob.exists()
    assert len(provider.rents) == 1
    assert state.running_job(j.session_id).job_id == j.job_id


def test_snapshot_write_failure_does_not_stop_evaluation_reconciliation(tmp_path, monkeypatch):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)

    def fail(*args, **kwargs):
        raise OSError("dashboard disk unavailable")

    monkeypatch.setattr("validator.controller.publish_snapshot", fail)
    controller.tick(21600)
    j = provider.jobs[0]
    result = evidence(j).model_copy(update={"started_at": 21601, "finished_at": 21602})
    store.write_bytes(f"jobs/{j.job_id}/result.json", result.model_dump_json().encode())
    controller.tick(21603)
    assert len(state.awards()) == 1


def test_pending_profile_keeps_frozen_workloads_after_config_change(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(1000)  # Discover and admit without reaching the six-hour trigger.
    changed = t.model_copy(update={"context_length": 4096})
    restarted = Controller(
        controller.settings.model_copy(update={"targets": (changed,)}),
        state,
        store,
        chain,
        provider,
    )
    restarted.tick(21600)
    assert provider.jobs[0].target.context_length == 8192
    assert provider.jobs[0].target.identity == t.identity
