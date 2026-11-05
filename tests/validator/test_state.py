from concurrent.futures import ThreadPoolExecutor

import pytest

from shared.contracts import Commitment, Decision, SubmissionManifest
from tests.factories import evidence, provenance, target
from validator.state import State


def admit(state, t, index, *, submitted_at=0):
    import struct
    import tempfile
    from pathlib import Path

    from shared.contracts import Commitment
    from tests.factories import make_checkpoint
    from validator.similarity import fingerprint

    c = Commitment(
        netuid=1,
        block=100 + index,
        block_hash=f"hash{index}",
        uid=index,
        hotkey=f"miner{index}",
        artifact=f"{index:064x}",
        submitted_at=submitted_at,
    )
    manifest = SubmissionManifest(
        target_id=t.model_id,
        target_revision=t.revision,
        competition_id=t.identity,
        files=[
            {"path": "config.json", "size": 1, "sha256": f"{index:064x}"},
            {"path": "model.safetensors", "size": 1, "sha256": "e" * 64},
        ],
        provenance=provenance(),
    )
    with tempfile.TemporaryDirectory() as temporary:
        head = make_checkpoint(Path(temporary) / "head")
        tensor = head / "model.safetensors"
        tensor.write_bytes(tensor.read_bytes()[:-8] + struct.pack("<ff", index, index + 1))
        evidence = fingerprint(head, t)
    state.admit(c, manifest, t, 1000, fingerprint=evidence)
    return c


def test_queue_nine_vs_ten_and_fifo(tmp_path):
    s = State(tmp_path / "state.sqlite")
    t = target()
    for i in range(9, 0, -1):
        admit(s, t, i)
    assert s.claim_session(t, 1000, 21600, 10, 3600) is None
    admit(s, t, 10)
    session = s.claim_session(t, 1000, 21600, 10, 3600)
    assert session is not None
    assert s.claim_session(t, 1000, 21600, 10, 3600) is None
    assert s.next_submission(session["id"])["commitment"].uid == 1


def test_schedule_boundary_and_recovery_after_restart(tmp_path):
    path = tmp_path / "state.sqlite"
    s = State(path)
    t = target()
    admit(s, t, 1)
    assert s.claim_session(t, 21599, 21600, 10, 3600) is None
    session = s.claim_session(t, 21600, 21600, 10, 3600)
    s.set_pod(session["id"], "pod-123")
    recovered = State(path)
    assert recovered.active_sessions()[0]["pod_id"] == "pod-123"
    assert recovered.claim_session(t, 22000, 21600, 10, 3600) is None
    recovered.finish_session(session["id"], 22001)
    assert recovered.ready_count(t.identity) == 1


def test_atomic_claim_under_competing_controllers(tmp_path):
    path = tmp_path / "state.sqlite"
    s = State(path)
    t = target()
    admit(s, t, 1)

    def claim(_):
        local = State(path)
        return local.claim_session(t, 21600, 21600, 10, 3600)

    with ThreadPoolExecutor(2) as pool:
        claims = list(pool.map(claim, range(2)))
    assert sum(x is not None for x in claims) == 1


def test_promotion_is_idempotent_and_inconclusive_preserves_priority(tmp_path):
    s = State(tmp_path / "state.sqlite")
    t = target()
    first, second = admit(s, t, 1), admit(s, t, 2)
    session = s.claim_session(t, 21600, 21600, 10, 3600)
    j = s.create_job(session["id"], candidate_url="file:///candidate", now=21601)
    assert j.commitment == first
    r = evidence(j).model_copy(update={"started_at": 21602, "finished_at": 21603})
    d = Decision(outcome="won", reason="qualified", score=2, threshold=1, lower_gain=2)
    s.complete_job(j, r, d, 21604)
    s.complete_job(j, r, d, 21605)
    assert len(s.awards()) == 1
    assert s.awards()[0].promoted_at == 21604
    assert s.incumbent(t) == first.artifact
    j2 = s.create_job(session["id"], candidate_url="file:///second", now=21606)
    assert j2.commitment == second
    assert j2.incumbent_artifact == first.artifact
    s.complete_job(j2, evidence(j2), Decision(outcome="inconclusive", reason="infra"), 21607)
    assert s.next_submission(session["id"])["commitment"] == second
    assert len(s.awards()) == 1


def test_stale_incumbent_and_duplicate_content_rejected(tmp_path):
    s = State(tmp_path / "state.sqlite")
    t = target()
    c = admit(s, t, 1)
    assert s.ready_count(t.identity) == 1
    # Rediscovery cannot requeue a commitment.
    admit(s, t, 1)
    assert s.ready_count(t.identity) == 1
    session = s.claim_session(t, 21600, 21600, 10, 3600)
    j = s.create_job(session["id"], candidate_url="file:///candidate", now=21601)
    bad = j.model_copy(update={"incumbent_artifact": "f" * 64})
    with pytest.raises(ValueError, match="stale"):
        s.complete_job(bad, evidence(bad), Decision(outcome="won", reason="bad", score=2), 21602)
    assert s.incumbent(t) is None
    assert c.identity == j.commitment.identity


def test_profile_transition_cannot_jump_ahead_of_older_same_target_submission(tmp_path):
    s = State(tmp_path / "state.sqlite")
    a = target()
    b = a.model_copy(update={"context_length": 4096})
    first = admit(s, a, 1)
    second = admit(s, b, 2)
    admit(s, a, 3)
    session = s.claim_session(a, 21600, 21600, 10, 3600)
    j = s.create_job(session["id"], candidate_url="file:///one", now=21601)
    assert j.commitment == first
    s.complete_job(j, evidence(j), Decision(outcome="lost", reason="insufficient"), 21602)
    # The session must end before changing hardware/profile; it cannot run A3 ahead of B2.
    assert s.next_submission(session["id"]) is None
    s.finish_session(session["id"], 21603)
    assert s.claim_session(a, 43200, 21600, 10, 3600) is None
    next_session = s.claim_session(b, 43200, 21600, 10, 3600)
    assert s.next_submission(next_session["id"])["commitment"] == second


def test_retrying_missing_artifacts_cannot_starve_later_chain_discoveries(tmp_path):
    state = State(tmp_path / "state.sqlite")
    commitments = tuple(
        Commitment(
            netuid=1,
            block=i + 1,
            block_hash=f"hash{i}",
            hotkey=f"miner{i}",
            uid=i,
            artifact=f"{i:064x}",
        )
        for i in range(101)
    )
    state.discover(commitments, 200)
    first = state.pending_admissions()
    assert len(first) == 100
    # None of the first 100 resolves (e.g. storage outage or unknown profile).
    second = state.pending_admissions()
    assert commitments[-1].identity in {c.identity for c in second}


def test_six_hours_is_measured_from_the_oldest_submission_not_last_run(tmp_path):
    state = State(tmp_path / "state.sqlite")
    config = target()
    admit(state, config, 1, submitted_at=1000)
    # Existing helper admits at time 1000; a controller's age is irrelevant.
    assert state.claim_session(config, 21600, 21600, 10, 3600) is None
    assert state.claim_session(config, 22599, 21600, 10, 3600) is None
    assert state.claim_session(config, 22600, 21600, 10, 3600) is not None


def test_queue_age_is_retained_across_restart_and_failed_session(tmp_path):
    path = tmp_path / "state.sqlite"
    state = State(path)
    config = target()
    admit(state, config, 1, submitted_at=1000)
    recovered = State(path)
    session = recovered.claim_session(config, 22600, 21600, 10, 3600)
    assert session is not None
    recovered.finish_session(session["id"], 22601)
    assert recovered.claim_session(config, 22602, 21600, 10, 3600) is not None
