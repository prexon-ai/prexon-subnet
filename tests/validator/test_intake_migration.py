"""Legacy work loses readiness until mandatory CPU intake has completed."""

import hashlib
import json

import pytest

from tests.factories import target
from tests.validator.test_similarity_admission import calibrated_target, candidate
from validator.similarity import compare_full
from validator.state import State


def legacy_v4_database(tmp_path, *, session_phase=None):
    path = tmp_path / "legacy.sqlite"
    state, config = State(path), target()
    commitments = []
    for index in (1, 2):
        claim, manifest, evidence = candidate(tmp_path, config, index, value=float(index))
        state.discover((claim,), claim.block, now=1000)
        state.admit(claim, manifest, config, 1000, fingerprint=evidence)
        state.resolve_admission(claim, "admitted")
        commitments.append(claim)
    session = job = None
    if session_phase:
        session = state.claim_session(config, 22600, 21600, 10, 3600)
        state.set_pod(session["id"], "legacy-pod")
        if session_phase == "running":
            job = state.create_job(session["id"], candidate_url="file:///legacy", now=22601)
    # Recreate the genuine pre-intake table shape and version. Legacy rows had no
    # mandatory fingerprint/verdict, discovery time, or submitted-at columns.
    state.connection.execute("DROP TABLE similarity_verdicts")
    state.connection.execute("DROP TABLE fingerprints")
    state.connection.execute("ALTER TABLE submissions DROP COLUMN submitted_at")
    state.connection.execute("ALTER TABLE discoveries DROP COLUMN seen_at")
    state.connection.execute("PRAGMA user_version=4")
    state.close()
    return path, config, commitments, session, job


def test_v4_ready_rows_are_rediscovered_and_cannot_start_unchecked_sessions(tmp_path):
    path, config, commitments, _, _ = legacy_v4_database(tmp_path)
    state = State(path)
    assert state.ready_count(config.identity) == state.target_ready_count(config) == 0
    assert all(state.needs_similarity(claim.identity) for claim in commitments)
    assert {row["status"] for row in state.submissions()} == {"pending_similarity"}
    assert state.pending_admissions() == commitments
    assert state.pending_profiles((config,)) == []
    assert state.claim_session(config, 10**12, 21600, 10, 3600) is None
    state.close()
    reopened = State(path)
    assert reopened.pending_admissions() == commitments
    assert reopened.target_ready_count(config) == 0


@pytest.mark.parametrize("phase", ["claimed", "running"])
def test_v4_active_session_stops_and_running_work_is_abandoned(tmp_path, phase):
    path, config, commitments, session, job = legacy_v4_database(tmp_path, session_phase=phase)
    state = State(path)
    recovered = state.session(session["id"])
    assert recovered["state"] == "stopping"
    assert recovered["pod_id"] == "legacy-pod"  # Retain ownership for cleanup.
    assert state.running_job(session["id"]) is None
    if job is not None:
        stored = state.connection.execute("SELECT status FROM jobs WHERE id=?", (job.job_id,))
        assert stored.fetchone()[0] == "abandoned"
    assert state.pending_admissions() == commitments
    assert state.target_ready_count(config) == 0
    with pytest.raises(ValueError, match="not active"):
        state.create_job(session["id"], candidate_url="file:///unchecked", now=22602)
    state.finish_session(session["id"], 22603)
    assert state.target_ready_count(config) == 0
    assert all(state.needs_similarity(claim.identity) for claim in commitments)


def test_discovery_age_survives_similarity_retry_restart_and_rediscovery(tmp_path):
    path, config = tmp_path / "state.sqlite", calibrated_target()
    state = State(path)
    source, _, previous = candidate(tmp_path, config, 1)
    claim, manifest, evidence = candidate(tmp_path, config, 2, value=1.01)
    assert claim.submitted_at is None
    state.store_fingerprint(source.artifact, previous)
    state.discover((claim,), claim.block, now=1000)
    assert state.pending_admissions() == [claim]
    state.admit(claim, manifest, config, 2000, fingerprint=evidence)
    assert state.needs_similarity(claim.identity)
    assert state.submissions()[0]["submitted_at"] == 1000
    state.close()
    state = State(path)
    state.discover((claim,), claim.block + 10, now=3000)
    assert state.pending_admissions() == [claim]
    full = compare_full(tmp_path / "head-2", tmp_path / "head-1", config)
    state.admit(
        claim,
        manifest,
        config,
        4000,
        fingerprint=evidence,
        full_matches={source.artifact: full},
    )
    assert state.submissions()[0]["submitted_at"] == 1000
    assert state.target_ready_count(config) == 1
    assert state.claim_session(config, 22599, 21600, 10, 3600) is None
    assert state.claim_session(config, 22600, 21600, 10, 3600) is not None


def test_legacy_queued_inline_profiles_retain_their_competition_identity(tmp_path):
    from tests.factories import target
    from validator.state import State

    legacy = target().model_dump(mode="json")
    legacy.pop("datasets")
    legacy.pop("similarity")
    for workload in legacy["workloads"]:
        workload.pop("input_ids")
    old_id = hashlib.sha256(
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    state = State(tmp_path / "state.sqlite")
    state.connection.execute(
        "INSERT INTO profiles(id,model_id,revision,config) VALUES(?,?,?,?)",
        (old_id, legacy["model_id"], legacy["revision"], json.dumps(legacy)),
    )
    assert state.profile(old_id).identity == old_id
