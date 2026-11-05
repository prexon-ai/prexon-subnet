import struct

import pytest

from shared.artifacts import package_checkpoint
from shared.checkpoints import sha256_file
from shared.contracts import Commitment
from tests.factories import make_checkpoint, provenance, target
from validator.similarity import fingerprint
from validator.state import State


def candidate(tmp_path, config, index, *, value=1.0):
    directory = make_checkpoint(tmp_path / f"head-{index}")
    tensor = directory / "model.safetensors"
    tensor.write_bytes(tensor.read_bytes()[:-8] + struct.pack("<ff", value, value + 1))
    package = tmp_path / f"head-{index}.tar.gz"
    manifest = package_checkpoint(
        directory,
        package,
        target_id=config.model_id,
        target_revision=config.revision,
        competition_id=config.identity,
        provenance=provenance().model_copy(update={"training_description": f"run {index}"}),
    )
    commitment = Commitment(
        netuid=1,
        block=100 + index,
        block_hash=f"block-{index}",
        hotkey=f"miner-{index}",
        uid=index,
        artifact=sha256_file(package),
    )
    return commitment, manifest, fingerprint(directory, config)


def test_queue_admission_requires_cpu_fingerprint_evidence(tmp_path):
    state, config = State(tmp_path / "state.sqlite"), target()
    commitment, manifest, _ = candidate(tmp_path, config, 1)
    with pytest.raises(TypeError):
        state.admit(commitment, manifest, config, 1000)
    assert state.target_ready_count(config) == 0


def test_repackaged_copies_across_hotkeys_and_profile_versions_never_enter_queue(tmp_path):
    state, original = State(tmp_path / "state.sqlite"), target()
    changed = original.model_copy(update={"context_length": 4096})
    a, ma, fa = candidate(tmp_path, original, 1)
    b, mb, fb = candidate(tmp_path, changed, 2)
    assert a.artifact != b.artifact and fa.tensor_digest == fb.tensor_digest
    state.admit(a, ma, original, 1000, fingerprint=fa)
    state.admit(b, mb, changed, 1001, fingerprint=fb)
    assert state.target_ready_count(original) == 1
    assert state.submissions()[-1]["status"] == "rejected"
    assert state.similarity_verdict(b.identity)["exact_duplicate"] is True
    assert state.awards() == []


def test_changed_heads_are_compared_but_raw_similarity_does_not_reduce_score(tmp_path):
    state, config = State(tmp_path / "state.sqlite"), target()
    a, ma, fa = candidate(tmp_path, config, 1)
    b, mb, fb = candidate(tmp_path, config, 2, value=1.00001)
    state.admit(a, ma, config, 1000, fingerprint=fa)
    state.admit(b, mb, config, 1001, fingerprint=fb)
    assert state.target_ready_count(config) == 2
    verdict = state.similarity_verdict(b.identity)
    assert verdict["cosine"] > 0.99
    assert verdict["outcome"] == "pass"
    assert state.submissions()[-1]["score"] is None


def test_public_reference_fingerprint_is_a_duplicate_source(tmp_path):
    state, config = State(tmp_path / "state.sqlite"), target()
    a, ma, fa = candidate(tmp_path, config, 1)
    state.store_fingerprint("e" * 64, fa)
    state.admit(a, ma, config, 1000, fingerprint=fa)
    assert state.target_ready_count(config) == 0
    assert state.similarity_verdict(a.identity)["source_artifact"] == "e" * 64


def calibrated_target():
    from shared.config import SimilarityPolicy

    return target().model_copy(
        update={
            "similarity": SimilarityPolicy(
                reject_near_copies=True,
                calibration_artifact="c" * 64,
            ),
        }
    )


@pytest.mark.parametrize("confirmation", ["missing", "sampled"])
def test_calibrated_policy_keeps_incomplete_full_evidence_pending(tmp_path, confirmation):
    from validator.similarity import compare

    state, config = State(tmp_path / "state.sqlite"), calibrated_target()
    source, _, previous = candidate(tmp_path, config, 1)
    claim, manifest, evidence = candidate(tmp_path, config, 2, value=1.0000001)
    state.store_fingerprint(source.artifact, previous)
    matches = (
        {}
        if confirmation == "missing"
        else {
            source.artifact: compare(evidence, previous),
        }
    )
    state.admit(claim, manifest, config, 1000, fingerprint=evidence, full_matches=matches)
    assert state.needs_similarity(claim.identity)
    assert state.target_ready_count(config) == 0
    assert state.submissions()[0]["status"] == "pending_similarity"
    assert claim.artifact not in state.fingerprint_bank()
    assert state.similarity_verdict(claim.identity)["outcome"] == "inconclusive"
    assert state.claim_session(config, 100000, 21600, 10, 3600) is None


def test_calibrated_full_near_match_is_rejected_before_ready(tmp_path):
    from validator.similarity import compare_full

    state, config = State(tmp_path / "state.sqlite"), calibrated_target()
    source, _, previous = candidate(tmp_path, config, 1)
    claim, manifest, evidence = candidate(tmp_path, config, 2, value=1.0000001)
    state.store_fingerprint(source.artifact, previous)
    full = compare_full(tmp_path / "head-2", tmp_path / "head-1", config)
    assert full.full_comparison and not full.exact_duplicate
    assert full.relative_distance < config.similarity.max_relative_distance
    state.admit(
        claim,
        manifest,
        config,
        1000,
        fingerprint=evidence,
        full_matches={source.artifact: full},
    )
    assert state.submissions()[0]["status"] == "rejected"
    assert state.submissions()[0]["reason"] == "near_duplicate_checkpoint"
    assert state.target_ready_count(config) == 0
    verdict = state.similarity_verdict(claim.identity)
    assert verdict["outcome"] == "near_duplicate" and verdict["full_comparison"]
    assert verdict["source_artifact"] == source.artifact
    assert state.awards() == []


def test_calibrated_policy_admits_meaningful_change_after_full_comparison(tmp_path):
    from validator.similarity import compare_full

    state, config = State(tmp_path / "state.sqlite"), calibrated_target()
    source, _, previous = candidate(tmp_path, config, 1)
    claim, manifest, evidence = candidate(tmp_path, config, 2, value=1.01)
    state.store_fingerprint(source.artifact, previous)
    full = compare_full(tmp_path / "head-2", tmp_path / "head-1", config)
    assert full.relative_distance > config.similarity.max_relative_distance
    state.admit(
        claim,
        manifest,
        config,
        1000,
        fingerprint=evidence,
        full_matches={source.artifact: full},
    )
    assert state.target_ready_count(config) == 1
    assert not state.needs_similarity(claim.identity)
    assert state.similarity_verdict(claim.identity)["outcome"] == "pass"
    assert state.submissions()[0]["score"] is None
    assert claim.artifact in state.fingerprint_bank()
