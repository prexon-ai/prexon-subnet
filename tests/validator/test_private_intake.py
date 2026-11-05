"""Private staging and CPU admission use real packages, fingerprints and local state."""

import json
import shutil
import struct
import threading
from types import SimpleNamespace

import pytest

from shared.artifacts import ArtifactStore, package_checkpoint
from shared.checkpoints import checkpoint_schema, sha256_file
from shared.config import PublicReference, Settings, SimilarityPolicy
from shared.contracts import Commitment, Qualification
from shared.uploads import submission_key
from tests.factories import make_checkpoint, provenance, target
from validator.similarity import fingerprint


def setup_intake(
    tmp_path,
    *,
    reference=False,
    near=False,
    dtype="F32",
    reference_weights=None,
):
    store = ArtifactStore(str(tmp_path / "canonical"))
    staging = ArtifactStore(str(tmp_path / "staging"))
    head = make_checkpoint(tmp_path / "head")
    if dtype != "F32":
        header = json.dumps(
            {"weight": {"dtype": dtype, "shape": [2], "data_offsets": [0, 2]}}
        ).encode()
        (head / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
    profile = target()
    reference_artifact = None
    if reference:
        reference_package = tmp_path / "reference.tar.gz"
        reference_head = head
        if reference_weights is not None:
            reference_head = make_checkpoint(tmp_path / "reference-head")
            weights = reference_head / "model.safetensors"
            weights.write_bytes(weights.read_bytes()[:-8] + struct.pack("<ff", *reference_weights))
        package_checkpoint(
            reference_head,
            reference_package,
            target_id=profile.model_id,
            target_revision=profile.revision,
            competition_id=profile.identity,
            provenance=provenance(),
        )
        reference_artifact = store.put_file(reference_package)
        profile = profile.model_copy(
            update={"public_reference": PublicReference(artifact=reference_artifact)}
        )
    if near:
        profile = profile.model_copy(
            update={
                "similarity": SimilarityPolicy(
                    reject_near_copies=True, calibration_artifact="c" * 64
                )
            }
        )
    qualification = Qualification(
        profile_id=profile.qualification_identity,
        engine_version=profile.engine_version,
        approved_by="operator",
        evidence_description="Reviewed exact runtime and schema",
        target_preserving_verification_reviewed=True,
        checkpoint_schemas=(checkpoint_schema(head),),
    )
    qualification_artifact = store.put_bytes(qualification.model_dump_json().encode())
    profile = profile.model_copy(update={"qualification_artifact": qualification_artifact})
    package = tmp_path / "candidate.tar.gz"
    package_checkpoint(
        head,
        package,
        target_id=profile.model_id,
        target_revision=profile.revision,
        competition_id=profile.identity,
        provenance=provenance(),
    )
    commitment = Commitment(
        netuid=17,
        block=101,
        block_hash="finalized",
        hotkey="alice",
        uid=1,
        artifact=sha256_file(package),
    )
    settings = Settings(
        role="validator",
        chain="file",
        network="local",
        netuid=17,
        targets=(profile,),
        artifact_root=store.root,
        upload_staging_root=staging.root,
        state_path=tmp_path / "state.sqlite3",
    )
    key = submission_key(
        settings.network, commitment.netuid, commitment.hotkey, commitment.artifact
    )
    staging.write_file(key, package)
    return SimpleNamespace(
        store=store,
        staging=staging,
        profile=profile,
        package=package,
        head=head,
        commitment=commitment,
        settings=settings,
        profiles={profile.identity: profile},
        key=key,
        reference_artifact=reference_artifact,
    )


def prepare(intake, **kwargs):
    from validator.submissions import prepare_commitment

    return prepare_commitment(
        intake.store,
        intake.commitment,
        intake.profiles,
        settings=intake.settings,
        fingerprints=kwargs.pop("fingerprints", {}),
        **kwargs,
    )


def test_different_approved_layout_is_not_blocked_by_public_reference(tmp_path):
    intake = setup_intake(tmp_path, reference=True)
    original_schema = checkpoint_schema(intake.head)
    weights = intake.head / "model.safetensors"
    data = weights.read_bytes()
    header_size = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_size])
    header["different_weight"] = header.pop("weight")
    encoded = json.dumps(header).encode()
    weights.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data[8 + header_size :])
    qualification = Qualification.model_validate_json(
        intake.store.read_bytes(f"objects/{intake.profile.qualification_artifact}")
    )
    qualification = qualification.model_copy(
        update={"checkpoint_schemas": (original_schema, checkpoint_schema(intake.head))}
    )
    digest = intake.store.put_bytes(qualification.model_dump_json().encode())
    profile = intake.profile.model_copy(update={"qualification_artifact": digest})
    package_checkpoint(
        intake.head,
        intake.package,
        target_id=profile.model_id,
        target_revision=profile.revision,
        competition_id=profile.identity,
        provenance=provenance(),
    )
    intake.commitment = intake.commitment.model_copy(
        update={"artifact": sha256_file(intake.package)}
    )
    intake.staging.write_file(
        submission_key(
            intake.settings.network,
            intake.commitment.netuid,
            intake.commitment.hotkey,
            intake.commitment.artifact,
        ),
        intake.package,
    )
    intake.profile = profile
    intake.profiles = {profile.identity: profile}
    prepared = prepare(intake)
    assert prepared is not None
    assert prepared.fingerprint.scope != prepared.references[intake.reference_artifact].scope


@pytest.mark.parametrize("authorization", [None, "different", "matching"])
def test_remote_staging_binds_api_authorized_competition(tmp_path, monkeypatch, authorization):
    intake = setup_intake(tmp_path)
    metadata = (
        {}
        if authorization is None
        else {
            "prexon-competition-id": intake.profile.identity
            if authorization == "matching"
            else "f" * 64
        }
    )

    class Hippius:
        def head_object(self, **kwargs):
            assert kwargs["Bucket"] == "staging"
            assert kwargs["Key"] == intake.key
            return {"Metadata": metadata, "ContentLength": intake.package.stat().st_size}

    remote = ArtifactStore("s3://staging", s3_client=Hippius())
    intake.settings = intake.settings.model_copy(update={"upload_staging_root": remote.root})
    monkeypatch.setattr("validator.submissions.ArtifactStore", lambda *_args, **_kwargs: remote)
    monkeypatch.setattr(
        "validator.submissions._download_package",
        lambda _store, _key, destination, **_: shutil.copyfile(intake.package, destination),
    )
    if authorization == "matching":
        assert prepare(intake) is not None
    else:
        with pytest.raises(ValueError, match="authorized competition"):
            prepare(intake)
        assert not intake.store.exists(f"objects/{intake.commitment.artifact}")


def test_private_candidate_is_fingerprinted_and_same_verified_bytes_frozen(tmp_path):
    intake = setup_intake(tmp_path)
    prepared = prepare(intake)
    assert prepared is not None
    assert prepared.commitment == intake.commitment
    assert prepared.target == intake.profile
    assert prepared.fingerprint == fingerprint(intake.head, intake.profile)
    assert prepared.references == {}
    assert prepared.full_matches == {}
    assert (
        intake.store.read_bytes(f"objects/{intake.commitment.artifact}")
        == intake.package.read_bytes()
    )
    staged = intake.staging.root + "/" + intake.key
    from pathlib import Path

    Path(staged).write_bytes(b"miner changed its staging object")
    assert (
        intake.store.read_bytes(f"objects/{intake.commitment.artifact}")
        == intake.package.read_bytes()
    )


def test_other_hotkey_cannot_claim_digest_even_when_canonical_object_exists(tmp_path):
    intake = setup_intake(tmp_path)
    intake.store.put_file(intake.package)
    intake.commitment = intake.commitment.model_copy(update={"hotkey": "mallory", "uid": 2})
    assert prepare(intake) is None


def test_production_never_falls_back_to_digest_only_canonical_lookup(tmp_path):
    intake = setup_intake(tmp_path)
    intake.store.put_file(intake.package)
    intake.settings = intake.settings.model_copy(
        update={"chain": "bittensor", "upload_staging_root": ""}
    )
    assert prepare(intake) is None


@pytest.mark.parametrize("missing", ["profile", "qualification", "public_reference"])
def test_missing_operator_inputs_leave_candidate_pending(tmp_path, missing):
    from pathlib import Path

    intake = setup_intake(tmp_path, reference=True)
    if missing == "profile":
        intake.profiles = {}
    else:
        artifact = (
            intake.profile.qualification_artifact
            if missing == "qualification"
            else intake.reference_artifact
        )
        (Path(intake.store.root) / "objects" / artifact).unlink()
    assert prepare(intake) is None
    assert not intake.store.exists(f"objects/{intake.commitment.artifact}")


def test_calibrated_near_copy_evidence_uses_full_tensor_comparison(tmp_path):
    intake = setup_intake(tmp_path, reference=True, near=True, reference_weights=(1.0, 2.0))
    prepared = prepare(intake)
    evidence = prepared.full_matches[intake.reference_artifact]
    assert evidence.full_comparison is True
    assert evidence.comparable is True
    assert evidence.exact_duplicate is False
    assert evidence.tensor_comparisons


def test_missing_shortlisted_canonical_source_stays_pending(tmp_path):
    intake = setup_intake(tmp_path, near=True)
    previous_head = make_checkpoint(tmp_path / "previous")
    weights = previous_head / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-8] + struct.pack("<ff", 1.0, 2.0))
    previous = fingerprint(previous_head, intake.profile)
    assert prepare(intake, fingerprints={"e" * 64: previous}) is None


def test_unsupported_similarity_dtype_is_explicit_pending(tmp_path):
    from validator.submissions import AdmissionPending

    intake = setup_intake(tmp_path, dtype="F8_E4M3")
    with pytest.raises(AdmissionPending, match="unsupported_cpu_similarity"):
        prepare(intake)
    assert not intake.store.exists(f"objects/{intake.commitment.artifact}")


def test_digest_tampering_is_invalid_before_canonical_freeze(tmp_path):
    intake = setup_intake(tmp_path)
    intake.staging.write_bytes(intake.key, b"tampered")
    with pytest.raises(ValueError, match="digest"):
        prepare(intake)
    assert not intake.store.exists(f"objects/{intake.commitment.artifact}")


def test_synchronous_admission_persists_private_fingerprint_before_ready(tmp_path):
    from validator.state import State
    from validator.submissions import admit_commitment

    intake = setup_intake(tmp_path)
    state = State(intake.settings.state_path)
    try:
        state.register_target(intake.profile)
        state.discover((intake.commitment,), 101, now=100)
        admit_commitment(state, intake.store, intake.commitment, 100, settings=intake.settings)
        assert intake.commitment.artifact in state.fingerprint_bank()
        assert state.ready_count(intake.profile.identity) == 1
        assert state.needs_similarity(intake.commitment.identity) is False
        assert state.pending_admissions() == []
    finally:
        state.close()


def test_background_worker_does_not_block_chain_polling_or_touch_sqlite(tmp_path, monkeypatch):
    from validator.admission import AdmissionService
    from validator.state import State
    from validator.submissions import prepare_commitment

    intake = setup_intake(tmp_path)
    state = State(intake.settings.state_path)
    state.register_target(intake.profile)
    state.discover((intake.commitment,), 101, now=100)
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()
    observed = []

    def blocked(*args, **kwargs):
        assert threading.get_ident() != main_thread
        # Only immutable/profile/bank inputs reach the worker, never State.
        assert all(not isinstance(arg, State) for arg in args)
        entered.set()
        assert release.wait(10)
        return prepare_commitment(*args, **kwargs)

    monkeypatch.setattr("validator.admission.prepare_commitment", blocked)
    service = AdmissionService()
    try:
        service.pump(state, intake.store, intake.settings, 100)
        assert entered.wait(2)
        for _ in range(3):
            service.pump(state, intake.store, intake.settings, 101)
            observed.append("chain scan")
            state.discover((), 102, now=101)
        assert observed == ["chain scan"] * 3
        assert state.ready_count(intake.profile.identity) == 0
        release.set()
        # Waiting for worker completion is only a test synchronization boundary.
        service.close()
        service.pump(state, intake.store, intake.settings, 102)
        assert state.ready_count(intake.profile.identity) == 1
        assert intake.commitment.artifact in state.fingerprint_bank()
    finally:
        release.set()
        service.close()
        state.close()


def test_public_reference_identity_is_screened_before_ready(tmp_path):
    from validator.state import State
    from validator.submissions import admit_commitment

    intake = setup_intake(tmp_path, reference=True)
    state = State(intake.settings.state_path)
    try:
        state.register_target(intake.profile)
        state.discover((intake.commitment,), 101, now=100)
        admit_commitment(state, intake.store, intake.commitment, 100, settings=intake.settings)
        assert intake.reference_artifact in state.fingerprint_bank()
        assert state.ready_count(intake.profile.identity) == 0
        assert state.pending_admissions() == []
    finally:
        state.close()


def test_historical_profile_sources_seed_private_bank_before_admission(tmp_path):
    intake = setup_intake(tmp_path)
    previous_profile = intake.profile.model_copy(
        update={"policy": intake.profile.policy.model_copy(update={"improvement_margin": 0.02})}
    )
    package = tmp_path / "previous-profile.tar.gz"
    package_checkpoint(
        intake.head,
        package,
        target_id=previous_profile.model_id,
        target_revision=previous_profile.revision,
        competition_id=previous_profile.identity,
        provenance=provenance(),
    )
    artifact = intake.store.put_file(package)
    prepared = prepare(intake, historical_sources={artifact: previous_profile})
    assert artifact in prepared.references
    assert prepared.references[artifact] == prepared.fingerprint


def test_missing_historical_source_is_explicit_pending(tmp_path):
    from validator.submissions import AdmissionPending

    intake = setup_intake(tmp_path)
    with pytest.raises(AdmissionPending, match="historical_similarity_source_unavailable"):
        prepare(intake, historical_sources={"e" * 64: intake.profile})
    assert not intake.store.exists(f"objects/{intake.commitment.artifact}")


def test_owned_migration_snapshot_can_be_rechecked_without_miner_staging(tmp_path):
    from pathlib import Path

    intake = setup_intake(tmp_path)
    intake.store.put_file(intake.package)
    (Path(intake.staging.root) / intake.key).unlink()
    intake.settings = intake.settings.model_copy(update={"chain": "bittensor"})
    assert prepare(intake) is None
    prepared = prepare(
        intake,
        allow_canonical_snapshot=True,
        historical_sources={intake.commitment.artifact: intake.profile},
    )
    assert prepared is not None
    assert prepared.references == {}


def test_exact_duplicate_can_finalize_without_unavailable_near_sources(tmp_path):
    intake = setup_intake(tmp_path, near=True)
    previous_head = make_checkpoint(tmp_path / "previous")
    weights = previous_head / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-8] + struct.pack("<ff", 1.0, 2.0))
    bank = {
        "e" * 64: fingerprint(intake.head, intake.profile),
        "f" * 64: fingerprint(previous_head, intake.profile),
    }
    prepared = prepare(intake, fingerprints=bank)
    assert prepared is not None
    assert prepared.full_matches == {}


@pytest.mark.parametrize("failure", ["malformed", "unsupported", "storage"])
def test_sync_failures_distinguish_invalid_candidates_from_retryable_inputs(
    tmp_path, monkeypatch, failure
):
    import hashlib

    from validator.state import State
    from validator.submissions import admit_commitment

    intake = setup_intake(tmp_path, dtype="F8_E4M3" if failure == "unsupported" else "F32")
    if failure == "malformed":
        raw = b"this is not gzip"
        intake.commitment = intake.commitment.model_copy(
            update={"artifact": hashlib.sha256(raw).hexdigest()}
        )
        key = submission_key(
            intake.settings.network,
            intake.commitment.netuid,
            intake.commitment.hotkey,
            intake.commitment.artifact,
        )
        intake.staging.write_bytes(key, raw)
    elif failure == "storage":

        def unavailable(*args, **kwargs):
            raise OSError("private endpoint details")

        monkeypatch.setattr("validator.submissions.download_url", unavailable)
    state = State(intake.settings.state_path)
    try:
        state.register_target(intake.profile)
        state.discover((intake.commitment,), 101, now=100)
        admit_commitment(state, intake.store, intake.commitment, 100, settings=intake.settings)
        assert state.ready_count(intake.profile.identity) == 0
        row = state.connection.execute(
            "SELECT status,reason FROM discoveries WHERE id=?", (intake.commitment.identity,)
        ).fetchone()
        if failure == "malformed":
            assert tuple(row) == ("rejected", "invalid_checkpoint_package")
        elif failure == "unsupported":
            assert tuple(row) == ("pending", "unsupported_cpu_similarity")
        else:
            assert row["status"] == "pending"
            assert "private endpoint details" not in (row["reason"] or "")
    finally:
        state.close()


def test_background_missing_object_retry_does_not_starve_next_discovery(tmp_path):
    from validator.admission import AdmissionService
    from validator.state import State

    intake = setup_intake(tmp_path)
    missing = intake.commitment.model_copy(update={"artifact": "e" * 64, "block": 100})
    state = State(intake.settings.state_path)
    service = AdmissionService()
    try:
        state.register_target(intake.profile)
        state.discover((missing, intake.commitment), 101, now=100)
        service.pump(state, intake.store, intake.settings, 100)
        assert service.future.result(timeout=5) is None
        service.pump(state, intake.store, intake.settings, 101)
        assert service.future.result(timeout=5).commitment == intake.commitment
        service.close()
        service.pump(state, intake.store, intake.settings, 102)
        assert state.ready_count(intake.profile.identity) == 1
        assert state.pending_admissions(limit=1) == [missing]
    finally:
        service.close()
        state.close()
