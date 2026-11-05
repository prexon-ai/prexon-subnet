"""Freeze private uploads and finish CPU checks before controller-owned admission."""

from __future__ import annotations

import gzip
import hashlib
import tarfile
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path

from shared.artifacts import ArtifactStore, download_url, extract_package
from shared.checkpoints import validate_checkpoint
from shared.config import Settings, TargetConfig, read_environment
from shared.contracts import Commitment, Qualification, SubmissionManifest
from shared.uploads import submission_key
from validator.similarity import (
    Fingerprint,
    SimilarityComparison,
    compare,
    compare_full,
    comparison_shortlist,
    fingerprint,
)
from validator.state import State


class AdmissionPending(RuntimeError):
    """An operator/infrastructure limitation cannot become a miner rejection."""


@dataclass(frozen=True)
class PreparedSubmission:
    commitment: Commitment
    manifest: SubmissionManifest
    target: TargetConfig
    fingerprint: Fingerprint
    references: dict[str, Fingerprint]
    full_matches: dict[str, SimilarityComparison]


def _candidate_source(
    store: ArtifactStore,
    commitment: Commitment,
    settings: Settings,
    allow_canonical_snapshot: bool,
) -> tuple[ArtifactStore, str] | None:
    if settings.upload_staging_root:
        try:
            staging = ArtifactStore(
                settings.upload_staging_root, environment=read_environment(settings.env_file)
            )
        except ValueError:
            raise AdmissionPending("private_staging_unavailable") from None
        key = submission_key(
            settings.network, commitment.netuid, commitment.hotkey, commitment.artifact
        )
        if staging.exists(key):
            return staging, key
        if not allow_canonical_snapshot:
            return None
    elif not allow_canonical_snapshot and not (settings.chain == "file" and not store.bucket):
        return None
    key = f"objects/{commitment.artifact}"
    return (store, key) if store.exists(key) else None


def _download_package(store: ArtifactStore, key: str, destination: Path, *, max_bytes: int) -> None:
    download_url(store.url(key), destination, max_bytes, allow_file=not store.bucket)


def _matches_target(manifest: SubmissionManifest, target: TargetConfig) -> bool:
    return (
        manifest.target_id == target.model_id
        and manifest.target_revision == target.revision
        and manifest.method in target.methods
    )


def _fingerprint(directory: Path, target: TargetConfig) -> Fingerprint:
    try:
        return fingerprint(directory, target)
    except ValueError:
        # Generic/qualified checkpoint validation already passed. Unsupported CPU
        # dtypes/statistics need an operator fix, never unchecked GPU admission.
        raise AdmissionPending("unsupported_cpu_similarity") from None


def _qualification(store: ArtifactStore, target: TargetConfig) -> Qualification | None:
    artifact = target.qualification_artifact
    if artifact is None:
        return None
    try:
        data = store.read_bytes(f"objects/{artifact}", 1024**2)
        if hashlib.sha256(data).hexdigest() != artifact:
            return None
        qualification = Qualification.model_validate_json(data)
    except (OSError, ValueError):
        return None
    if (
        qualification.profile_id != target.qualification_identity
        or qualification.engine_version != target.engine_version
        or qualification.method not in target.methods
    ):
        return None
    return qualification


def _reference_fingerprint(
    store: ArtifactStore,
    artifact: str,
    target: TargetConfig,
    qualification: Qualification,
    directory: Path,
) -> Fingerprint | None:
    key = f"objects/{artifact}"
    if not store.exists(key):
        return None
    try:
        package = directory / "reference.tar.gz"
        _download_package(
            store, key, package, max_bytes=target.policy.max_checkpoint_bytes + 2 * 1024**2
        )
        head = directory / "reference"
        manifest = extract_package(package, head, artifact, target.policy.max_checkpoint_bytes)
        if not _matches_target(manifest, target):
            return None
        validate_checkpoint(
            head, target.policy.max_checkpoint_bytes, schemas=qualification.checkpoint_schemas
        )
        return _fingerprint(head, target)
    except (OSError, ValueError, tarfile.TarError, EOFError, UnicodeError, zlib.error):
        return None


def prepare_commitment(
    store: ArtifactStore,
    commitment: Commitment,
    profiles: dict[str, TargetConfig],
    *,
    settings: Settings,
    fingerprints: dict[str, Fingerprint],
    allow_canonical_snapshot: bool = False,
    historical_sources: dict[str, TargetConfig] | None = None,
) -> PreparedSubmission | None:
    """Perform storage/CPU work with snapshots; never access controller SQLite.

    Canonical fallback is reserved for the controller's previously owned migrated
    rows. A newly discovered production commitment must prove its hotkey-scoped
    staging object even if that digest already exists in canonical storage.
    """
    source = _candidate_source(store, commitment, settings, allow_canonical_snapshot)
    if source is None:
        return None
    source_store, key = source
    authorized_competition = None
    staged_remote = bool(source_store.bucket and key.startswith("submissions/"))
    if staged_remote:
        details = source_store.client.head_object(
            Bucket=source_store.bucket, Key=source_store._key(key)
        )
        authorized_competition = details.get("Metadata", {}).get("prexon-competition-id")
    with tempfile.TemporaryDirectory(prefix="prexon-private-admission-") as temporary:
        directory = Path(temporary)
        package, head = directory / "candidate.tar.gz", directory / "candidate"
        _download_package(source_store, key, package, max_bytes=settings.upload_max_bytes)
        parse_limit = max(
            (t.policy.max_checkpoint_bytes for t in profiles.values()),
            default=settings.upload_max_bytes,
        )
        manifest = extract_package(package, head, commitment.artifact, parse_limit)
        if staged_remote and authorized_competition != manifest.competition_id:
            raise ValueError("Package does not match the API-authorized competition")
        target = profiles.get(manifest.competition_id)
        if target is None:
            return None
        if manifest.competition_id != target.identity or not _matches_target(manifest, target):
            raise ValueError("Submission does not match its competition")
        if sum(file.size for file in manifest.files) > target.policy.max_checkpoint_bytes:
            raise ValueError("Checkpoint exceeds the competition size limit")
        qualification = _qualification(store, target)
        if qualification is None:
            return None
        validate_checkpoint(
            head, target.policy.max_checkpoint_bytes, schemas=qualification.checkpoint_schemas
        )
        candidate = _fingerprint(head, target)
        references: dict[str, Fingerprint] = {}
        if target.public_reference is not None:
            artifact = target.public_reference.artifact
            reference = fingerprints.get(artifact)
            if reference is None:
                reference = _reference_fingerprint(
                    store, artifact, target, qualification, directory
                )
            if reference is None:
                return None
            references[artifact] = reference
        for artifact, previous_target in (historical_sources or {}).items():
            if (
                artifact == commitment.artifact
                or previous_target.model_id != target.model_id
                or previous_target.revision != target.revision
                or manifest.method not in previous_target.methods
            ):
                continue
            previous = fingerprints.get(artifact)
            if previous is None:
                source_key = f"objects/{artifact}"
                if not store.exists(source_key):
                    raise AdmissionPending("historical_similarity_source_unavailable")
                try:
                    previous_package = directory / f"historical-{artifact}.tar.gz"
                    previous_head = directory / f"historical-{artifact}"
                    _download_package(
                        store,
                        source_key,
                        previous_package,
                        max_bytes=previous_target.policy.max_checkpoint_bytes + 2 * 1024**2,
                    )
                    previous_manifest = extract_package(
                        previous_package,
                        previous_head,
                        artifact,
                        previous_target.policy.max_checkpoint_bytes,
                    )
                    if (
                        previous_manifest.competition_id != previous_target.identity
                        or not _matches_target(previous_manifest, previous_target)
                    ):
                        raise AdmissionPending("historical_similarity_source_invalid")
                    previous = _fingerprint(previous_head, previous_target)
                except (OSError, ValueError, tarfile.TarError, EOFError, UnicodeError, zlib.error):
                    raise AdmissionPending("historical_similarity_source_unavailable") from None
            references[artifact] = previous
        full_matches: dict[str, SimilarityComparison] = {}
        if target.similarity is not None and target.similarity.reject_near_copies:
            if target.similarity.calibration_artifact is None:
                raise AdmissionPending("uncalibrated_similarity_policy")
            bank = {**fingerprints, **references}
            comparisons = {
                source: compare(candidate, previous)
                for source, previous in bank.items()
                if source != commitment.artifact
            }
            shortlist = (
                []
                if any(result.exact_duplicate for result in comparisons.values())
                else comparison_shortlist(comparisons, manifest.provenance.parent_artifacts)
            )
            for source_artifact in shortlist:
                source_key = f"objects/{source_artifact}"
                if not store.exists(source_key):
                    return None
                try:
                    previous_target = (historical_sources or {}).get(source_artifact, target)
                    previous_package = directory / f"{source_artifact}.tar.gz"
                    previous_head = directory / source_artifact
                    _download_package(
                        store,
                        source_key,
                        previous_package,
                        max_bytes=previous_target.policy.max_checkpoint_bytes + 2 * 1024**2,
                    )
                    previous_manifest = extract_package(
                        previous_package,
                        previous_head,
                        source_artifact,
                        previous_target.policy.max_checkpoint_bytes,
                    )
                    if not _matches_target(previous_manifest, target):
                        return None
                    full_matches[source_artifact] = compare_full(head, previous_head, target)
                except (OSError, ValueError, tarfile.TarError, EOFError, UnicodeError, zlib.error):
                    return None
        # Freeze the same verified archive, never re-read mutable miner staging.
        store.write_file(f"objects/{commitment.artifact}", package)
        return PreparedSubmission(commitment, manifest, target, candidate, references, full_matches)


def finalize_prepared(state: State, prepared: PreparedSubmission, now: float) -> None:
    for artifact, reference in prepared.references.items():
        state.store_fingerprint(artifact, reference)
    state.admit(
        prepared.commitment,
        prepared.manifest,
        prepared.target,
        now,
        fingerprint=prepared.fingerprint,
        full_matches=prepared.full_matches,
    )
    if not state.needs_similarity(prepared.commitment.identity):
        state.resolve_admission(prepared.commitment, "admitted")


def resolve_preparation_failure(state: State, commitment: Commitment, error: Exception) -> None:
    if isinstance(error, AdmissionPending):
        state.resolve_admission(commitment, "pending", str(error))
    elif isinstance(
        error, (ValueError, tarfile.TarError, EOFError, UnicodeError, gzip.BadGzipFile, zlib.error)
    ):
        state.resolve_admission(commitment, "rejected", "invalid_checkpoint_package")
    # Storage/network failures remain pending without publishing sensitive details.


def admit_commitment(
    state: State,
    store: ArtifactStore,
    commitment: Commitment,
    now: float,
    *,
    settings: Settings | None = None,
) -> None:
    if state.known(commitment.identity) and not state.needs_similarity(commitment.identity):
        state.resolve_admission(commitment, "admitted")
        return
    if settings is None:
        if store.bucket:
            return  # Remote embedding requires explicit private-staging settings.
        settings = Settings(role="validator", chain="file", artifact_root=store.root)
    try:
        prepared = prepare_commitment(
            store,
            commitment,
            state.registered_profiles(),
            settings=settings,
            fingerprints=state.fingerprint_bank(),
            allow_canonical_snapshot=state.needs_similarity(commitment.identity),
            historical_sources=state.comparison_sources(),
        )
        if prepared is not None:
            finalize_prepared(state, prepared, now)
    except Exception as error:
        resolve_preparation_failure(state, commitment, error)
