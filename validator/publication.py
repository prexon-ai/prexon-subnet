"""Copy only verified historical winners into a separate public namespace."""

from __future__ import annotations

import logging
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from shared.artifacts import ArtifactStore, download_url, extract_package
from shared.checkpoints import sha256_file
from shared.contracts import (
    Award,
    Commitment,
    Decision,
    EvaluationJob,
    EvaluationResult,
    SubmissionManifest,
)
from shared.releases import WinnerRelease, public_release_key
from validator.state import State

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PublicationSnapshot:
    """Verified immutable inputs; workers never receive SQLite or State."""

    award: Award
    commitment: Commitment
    manifest: SubmissionManifest
    max_checkpoint_bytes: int


def _separate_namespaces(private: ArtifactStore, public: ArtifactStore) -> None:
    if private.bucket and public.bucket:
        if private.bucket != public.bucket:
            return
        # Separate buckets prevent a bucket-wide public ACL from exposing private prefixes.
    elif not private.bucket and not public.bucket:
        a_path, b_path = Path(private.root).resolve(), Path(public.root).resolve()
        if not a_path.is_relative_to(b_path) and not b_path.is_relative_to(a_path):
            return
    else:
        return
    raise ValueError("Public artifacts must use a separate private/public storage namespace")


def _checkpoint_url(public: ArtifactStore, base_url: str, award: Award) -> str:
    if not base_url:
        if public.bucket:
            raise ValueError("Public object storage requires a stable HTTPS release base URL")
        return f"/api/v1/winners/{award.submission_id}/checkpoint"
    parsed = urlparse(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "\\" in base_url
    ):
        raise ValueError("Public release base URL must be HTTPS without credentials or query")
    return f"{base_url.rstrip('/')}/{public_release_key(award.artifact)}"


def _verify_award(state: State, award: Award) -> _PublicationSnapshot:
    rows = [row for row in state.submissions() if row["id"] == award.submission_id]
    if len(rows) != 1:
        raise ValueError("Winner has no unique admitted submission")
    row = rows[0]
    commitment = Commitment.model_validate_json(row["commitment"])
    manifest = SubmissionManifest.model_validate_json(row["manifest"])
    target = state.profile(award.competition_id)
    if (
        commitment.identity != award.submission_id
        or commitment.artifact != award.artifact
        or commitment.hotkey != award.hotkey
        or commitment.uid != award.uid
        or row["profile_id"] != award.competition_id
        or row["status"] != "won"
        or manifest.competition_id != target.identity
        or manifest.target_id != target.model_id
        or manifest.target_revision != target.revision
        or manifest.method not in target.methods
    ):
        raise ValueError("Winner admission identity does not match its award")
    for evaluation in state.evaluations():
        if evaluation["submission_id"] != award.submission_id:
            continue
        decision = Decision.model_validate_json(evaluation["decision"])
        if decision.outcome != "won":
            continue
        job = EvaluationJob.model_validate_json(evaluation["job"])
        result = EvaluationResult.model_validate_json(evaluation["result"])
        if (
            job.commitment == commitment
            and job.target == target
            and evaluation["id"] == job.job_id == result.job_id
            and result.submission_id == award.submission_id
            and result.competition_id == award.competition_id
            and result.status == "complete"
        ):
            return _PublicationSnapshot(
                award, commitment, manifest, target.policy.max_checkpoint_bytes
            )
    raise ValueError("Winner has no matching successful evaluated package")


def publish_winners(
    state: State,
    private_store: ArtifactStore,
    public_store: ArtifactStore,
    public_base_url: str,
    now: float,
) -> None:
    """Publication is retriable and never mutates promotion or earning windows."""
    _separate_namespaces(private_store, public_store)
    published = state.publications()
    for award in state.awards():
        try:
            if award.submission_id in published:
                release = published[award.submission_id]
                if release.source_artifact != award.artifact or release.artifact != award.artifact:
                    raise ValueError("Recorded release digest does not match its award")
                continue
            release = _publish_snapshot(
                _verify_award(state, award), private_store, public_store, public_base_url, now
            )
            state.record_publication(release)
        except Exception as exc:
            # Diagnostic strings may contain private URLs; log only identity and failure class.
            logger.warning(
                "Winner publication failed for %s (%s)", award.submission_id, type(exc).__name__
            )


def _publish_snapshot(
    snapshot: _PublicationSnapshot,
    private_store: ArtifactStore,
    public_store: ArtifactStore,
    public_base_url: str,
    now: float,
) -> WinnerRelease:
    _separate_namespaces(private_store, public_store)
    award, commitment = snapshot.award, snapshot.commitment
    release = WinnerRelease(
        submission_id=award.submission_id,
        source_artifact=award.artifact,
        artifact=award.artifact,
        checkpoint_url=_checkpoint_url(public_store, public_base_url, award),
        published_at=now,
    )
    with tempfile.TemporaryDirectory(prefix="prexon-winner-") as temporary:
        directory = Path(temporary)
        package = directory / "checkpoint.tar.gz"
        download_url(
            private_store.url(f"objects/{commitment.artifact}"),
            package,
            snapshot.max_checkpoint_bytes + 2 * 1024**2,
            allow_file=not private_store.bucket,
        )
        manifest = extract_package(
            package, directory / "head", commitment.artifact, snapshot.max_checkpoint_bytes
        )
        if manifest != snapshot.manifest:
            raise ValueError("Verified package manifest differs from immutable admission")
        if sha256_file(package) != release.artifact:
            raise ValueError("Release digest does not match the verified package")
        public_store.write_file(public_release_key(release.source_artifact), package)
    return release


class PublicationService:
    """One background copy at a time; callers collect and persist results on their thread.

    Unpublished awards get one attempt per fair round so an unavailable old source
    cannot starve newer winners. A failed state commit retains the verified release.
    A final pump after close may collect the last result without scheduling more work.
    """

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prexon-publication")
        self._future: Future[WinnerRelease] | None = None
        self._active_submission: str | None = None
        self._ready: WinnerRelease | None = None
        self._attempted: set[str] = set()
        self._closed = False

    def pump(
        self,
        state: State,
        private_store: ArtifactStore,
        public_store: ArtifactStore,
        public_base_url: str,
        now: float,
    ) -> None:
        if self._future is not None:
            if not self._future.done():
                return
            future, self._future = self._future, None
            try:
                self._ready = future.result()
            except Exception as exc:
                logger.warning(
                    "Winner publication failed for %s (%s)",
                    self._active_submission,
                    type(exc).__name__,
                )
            self._active_submission = None
        if self._ready is not None:
            try:
                state.record_publication(self._ready)
            except Exception as exc:
                logger.warning(
                    "Winner publication commit retry for %s (%s)",
                    self._ready.submission_id,
                    type(exc).__name__,
                )
                return
            self._ready = None
        if self._closed:
            return
        published = state.publications()
        pending = [a for a in state.awards() if a.submission_id not in published]
        self._attempted.intersection_update(a.submission_id for a in pending)
        if not pending:
            return
        if all(a.submission_id in self._attempted for a in pending):
            self._attempted.clear()
        award = next(a for a in pending if a.submission_id not in self._attempted)
        self._attempted.add(award.submission_id)
        try:
            snapshot = _verify_award(state, award)
        except Exception as exc:
            logger.warning(
                "Winner publication snapshot retry for %s (%s)",
                award.submission_id,
                type(exc).__name__,
            )
            return
        self._active_submission = award.submission_id
        self._future = self._executor.submit(
            _publish_snapshot, snapshot, private_store, public_store, public_base_url, now
        )

    def close(self) -> None:
        self._closed = True
        self._executor.shutdown(wait=True)
