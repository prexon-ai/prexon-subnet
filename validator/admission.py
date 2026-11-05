"""One bounded CPU/storage worker; all state finalization stays on the controller thread."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor

from shared.artifacts import ArtifactStore
from shared.config import Settings
from shared.contracts import Commitment
from validator.state import State
from validator.submissions import (
    PreparedSubmission,
    finalize_prepared,
    prepare_commitment,
    resolve_preparation_failure,
)


class AdmissionService:
    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prexon-admission")
        self.future: Future[PreparedSubmission | None] | None = None
        self.commitment: Commitment | None = None
        self.closed = False

    def pump(self, state: State, store: ArtifactStore, settings: Settings, now: float) -> None:
        if self.future is not None:
            if not self.future.done():
                return
            future, commitment = self.future, self.commitment
            self.future = self.commitment = None
            assert commitment is not None
            try:
                prepared = future.result()
                if prepared is not None:
                    finalize_prepared(state, prepared, now)
            except Exception as error:
                resolve_preparation_failure(state, commitment, error)
        if self.closed:
            return
        pending = state.pending_admissions(limit=1)
        if not pending:
            return
        commitment = pending[0]
        needs_similarity = state.needs_similarity(commitment.identity)
        if state.known(commitment.identity) and not needs_similarity:
            state.resolve_admission(commitment, "admitted")
            return
        # Snapshot SQLite-owned inputs before dispatch; workers never receive State.
        profiles = state.registered_profiles()
        fingerprints = state.fingerprint_bank()
        historical_sources = state.comparison_sources()
        self.commitment = commitment
        self.future = self.executor.submit(
            prepare_commitment,
            store,
            commitment,
            profiles,
            settings=settings,
            fingerprints=fingerprints,
            allow_canonical_snapshot=needs_similarity,
            historical_sources=historical_sources,
        )

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.executor.shutdown(wait=True)
