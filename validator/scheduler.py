from __future__ import annotations

from typing import Any

from shared.targets import TargetConfig
from validator.config import (
    EARLY_QUEUE_DEPTH,
    SCHEDULE_SECONDS,
    SESSION_RETRY_SECONDS,
    SESSION_SECONDS,
)
from validator.state import State


def claim_due_sessions(
    state: State,
    targets: tuple[TargetConfig, ...],
    now: float,
    qualified_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    claimed = []
    for target in state.pending_profiles(targets):
        if qualified_ids is not None and target.identity not in qualified_ids:
            continue
        session = state.claim_session(
            target,
            now,
            SCHEDULE_SECONDS,
            EARLY_QUEUE_DEPTH,
            SESSION_SECONDS,
            retry_seconds=SESSION_RETRY_SECONDS,
        )
        if session:
            claimed.append(session)
    return claimed
