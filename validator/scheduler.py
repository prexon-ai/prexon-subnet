from __future__ import annotations

from typing import Any

from shared.config import Settings
from validator.state import State


def claim_due_sessions(
    state: State, settings: Settings, now: float, qualified_ids: set[str] | None = None
) -> list[dict[str, Any]]:
    claimed = []
    for target in state.pending_profiles(settings.targets):
        if qualified_ids is not None and target.identity not in qualified_ids:
            continue
        session = state.claim_session(
            target,
            now,
            settings.schedule_seconds,
            settings.early_queue_depth,
            settings.session_seconds,
        )
        if session:
            claimed.append(session)
    return claimed
