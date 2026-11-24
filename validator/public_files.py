"""The validator's public files, kept in public storage where the website reads them.

Nothing public is served from the validator's machine. A file is uploaded when what
it shows changes; a file that carries the time is also uploaded again once its copy
is REFRESH_SECONDS old, so readers can tell a quiet subnet from a stopped validator.
"""

from __future__ import annotations

import hashlib
import json
import logging

from shared.artifacts import ArtifactStore
from validator.state import State

log = logging.getLogger(__name__)

REFRESH_SECONDS = 300


def _shown(data: bytes) -> str:
    """Identify what a document shows, apart from the time and block it was written at."""
    document = json.loads(data)
    for moving in ("updated_at", "finalized_block"):
        document.pop(moving, None)
    return hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def publish_file(
    state: State, store: ArtifactStore, key: str, data: bytes, now: float, *, timed: bool = False
) -> None:
    """Upload a public JSON file when it is due; a failed upload stays owed and is retried."""
    marker = f"published:{key}"
    content = _shown(data) if timed else hashlib.sha256(data).hexdigest()
    last = json.loads(state.metadata(marker) or "null")
    if last and last["content"] == content and (not timed or now - last["at"] < REFRESH_SECONDS):
        return
    try:
        store.write_bytes(key, data, content_type="application/json")
    except Exception as exc:
        log.warning("public_file_retry key=%s error=%s", key, type(exc).__name__)
        return
    state.set_metadata(marker, json.dumps({"content": content, "at": now}))
