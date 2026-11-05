from __future__ import annotations

from pydantic import BaseModel


class Collection[T](BaseModel):
    schema_version: int = 1
    updated_at: float
    total: int
    items: tuple[T, ...]


class Status(BaseModel):
    schema_version: int = 1
    updated_at: float
    finalized_block: int
    age_seconds: float
    stale: bool
    ready_submissions: int
    active_sessions: int
