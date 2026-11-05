from __future__ import annotations

import time
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from api.queries import PublishedViews
from api.schemas import Collection, Status
from shared.contracts import (
    DashboardSnapshot,
    Emission,
    PublicEvaluation,
    PublicSubmission,
    PublicTarget,
    PublicWinner,
)

Offset = Annotated[int, Query(ge=0)]
Limit = Annotated[int, Query(ge=1, le=1000)]


def dashboard_routes(views: PublishedViews) -> APIRouter:
    router = APIRouter(prefix="/v1")

    def snapshot() -> DashboardSnapshot:
        try:
            return views.snapshot()
        except (OSError, ValueError) as exc:
            raise HTTPException(503, "Published dashboard view is unavailable") from exc

    @router.get("/snapshot", response_model=DashboardSnapshot)
    def full_snapshot() -> DashboardSnapshot:
        return snapshot()

    @router.get("/status", response_model=Status)
    def status() -> Status:
        data = snapshot()
        age = max(0, time.time() - data.updated_at)
        return Status(
            updated_at=data.updated_at,
            finalized_block=data.finalized_block,
            age_seconds=age,
            stale=age > 180,
            ready_submissions=sum(t.ready_submissions for t in data.targets),
            active_sessions=sum(t.session_state is not None for t in data.targets),
        )

    def collection(name: str, offset: int, limit: int) -> Collection[Any]:
        data = snapshot()
        items = getattr(data, name)
        return Collection(
            updated_at=data.updated_at, total=len(items), items=items[offset : offset + limit]
        )

    @router.get("/targets", response_model=Collection[PublicTarget])
    def targets(offset: Offset = 0, limit: Limit = 100) -> Collection[Any]:
        return collection("targets", offset, limit)

    @router.get("/submissions", response_model=Collection[PublicSubmission])
    def submissions(offset: Offset = 0, limit: Limit = 100) -> Collection[Any]:
        return collection("submissions", offset, limit)

    @router.get("/evaluations", response_model=Collection[PublicEvaluation])
    def evaluations(offset: Offset = 0, limit: Limit = 100) -> Collection[Any]:
        return collection("evaluations", offset, limit)

    @router.get("/winners", response_model=Collection[PublicWinner])
    def winners(offset: Offset = 0, limit: Limit = 100) -> Collection[Any]:
        return collection("winners", offset, limit)

    @router.get("/emissions", response_model=Collection[Emission])
    def emissions(offset: Offset = 0, limit: Limit = 100) -> Collection[Any]:
        return collection("emissions", offset, limit)

    @router.get("/evaluations/{job_id}", response_model=PublicEvaluation)
    def report(job_id: str) -> PublicEvaluation:
        try:
            value = views.report(job_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(503, "Published report view is unavailable") from exc
        if value is None:
            raise HTTPException(404, "Evaluation report not found")
        return value

    return router
