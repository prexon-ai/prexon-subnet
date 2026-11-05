from __future__ import annotations

import re
from pathlib import Path

from shared.contracts import DashboardSnapshot, PublicEvaluation
from shared.releases import WinnerRelease


class PublishedViews:
    def __init__(self, path: Path) -> None:
        self.path = path

    def snapshot(self) -> DashboardSnapshot:
        if self.path.stat().st_size > 32 * 1024**2:
            raise ValueError("Dashboard snapshot exceeds size limit")
        snapshot = DashboardSnapshot.model_validate_json(self.path.read_bytes())
        releases = {r.submission_id: r for r in snapshot.releases}
        submissions = tuple(
            s.model_copy(update={"reason": None})
            for s in snapshot.submissions
            if s.submission_id in releases
            and s.artifact == releases[s.submission_id].source_artifact
            and s.status == "won"
        )
        evaluations = tuple(
            e.model_copy(update={"reason": "promoted_winner"})
            for e in snapshot.evaluations
            if e.submission_id in releases and e.outcome == "won"
        )
        winners = tuple(
            w.model_copy(
                update={
                    "checkpoint_url": releases[w.submission_id].checkpoint_url,
                    "published_at": releases[w.submission_id].published_at,
                }
            )
            for w in snapshot.winners
            if w.submission_id in releases
            and w.artifact == releases[w.submission_id].source_artifact
        )
        return snapshot.model_copy(
            update={"submissions": submissions, "evaluations": evaluations, "winners": winners}
        )

    def release(self, submission_id: str) -> WinnerRelease | None:
        if not re.fullmatch(r"[0-9a-f]{64}", submission_id):
            return None
        return next((r for r in self.snapshot().releases if r.submission_id == submission_id), None)

    def report(self, job_id: str) -> PublicEvaluation | None:
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            return None
        snapshot = self.snapshot()
        for evaluation in snapshot.evaluations:
            if evaluation.job_id == job_id:
                return evaluation
        path = self.path.parent / "reports" / f"{job_id}.json"
        if not path.exists():
            return None
        if path.stat().st_size > 8 * 1024**2:
            raise ValueError("Public report exceeds size limit")
        evaluation = PublicEvaluation.model_validate_json(path.read_bytes())
        allowed = {r.submission_id for r in snapshot.releases}
        if (
            evaluation.job_id != job_id
            or evaluation.submission_id not in allowed
            or evaluation.outcome != "won"
        ):
            return None
        return evaluation.model_copy(update={"reason": "promoted_winner"})
