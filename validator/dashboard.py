"""Publish explicit public views; private state and raw jobs never leave this boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from shared.config import TargetConfig
from shared.contracts import (
    Commitment,
    DashboardSnapshot,
    Decision,
    Emission,
    EvaluationResult,
    PublicEvaluation,
    PublicSubmission,
    PublicTarget,
    PublicWinner,
)
from validator.state import State, target_key


def publish_snapshot(
    state: State,
    path: Path,
    targets: tuple[TargetConfig, ...],
    owners: Mapping[int, str],
    weights: dict[int, float],
    now: float,
    block: int,
) -> DashboardSnapshot:
    awards_by_id = {a.submission_id: a for a in state.awards()}
    releases = {
        submission_id: release
        for submission_id, release in state.publications().items()
        if submission_id in awards_by_id
        and release.source_artifact == awards_by_id[submission_id].artifact
        and release.artifact == awards_by_id[submission_id].artifact
    }
    sessions = {s["target_key"]: s["state"] for s in state.active_sessions()}
    public_targets = tuple(
        PublicTarget(
            competition_id=t.identity,
            slot=t.slot,
            model_id=t.model_id,
            revision=t.revision,
            methods=t.methods,
            allocation=t.allocation,
            gpu_type=t.hardware.gpu_type,
            gpu_count=t.hardware.gpu_count,
            ready_submissions=state.target_ready_count(t),
            session_state=sessions.get(target_key(t)),
        )
        for t in targets
    )
    submissions = []
    for row in state.submissions():
        if row["id"] not in releases or row["status"] != "won":
            continue
        c = Commitment.model_validate_json(row["commitment"])
        if c.artifact != releases[row["id"]].source_artifact:
            continue
        submissions.append(
            PublicSubmission(
                submission_id=c.identity,
                competition_id=row["profile_id"],
                artifact=c.artifact,
                hotkey=c.hotkey,
                uid=c.uid,
                block=c.block,
                status=row["status"],
                score=row["score"],
                reason=None,
            )
        )
    evaluations = []
    for row in state.evaluations():
        if row["submission_id"] not in releases or not re.fullmatch(r"[0-9a-f]{32}", row["id"]):
            continue
        result = EvaluationResult.model_validate_json(row["result"])
        decision = Decision.model_validate_json(row["decision"])
        if decision.outcome != "won" or result.submission_id != row["submission_id"]:
            continue
        evaluations.append(
            PublicEvaluation(
                job_id=row["id"],
                submission_id=row["submission_id"],
                competition_id=result.competition_id,
                outcome=decision.outcome,
                reason="promoted_winner",
                score=decision.score,
                threshold=decision.threshold,
                lower_gain=decision.lower_gain,
                measurements=result.measurements,
                timings=result.timings,
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    report_dir = path.parent / "reports"
    report_dir.mkdir(exist_ok=True)
    allowed_reports = {f"{e.job_id}.json" for e in evaluations}
    for old_report in (*report_dir.glob("*.json"), *report_dir.glob("*.tmp")):
        if old_report.name not in allowed_reports:
            old_report.unlink()
    for evaluation in evaluations:
        report = report_dir / f"{evaluation.job_id}.json"
        temp_report = report.with_suffix(".tmp")
        temp_report.write_text(evaluation.model_dump_json() + "\n")
        temp_report.chmod(0o644)
        temp_report.replace(report)
    winners = []
    awards = state.awards(targets)
    for target in targets:
        recent = [a for a in awards if a.competition_id == target.identity][-5:]
        for award in recent:
            release = releases.get(award.submission_id)
            if release is None:
                continue
            winners.append(
                PublicWinner(
                    **award.model_dump(),
                    checkpoint_url=release.checkpoint_url,
                    published_at=release.published_at,
                    eligible=(
                        award.promoted_at <= now < award.expires_at
                        and owners.get(award.uid) == award.hotkey
                    ),
                )
            )
    snapshot = DashboardSnapshot(
        updated_at=now,
        finalized_block=block,
        targets=public_targets,
        submissions=tuple(submissions[-1000:]),
        evaluations=tuple(evaluations[-1000:]),
        winners=tuple(winners),
        releases=tuple(releases.values()),
        emissions=tuple(Emission(uid=uid, share=share) for uid, share in sorted(weights.items())),
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(snapshot.model_dump_json() + "\n")
    temporary.chmod(0o644)
    temporary.replace(path)
    return snapshot
