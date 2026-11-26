"""Publish the public snapshot; private state and raw jobs never leave this boundary."""

from __future__ import annotations

import re
from collections.abc import Mapping

from shared.artifacts import ArtifactStore
from shared.contracts import (
    Commitment,
    DashboardSnapshot,
    Decision,
    Emission,
    EvaluationJob,
    EvaluationResult,
    PublicEvaluation,
    PublicSubmission,
    PublicTarget,
    PublicWinner,
)
from shared.targets import TargetConfig
from validator.competition import reward_shares
from validator.public_files import publish_file
from validator.state import State, target_key


def publish_snapshot(
    state: State,
    store: ArtifactStore,
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
        measured = EvaluationJob.model_validate_json(row["job"]).target
        evaluations.append(
            PublicEvaluation(
                job_id=row["id"],
                submission_id=row["submission_id"],
                competition_id=result.competition_id,
                model_id=measured.model_id,
                revision=measured.revision,
                outcome=decision.outcome,
                reason="promoted_winner",
                score=decision.score,
                threshold=decision.threshold,
                lower_gain=decision.lower_gain,
                measurements=result.measurements,
                timings=result.timings,
            )
        )
    winners = []
    awards = state.awards(targets)
    for target in targets:
        own = [a for a in awards if a.competition_id == target.identity]
        for award, share in reward_shares(own, owners, now):
            release = releases.get(award.submission_id)
            if release is None:
                continue
            winners.append(
                PublicWinner(
                    **award.model_dump(),
                    share=share,
                    checkpoint_url=release.checkpoint_url,
                    published_at=release.published_at,
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
    data = (snapshot.model_dump_json() + "\n").encode()
    publish_file(state, store, "dashboard/snapshot.json", data, now, timed=True)
    return snapshot
