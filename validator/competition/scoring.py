"""Production identity, qualification and expiry gates around common benchmark scoring."""

from __future__ import annotations

from evaluation.datasets.sampling import seed_digest
from evaluation.scoring import score_benchmark
from shared.contracts import Decision, EvaluationJob, EvaluationResult


def score_evidence(job: EvaluationJob, result: EvaluationResult) -> Decision:
    def inconclusive(reason: str) -> Decision:
        return Decision(outcome="inconclusive", reason=reason)

    target = job.target
    if (
        result.job_id != job.job_id
        or result.session_id != job.session_id
        or result.competition_id != target.identity
        or result.submission_id != job.commitment.identity
        or result.incumbent_artifact != job.incumbent_artifact
    ):
        return inconclusive("result_identity_mismatch")
    if (
        result.status != "complete"
        or result.finished_at > job.deadline
        or result.finished_at <= result.started_at
    ):
        return inconclusive("incomplete_or_expired_evidence")
    if not result.qualification_passed or not target.qualification_artifact:
        return inconclusive("unqualified_runtime")
    if target.datasets is not None:
        selection = result.dataset_selection
        context = job.round_context
        if (
            context is None
            or result.round_context != context
            or context.seed_block != context.cutoff_block + target.datasets.seed_block_delay
            or job.commitment.block > context.cutoff_block
            or selection is None
            or job.expected_dataset_selection is None
            or selection != job.expected_dataset_selection
            or selection.manifest != target.datasets.manifest
            or selection.plan_digest != job.dataset_plan_digest
            or selection.seed != seed_digest(target, context.seed(target.identity))
        ):
            return inconclusive("dataset_selection_mismatch")
    arms = ["target", "candidate"]
    if target.public_reference is not None:
        arms.append("public")
    if job.incumbent_artifact is not None:
        arms.append("incumbent")
    return score_benchmark(target, result, tuple(arms))
