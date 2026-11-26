"""Public history of every model: its queue and each evaluated submission."""

from __future__ import annotations

import re
from typing import Any

from shared.artifacts import ArtifactStore
from shared.contracts import (
    Commitment,
    Decision,
    EvaluatedSubmission,
    ModelHistory,
    ModelIndex,
    ModelSummary,
    QueuedSubmission,
)
from shared.targets import TargetConfig
from validator.competition import reward_shares
from validator.public_files import publish_file
from validator.state import State, target_key


def model_slug(model_id: str, revision: str, length: int = 7) -> str:
    """A stable, URL-safe name for one model revision, e.g. `qwen3-8b-1a2b3c4`."""
    name = re.sub(r"[^a-z0-9]+", "-", model_id.rsplit("/", 1)[-1].lower()).strip("-")
    return f"{name or 'model'}-{revision[:length]}"


def _history(
    state: State,
    target: TargetConfig,
    slug: str,
    *,
    active: bool,
    period: dict[str, Any],
    shares: dict[str, float],
    releases: dict[str, Any],
    now: float,
) -> ModelHistory:
    submissions = []
    for row in state.result_rows(target):
        commitment = Commitment.model_validate_json(row["commitment"])
        decision = Decision.model_validate_json(row["decision"])
        crowned = row["status"] == "won"
        evaluated_at = row["decided_at"]
        release = releases.get(row["id"]) if crowned else None
        submissions.append(
            EvaluatedSubmission(
                submission_id=row["id"],
                hotkey=commitment.hotkey,
                uid=commitment.uid,
                submitted_at=min(row["submitted_at"] or evaluated_at, evaluated_at),
                evaluated_at=evaluated_at,
                crowned=crowned,
                speedup=decision.score,
                previous_best=decision.threshold,
                share=shares.get(row["id"], 0),
                checkpoint_url=release.checkpoint_url if release else None,
            )
        )
    submissions.sort(key=lambda s: (s.evaluated_at, s.submission_id), reverse=True)
    # A retired model evaluates nothing more, so it shows no queue.
    queue = []
    for row in state.queue_rows(target) if active else ():
        commitment = Commitment.model_validate_json(row["commitment"])
        queue.append(
            QueuedSubmission(
                submission_id=row["id"],
                hotkey=commitment.hotkey,
                uid=commitment.uid,
                submitted_at=row["submitted_at"] or now,
                status="evaluating" if row["status"] == "running" else "waiting",
            )
        )
    crowned_speedups = [s.speedup for s in submissions if s.crowned and s.speedup is not None]
    return ModelHistory(
        id=slug,
        model_id=target.model_id,
        revision=target.revision,
        slot=target.slot if active else period["slot"],
        active_from=period["active_from"],
        active_to=None if active else period["active_to"],
        allocation=target.allocation if active else period["allocation"],
        methods=target.methods,
        gpu_type=target.hardware.gpu_type,
        gpu_count=target.hardware.gpu_count,
        queued=len(queue),
        evaluated=len(submissions),
        winners=sum(s.crowned for s in submissions),
        best_speedup=max(crowned_speedups, default=None),
        reward_shares=tuple(sorted((s.share for s in submissions if s.share), reverse=True)),
        queue=tuple(queue),
        submissions=tuple(submissions),
    )


def build_history(
    state: State,
    targets: tuple[TargetConfig, ...],
    owners: dict[int, str],
    now: float,
    block: int,
) -> tuple[ModelIndex, dict[str, ModelHistory]]:
    """Every model the validator has run, active or not, from its own database."""
    enabled = {target_key(t): t for t in targets}
    # An enabled model is described by its current profile; a past one by its last.
    known = {target_key(t): t for t in state.models()} | enabled
    periods = state.model_periods()
    releases = state.publications()
    awards = state.awards(targets)
    slugs: dict[str, str] = {}
    for key, target in known.items():
        length = 7
        while (slug := model_slug(target.model_id, target.revision, length)) in slugs.values():
            length += 5  # two models sharing a name and a revision prefix
        slugs[key] = slug
    histories: dict[str, ModelHistory] = {}
    for key, target in known.items():
        active = key in enabled
        shares = {
            award.submission_id: share
            for award, share in reward_shares(
                [a for a in awards if active and a.competition_id == target.identity], owners, now
            )
        }
        histories[slugs[key]] = _history(
            state,
            target,
            slugs[key],
            active=active,
            period=periods[key],
            shares=shares,
            releases=releases,
            now=now,
        )
    ordered = sorted(
        histories.values(),
        key=lambda m: (m.active_to is not None, m.slot if m.active_to is None else -m.active_to),
    )
    summaries = tuple(
        ModelSummary.model_validate(m.model_dump(include=set(ModelSummary.model_fields)))
        for m in ordered
    )
    index = ModelIndex(updated_at=now, finalized_block=block, models=summaries)
    return index, histories


def publish_history(
    state: State,
    store: ArtifactStore,
    targets: tuple[TargetConfig, ...],
    owners: dict[int, str],
    now: float,
    block: int,
) -> ModelIndex:
    index, histories = build_history(state, targets, owners, now, block)
    for model_id, history in histories.items():
        data = (history.model_dump_json() + "\n").encode()
        publish_file(state, store, f"dashboard/models/{model_id}.json", data, now)
    data = (index.model_dump_json() + "\n").encode()
    publish_file(state, store, "dashboard/index.json", data, now, timed=True)
    return index
