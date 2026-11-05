from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

from shared.config import TargetConfig
from shared.contracts import Award


def emission_weights(
    awards: Iterable[Award],
    targets: tuple[TargetConfig, ...],
    owners: Mapping[int, str],
    now: float,
    burn_uid: int,
) -> dict[int, float]:
    """Consume awards in their durable promotion sequence, including timestamp ties."""
    weights: dict[int, float] = defaultdict(float)
    groups: dict[str, list[Award]] = defaultdict(list)
    for award in awards:
        groups[award.competition_id].append(award)
    unassigned = 1.0
    for target in targets:
        share = target.allocation / 100 / 5
        recent = groups[target.identity][-5:]
        for award in recent:
            if (
                award.promoted_at <= now < award.expires_at
                and owners.get(award.uid) == award.hotkey
            ):
                weights[award.uid] += share
                unassigned -= share
    if unassigned > 1e-12:
        weights[burn_uid] += unassigned
    return dict(sorted(weights.items()))
