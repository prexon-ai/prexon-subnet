from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

from shared.contracts import Award
from shared.targets import TargetConfig

# Share of a target's allocation by winner rank, newest first. The whole
# allocation is always paid out while the target has at least one winner.
REWARD_SPLITS: dict[int, tuple[float, ...]] = {
    1: (1.0,),
    2: (0.7, 0.3),
    3: (0.5, 0.35, 0.15),
}


def reward_shares(
    awards: Sequence[Award], owners: Mapping[int, str], now: float
) -> list[tuple[Award, float]]:
    """A target's rewarded winners, newest first, with each one's share of its allocation.

    `awards` are one target's awards in their durable promotion sequence. A winner
    whose hotkey no longer owns its UID is passed over, so an older winner moves up.
    Rewards do not expire: a winner is paid until three newer ones replace it.
    """
    eligible = [
        award
        for award in awards
        if award.promoted_at <= now and owners.get(award.uid) == award.hotkey
    ]
    rewarded = eligible[-3:][::-1]
    return list(zip(rewarded, REWARD_SPLITS.get(len(rewarded), ()), strict=True))


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
        groups[award.profile_id].append(award)
    unassigned = 1.0
    for target in targets:
        for award, fraction in reward_shares(groups[target.identity], owners, now):
            share = target.allocation / 100 * fraction
            weights[award.uid] += share
            unassigned -= share
    # Only a target with no eligible winner (or no target at all) leaves a share to burn.
    if unassigned > 1e-12:
        weights[burn_uid] += unassigned
    return dict(sorted(weights.items()))
