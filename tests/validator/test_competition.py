import pytest

from shared.contracts import (
    Award,
)
from tests.factories import evidence, job, target
from validator.competition import emission_weights, score_evidence


def test_target_only_denominator_and_strict_incumbent_threshold():
    j = job(incumbent="e" * 64)
    d = score_evidence(j, evidence(j, candidate=2, incumbent=1.8))
    assert d.outcome == "won"
    assert d.score == pytest.approx(2)
    assert d.threshold == pytest.approx(1.8)
    assert d.lower_gain == pytest.approx(2 / 1.8)
    assert score_evidence(j, evidence(j, candidate=1.8 * 1.015, incumbent=1.8)).outcome == "lost"


def test_geometric_weighting_and_missing_evidence():
    j = job()
    r = evidence(j)
    rows = [
        m.model_copy(update={"elapsed_seconds": 4.0})
        if m.arm == "candidate" and m.workload == "high"
        else m
        for m in r.measurements
    ]
    d = score_evidence(j, r.model_copy(update={"measurements": tuple(rows)}))
    assert d.score == pytest.approx(2**0.2 * 4**0.8)
    assert (
        score_evidence(j, r.model_copy(update={"measurements": r.measurements[:-1]})).outcome
        == "inconclusive"
    )


def test_correctness_and_reference_drift_gate():
    j = job()
    r = evidence(j)
    assert score_evidence(j, r.model_copy(update={"correctness_passed": False})).outcome == "lost"
    assert (
        score_evidence(j, r.model_copy(update={"reference_after": (1.2, 1.0)})).outcome
        == "inconclusive"
    )
    assert score_evidence(j, r.model_copy(update={"job_id": "f" * 32})).outcome == "inconclusive"


def test_noisy_gain_requires_confidence():
    j = job()
    r = evidence(j, candidate=1.1)
    rows = [
        m.model_copy(update={"elapsed_seconds": 16 / (0.5 if m.repeat == 0 else 1.4)})
        if m.arm == "candidate"
        else m
        for m in r.measurements
    ]
    assert score_evidence(j, r.model_copy(update={"measurements": tuple(rows)})).outcome == "lost"


def _award(t, i, *, promoted_at=None, uid=None):
    return Award(
        submission_id=f"{i:064x}",
        profile_id=t.identity,
        hotkey=f"miner{uid or i}",
        uid=uid or i,
        promoted_at=promoted_at or 100 + i,
        artifact="c" * 64,
    )


@pytest.mark.parametrize(
    "winners,expected",
    [
        (1, {1: 1.0}),
        (2, {2: 0.7, 1: 0.3}),
        (3, {3: 0.5, 2: 0.35, 1: 0.15}),
        # Older winners drop out once three newer ones exist.
        (6, {6: 0.5, 5: 0.35, 4: 0.15}),
    ],
)
def test_latest_winners_share_the_whole_allocation_newest_first(winners, expected):
    t = target()
    awards = [_award(t, i) for i in range(1, winners + 1)]
    owners = {i: f"miner{i}" for i in range(1, winners + 1)}
    weights = emission_weights(awards, (t,), owners, 200, 0)
    assert weights == pytest.approx(expected)
    assert 0 not in weights  # nothing is burned while a winner exists


def test_rewards_do_not_expire_while_no_newer_winner_arrives():
    t = target()
    ten_years_later = 100 + 10 * 365 * 86400
    weights = emission_weights([_award(t, 1)], (t,), {1: "miner1"}, ten_years_later, 0)
    assert weights == {1: 1.0}


def test_winner_that_left_the_subnet_is_passed_over_for_the_next_latest():
    t = target()
    awards = [_award(t, i) for i in range(1, 5)]
    owners = {i: f"miner{i}" for i in range(1, 5)}
    owners[4] = "replacement"  # the newest winner's UID now belongs to someone else
    weights = emission_weights(awards, (t,), owners, 200, 0)
    assert weights == pytest.approx({3: 0.5, 2: 0.35, 1: 0.15})


def test_one_miner_holding_several_ranks_collects_each_share():
    t = target()
    awards = [_award(t, 1, uid=7), _award(t, 2, uid=8), _award(t, 3, uid=7)]
    weights = emission_weights(awards, (t,), {7: "miner7", 8: "miner8"}, 200, 0)
    assert weights == pytest.approx({7: 0.65, 8: 0.35})


def test_target_without_an_eligible_winner_burns_only_its_own_allocation():
    a, b = target(1, 20), target(2, 80)
    weights = emission_weights([_award(a, 1)], (a, b), {1: "miner1"}, 200, 0)
    assert weights == pytest.approx({1: 0.2, 0: 0.8})
    assert emission_weights([], (), {}, 200, 0) == {0: 1.0}


def test_equal_promotion_times_preserve_durable_succession_order():
    t = target()
    awards = [_award(t, i, promoted_at=100) for i in range(1, 7)]
    owners = {i: f"miner{i}" for i in range(1, 7)}
    weights = emission_weights(awards, (t,), owners, 200, 0)
    assert weights == pytest.approx({6: 0.5, 5: 0.35, 4: 0.15})
