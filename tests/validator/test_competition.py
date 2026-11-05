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


def test_five_successive_awards_expiry_and_ownership():
    t = target()
    awards = [
        Award(
            submission_id=f"{i:064x}",
            competition_id=t.identity,
            hotkey=f"miner{i}",
            uid=i,
            promoted_at=100 + i,
            expires_at=1000,
            artifact="c" * 64,
        )
        for i in range(1, 7)
    ]
    owners = {i: f"miner{i}" for i in range(1, 7)}
    assert emission_weights(awards, (t,), owners, 200, 0) == {
        2: 0.2,
        3: 0.2,
        4: 0.2,
        5: 0.2,
        6: 0.2,
    }
    owners[6] = "replacement"
    weights = emission_weights(awards, (t,), owners, 200, 0)
    assert weights[0] == pytest.approx(0.2)
    assert 1 not in weights and 6 not in weights
    assert emission_weights(awards, (t,), owners, 1000, 0) == {0: 1.0}


def test_two_targets_split_and_empty_slots_burn():
    a, b = target(1, 20), target(2, 80)
    awards = [
        Award(
            submission_id="a" * 64,
            competition_id=a.identity,
            hotkey="miner",
            uid=1,
            promoted_at=100,
            expires_at=1000,
            artifact="c" * 64,
        ),
        Award(
            submission_id="b" * 64,
            competition_id=b.identity,
            hotkey="miner",
            uid=1,
            promoted_at=100,
            expires_at=1000,
            artifact="d" * 64,
        ),
    ]
    weights = emission_weights(awards, (a, b), {1: "miner"}, 200, 0)
    assert weights[1] == pytest.approx(0.2)
    assert weights[0] == pytest.approx(0.8)
    assert emission_weights([], (), {}, 200, 0) == {0: 1.0}


def test_equal_promotion_times_preserve_durable_succession_order():
    t = target()
    awards = [
        Award(
            submission_id=f"{100 - i:064x}",
            competition_id=t.identity,
            hotkey=f"miner{i}",
            uid=i,
            promoted_at=100,
            expires_at=1000,
            artifact="c" * 64,
        )
        for i in range(1, 7)
    ]
    owners = {i: f"miner{i}" for i in range(1, 7)}
    assert emission_weights(awards, (t,), owners, 200, 0) == {
        2: 0.2,
        3: 0.2,
        4: 0.2,
        5: 0.2,
        6: 0.2,
    }
