"""Numerical summaries shared by authoritative scoring and local previews."""

from __future__ import annotations

import math
from statistics import mean, stdev

from scipy.stats import t as student_t

from shared.config import FrozenModel, TargetConfig
from shared.contracts import Measurement


class ThroughputSummary(FrozenModel):
    scores: dict[str, float]
    threshold: float
    lower_gain: float


def throughput_summary(
    target: TargetConfig,
    measurements: tuple[Measurement, ...],
    arms: tuple[str, ...],
) -> ThroughputSummary:
    if (
        not {"target", "candidate"}.issubset(arms)
        or len(set(arms)) != len(arms)
        or not set(arms).issubset({"target", "candidate", "public", "incumbent"})
    ):
        raise ValueError("Invalid measured configurations")
    repeats = target.policy.repeats
    expected = {(w.name, r, arm) for w in target.workloads for r in range(repeats) for arm in arms}
    rows: dict[tuple[str, int, str], Measurement] = {
        (m.workload, m.repeat, m.arm): m for m in measurements
    }
    if len(rows) != len(measurements) or rows.keys() != expected:
        raise ValueError("Missing, duplicate, or unexpected measurements")
    for w in target.workloads:
        for r in range(repeats):
            for arm in arms:
                m = rows[w.name, r, arm]
                if (
                    m.output_tokens != w.requests * w.output_tokens
                    or m.expected_tokens != m.output_tokens
                ):
                    raise ValueError("Incomplete output budget")
    logs = {
        arm: [
            sum(w.weight * math.log(rows[w.name, r, arm].throughput) for w in target.workloads)
            for r in range(repeats)
        ]
        for arm in arms
    }
    baseline = mean(logs["target"])
    scores = {arm: math.exp(mean(values) - baseline) for arm, values in logs.items()}
    references = [arm for arm in arms if arm != "candidate"]
    quantile = float(
        student_t.ppf(1 - (1 - target.policy.confidence) / len(references), repeats - 1)
    )
    lower_bounds = []
    for arm in references:
        differences = [c - ref for c, ref in zip(logs["candidate"], logs[arm], strict=True)]
        lower = mean(differences) - quantile * stdev(differences) / math.sqrt(repeats)
        lower_bounds.append(math.exp(lower))
    return ThroughputSummary(
        scores=scores,
        threshold=max(1.0, *(scores[a] for a in references)),
        lower_gain=min(lower_bounds),
    )
