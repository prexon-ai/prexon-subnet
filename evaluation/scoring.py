"""Benchmark evidence decisions reusable independently of subnet authority."""

from __future__ import annotations

import math
from statistics import mean

from evaluation.datasets.sampling import validate_selection
from evaluation.metrics import throughput_summary
from shared.config import TargetConfig
from shared.contracts import BenchmarkEvidence, Decision, Measurement


def score_benchmark(
    target: TargetConfig, result: BenchmarkEvidence, arms: tuple[str, ...]
) -> Decision:
    """Common benchmark decision; production adds identity, expiry and qualification gates."""

    def inconclusive(reason: str) -> Decision:
        return Decision(outcome="inconclusive", reason=reason)

    policy = target.policy
    if result.status != "complete" or result.finished_at <= result.started_at:
        return inconclusive("incomplete_or_expired_evidence")
    try:
        validate_selection(target, result.dataset_selection)
    except ValueError:
        return inconclusive("dataset_selection_mismatch")
    if (
        not {"target", "candidate"} <= set(arms)
        or len(set(arms)) != len(arms)
        or set(arms) - {"target", "candidate", "public", "incumbent"}
    ):
        return inconclusive("invalid_comparison_arms")
    if not result.reference_correctness_passed:
        return inconclusive("reference_correctness_failed")
    hardware = result.hardware
    if (
        hardware is None
        or hardware.engine_version != target.engine_version
        or len(hardware.gpu_names) != target.hardware.gpu_count
        or len(hardware.gpu_memory_gb) != target.hardware.gpu_count
        or hardware.tensor_parallel_size != target.tensor_parallel_size
        or any(
            target.hardware.gpu_type.lower().replace(" ", "") not in n.lower().replace(" ", "")
            for n in hardware.gpu_names
        )
        or any(v < target.hardware.min_vram_gb for v in hardware.gpu_memory_gb)
    ):
        return inconclusive("hardware_or_runtime_mismatch")
    if target.hardware.require_nvlink and "NV" not in hardware.topology:
        return inconclusive("interconnect_not_verified")
    if (
        len(result.reference_before) != len(target.workloads)
        or len(result.reference_after) != len(target.workloads)
        or any(
            abs(after / before - 1) > policy.max_reference_drift
            for before, after in zip(result.reference_before, result.reference_after, strict=True)
        )
    ):
        return inconclusive("reference_drift")
    references = [arm for arm in arms if arm != "candidate"]
    expected = {
        (w.name, r, arm) for w in target.workloads for r in range(policy.repeats) for arm in arms
    }
    rows: dict[tuple[str, int, str], Measurement] = {}
    for m in result.measurements:
        key = (m.workload, m.repeat, m.arm)
        if key in rows or key not in expected:
            return inconclusive("unexpected_or_duplicate_measurement")
        rows[key] = m
    if rows.keys() != expected:
        return inconclusive("missing_measurement")
    for w in target.workloads:
        for r in range(policy.repeats):
            for arm in arms:
                m = rows[w.name, r, arm]
                budget = w.requests * w.output_tokens
                if m.output_tokens != budget or m.expected_tokens != budget:
                    return inconclusive("incomplete_output_budget")
    if not result.correctness_passed:
        return Decision(outcome="lost", reason="correctness_gate_failed")
    for w in target.workloads:
        candidate = [rows[w.name, r, "candidate"] for r in range(policy.repeats)]
        if w.max_p95_latency_seconds and any(
            m.p95_latency_seconds > w.max_p95_latency_seconds for m in candidate
        ):
            return Decision(outcome="lost", reason="service_requirement_failed")
        log_candidate = mean(math.log(m.throughput) for m in candidate)
        strongest = max(
            mean(math.log(rows[w.name, r, arm].throughput) for r in range(policy.repeats))
            for arm in references
        )
        if log_candidate < strongest + math.log1p(-policy.max_regression):
            return Decision(outcome="lost", reason="required_workload_regression")
    summary = throughput_summary(target, result.measurements, tuple(arms))
    scores, threshold, lower_gain = summary.scores, summary.threshold, summary.lower_gain
    gain = 1 + policy.improvement_margin
    won = scores["candidate"] > gain * threshold + 1e-12 and lower_gain > gain + 1e-12
    return Decision(
        outcome="won" if won else "lost",
        reason="qualified_improvement" if won else "insufficient_confident_improvement",
        score=scores["candidate"],
        threshold=threshold,
        lower_gain=lower_gain,
    )
