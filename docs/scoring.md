# Scoring and emissions

Prexon measures **greedy decoding** with a fixed output-token budget, temperature
zero, and EOS ignored equally for every arm. A target profile pins the target
checkpoint, dtype, engine version, worker image digest, total GPU allocation,
workload settings, methods, and evaluation policy. An operator qualification
artifact is mandatory; finite output comparisons supplement review of the
runtime's verification algorithm.

## Measurement

For workload j, Q is complete output tokens divided by the independently measured
wall time of the full closed batch. Requests run up to the configured concurrency.
Request latency excludes client-side waiting for a concurrency slot; full batch
throughput includes it. Warmup and model loading are excluded from steady-state Q.
Every arm uses the same prompts, request count, output budget, and warmup procedure.
Dataset workloads use the [pinned general-purpose pools](datasets.md), sampled
without replacement once per session after the submission cutoff. Every candidate
and reference uses that session's frozen input token IDs; reports bind measurements
to the manifest, plan and sampling seed. Missing/mismatched selection evidence is
inconclusive. Inline text workloads remain available for development smoke tests.
Startup/loading remain operational costs and must be included when estimating the
cost and duration of a rental pilot. There are no guaranteed evaluation durations.

Worker reports separately record head-artifact staging, target staging, engine
startup, warmup, reference-anchor measurement, scheduled measurement, and total job
seconds. These timings are included in sanitized dashboard reports for pilot budgeting.

Each round measures target-only, candidate, and any public/incumbent references.
Arm order is deterministically shuffled from the validator's random job identity.
The evaluator restarts the engine for each configuration and warms it consistently.
Persistent Hugging Face downloads and a per-rental local snapshot cache avoid
downloading/staging the large target for every head. GPU
weight loading still happens on engine restart. Draft-only hot swapping is not
implemented or assumed; it requires separate runtime qualification.

## Score and promotion

The score is the weighted geometric mean of candidate throughput ratios against
target-only across workloads and rounds:

```text
score = exp(mean_round(sum_j(weight_j * log(Q_candidate / Q_target_only))))
threshold = max(1, score_public_reference, score_incumbent)
```

Absent public/incumbent references contribute 1. The qualifying point estimate
must strictly exceed `(1 + improvement_margin) * threshold`. Each paired comparison
against target-only/public/incumbent must also establish that gain with a one-sided
Student t lower bound on per-round weighted log-throughput differences. Bonferroni
adjustment covers the required comparisons jointly at the configured confidence.
Rounds are the statistical units; requests within a batch are not independent
replicates. This protocol assumes sufficiently stable round-level log differences;
its coverage, repeat count, workload mix, and margins need pilot calibration.

Correctness, complete output budgets, hardware/runtime matching, bounded reference
drift, configured p95 latency SLOs, and required-cell regression checks are gates.
Candidate output IDs must match the reference greedy output IDs. Public, incumbent,
or target-reference inconsistencies make evidence inconclusive. Infrastructure
failures/missing evidence do not become fabricated miner losses. The cell regression
check compares each candidate workload's geometric throughput against the strongest
reference on that workload. All scheduled measurements are retained.

SGLang counters, when provided, report strict accepted draft tokens, proposed draft
tokens, and verification steps. Acceptance rate is accepted/proposed; average
acceptance length is completed tokens/verification steps and **includes the
correction/bonus token**. Missing runtime counters remain null. These diagnostics
have no reward weight.

The example defaults are a 1.5% margin, 95% confidence, 5% maximum regression/drift,
five rounds, and 20/30/50 workload weights. Calibrate them with the intended
target, runtime, and hardware before production.

## Emissions

Each target's five most recent promoted winners occupy its reward slots. Every
eligible slot earns one fifth of that target's allocation. An award ends at its
recorded expiry (five days by default) or displacement by five later winners.
Expiry does not reset the incumbent. Missing, expired, or UID/hotkey-ineligible
shares burn; no shares are redistributed. Awards aggregate across targets. Retries
do not create duplicate awards or reset their start times. Weight publication is a
separate retryable chain operation, not a new promotion.

Runtime references:

- [SpecForge DFlash2](https://sgl-project.github.io/SpecForge/concepts/DFlash2.html)
- [SpecForge training/export](https://github.com/sgl-project/SpecForge/blob/main/docs/sections/basic_usage/training.md)
- [SGLang speculative counters](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/speculative/spec_decoding_metrics.py)
