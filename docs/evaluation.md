# Evaluation

`prexon-evaluate` benchmarks one DFlash2 head using the same measurement and
scoring library as the validator. Standalone reports are development results;
they do not admit submissions, promote winners, or publish weights.

## Run a benchmark

```bash
uv sync --frozen
mkdir -p profiles
cp docs/examples/local-evaluation.json profiles/local.json
```

Configure the target, image digest, GPU requirements, head path, and prepared
[dataset manifest](datasets.md), then run:

```bash
uv run prexon-evaluate --profile profiles/local.json
uv run prexon-evaluate --profile profiles/local.json --mode full
```

| Mode | Measurements | Result |
| --- | --- | --- |
| `head` (default) | Candidate speculative configuration only | Throughput, latency, and acceptance metrics |
| `full` | Target-only, candidate, and supplied references | Correctness, speedup, confidence, and a benchmark preview |

Head mode still needs the target LLM; it measures serving with that head rather
than the draft network alone. Full mode measures only supplied references and
uses no saved baseline scores. See [scoring](scoring.md) for comparison rules.

## Profile settings

The template contains a complete `target` configuration. It does not use dotenv
model slots. Paths are relative to the profile file, except `target_path` inside
a Lium pod, which must be absolute.

| Field | Purpose / default |
| --- | --- |
| `target` | Pinned model, runtime, hardware, workloads, and policy |
| `draft_head` | Exported candidate directory |
| `baseline_head`, `incumbent_head` | Optional references for full mode |
| `mode` | `head` or `full`; CLI `--mode` overrides it |
| `backend` | `lium` (default) or `pod` |
| `output_dir` | Report and rental-journal directory; default `runtime/evaluations` |
| `target_path` | Optional existing target snapshot in the GPU environment |
| `seed` | Reproducible dataset selection/order seed; default `"42"` |
| `port` | SGLang port; default 30000 |
| `session_seconds`, `session_max_cost` | Optional rental-budget overrides |

Baseline/incumbent paths are ignored in head mode. Unknown fields fail validation.
A local target path does not authenticate the checkpoint revision; use a verified
snapshot. Template values are placeholders, not qualified model/GPU combinations.

## Lium or an existing pod

For Lium, configure host credentials in `.env` or select another file with
`--env-file`. Relevant settings are:

```dotenv
LIUM_API_KEY=your-private-key
PREXON_LIUM_TEMPLATE_ID=your-template-id
PREXON_SSH_PRIVATE_KEY=~/.ssh/id_ed25519
PREXON_SSH_PUBLIC_KEY=~/.ssh/id_ed25519.pub
PREXON_LIUM_VOLUME_ID=
```

A target-specific `lium_template_id` overrides the global template. The host can
be CPU-only. A [qualified worker image](docker.md) must supply SGLang/Torch and
working SSH. Frozen heads and the selected plan travel over SSH; host credentials
and the full dataset corpus stay on the host. The target is downloaded or staged
inside the GPU pod. Persistent `HF_HOME` storage reuses model downloads.

Set `backend: "pod"` to evaluate in an existing GPU environment. If the runtime is
installed system-wide, create the project venv with
`uv venv --system-site-packages .venv` before syncing. Prexon does not install the
CUDA runtime. Both backends validate and freeze data-only heads before use.

## Reports and recovery

Completed invocations write unique reports under `output_dir/head/` or
`output_dir/full/`. The CLI prints the report path and candidate metrics; full
mode also prints target-only speedup. Throughput is the geometric mean over
repeats; reported p95 is the worst repeat. Acceptance length includes the target's
correction/bonus token. Missing engine counters are null.

Lium runs keep private `.lium.json` journals and adjacent log archives. Inputs,
mode, exact rental request, and execution identity are saved before paid actions.
A directory lock prevents concurrent ownership of an output directory. Compatible
unfinished runs resume; changed inputs require cleanup first.

```bash
# Validate and quote without renting.
uv run prexon-evaluate --profile profiles/local.json --dry-run

# Retry cleanup of unfinished rentals.
uv run prexon-evaluate --profile profiles/local.json --cleanup-only
```

An unresolved create attempt is recovered by its saved identity. If no pod is
identifiable, the run is inconclusive and cleanup remains pending. Once the saved
deadline plus the requested provider lifetime and a 60-second allowance have
elapsed, a successful lookup with no matching pod closes the journal as
`expiry_unconfirmed`. This records presumed automatic expiry, not confirmed
deletion. Failed lookups or deletion attempts remain pending. Head files are not
required for cleanup; journals are retained for reconciliation.

Deletion is attempted after completion, failure, SIGTERM, and Ctrl-C. Price,
observed spend, and deadlines are monitored; provider-side budgets are still
needed because billing can lag. Engine/staging cleanup is awaited on cancellation.

| Exit status | Meaning |
| --- | --- |
| 0 | Completed head measurements or passing full preview |
| 1 | Completed full evaluation failed benchmark gates |
| 2 | Invalid input, incomplete evidence, or pending cleanup |

Python callers use `evaluation.profile.load_profile` and
`evaluation.runner.evaluate`. Production jobs wrap shared measurement/scoring with
validator-specific identity, qualification, and deadline checks.
