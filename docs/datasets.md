# Evaluation datasets

Each round takes its prompts by the competition's `datasets.mix`: a percent per category
and length bucket, adding up to 100. In the default (in `competitions/shared.json`) long
prompts all come from chat and maths prompts are all short, because coding questions are
almost never long and maths problems rarely pass 512 tokens (the Qwen3.8-27B bank has
215,504 long chat prompts but 1 long code, 111 medium maths and 3 long maths prompts):

| % of a round | short | medium | long | total |
| --- | --- | --- | --- | --- |
| chat | 10 | 10 | 30 | 50 |
| code | 10 | 20 | — | 30 |
| math | 20 | — | — | 20 |
| total | 40 | 30 | 30 | 100 |

Keep every share well below its shelf: a share that takes most of a small shelf repeats
the same prompts round after round.

These percentages are separate from emission allocations and concurrency score
weights. Counts per shelf are fixed per workload using largest-remainder rounding with
stable ties.

| Source | Input used | Preparation rule |
| --- | --- | --- |
| [WildChat](https://huggingface.co/datasets/allenai/WildChat-4.8M) | Conversation through its last user turn | One input per conversation; exclude the final assistant answer. Earlier assistant turns remain conversation context. |
| [OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) | `input` | Ignore stored code, tests and judgments. |
| [OpenMathReasoning](https://huggingface.co/datasets/nvidia/OpenMathReasoning) | `problem` | Deduplicate problems across solution rows; ignore solutions/expected answers. |
| [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) (optional) | `article` with a summarization instruction | Ignore stored highlights. |

All references and candidates generate live with the target. Dataset answers are
never the speculative-decoding correctness oracle. These corpora are public
training material: rotating samples do not guarantee unseen inputs or prevent
training overlap. Exact normalized questions and identical tokenized inputs are
deduplicated globally; semantic near-duplicates and unknown source families are
not claimed to be removed.

## Prepare once on CPU

Install preparation dependencies without adding them to the API or GPU worker:

```bash
uv sync --frozen --extra datasets
```

The recipe is the competition's own settings: its model and revision, the pinned
`sources` in `competitions/shared.json`, and the model's `chat_template_kwargs` (for
example `{"reasoning_effort": "xhigh"}` for Qwen3.8) in its file in `competitions/`.
Preparation streams every configured split, reading only the column a prompt uses; the
first pass can take substantial CPU time, bandwidth and disk. It does not download model
weights or rent GPUs. It loads the target tokenizer with remote code disabled and applies
its chat template with `add_generation_prompt=true`. Tokenizing runs on every CPU by
default (`--workers`); the bank is the same byte for byte for any number of workers.

```bash
uv run --extra datasets prexon-evaluate prepare-datasets --competition Qwen/Qwen3.8-27B
```

The command prints a manifest SHA-256. The completed snapshot lives at
`runtime/datasets/<manifest>/`. It contains JSONL prompt files, fixed-width
random-access indexes and `manifest.json`, including source revisions,
normalization version, tokenizer/model revision, template options, library
versions, counts and file hashes. A temporary on-disk SQLite uniqueness index
bounds preparation memory; it is removed before publication. Failed preparation
never publishes a partial snapshot. Keep completed snapshots immutable.

Default token-length buckets include chat-template tokens:

| Bucket | Input tokens |
| --- | --- |
| short | 1–512 |
| medium | 513–2048 |
| long | 2049–6144 |

Configure `buckets` in preparation and the target's `datasets` object to change
these ranges. Names/ranges must match; ranges cannot overlap. Out-of-range rows
are skipped, not truncated. A shelf no prompt reaches (for example long maths) is left
out of the bank; a mix that asks for a missing or undersized shelf fails selection instead
of shrinking samples, redistributing shares or repeating prompts. Check prepared counts
before publishing a competition.

For optional document coverage, add a source with `kind: "document"`,
`category: "document"`, `repo_id: "abisee/cnn_dailymail"`, subset `"3.0.0"`,
revision `"96df5e686bee6baa90b8bee7c28b81fa3fa6223d"`, and desired splits.
Give it percentages in the `mix` (rebalancing the rest to add up to 100) and prepare a new pool.

## Configure a target

Both validator target profiles and standalone `profile.target` accept:

```json
{
  "datasets": {
    "manifest": "REPLACE_WITH_PRINTED_64_CHARACTER_SHA256",
    "mix": {"chat": {"short": 10, "medium": 10, "long": 30},
            "code": {"short": 10, "medium": 20},
            "math": {"short": 20}},
    "min_unique_prompts": 512,
    "seed_block_delay": 1
  },
  "workloads": [
    {"name": "low", "concurrency": 1, "requests": 512,
     "output_tokens": 256, "weight": 0.2},
    {"name": "medium", "concurrency": 8, "requests": 512,
     "output_tokens": 256, "weight": 0.3},
    {"name": "high", "concurrency": 32, "requests": 512,
     "output_tokens": 256, "weight": 0.5}
  ]
}
```

Remove inline `prompts` when configuring datasets. Each workload needs at least
`min_unique_prompts` requests and enough context for its longest input bucket
plus `output_tokens`. The 512-prompt default is a **pilot**, not a proven sample
size for detecting a 1.5% gain. Calibrate diversity, total output work, repeats,
variance and GPU cost before production. Repeats reuse the selected prompts to
measure execution noise; they do not count as additional unique inputs. Output
budgets remain configurable per workload: add cells for short/long outputs or
different serving loads, with published weights.

Set `DATASET_CACHE` in `evaluation/config.py`; the controller and the evaluator both
use it (default: `runtime/datasets` in the checkout). Changing sources,
rendering or sampling policy changes the competition's profile hash and requires
reference remeasurement and runtime qualification. Inline text prompts
remain available for small smoke tests; dataset workloads never cycle prompts.

## Round reproducibility

Production freezes ready submission membership when claiming a session, then
reads a fresh finalized chain height and persists it as the sampling cutoff.
Reading the head after membership freezes prevents slow admission/qualification
work from turning future entropy into an already-known block. New submissions,
including older late admissions, wait for another round without blocking the
frozen members. The controller waits until the exact
`cutoff + seed_block_delay` block is finalized, then combines its hash, chain
network/netuid and the competition's profile hash into a versioned seed. It does not start
new dataset rounds while replaying historical chain snapshots.

The controller samples without replacement using SHA-256 counter sampling,
verifies indexes and selected record hashes, and writes one content-addressed
plan before provisioning. Equal request counts share the same sample across
concurrency/output settings. All submissions, arms, drift anchors and repeats
use that frozen plan. State persists its identity and block context across
restarts; jobs get a scoped plan URL and the exact expected selection summary.
The worker and scoring gate compare selected prompt IDs with that summary, not
just a reported plan digest. Full corpora remain on CPU/shared storage;
only selected token IDs travel to GPU pods.

The file-chain development adapter requires an explicit `block_hashes` mapping
for sampling blocks, e.g. `"111": "0x<64 hex characters>"`, alongside a snapshot
whose `block` has reached that height. It never invents network entropy.

Standalone pod/Lium evaluation uses the profile's explicit `seed` instead of
chain entropy and remains non-authoritative. Lium freezes the plan before renting
and transfers it with the heads. The remote host needs neither the full dataset
cache nor preparation dependencies. Resume reuses and validates the journal's
frozen plan even if the original corpus is unavailable. Changed inputs require
cleanup/new execution as before.

Private reports record manifest/plan/seed identities, selected prompt IDs,
category/length counts and production chain context. Private logs retain the
tokenized plan for replay. Dashboard responses expose aggregated measurements,
not prompt inputs. Missing/mismatched dataset evidence makes results inconclusive.
