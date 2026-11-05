# Evaluation datasets

The general-purpose source mix is 40% chat, 35% code and 25% math by distinct
prompt count. These are configurable sampling shares, separate from target
emission allocations and concurrency score weights. Category shares are fixed
per workload using largest-remainder rounding with stable ties.

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
mkdir -p profiles
cp docs/examples/datasets.json profiles/datasets.json
```

Set the **exact target model ID and 40-character revision** in that file. They
must match the evaluation target. Source revisions in the example are pinned
snapshots. Preparation streams every configured split: the first pass can take
substantial CPU time, bandwidth and disk. It does not download model weights or
rent GPUs. It loads the target tokenizer with remote code disabled and applies its chat template with
`add_generation_prompt=true`. Model-specific options such as `enable_thinking`
belong in `chat_template_kwargs`; they become part of the manifest identity.

```bash
uv run --extra datasets prexon-evaluate prepare-datasets \
  --config profiles/datasets.json --cache-dir runtime/datasets
```

The command prints a manifest SHA-256. The completed snapshot lives at
`runtime/datasets/<manifest>/`. It contains JSONL prompt files, fixed-width
random-access indexes and `manifest.json`, including source revisions,
normalization version, tokenizer/model revision, template options, library
versions, counts and file hashes. A temporary on-disk SQLite uniqueness index
bounds preparation memory; it is removed before publication. Failed preparation
never publishes a partial snapshot. Keep completed snapshots immutable.

Default token-length buckets include chat-template tokens:

| Bucket | Input tokens | Share within each category |
| --- | --- | --- |
| short | 1–512 | 30% |
| medium | 513–2048 | 40% |
| long | 2049–6144 | 30% |

Configure `buckets` in preparation and the target's `datasets` object to change
these ranges. Names/ranges must match; ranges cannot overlap. Out-of-range rows
are skipped, not truncated. Empty prepared strata fail preparation. Undersized
strata fail selection instead of shrinking samples, redistributing shares or
repeating prompts. Check prepared counts before publishing a competition.

For optional document coverage, add a source with `kind: "document"`,
`category: "document"`, `repo_id: "abisee/cnn_dailymail"`, subset `"3.0.0"`,
revision `"96df5e686bee6baa90b8bee7c28b81fa3fa6223d"`, and desired splits.
Explicitly rebalance `category_weights` to sum to one and prepare a new pool.

## Configure a target

Both dotenv-selected validator profiles and standalone `profile.target` accept:

```json
{
  "datasets": {
    "manifest": "REPLACE_WITH_PRINTED_64_CHARACTER_SHA256",
    "category_weights": {"chat": 0.4, "code": 0.35, "math": 0.25},
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

Set `PREXON_DATASET_CACHE=runtime/datasets` in dotenv for the controller or local
evaluator. Unlike profile head paths, this operational cache path is relative to
the working directory. Docker Compose mounts it at `/data/datasets`. Changing
sources, rendering or sampling policy changes the competition identity and
requires reference remeasurement and runtime qualification. Inline text prompts
remain available for small smoke tests; dataset workloads never cycle prompts.

## Round reproducibility

Production freezes ready submission membership when claiming a session, then
reads a fresh finalized chain height and persists it as the sampling cutoff.
Reading the head after membership freezes prevents slow admission/qualification
work from turning future entropy into an already-known block. New submissions,
including older late admissions, wait for another round without blocking the
frozen members. The controller waits until the exact
`cutoff + seed_block_delay` block is finalized, then combines its hash, chain
network/netuid and competition identity into a versioned seed. It does not start
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
