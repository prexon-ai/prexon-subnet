# Miner workflow

Install `uv sync --frozen --extra miner`. Training and GPU benchmarking additionally
need a compatible CUDA environment with the selected SGLang build and
`specforge==0.2.0`. Keep training/inference libraries in the GPU environment; they
are deliberately absent from the CPU role dependency lock. Check official recipes
for the chosen model and hardware rather than assuming every target is supported.

Configure miners with command arguments. The default API base is
`https://prexon.ai/api/v1`; submission authorization goes to
`https://prexon.ai/api/v1/uploads`. Miners use their Bittensor wallet and do not need
storage credentials or the validator's private benchmark profile.

| Option | Purpose / default |
| --- | --- |
| `--api-url` | Full versioned API base; default `https://prexon.ai/api/v1` |
| `--network` | Bittensor network or RPC endpoint; default `finney` |
| `--netuid` | Supply the actual Prexon subnet ID |
| `--wallet-name`, `--wallet-hotkey` | Both default to `default` |
| `--wallet-path` | Default `~/.bittensor/wallets` |
| `--max-upload-bytes` | Compressed upload limit; default 4 GiB + 2 MiB |
| `--max-checkpoint-bytes` | Local checkpoint limit; default 4 GiB without a profile |
| `--env-file` | Optional miner dotenv, explicitly selected; command arguments take precedence |

Common connection/wallet options work before or after the subcommand. Miner
commands do not automatically load the server's `.env`. To use a separate file,
pass `--env-file .env.miner`; a template is not required. `--profile PATH` remains
available for a full target profile, and `--slot` selects a target from that file or
an explicitly supplied dotenv. Direct training/packaging uses `--model-id` and
`--revision`; packaging also requires the published `--competition-id`.

In the commands below, supply the real `TARGET_REVISION`, `COMPETITION_ID` and
`NETUID` values. Revisions are exact 40-character checkpoint commits; competition
IDs are the validator's published 64-character digests.

Adapt a complete upstream SpecForge YAML recipe with `training.strategy: dflash`
and a draft config declaring `DFlash2DraftModel`. Include dataset identities,
licenses, revisions, training settings, lineage, and reproducible deployment notes.
The wrapper resolves the exact target revision, disables remote code, records the
resolved recipe, invokes the public trainer, and exports a Hugging Face checkpoint:

```bash
uv run prexon-miner train --model-id org/model --revision "$TARGET_REVISION" \
  --recipe /path/recipe.yaml \
  --run-dir /path/run --checkpoint /path/run/training/your-saved-checkpoint \
  --output /path/exported-head
```

The saved checkpoint path must match the upstream recipe's actual output naming.
Training errors stop export. Inspect `training-record.json` in the run directory
and use the resolved recipe hash in your provenance record.
Training can precede competition selection; its record has no competition ID
unless supplied by `--competition-id` or a full profile.

Use the standalone evaluator for either miner benchmarking or subnet development:

```bash
uv run prexon-evaluate --profile profiles/local.json
uv run prexon-evaluate --profile profiles/local.json --mode full
```

See [standalone evaluation](evaluation.md) for the profile, head/full modes,
Lium or existing-pod backends, reports and cleanup. The implementation is in the
root `evaluation/` package and is imported by the validator. Miner commands retain
training/export, inspection, packaging and chain submission.

Prepare provenance using `docs/examples/provenance.json` as a field template. Replace
all placeholders. Packaging permits only the DFlash2 config and safetensors files
(plus an optional shard index); executable code and pickle formats are rejected.

```bash
uv run prexon-miner package --model-id org/model --revision "$TARGET_REVISION" \
  --competition-id "$COMPETITION_ID" --checkpoint /path/exported-head \
  --provenance /path/provenance.json --output /path/head.tar.gz
uv run prexon-miner submit /path/head.tar.gz --netuid "$NETUID" \
  --wallet-name miner --wallet-hotkey default --upload-only
uv run prexon-miner submit /path/head.tar.gz --netuid "$NETUID" \
  --wallet-name miner --wallet-hotkey default
```

The submit command performs this sequence:

1. Copy the package to private temporary storage and validate its archive, file
   hashes, checkpoint, target/method and configured size limit.
2. Sign an authorization request binding your hotkey, network/netuid, published
   competition ID, exact compressed digest/size, timestamp and fresh nonce.
3. Request a short-lived PUT capability from the API base plus `/uploads`, then
   stream the frozen bytes directly to private Hippius staging with
   the required `application/gzip`, exact `Content-Length` and signed
   `x-amz-meta-prexon-competition-id` headers.
4. After PUT succeeds, reveal `px1:<sha256>` on Bittensor with the configured wallet.

`--upload-only` still uses your wallet to sign upload authorization, but sends no
chain transaction. A failed or redirected PUT prevents commitment; the CLI does
not retry a failed chain transaction automatically. The uploaded package remains
available after a chain failure. Preserve the original package/digest for a later
retry; submitting again requests a fresh capability and consumes another grant.
Training can overwrite its source path after freezing without changing uploaded or
committed bytes. Capability URLs and authorization signatures stay private.
The package already records the model, revision and competition, so `submit` works
without a model profile. If a profile is supplied, its target and byte limits are
checked as well; the validator always applies its authoritative admission policy.

Remote API URLs must use HTTPS. Explicit `http://localhost`, loopback IPv4 or
loopback IPv6 URLs support local development; granted storage URLs still require
HTTPS. Upload authorization defaults to a 900-second PUT URL, a request no older
than 60 seconds, and four grants per hotkey per rolling hour. Keep clocks
synchronized. The API also requires a fresh validator policy, current subnet
registration, an accepted competition, and a package within its target/global
limits. See [API settings and responses](api.md) for retryable failures.

For local development, pass `--api-url http://127.0.0.1:8000/api/v1`. The URL is the
versioned base; the client adds only `/uploads`, including when the base ends in `/`.

Uploading creates no ready-queue slot. After discovering your finalized commitment,
the validator reads only your hotkey-scoped staging key, verifies/freezes the bytes,
and completes CPU qualification and similarity checks. Default target sessions
become due at **at least 10 ready candidates** or **six hours after the oldest
unevaluated ready commitment**; that age survives retries. This is a scheduling
trigger, not a completion-time promise.

Your checkpoint and individual evaluation details remain private unless promoted
and published as a winner. The public dashboard exposes anonymous queue counts and
released winners; the chain itself still exposes commitment information.

The competition rewards measured improvements, not package metadata changes.
Repackaging identical config/weights does not create another eligible claim.
