# Prexon

Prexon is a Bittensor subnet for improving speculative decoding. Miners train
DFlash2 draft heads; validators measure their serving speedup against the target
model and reward successive improvements.

The project includes miner tools, a standalone evaluator, a CPU validator,
temporary GPU workers, and a small upload API. The validator publishes the
dashboard's data to public storage, where the website reads it. Training uses
SpecForge; evaluation uses SGLang. Model and GPU combinations must be qualified before
production use.

## How it works

1. A miner packages a head, requests a signed upload URL, uploads to private
   Hippius storage, and commits the package hash on chain.
2. The validator discovers finalized commitments and checks the package,
   checkpoint compatibility, and CPU similarity before queue admission.
3. A target's GPU session starts with **10 ready submissions** or when its oldest
   unevaluated submission is **six hours old**. Submissions share a frozen workload.
4. A candidate wins by beating target-only decoding, the public reference, and
   the incumbent under correctness, reliability, and measurement gates.
5. Only winners' checkpoints and detailed results are published; every evaluated
   submission's result and speedup appear in the public model history. Each target's
   whole allocation goes to its latest winners: one winner takes all of it, two
   split it 70/30, and the three latest split it 50/35/15, newest first.

Competitions are files in `competitions/`, each named by its target model. Every
active one runs independently, with its own hardware, queue and share of emissions.
See [scoring and emissions](docs/scoring.md) for the formula and reward rules.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra validator --extra api
cp .env.example .env  # then fill in the netuid, wallet and Hippius key
uv run prexon-validator --check-config
uv run prexon-validator --once
uv run prexon-api
```

No competition is active by default. `--once` then publishes an idle dashboard
to the public bucket named in `validator/config.py` without chain access or a GPU; it
needs the Hippius key in `.env`. To try it without Hippius, set `ARTIFACT_ROOT` and
`PUBLIC_ARTIFACT_ROOT` to local directories and `UPLOAD_STAGING_ROOT = ""`. The local
upload API is `http://127.0.0.1:8080/api/v1`; interactive documentation is at
`/api/docs`. It only authorizes miners' uploads and serves no data.

For continuous operation, run the validator under pm2 as described in the
[validator guide](docs/validator.md). `.env` holds the secrets in two sections:
Validator (network, netuid, wallet, S3 key) and Evaluation (Lium account, SSH key). Every
other setting is a plain value in `validator/config.py` (validator and upload API) or
`evaluation/config.py` (GPU rentals), and tests check the committed values. Configure targets, storage, the chain, and a qualified worker image using
the same guide. [Docker deployment](docs/docker.md)
runs the upload API and documents the `https://prexonai.com/api/v1` routing.

## Miner workflow

Install miner dependencies on the training/submission machine:

```bash
uv sync --frozen --extra miner
```

Package an exported head for a competition in `competitions/` (its model ID), then
submit with your subnet and wallet:

```bash
uv run prexon-miner competitions
uv run prexon-miner package --competition Qwen/Qwen3-8B \
  --checkpoint /path/to/head --provenance provenance.json --output head.tar.gz

uv run prexon-miner submit head.tar.gz \
  --netuid "$NETUID" --wallet-name miner --wallet-hotkey default
```

Miner commands take every setting as a flag and read no `.env`. The API base
defaults to
`https://prexonai.com/api/v1` and can be changed with `--api-url`. Miners need no S3
credentials. See [training and submission](docs/miner.md) for export, provenance,
limits, and command options.

## Evaluate a head

The same evaluator is available to miners and validators:

```bash
uv run prexon-evaluate --competition Qwen/Qwen3.8-27B --head /path/exported-head
uv run prexon-evaluate --competition Qwen/Qwen3.8-27B --head /path/exported-head \
  --mode full --public-head /path/public-reference
```

`head` mode measures candidate throughput, latency, and acceptance metrics.
`full` mode also measures target-only and supplied references to calculate
speedup. Lium is the default backend; `pod` uses an existing GPU environment.
Standalone reports are development results and do not affect emissions.

See [evaluation](docs/evaluation.md) for every option and rental recovery, and
[datasets](docs/datasets.md) for preparing the pinned prompt pool. Templates
require real revisions, image digests, and a prepared dataset manifest.

## Project layout

| Package | Purpose |
| --- | --- |
| `validator/` | Chain discovery, admission, scheduling, scoring, and GPU workers |
| `miner/` | Training, inspection, packaging, and submission |
| `evaluation/` | Shared benchmark library and standalone CLI |
| `api/` | Signed upload URLs for miners |
| `shared/` | Target profiles, contracts, checkpoints and storage |
| `docker/` | Upload API and GPU worker images |
| `tests/` | CPU integration tests and explicit GPU/network checks |

[Architecture](docs/architecture.md) explains the trust boundaries.
[Published data and the upload API](docs/api.md) documents the public files and
upload signing.

## Development

```bash
uv sync --frozen --all-extras
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build
```

The default suite runs on CPU and replaces external chain, rental, and GPU
boundaries. [Testing](docs/testing.md) describes GPU/network checks and runtime
qualification. CPU tests do not establish live provider behavior or GPU performance.

## License

[MIT](LICENSE).
