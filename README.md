# Prexon

Prexon is a Bittensor subnet for improving speculative decoding. Miners train
DFlash2 draft heads; validators measure their serving speedup against the target
model and reward successive improvements.

The project includes miner tools, a standalone evaluator, a CPU validator,
temporary GPU workers, and a dashboard/upload API. Training uses SpecForge;
evaluation uses SGLang. Model and GPU combinations must be qualified before
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
5. Only winners' checkpoints and detailed results are published. Each target's
   five latest promoted winners occupy equal reward slots; expired or ineligible
   shares burn.

Up to two targets can run independently, with separate hardware and queues.
See [scoring and emissions](docs/scoring.md) for the formula and reward rules.

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --frozen --extra validator --extra api
cp .env.example .env
uv run prexon-validator --check-config
uv run prexon-validator --once
uv run prexon-api
```

Both target slots are disabled by default. This creates an idle dashboard without
chain access, storage credentials, or a GPU. The local API is
`http://127.0.0.1:8000/api/v1`; interactive documentation is at `/api/docs`.

For continuous operation, run `uv run prexon-validator` in a separate terminal.
The validator and API share one `.env` on the same machine. Configure targets,
storage, chain credentials, and a qualified worker image using the
[validator guide](docs/validator.md). [Docker deployment](docs/docker.md) runs
both CPU services and documents the `https://prexon.ai/api/v1` routing.

## Miner workflow

Install miner dependencies on the training/submission machine:

```bash
uv sync --frozen --extra miner
```

Package an exported head with the actual model revision and published competition
ID, then submit with your subnet and wallet:

```bash
uv run prexon-miner package \
  --model-id org/model --revision "$TARGET_REVISION" \
  --competition-id "$COMPETITION_ID" \
  --checkpoint /path/to/head --provenance provenance.json --output head.tar.gz

uv run prexon-miner submit head.tar.gz \
  --netuid "$NETUID" --wallet-name miner --wallet-hotkey default
```

Miner configuration uses command arguments. `--env-file .env.miner` is optional;
the server's `.env` is not loaded automatically. The API base defaults to
`https://prexon.ai/api/v1` and can be changed with `--api-url`. Miners need no S3
credentials. See [training and submission](docs/miner.md) for export, provenance,
limits, and command options.

## Evaluate a head

The same evaluator is available to miners and validators:

```bash
mkdir -p profiles
cp docs/examples/local-evaluation.json profiles/local.json
# Configure the target, checkpoint path, dataset manifest, and backend first.
uv run prexon-evaluate --profile profiles/local.json
uv run prexon-evaluate --profile profiles/local.json --mode full
```

`head` mode measures candidate throughput, latency, and acceptance metrics.
`full` mode also measures target-only and supplied references to calculate
speedup. Lium is the default backend; `pod` uses an existing GPU environment.
Standalone reports are development results and do not affect emissions.

See [evaluation](docs/evaluation.md) for profiles and rental recovery, and
[datasets](docs/datasets.md) for preparing the pinned prompt pool. Templates
require real revisions, image digests, and a prepared dataset manifest.

## Project layout

| Package | Purpose |
| --- | --- |
| `validator/` | Chain discovery, admission, scheduling, scoring, and GPU workers |
| `miner/` | Training, inspection, packaging, and submission |
| `evaluation/` | Shared benchmark library and standalone CLI |
| `api/` | Public winner views and signed upload URLs |
| `shared/` | Configuration, contracts, checkpoints, storage, and transports |
| `docker/` | CPU service and GPU worker images |
| `tests/` | CPU integration tests and explicit GPU/network checks |

[Architecture](docs/architecture.md) explains the trust boundaries.
[API reference](docs/api.md) documents endpoints and upload signing.

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
