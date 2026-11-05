# Testing and runtime qualification

## CPU checks

Use Python 3.12+ and the locked development dependencies:

```bash
uv sync --frozen --all-extras
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
uv build
```

The default pytest selection excludes `gpu` and `network` markers. Tests exercise
real checkpoint archives, signatures, local storage, SQLite state, API routes,
dataset plans, and controller lifecycles. External providers, chain RPCs, serving
engines, and paid operations are replaced at their boundaries.

Tests follow the runtime packages under `tests/`; `tests/integration/` covers
workflows across roles. Shared input factories live in `tests/factories.py` and
the simulated GPU fixture in `tests/conftest.py`. Related regressions stay in
the same module, with parameterized cases for input variations.

Focus coverage on scoring rules, private artifact handling, dataset fairness,
durable queue transitions, and worker cleanup. Skip tests that repeat library
validation, check simple getters or argument forwarding, or mirror the
implementation. Use representative input variations for each distinct behavior.

CI also checks installed entrypoints and empty-target configuration. Docker
configuration can be checked without starting services:

```bash
docker compose --env-file .env.example -f docker/compose.yaml config --quiet
```

## Explicit integration checks

```bash
PREXON_GPU_JOB_PATH=/absolute/path/job.json uv run pytest -m gpu
PREXON_NETWORK_TESTS=1 uv run pytest -m network
```

The GPU test runs a prepared worker job in an existing compatible environment.
The network test reads Lium availability; it does not rent hardware or write the
chain. These tests require deliberate configuration and are not part of CPU CI.

## Before enabling a target

Qualify the exact target revision, DFlash2 export, serving engine, worker image,
and GPU topology. Review target-preserving verification, then measure complete
outputs, reference drift, runtime variance, memory use, and startup cost. Calibrate
the workload mix, repeat count, confidence threshold, and improvement margin.

A provider pilot must also verify Hippius PUT/HEAD/GET metadata and permissions,
finalized chain replay and transactions, SSH/template startup, rental recovery,
and termination. Keep provider-side spending limits enabled during paid checks.

Record the evidence in an operator qualification artifact as described in
[validator operations](validator.md). A passing CPU suite or example profile is
not evidence that a model/runtime combination is qualified.
