# Validator operation

The validator is an always-on CPU controller. It discovers chain commitments,
checks candidates, rents temporary GPUs, evaluates improvements, and publishes
emission weights. The API runs alongside it with the same `.env`.

## Configure the controller

```bash
uv sync --frozen --extra validator --extra api
cp .env.example .env
mkdir -p profiles
cp docs/examples/profile.json profiles/model-1.json
```

Leave unused model IDs empty. One enabled target receives allocation 100; two
allocations must sum to 100. Each target profile pins its revision, engine/image,
hardware, workloads, policy, and optional public reference. A useful public
reference is `{"artifact":"PACKAGE_SHA256","method":"dflash2"}`; absent references
contribute 1.0 to the winning threshold.

Prepare the [dataset pool](datasets.md), replace placeholder digests, and configure
the subnet netuid, wallet, and [Hippius/Lium deployment](docker.md). Production
requires separate staging, canonical, and winner buckets plus a stable public
winner URL. Store S3 and rental credentials in `.env`. The API receives selected
settings in Compose; it does not mount the validator database or wallet files.

Keep `PREXON_STATE_PATH` on persistent local storage. A process lock permits one
controller per state file. Use separate state for different chain/subnet contexts.
Disabling all targets retains an initialized chain connection so unused emissions
can be burned; a fresh empty installation needs no chain credentials.

## Qualify and publish a target

Test the actual model/runtime/image on the required hardware and review its
output-preserving greedy verification path. Finite output comparisons alone do
not establish correctness. Generate the exact identities and checkpoint schema:

```bash
uv run prexon-validator --check-config
uv run prexon-validator --qualification-profile 1
uv run prexon-miner inspect --checkpoint /path/verified-export --output schema.json
```

The first command prints the competition ID; the second prints its qualification
identity with `qualification_artifact` unset. Create an operator record:

```json
{
  "schema_version": 1,
  "profile_id": "QUALIFICATION_ID",
  "engine_version": "QUALIFIED_ENGINE_VERSION",
  "method": "dflash2",
  "sampling": "greedy",
  "approved_by": "operator",
  "evidence_description": "Immutable pilot evidence and verification review references",
  "target_preserving_verification_reviewed": true,
  "checkpoint_schemas": []
}
```

Insert the complete generated schema object into `checkpoint_schemas`; an empty
list is rejected. Include every tensor name, shape, and dtype from verified
exports. Publish the record, place its returned digest in the target profile's
`qualification_artifact`, and publish the final competition ID:

```bash
uv run prexon-validator --publish-qualification qualification.json
uv run prexon-validator --check-config
```

Unqualified targets cannot rent workers. A record captures operator approval;
software cannot verify the truthfulness of its human-review description.

## Start and monitor

```bash
uv run prexon-validator
# In a second terminal:
uv run prexon-api
```

`--once` performs one synchronous controller tick. Continuous mode uses bounded
background admission/publication workers; the controller alone finalizes SQLite.

Initial chain discovery reads available finalized reveal history. Subsequent
scans replay up to 120 blocks per tick after the durable cursor. Archive access
is needed after long downtime; pruned reveals cannot be reconstructed from a fresh
state file. Same-block ties use hotkey then commitment ID.

The validator publishes a fresh private upload policy after checking ownership
and qualifications. The API rejects stale/missing policies. Incoming packages
must exist at the committing hotkey's staging key, with signed competition
metadata matching the archive. Hashes, size bounds, qualified config/tensor
schemas, and CPU similarity are checked before queue admission.

Exact copies receive no new slot or award. Optional near-copy rejection requires
a calibrated `similarity` policy and full-value comparisons after screening.
Shared embeddings/output heads are excluded from near metrics. Raw cosine does
not change speedup scores. Missing operator/storage inputs remain pending;
malformed packages are rejected before GPU rental.

## Scheduling and persistence

Sessions start at `PREXON_EARLY_QUEUE_DEPTH=10` ready candidates or
`PREXON_SCHEDULE_SECONDS=21600` seconds since the oldest unevaluated ready
commitment. Chain time is used when available; first discovery is the fallback.
Retries and restarts preserve age. Session cost/duration bounds are independent
of this trigger.

Each target has one active session. Frozen profile versions drain in finalized
order, preserving benchmark settings. Dataset membership and later finalized
entropy freeze one plan before provisioning. See [datasets](datasets.md) and
[scoring](scoring.md) for the measurement and promotion protocol.

Back up validator SQLite and private artifacts. Preserve state across upgrades;
legacy unchecked work is revalidated before rental. Only trusted owned records
may reuse canonical bytes when their staging upload is unavailable.

## Rentals and publication

Lium templates must match the target's image digest. Configure SSH keys, optional
persistent volumes, session budgets, and provider-side key limits. Workers use
`/opt/prexon/.venv/bin/python` and receive only scoped job/plan/result URLs.
Persistent model downloads are staged locally; engines restart between
configurations. Draft-only hot swapping is not enabled.

Requests and session ownership are persisted before provisioning. Ambiguous rent
outcomes reuse the saved request/key. Shutdown reconciles rentals before closing
background workers. Failed deletion retains stopping state until confirmed cleanup
or provider expiry. Provider polling and billing can lag; use account limits.

Promotion, emission publication, and public copying are separate retryable steps.
A publication worker verifies the exact evaluated package before releasing it.
Failures do not duplicate awards or extend earning windows. Historical winner
releases remain public after reward expiry; pending/losing details remain private.

For development, `PREXON_CHAIN=file` and `PREXON_PROVIDER=local` use explicit chain
fixtures and local storage. Real evaluation still requires a compatible GPU
runtime. See [testing](testing.md) and [standalone evaluation](evaluation.md).
