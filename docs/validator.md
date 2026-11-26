# Validator operation

The validator is an always-on CPU controller. It discovers chain commitments,
checks candidates, rents temporary GPUs, evaluates improvements, and publishes
emission weights. It publishes the dashboard's data to public storage. The
upload API runs alongside it with the same `.env` and `validator/config.py`.

## Configure the controller

```bash
uv sync --frozen --extra validator --extra api
cp .env.example .env
mkdir -p profiles
cp docs/examples/profile.json profiles/model-1.json
```

`.env` has two sections: Validator (`PREXON_NETWORK`, `finney` or `test`,
`PREXON_NETUID`, the wallet and the S3 key) and Evaluation (the Lium account and SSH
key, used by GPU rentals). Every other setting is a plain value in
`validator/config.py` (validator and upload API) or `evaluation/config.py` (GPU
rentals), commented Python files you edit in place. Values are not validated at
startup; `uv run pytest tests/validator/test_config.py` checks the committed values.
Two safety checks remain: enabled targets need `PREXON_NETUID` and the three Hippius
buckets. Configure each
target as an entry of its `TARGETS` list and leave unused model IDs empty. One enabled target receives allocation 100; two
allocations must sum to 100. Each target profile pins its revision, engine/image,
hardware, workloads, policy, and optional public reference. A useful public
reference is `{"artifact":"PACKAGE_SHA256","method":"dflash2"}`; absent references
contribute 1.0 to the winning threshold.

Prepare the [dataset pool](datasets.md), replace placeholder digests, and set
`PREXON_NETUID` and the wallet in `.env`, and configure the [Hippius/Lium deployment](docker.md). Production
requires separate uploads, private, and public buckets plus a stable public
winner URL. The upload API container mounts `validator/config.py` and receives only
the S3 keys; it does not mount the validator database or wallet files.

Keep `STATE_PATH` on persistent local storage. A process lock permits one
controller per state file. Use separate state for different chain/subnet contexts.
Disabling all targets retains an initialized chain connection so unused emissions
can be burned; a fresh empty installation needs no chain credentials.

**Upgrading from an all-in-one `.env`.** Old keys in `.env` are ignored, not
reported. Keep only the keys of `.env.example`, move every other value to its constant
in the config files (`PREXON_MODEL_{1,2}_*` become `TARGETS` entries) and run the
config tests. There is no file chain: the validator always reads the Bittensor
network in `PREXON_NETWORK`.

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

Run the validator with [pm2](https://pm2.keymetrics.io/) from the repository
checkout, whose `.venv` the command uses. `validator/config.py` anchors its paths
and the default `.env` to the checkout:

```bash
pm2 start .venv/bin/prexon-validator --name prexon-validator --interpreter none --kill-timeout 120000
pm2 save
pm2 logs prexon-validator
```

`--kill-timeout 120000` gives the validator two minutes to stop its GPU sessions
and release the Lium rentals when pm2 stops or restarts it; pm2's default is 1.6
seconds. Run `pm2 startup` once so pm2 restarts the validator after a reboot, and
consider `pm2-logrotate` for log rotation. For a foreground run, use
`uv run prexon-validator`. Start the upload API with [Docker](docker.md).

`--once` performs one synchronous controller tick. Continuous mode uses bounded
background admission/publication workers; the controller alone finalizes SQLite.

Initial chain discovery reads available finalized reveal history. Subsequent
scans replay up to 120 blocks per tick after the durable cursor. Archive access
is needed after long downtime; pruned reveals cannot be reconstructed from a fresh
state file. Same-block ties use hotkey then commitment ID. A reveal's block time is
read once, when it is first discovered, so reveals that outlive a non-archive
node's history do not interrupt scanning.

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

Sessions start at `EARLY_QUEUE_DEPTH = 10` ready candidates or
`SCHEDULE_SECONDS = 21600` seconds since the oldest unevaluated ready
commitment. Chain time is used when available; first discovery is the fallback.
Retries and restarts preserve age. Session cost/duration bounds are independent
of this trigger.

An evaluation that ends without a verdict counts as a failed attempt: an unusable
result, or a job that used its whole time budget without reporting. After
`MAX_EVALUATION_ATTEMPTS = 3` failed attempts the submission is set aside as
`stalled`. That is not a loss; the submission simply stops holding up the queue.
Jobs cut short by a session ending, a pod failure, or a controller shutdown do not
count. After fixing an infrastructure problem, stop the controller and run
`uv run prexon-validator --requeue-stalled` to give stalled submissions a fresh
set of attempts.

When a session ends without any verdict, the next rental for that target waits
`SESSION_RETRY_SECONDS = 300`. The wait doubles after each further rental
that produced no verdict, up to one hour, and clears once a session reaches a
verdict. Refused rentals cost nothing and are retried at the first wait.

Each target has one active session. Frozen profile versions drain in finalized
order, preserving benchmark settings. Dataset membership and later finalized
entropy freeze one plan before provisioning. See [datasets](datasets.md) and
[scoring](scoring.md) for the measurement and promotion protocol.

Back up validator SQLite and private artifacts. Only trusted owned records may
reuse canonical bytes when their staging upload is unavailable.

## Rentals and publication

Lium templates must match the target's image digest. Configure SSH keys, optional
persistent volumes, session budgets, and provider-side key limits. Workers use
`/opt/prexon/.venv/bin/python` and receive only scoped job/plan/result URLs.
Persistent model downloads are staged locally; engines restart between
configurations. Draft-only hot swapping is not enabled.

Requests and session ownership are persisted before provisioning. Ambiguous rent
outcomes reuse the saved request/key. A definite refusal, such as no matching GPU,
ends the session at once; the next session retries under a new rental identity
after the retry wait. Shutdown reconciles rentals before closing
background workers. Failed deletion retains stopping state until confirmed cleanup
or provider expiry. Provider polling and billing can lag; use account limits.

Promotion, emission publication, and public copying are separate retryable steps.
A publication worker verifies the exact evaluated package before releasing it.
Failures do not duplicate awards or change their order. Historical winner
releases remain public after a winner drops out of the reward split. The model
history lists every queued and evaluated submission with its miner, time,
result and speedup; pending and losing checkpoints and loss reasons remain
private. The snapshot and history files are written to public storage under
`dashboard/`, where the website reads them: when they change, and at least every
five minutes for the two that carry the time. A failed upload is retried.

There is no local chain. Enabled targets always mean the Bittensor network in
`PREXON_NETWORK` and the three Hippius stores, so try changes on the testnet
(`PREXON_NETWORK=test`) first; the tests drive the controller with fixtures instead.
Real evaluation still requires a compatible GPU runtime. See [testing](testing.md) and [standalone evaluation](evaluation.md).
