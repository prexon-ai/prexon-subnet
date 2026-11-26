# Published data and the upload API

Nothing public is served from the validator's machine. The validator publishes
JSON files to public storage, where the website and anyone else reads them. The
only service is a small API that authorizes miners' private uploads.

## Published files

The validator writes these under its public root (`PUBLIC_ARTIFACT_ROOT` in `validator/config.py`).
With `s3://prexon-public` and the public base
`https://s3.hippius.com/prexon-public`, they are public at
`https://s3.hippius.com/prexon-public/<object>`.

| Object | Contents |
| --- | --- |
| `dashboard/snapshot.json` | Targets, rewarded winners and their shares, emissions, released winners' reports, and the release manifest |
| `dashboard/index.json` | Every model the subnet has run, active first, with its summary |
| `dashboard/models/{id}.json` | One model's queue and every evaluated submission |
| `winners/{digest}.tar.gz` | Released winner packages |

A file is uploaded when what it shows changes. `snapshot.json` and `index.json`
also carry `updated_at` (epoch UTC seconds) and `finalized_block`, and are
uploaded again at least every five minutes, so a reader can tell a quiet subnet
from a stopped validator: treat a copy older than 15 minutes as stale. A failed
upload is retried on the next tick. While the chain is unreachable nothing is
uploaded and the last copies stay in place.

The website reads the `dashboard/` folder through its `PREXON_DATA_URL` setting.
With a local public root (for example `runtime/public/artifacts`) the same files
are written under that directory, and packages have no public address.

Only crowned heads have a public checkpoint and detailed measurements. For every
queued or evaluated submission the model history shows the miner (hotkey and
UID), the submission time and, once evaluated, the result and measured speedup.
Checkpoints of pending and losing heads, loss reasons, rejected submissions, and
similarity evidence stay private. Historical winner downloads stay available
after a winner drops out of the reward split.

### Snapshot

`snapshot.json` holds `targets` (identity, allocation, hardware, and queue
summary), `winners` (the currently rewarded heads and each one's share of its
target), `emissions` (UID shares, including burn), `submissions` and
`evaluations` (released winners with their decisions and aggregate metrics, the
1,000 most recent), and `releases` (every released package with its digest and
address).

Each evaluation's `competition_id` is the one it was scored under. Changing a
target's benchmark profile gives it a new competition ID, so evaluations also
carry the `model_id` and `revision` they measured; match those to a target to
keep its earlier results.

### Model history

`index.json` has `updated_at`, `finalized_block`, and one summary per model: its
URL-safe `id`, `model_id`, `revision`, `slot`, `active_from`, `active_to` (null
while active), `allocation`, hardware, queue length, evaluated and winner counts,
`best_speedup`, and the current `reward_shares`. Active models come first; a
model that left its slot stays listed with the time it was last active.

`models/{id}.json` adds the model's `queue` (the head being evaluated first, then
the rest in the order they will run) and its `submissions`, newest result first.
Each submission has `crowned`, `speedup`, `previous_best` (the speedup it had to
beat, measured in the same evaluation: the faster of the incumbent and the public
reference, or 1 when neither beat target-only decoding), its current `share` of
the model's emissions, and a `checkpoint_url` when it was crowned and its package
is in public storage. A retired model keeps its submissions but has no queue and
pays no rewards.

## Upload API

Production base: **`https://prexonai.com/api/v1`**. Local base:
`http://127.0.0.1:8080/api/v1`. Reverse proxies must preserve `/api`.

The validator and upload API share `.env` and `validator/config.py` on one machine:

```bash
uv sync --frozen --extra validator --extra api
uv run prexon-api
```

Its one endpoint is `POST /uploads`, which returns an authenticated short-lived
private PUT capability. Interactive documentation is at `/api/docs`; the schema is
at `/api/openapi.json`. `GET /api/health` checks process liveness.

## Upload configuration

Set `UPLOADS_ENABLED = True` in `validator/config.py` after configuring the three
Hippius buckets and qualification in the [validator guide](validator.md). The
shared S3 key, `S3_ACCESS_KEY_ID` and `S3_SECRET_ACCESS_KEY`, is in the Validator
section of `.env`. The client always uses `https://s3.hippius.com`, region
`decentralized`, SigV4 and path-style addresses.

The controller publishes a private upload policy with fresh subnet ownership,
accepted competition IDs, and byte limits. The API reads it per request and owns
a separate replay/quota SQLite file. Neither upload authorization nor successful
storage upload creates an evaluation queue entry.

| `validator/config.py` setting | Default |
| --- | --- |
| `UPLOADS_ENABLED` | `False` |
| `UPLOAD_POLICY_PATH` | `runtime/intake-policy/upload-policy.json` |
| `UPLOAD_STATE_PATH` | `runtime/api/uploads.sqlite3` |
| `UPLOAD_POLICY_MAX_AGE_SECONDS` | 300 |
| `UPLOAD_URL_TTL_SECONDS` | 900 |
| `UPLOAD_REQUEST_MAX_AGE_SECONDS` | 60 |
| `UPLOAD_MAX_REQUESTS_PER_HOUR` | 4 per hotkey |
| `UPLOAD_MAX_BYTES` (in `shared/config.py`) | 4 GiB + 2 MiB |
| `API_HOST`, `API_PORT` | `127.0.0.1`, 8080; `prexon-api --host` overrides the host |

Paths are inside the checkout's `runtime/` folder.

The effective size ceiling is the smaller of the global and competition limits.
Persist the API database on local storage and synchronize clocks. API replicas
with separate databases do not share quotas. Mount the policy **directory**
read-only so the validator's atomic file replacement remains visible.

## Signed upload protocol

`POST /uploads` accepts a hotkey-signed `UploadRequest`:

| Field | Encoding |
| --- | --- |
| `hotkey` | Registered SS58 address |
| `network`, `netuid` | Exact subnet context |
| `competition_id`, `artifact` | Competition and compressed-package SHA-256 digests |
| `size`, `timestamp` | Positive byte count and integer epoch UTC seconds |
| `nonce` | Fresh 16-byte random value as 32 lowercase hex characters |
| `signature` | 64-byte signature as 128 hex characters; `0x` prefix accepted |

Signing bytes are `b"prexon-upload-v1\n"` followed by canonical UTF-8 JSON of every
field except `signature`, with sorted keys and separators `(',', ':')`.

The `UploadGrant` response contains `url`, required `headers`, `expires_at`, and
`artifact`. Send the exact bytes with PUT and the supplied `Content-Length`,
`Content-Type: application/gzip`, and `x-amz-meta-prexon-competition-id` headers.
The signed metadata must match the package manifest during validator admission.
The server chooses the object key:

```text
<staging-prefix>/submissions/<sha256(network)[:16]>/<netuid>/<sha256(hotkey)>/<artifact>
```

The grant authorizes no private GET or canonical-store access. Keep its URL and
signatures out of proxy/application logs. Commit the digest on chain only after
PUT succeeds; the validator independently checks the uploaded package.

| Status | Meaning |
| --- | --- |
| 400 | Malformed authorization |
| 401 | Invalid signature or stale/future timestamp |
| 403 | Wrong subnet, unregistered hotkey, or closed competition |
| 409 | Nonce already used |
| 413 | Size limit exceeded |
| 429 | Hourly quota exceeded; includes `Retry-After` |
| 503 | Disabled/misconfigured uploads, stale policy, or backend/state failure |
