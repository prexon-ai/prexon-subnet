# API reference

Production base: **`https://prexon.ai/api/v1`**. Local base:
`http://127.0.0.1:8000/api/v1`. Reverse proxies must preserve `/api`.

The validator and API share `.env` on one machine:

```bash
uv sync --frozen --extra validator --extra api
uv run prexon-api
```

Interactive documentation is at `/api/docs`; the schema is at `/api/openapi.json`.
`GET /api/health` checks process liveness. Public reads work with uploads disabled.

## Public endpoints

Paths below are relative to the versioned base.

| Method and path | Response |
| --- | --- |
| `GET /status` | View age, finalized block, anonymous queue/session counts |
| `GET /targets` | Target identity, allocation, hardware, and queue summary |
| `GET /submissions` | Released winning submissions |
| `GET /evaluations` | Released winners' decisions and aggregate metrics |
| `GET /evaluations/{job_id}` | A released winner's report |
| `GET /winners` | Published five-slot history, eligibility, and expiry |
| `GET /winners/{submission_id}/checkpoint` | Verified local package or public HTTPS redirect |
| `GET /emissions` | UID shares, including burn |
| `GET /snapshot` | Public view and historical release manifest |
| `POST /uploads` | Authenticated short-lived private PUT capability |

Collections support `offset>=0` and `1<=limit<=1000`; responses contain
`schema_version`, `updated_at`, `total`, and `items`. The snapshot retains 1,000
recent winning submissions/evaluations; older winner reports remain individually
accessible. `updated_at` is epoch UTC seconds. Views older than 180 seconds are
marked stale. Missing/invalid views return 503; private or unknown records return 404.

Only released winners' checkpoints and detailed results are public. Losing and
pending heads, individual statuses, and similarity evidence stay private. Public
queue counts are anonymous. Historical winner downloads survive award expiry.

## Upload configuration

Set `PREXON_UPLOADS_ENABLED=true` after configuring the three Hippius buckets and
qualification in the [validator guide](validator.md). Shared storage settings are
`S3_ENDPOINT_URL=https://s3.hippius.com`, `S3_REGION=decentralized`,
`S3_ACCESS_KEY_ID`, and `S3_SECRET_ACCESS_KEY`. Endpoint/region defaults are applied
when omitted; incompatible values are rejected. The client uses SigV4 and
path-style addresses.

The controller publishes a private upload policy with fresh subnet ownership,
accepted competition IDs, and byte limits. The API reads it per request and owns
a separate replay/quota SQLite file. Neither upload authorization nor successful
storage upload creates an evaluation queue entry.

| Setting | Default |
| --- | --- |
| `PREXON_UPLOADS_ENABLED` | `false` |
| `PREXON_UPLOAD_POLICY_PATH` | `runtime/intake-policy/upload-policy.json` |
| `PREXON_UPLOAD_STATE_PATH` | `runtime/api/uploads.sqlite3` |
| `PREXON_UPLOAD_POLICY_MAX_AGE_SECONDS` | 300 |
| `PREXON_UPLOAD_URL_TTL_SECONDS` | 900 (allowed: 60–3,600) |
| `PREXON_UPLOAD_REQUEST_MAX_AGE_SECONDS` | 60 |
| `PREXON_UPLOAD_MAX_REQUESTS_PER_HOUR` | 4 per hotkey |
| `PREXON_UPLOAD_MAX_BYTES` | 4 GiB + 2 MiB |

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
