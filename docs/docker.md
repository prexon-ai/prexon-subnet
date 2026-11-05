# Docker deployment

Compose runs the CPU validator and API on one machine. Temporary GPU workers are
provisioned separately through Lium.

## Start the CPU services

```bash
cp .env.example .env
mkdir -p runtime/public/artifacts runtime/intake-policy runtime/api profiles
sudo chown -R 10001:10001 runtime
docker compose --env-file .env -f docker/compose.yaml config --quiet
docker compose --env-file .env -f docker/compose.yaml up --build -d
```

Images run as UID 10001. Empty target slots start an idle installation. Configure
production targets using the [validator guide](validator.md) before enabling them.
Compose uses one `.env`: the validator receives controller settings, while the API
receives selected API/storage values. It maps runtime paths to `/data`.

| Service | Host mount | Container path / access |
| --- | --- | --- |
| Validator | `runtime/` | `/data`, writable |
| Validator | `profiles/` | `/app/profiles`, read-only |
| API | `runtime/public/` | `/data/public`, read-only |
| API | `runtime/intake-policy/` | `/data/intake-policy`, read-only |
| API | `runtime/api/` | `/data/api`, writable |

The policy directory must be mounted as a directory because updates use atomic
rename. API replay state is separate from validator SQLite. Keep both databases
on local persistent storage. Wallet and SSH key mounts belong only in the
validator service; use container paths in its configuration.

## Storage and public routing

Create three separate [Hippius](https://console.hippius.com) buckets. Staging and
canonical buckets stay private; the winner bucket publishes only promoted heads.
Configure shared credentials and roots in `.env`:

```dotenv
S3_ENDPOINT_URL=https://s3.hippius.com
S3_REGION=decentralized
S3_ACCESS_KEY_ID=your-access-key
S3_SECRET_ACCESS_KEY=your-secret-key
PREXON_UPLOAD_STAGING_ROOT=s3://prexon-staging/intake
PREXON_ARTIFACT_ROOT=s3://prexon-private/artifacts
PREXON_PUBLIC_ARTIFACT_ROOT=s3://prexon-winners
PREXON_PUBLIC_ARTIFACT_BASE_URL=https://s3.hippius.com/prexon-winners
PREXON_UPLOADS_ENABLED=true
```

Credentials must cover staging, canonical, and winner operations. Use a read-only
anonymous `s3:GetObject` policy for winner downloads. A bucket-wide public ACL must
never expose either private bucket. Include the bucket name and any configured
prefix in the public base URL. See [Hippius access controls](https://docs.hippius.com/storage/s3/advanced).
Compose does not create buckets or change their permissions.

The API binds to host `127.0.0.1:8000` by default. Configure HTTPS on `prexon.ai` and
forward `/api/` to the local service while preserving the prefix. The public base
is `https://prexon.ai/api/v1`; the OpenAPI UI is `/api/docs`. Host bind/port follow
`PREXON_API_HOST` and `PREXON_API_PORT`; the container listens on port 8000.

## GPU worker image

Build against the exact qualified CUDA/SGLang image:

```bash
docker build -f docker/Dockerfile.worker \
  --build-arg ENGINE_IMAGE=your-qualified-image@sha256:ACTUAL_DIGEST \
  -t your-registry/prexon-worker:VERSION .
```

Use the built worker's registry digest in the target profile and Lium template.
The template must inject the SSH public key, start SSH, and keep the container
running. The image installs OpenSSH and uses the base image's SGLang/Torch through
a venv with system site packages. Verify startup and GPU topology in a pilot.

Set `HF_HOME=/mnt/huggingface` for persistent downloads and
`PREXON_TARGET_CACHE` for local staging (default `/tmp/prexon-targets`). Workers
receive scoped job URLs, never the controller's wallet or provider credentials.
Standalone evaluation uses the same image through `evaluation.remote`, with
heads and reports transferred over SSH. Rebuild and pin the worker image after
runtime/protocol changes.

CPU builds and Compose validation do not establish GPU compatibility or live
provider permissions. See [testing](testing.md) for qualification checks.
