# Docker: upload API and GPU worker image

Docker is used for two things: the upload API on the validator machine, and the
image that rented Lium GPUs run for evaluations. The validator itself runs on the
host under pm2; see the [validator guide](validator.md#start-and-monitor).

## Start the upload API

Run these from the repository checkout, as the same user that runs the validator:

```bash
mkdir -p runtime/intake-policy runtime/api
export API_CONTAINER_USER="$(id -u):$(id -g)"
docker compose --env-file .env -f docker/compose.yaml config --quiet
docker compose --env-file .env -f docker/compose.yaml up --build -d
```

The API container runs as the validator's user because the validator writes the
upload policy with mode 0640. Create both directories before the first start:
Compose refuses to start without them, instead of letting Docker create
root-owned directories in their place. Set `API_CONTAINER_USER` the same way
whenever you run `up` again after pulling changes. The container reads its settings
from the checkout's `validator/config.py`, mounted read-only, and receives only
the two S3 keys from `.env`. Restart it (`docker compose ... restart api`) after
editing `validator/config.py`.

| Host mount | Container path / access |
| --- | --- |
| `validator/config.py` | `/app/validator/config.py`, read-only |
| `runtime/intake-policy/` | `/app/runtime/intake-policy`, read-only |
| `runtime/api/` | `/app/runtime/api`, writable |

Keep `UPLOAD_POLICY_PATH` and `UPLOAD_STATE_PATH` under `ROOT / "runtime"` in
`validator/config.py`: paths are anchored to the checkout on the host and to `/app`
in the container, which is where the directories above are mounted.

**Upgrading a container started from an older image.** Older images read
`PREXON_API_*` and `PREXON_UPLOAD_*` environment variables and used mounts under
`/data`. The current image ignores those variables. Started with the old
`docker run` command, it would never see the upload policy, and its replay database
would live inside the container and be lost on the next recreate. Replace it with the
Compose setup above (or the same three mounts and only the two S3 keys).

The policy directory must be mounted as a directory because the validator replaces
the file atomically. API replay state is separate from validator SQLite; keep both
on local persistent storage. The API never mounts the validator's database, wallet
or SSH keys.

## Storage and public routing

Create three separate [Hippius](https://console.hippius.com) buckets:

| Bucket | Access | Holds |
| --- | --- | --- |
| `prexon-uploads` | Private | Miners' unchecked uploads |
| `prexon-private` | Private | Checked packages, qualification, jobs, and results |
| `prexon-public` | Public read | Winner packages and the dashboard's data |

`validator/config.py` already uses these three buckets (`UPLOAD_STAGING_ROOT`,
`ARTIFACT_ROOT`, `PUBLIC_ARTIFACT_ROOT`, and `PUBLIC_ARTIFACT_BASE_URL =
"https://s3.hippius.com/prexon-public"`); change them there if yours have other names.
Put one key in `.env`:

```dotenv
S3_ACCESS_KEY_ID=your-access-key
S3_SECRET_ACCESS_KEY=your-secret-key
```

Set `UPLOADS_ENABLED = True` in `validator/config.py` when miners may upload.

The client always uses `https://s3.hippius.com` and region `decentralized`.

The key must cover all three buckets; the validator and the upload API both use
it. Use a read-only anonymous `s3:GetObject` policy on the public bucket; new
Hippius buckets block public policies until Block Public Access is switched off
for that bucket. A bucket-wide public ACL must never
expose either private bucket. Include the bucket name and any configured
prefix in the public base URL. See [Hippius access controls](https://docs.hippius.com/storage/s3/advanced).
Compose does not create buckets or change their permissions.

The website reads the dashboard's data straight from the public bucket: give it
`PREXON_DATA_URL=<public base URL>/dashboard`. The anonymous read policy above
covers those files too.

The upload API listens on `127.0.0.1:8080` by default (`API_HOST` and
`API_PORT`). Compose publishes the container's port 8080 on `127.0.0.1:8080`; the
image starts the API with `--host 0.0.0.0`, which overrides `API_HOST` inside the
container. Configure HTTPS on `prexonai.com` and forward `/api/` to the local
service while preserving the prefix. The public base is
`https://prexonai.com/api/v1`; the OpenAPI UI is `/api/docs`.

## GPU worker image

Build against the exact qualified CUDA/SGLang image, repack it into one layer, and push it
to Docker Hub:

```bash
docker build -f docker/Dockerfile.worker \
  --build-arg ENGINE_IMAGE=lmsysorg/sglang:VERSION@sha256:ACTUAL_DIGEST \
  -t prexon-worker:VERSION-layers .
docker/flatten-image.sh prexon-worker:VERSION-layers your-namespace/prexon-worker:VERSION
docker push your-namespace/prexon-worker:VERSION
```

Lium machines run pods under Sysbox, which cannot start images of many layers, so Lium
re-packs any image of 30 or more layers into one before starting it. That needs room for a
full extra copy (twice the image plus 10 GB), and a one-GPU pod is given the machine's whole
disk, so the pod fails to start ("not enough free disk … to prepare this image"). SGLang
images have about 70 layers; `flatten-image.sh` makes one, keeping the image's environment,
command, working directory and labels. The single layer is about 16 GB compressed, which
Docker Hub accepts; GitHub's registry limits layers to 10 GB.

Use the pushed worker's registry digest in `competitions/shared.json` and the Lium template
(image, tag and digest).
The template must inject the SSH public key, start SSH, and keep the container
running. The image installs OpenSSH and installs Prexon into the base image's own
Python environment (`$VIRTUAL_ENV`, `/opt/sglang` in SGLang images), leaving SGLang's
packages as they are; `/opt/prexon/.venv` points at that environment. Verify startup
and GPU topology in a pilot.

Processes started over SSH do not see the image's environment, so the host sends the
worker its settings on stdin when it starts a job: `HF_XET_HIGH_PERFORMANCE=1`, the
`HF_TOKEN` from `.env` if set, and `PREXON_TARGET_VOLUME` when a volume is attached. The
target is staged on local disk (`PREXON_TARGET_CACHE`, default `/tmp/prexon-targets`).
Workers receive scoped job URLs, never the controller's wallet or provider credentials.
Standalone evaluation uses the same image through `evaluation.remote`, with
heads and reports transferred over SSH. Rebuild and pin the worker image after
runtime/protocol changes.

CPU builds and Compose validation do not establish GPU compatibility or live
provider permissions. See [testing](testing.md) for qualification checks.
