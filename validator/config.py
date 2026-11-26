"""Settings for the validator and the upload API.

Edit the values below, then restart both. The Validator section of `.env` (see
`.env.example`) supplies the network, netuid, wallet and S3 keys; GPU rentals use
`evaluation/config.py`; the competitions are in `competitions/`. Tests check the
committed values.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
_env = {**dotenv_values(ROOT / ".env"), **os.environ}

# .env: Validator section ---------------------------------------------------------------
NETWORK = _env.get("PREXON_NETWORK") or "finney"  # "finney" or "test"
NETUID = int(_env.get("PREXON_NETUID") or 0)  # Required before enabling targets
WALLET_NAME = _env.get("PREXON_WALLET_NAME") or "default"
WALLET_HOTKEY = _env.get("PREXON_WALLET_HOTKEY") or "default"
WALLET_PATH = _env.get("PREXON_WALLET_PATH") or "~/.bittensor/wallets"
S3_ACCESS_KEY_ID = _env.get("S3_ACCESS_KEY_ID") or ""
S3_SECRET_ACCESS_KEY = _env.get("S3_SECRET_ACCESS_KEY") or ""

# Chain ---------------------------------------------------------------------------------
BURN_UID = 0
WEIGHTS_VERSION = 1

# Schedule ------------------------------------------------------------------------------
POLL_SECONDS = 30
# A GPU session starts at EARLY_QUEUE_DEPTH ready candidates, or once the oldest
# unevaluated ready commitment is SCHEDULE_SECONDS old. Age comes from chain submission
# time, with durable first discovery as fallback; retries keep it.
SCHEDULE_SECONDS = 21600
EARLY_QUEUE_DEPTH = 10
SESSION_SECONDS = 14400
SESSION_MAX_COST = 100
# After a session without a verdict, wait before renting again (doubles per failed
# rental, up to one hour).
SESSION_RETRY_SECONDS = 300
# Evaluations without a verdict before a submission is set aside as stalled (not a loss).
MAX_EVALUATION_ATTEMPTS = 3
PUBLISH_SECONDS = 600

# Storage -------------------------------------------------------------------------------
STATE_PATH = ROOT / "runtime/validator.sqlite3"
# Hippius uses THREE DIFFERENT BUCKETS, never public/private prefixes of one. They need
# the S3 keys in .env.
# Private bucket: checked packages, qualification, jobs and results.
ARTIFACT_ROOT = "s3://prexon-private"
# Uploads bucket (private): miners' unchecked packages, written only through API-issued
# upload URLs.
UPLOAD_STAGING_ROOT = "s3://prexon-uploads"
# Public bucket: verified winners and the dashboard's data.
PUBLIC_ARTIFACT_ROOT = "s3://prexon-public"
# Stable public HTTPS base. Hippius uses path-style public URLs, so include the bucket.
PUBLIC_ARTIFACT_BASE_URL = "https://s3.hippius.com/prexon-public"
# Written by the validator, read by the upload API.
UPLOAD_POLICY_PATH = ROOT / "runtime/intake-policy/upload-policy.json"

# GPU rentals: "lium" rents GPUs; "local" is for development --------------------------
PROVIDER = "lium"

# Upload API ----------------------------------------------------------------------------
API_HOST = "127.0.0.1"
API_PORT = 8080
UPLOADS_ENABLED = False
# The API's own nonce and quota state, separate from the validator's database.
UPLOAD_STATE_PATH = ROOT / "runtime/api/uploads.sqlite3"
UPLOAD_POLICY_MAX_AGE_SECONDS = 300
UPLOAD_URL_TTL_SECONDS = 900
UPLOAD_REQUEST_MAX_AGE_SECONDS = 60
UPLOAD_MAX_REQUESTS_PER_HOUR = 4
# Largest package upload (4 GiB of checkpoint plus 2 MiB of archive overhead). The API
# refuses bigger uploads and the validator downloads no more. prexon-miner's default
# --max-upload-bytes repeats this value.
UPLOAD_MAX_BYTES = 4 * 1024**3 + 2 * 1024**2
