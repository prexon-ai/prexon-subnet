"""Settings for GPU evaluations: prexon-evaluate and the validator's rentals.

The Evaluation section of `.env` (see `.env.example`) supplies the Lium account and the
SSH key; validators and miners both fill it in.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
_env = {**dotenv_values(ROOT / ".env"), **os.environ}

# .env: Evaluation section --------------------------------------------------------------
LIUM_API_KEY = _env.get("LIUM_API_KEY") or ""
LIUM_TEMPLATE_ID = _env.get("LIUM_TEMPLATE_ID") or ""
# Each competition's Lium volume, as MODEL_ID=VOLUME_ID pairs separated by commas. A
# volume mounts at /mnt and keeps that model's download (HF_HOME=/mnt/huggingface) between
# rentals; it attaches to one pod at a time, so competitions do not share one.
LIUM_VOLUMES = dict(
    pair.strip().split("=", 1)
    for pair in (_env.get("LIUM_VOLUMES") or "").split(",")
    if pair.strip()
)
SSH_PRIVATE_KEY = Path(_env.get("SSH_PRIVATE_KEY") or "~/.ssh/id_ed25519")
SSH_PUBLIC_KEY = Path(_env.get("SSH_PUBLIC_KEY") or "~/.ssh/id_ed25519.pub")

# Values --------------------------------------------------------------------------------
LIUM_BASE_URL = "https://lium.io/api"
SSH_KNOWN_HOSTS = ROOT / "runtime/known_hosts"
DATASET_CACHE = ROOT / "runtime/datasets"
# Defaults for a prexon-evaluate round; an evaluation profile may override both.
SESSION_SECONDS = 14400
SESSION_MAX_COST = 100
