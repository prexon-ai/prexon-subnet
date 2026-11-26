"""The committed values in the config files, and the `.env` sections they read."""

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from validator.config import (
    ARTIFACT_ROOT,
    PUBLIC_ARTIFACT_BASE_URL,
    PUBLIC_ARTIFACT_ROOT,
    TARGETS,
    UPLOAD_STAGING_ROOT,
)

ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_SECTION = [
    "PREXON_NETWORK",
    "PREXON_NETUID",
    "PREXON_WALLET_NAME",
    "PREXON_WALLET_HOTKEY",
    "PREXON_WALLET_PATH",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "S3_SESSION_TOKEN",
]
EVALUATION_SECTION = [
    "LIUM_API_KEY",
    "LIUM_TEMPLATE_ID",
    "LIUM_VOLUME_ID",
    "SSH_PRIVATE_KEY",
    "SSH_PUBLIC_KEY",
]


def read(module, names, **environment):
    """Import a config module in a fresh interpreter and return the repr of each value."""
    code = (
        f"import json, {module} as c; "
        f"print(json.dumps({{n: repr(getattr(c, n)) for n in {names!r}}}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    return json.loads(result.stdout)


def test_validator_section_of_env_fills_the_validator_config():
    names = ["NETWORK", "NETUID", "WALLET_NAME", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"]
    environment = {
        "PREXON_NETWORK": "test",
        "PREXON_NETUID": "7",
        "PREXON_WALLET_NAME": "owner",
        "S3_ACCESS_KEY_ID": "key",
        "S3_SECRET_ACCESS_KEY": "secret",
    }
    assert read("validator.config", names, **environment) == {
        "NETWORK": "'test'",
        "NETUID": "7",
        "WALLET_NAME": "'owner'",
        "S3_ACCESS_KEY_ID": "'key'",
        "S3_SECRET_ACCESS_KEY": "'secret'",
    }


def test_empty_validator_section_falls_back_to_defaults():
    # tests/conftest.py blanks every .env key, as an empty .env would.
    names = ["NETWORK", "NETUID", "WALLET_NAME", "WALLET_HOTKEY", "WALLET_PATH", "S3_SESSION_TOKEN"]
    assert read("validator.config", names) == {
        "NETWORK": "'finney'",
        "NETUID": "0",
        "WALLET_NAME": "'default'",
        "WALLET_HOTKEY": "'default'",
        "WALLET_PATH": "'~/.bittensor/wallets'",
        "S3_SESSION_TOKEN": "''",
    }


def test_evaluation_section_of_env_fills_the_evaluation_config():
    names = ["LIUM_API_KEY", "LIUM_TEMPLATE_ID", "LIUM_VOLUME_ID", "SSH_PRIVATE_KEY"]
    environment = {
        "LIUM_API_KEY": "lium",
        "LIUM_TEMPLATE_ID": "template",
        "LIUM_VOLUME_ID": "volume",
        "SSH_PRIVATE_KEY": "/keys/prexon",
    }
    assert read("evaluation.config", names, **environment) == {
        "LIUM_API_KEY": "'lium'",
        "LIUM_TEMPLATE_ID": "'template'",
        "LIUM_VOLUME_ID": "'volume'",
        "SSH_PRIVATE_KEY": "PosixPath('/keys/prexon')",
    }
    defaults = read("evaluation.config", ["LIUM_API_KEY", "SSH_PRIVATE_KEY", "SSH_PUBLIC_KEY"])
    assert defaults == {
        "LIUM_API_KEY": "''",
        "SSH_PRIVATE_KEY": "PosixPath('~/.ssh/id_ed25519')",
        "SSH_PUBLIC_KEY": "PosixPath('~/.ssh/id_ed25519.pub')",
    }


def test_env_example_has_the_validator_and_evaluation_sections():
    example = (ROOT / ".env.example").read_text()
    assert list(dotenv_values(ROOT / ".env.example")) == VALIDATOR_SECTION + EVALUATION_SECTION
    assert example.index("# Validator") < example.index("# Evaluation")


def test_enabled_targets_have_distinct_slots_and_models_and_share_100():
    slots = [entry["slot"] for entry in TARGETS]
    assert len(slots) == len(set(slots)) and set(slots) <= {1, 2}
    enabled = [entry for entry in TARGETS if entry["model_id"]]
    models = [entry["model_id"] for entry in enabled]
    assert len(models) == len(set(models))
    if enabled:
        assert sum(100 if e["allocation"] is None else e["allocation"] for e in enabled) == 100
    for entry in enabled:
        assert Path(entry["profile"]).is_file(), entry["profile"]


def test_storage_uses_three_different_buckets_and_a_public_https_url():
    roots = [urlsplit(root) for root in (UPLOAD_STAGING_ROOT, ARTIFACT_ROOT, PUBLIC_ARTIFACT_ROOT)]
    assert all(root.scheme == "s3" and root.netloc for root in roots)
    assert len({root.netloc for root in roots}) == 3
    # No credentials, queries or relative paths in any address.
    public = urlsplit(PUBLIC_ARTIFACT_BASE_URL)
    for address in [*roots, public]:
        assert not (address.username or address.password or address.query or address.fragment)
        assert ".." not in address.path
    assert public.scheme == "https"
    assert public.path.strip("/") == urlsplit(PUBLIC_ARTIFACT_ROOT).netloc


def test_only_config_files_read_env():
    code = [
        path
        for folder in ("api", "evaluation", "miner", "shared", "validator")
        for path in (ROOT / folder).rglob("*.py")
    ]
    readers = sorted(str(p.relative_to(ROOT)) for p in code if "dotenv" in p.read_text())
    assert readers == ["evaluation/config.py", "validator/config.py"]
    keys = VALIDATOR_SECTION + EVALUATION_SECTION
    for path in code:
        if path.name != "config.py":
            assert not any(f'"{key}"' in path.read_text() for key in keys), path
