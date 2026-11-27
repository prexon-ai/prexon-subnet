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
]
EVALUATION_SECTION = [
    "LIUM_API_KEY",
    "LIUM_TEMPLATE_ID",
    "LIUM_VOLUMES",
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
    names = ["NETWORK", "NETUID", "WALLET_NAME", "WALLET_HOTKEY", "WALLET_PATH", "S3_ACCESS_KEY_ID"]
    assert read("validator.config", names) == {
        "NETWORK": "'finney'",
        "NETUID": "0",
        "WALLET_NAME": "'default'",
        "WALLET_HOTKEY": "'default'",
        "WALLET_PATH": "'~/.bittensor/wallets'",
        "S3_ACCESS_KEY_ID": "''",
    }


def test_evaluation_section_of_env_fills_the_evaluation_config():
    names = ["LIUM_API_KEY", "LIUM_TEMPLATE_ID", "LIUM_VOLUMES", "SSH_PRIVATE_KEY"]
    environment = {
        "LIUM_API_KEY": "lium",
        "LIUM_TEMPLATE_ID": "template",
        "LIUM_VOLUMES": "Qwen/Qwen3.8-27B=volume-a, org/other=volume-b",
        "SSH_PRIVATE_KEY": "/keys/prexon",
    }
    assert read("evaluation.config", names, **environment) == {
        "LIUM_API_KEY": "'lium'",
        "LIUM_TEMPLATE_ID": "'template'",
        "LIUM_VOLUMES": "{'Qwen/Qwen3.8-27B': 'volume-a', 'org/other': 'volume-b'}",
        "SSH_PRIVATE_KEY": "PosixPath('/keys/prexon')",
    }
    defaults = read(
        "evaluation.config", ["LIUM_API_KEY", "LIUM_VOLUMES", "SSH_PRIVATE_KEY", "SSH_PUBLIC_KEY"]
    )
    assert defaults == {
        "LIUM_API_KEY": "''",
        "LIUM_VOLUMES": "{}",
        "SSH_PRIVATE_KEY": "PosixPath('~/.ssh/id_ed25519')",
        "SSH_PUBLIC_KEY": "PosixPath('~/.ssh/id_ed25519.pub')",
    }


def test_env_example_has_the_validator_and_evaluation_sections():
    example = (ROOT / ".env.example").read_text()
    assert list(dotenv_values(ROOT / ".env.example")) == VALIDATOR_SECTION + EVALUATION_SECTION
    assert example.index("# Validator") < example.index("# Evaluation")


def test_committed_competitions_have_unique_models_and_active_shares_of_100(monkeypatch):
    from shared.targets import load_competitions

    monkeypatch.setattr("shared.targets.COMPETITIONS_DIR", ROOT / "competitions")
    competitions = load_competitions()
    models = [c.model_id for c in competitions]
    assert len(models) == len(set(models))
    active = [c for c in competitions if c.active]
    if active:
        assert sum(c.allocation for c in active) == 100


def test_the_example_competition_completes_the_shared_settings(tmp_path, monkeypatch):
    from shared.targets import load_competitions

    shared = ROOT / "competitions/shared.json"
    (tmp_path / "shared.json").write_text(shared.read_text())
    (tmp_path / "qwen3-8b.json").write_text((ROOT / "docs/examples/competition.json").read_text())
    monkeypatch.setattr("shared.targets.COMPETITIONS_DIR", tmp_path)
    (example,) = load_competitions()
    assert not example.active
    assert example.datasets is not None
    assert example.datasets.buckets  # from shared.json; the manifest is the competition's own
    assert example.workloads == load_competitions()[0].workloads


def test_a_competition_file_overrides_shared_settings_and_keeps_the_rest(tmp_path, monkeypatch):
    from shared.targets import load_competitions
    from tests.factories import target

    full = target().model_dump(mode="json")
    shared = {key: full.pop(key) for key in ("engine_version", "worker_image", "workloads")}
    shared["policy"] = {"improvement_margin": 0.015, "repeats": 7}
    (tmp_path / "shared.json").write_text(json.dumps(shared))
    full["policy"] = {"improvement_margin": 0.02}
    (tmp_path / "b.json").write_text(json.dumps({**full, "model_id": "org/b"}))
    (tmp_path / "a.json").write_text(json.dumps({**full, "context_length": 4096}))
    monkeypatch.setattr("shared.targets.COMPETITIONS_DIR", tmp_path)
    first, second = load_competitions()
    assert [first.model_id, second.model_id] == ["org/model1", "org/b"]  # file name order
    assert (first.policy.improvement_margin, first.policy.repeats) == (0.02, 7)
    assert (first.context_length, second.context_length) == (4096, 8192)
    assert first.engine_version == shared["engine_version"]


def test_competition_id_is_the_model_and_the_profile_hash_ignores_active_and_allocation():
    from tests.factories import target

    competition = target()
    assert competition.competition_id == competition.model_id
    reweighted = competition.model_copy(update={"active": False, "allocation": 20})
    assert reweighted.identity == competition.identity
    assert reweighted.qualification_identity == competition.qualification_identity
    assert competition.model_copy(update={"draft_tokens": 4}).identity != competition.identity


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


def test_miner_default_upload_limit_matches_the_api_limit():
    from miner.cli import DEFAULTS
    from validator.config import UPLOAD_MAX_BYTES

    assert DEFAULTS["max_upload_bytes"] == UPLOAD_MAX_BYTES == 4 * 1024**3 + 2 * 1024**2
