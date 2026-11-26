import os
import sys
from contextlib import asynccontextmanager

import pytest

# Tests never read the owner's .env: every key it may hold is blanked before any config
# file is imported (the environment wins over .env, and an empty value means the default).
ENV_KEYS = (
    "PREXON_NETWORK",
    "PREXON_NETUID",
    "PREXON_WALLET_NAME",
    "PREXON_WALLET_HOTKEY",
    "PREXON_WALLET_PATH",
    "S3_ACCESS_KEY_ID",
    "S3_SECRET_ACCESS_KEY",
    "LIUM_API_KEY",
    "LIUM_TEMPLATE_ID",
    "LIUM_VOLUME_ID",
    "SSH_PRIVATE_KEY",
    "SSH_PUBLIC_KEY",
)
for key in ENV_KEYS:
    os.environ[key] = ""

PACKAGES = {"api", "evaluation", "miner", "shared", "validator"}
CONFIGS = ("validator.config", "evaluation.config", "shared.config")


@pytest.fixture
def set_config(monkeypatch):
    """Change config values in every module that imported them with `from ... import`."""

    def apply(**values):
        known = set().union(*(vars(__import__(name, fromlist=["_"])) for name in CONFIGS))
        unknown = set(values) - known
        assert not unknown, f"not a config value: {sorted(unknown)}"
        for module in list(sys.modules.values()):
            if (getattr(module, "__name__", None) or "").split(".")[0] not in PACKAGES:
                continue
            for name, value in values.items():
                if name in vars(module):
                    monkeypatch.setattr(module, name, value)

    return apply


@pytest.fixture(autouse=True)
def local_config(set_config, tmp_path):
    """Every test starts on local storage with runtime files in tmp: never the owner's
    buckets, targets or this checkout's runtime/ folder."""
    set_config(
        TARGETS=[],
        STATE_PATH=tmp_path / "state.sqlite",
        ARTIFACT_ROOT=str(tmp_path / "artifacts"),
        UPLOAD_STAGING_ROOT="",
        PUBLIC_ARTIFACT_ROOT=str(tmp_path / "public"),
        PUBLIC_ARTIFACT_BASE_URL="",
        UPLOAD_POLICY_PATH=tmp_path / "intake-policy/upload-policy.json",
        UPLOAD_STATE_PATH=tmp_path / "api/uploads.sqlite3",
        DATASET_CACHE=tmp_path / "datasets",
        SSH_KNOWN_HOSTS=tmp_path / "known_hosts",
    )


@pytest.fixture
def pod(tmp_path, monkeypatch):
    from shared.contracts import HardwareEvidence, Measurement
    from tests.factories import make_checkpoint

    checkpoint = make_checkpoint(tmp_path / "head")
    snapshot = tmp_path / "target"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}")
    inventory = HardwareEvidence(
        gpu_names=("NVIDIA H100",),
        gpu_memory_gb=(80,),
        engine_version="0.5.23",
        cuda_version="12.9",
        tensor_parallel_size=1,
    )
    monkeypatch.setattr("evaluation.measurement.hardware_evidence", lambda _: inventory)

    async def stage(*args, **kwargs):
        return snapshot

    monkeypatch.setattr("evaluation.measurement.stage_target_async", stage)
    calls = []

    @asynccontextmanager
    async def server(config, draft, log_path, deadline, **kwargs):
        calls.append((draft, log_path, kwargs))

        class Engine:
            async def warmup(self, _):
                return None

            async def measure(self, workload, repeat, arm):
                speed = {"target": 10, "candidate": 20, "public": 18, "incumbent": 19}[arm]
                budget = workload.requests * workload.output_tokens
                return Measurement(
                    workload=workload.name,
                    repeat=repeat,
                    arm=arm,
                    output_tokens=budget,
                    expected_tokens=budget,
                    elapsed_seconds=budget / speed,
                    p95_latency_seconds=1,
                    accepted_tokens=8 if draft else None,
                    drafted_tokens=12 if draft else None,
                    verify_steps=4 if draft else None,
                ), tuple(tuple(range(workload.output_tokens)) for _ in range(workload.requests))

        yield Engine()

    monkeypatch.setattr("evaluation.measurement.launch_server", server)
    return checkpoint, snapshot, calls
