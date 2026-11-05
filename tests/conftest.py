from contextlib import asynccontextmanager

import pytest

from shared.contracts import HardwareEvidence, Measurement
from tests.factories import make_checkpoint


@pytest.fixture
def pod(tmp_path, monkeypatch):
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
