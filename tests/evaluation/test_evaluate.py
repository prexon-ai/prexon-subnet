import asyncio
from contextlib import asynccontextmanager

import pytest

from evaluation.runner import evaluate
from tests.factories import target


def test_failed_local_run_retains_partial_evidence_without_score(tmp_path, pod, monkeypatch):
    head, snapshot, calls = pod
    original = __import__("evaluation.measurement", fromlist=["launch_server"]).launch_server

    @asynccontextmanager
    async def failing(config, draft, *args, **kwargs):
        if draft is not None:
            raise RuntimeError("engine failed")
        async with original(config, draft, *args, **kwargs) as engine:
            yield engine

    monkeypatch.setattr("evaluation.measurement.launch_server", failing)
    report = asyncio.run(evaluate(target(), head, tmp_path / "logs"))
    assert report.status == "inconclusive"
    assert report.assessment == "inconclusive"
    assert report.score is None
    assert report.error_code == "RuntimeError"
    assert len(report.reference_before) == 2
    assert report.timings is not None


def test_correctness_failure_cannot_be_hidden_by_speedup(tmp_path, pod, monkeypatch):
    head, snapshot, calls = pod
    original = __import__("evaluation.measurement", fromlist=["launch_server"]).launch_server

    @asynccontextmanager
    async def incorrect(config, draft, *args, **kwargs):
        async with original(config, draft, *args, **kwargs) as engine:
            if draft:
                measure = engine.measure

                async def wrong(*values):
                    measurement, _ = await measure(*values)
                    return measurement, ((999,),)

                engine.measure = wrong
            yield engine

    monkeypatch.setattr("evaluation.measurement.launch_server", incorrect)
    report = asyncio.run(evaluate(target(), head, tmp_path / "logs"))
    assert report.score == pytest.approx(2)
    assert report.correctness_passed is False
    assert report.assessment == "fails_local_checks"


def test_expired_measurement_budget_saves_inconclusive_evidence_without_launching(tmp_path, pod):
    import time

    from evaluation.measurement import measure_configurations

    head, snapshot, calls = pod
    result = asyncio.run(
        measure_configurations(
            target(),
            {"target": None, "candidate": head},
            tmp_path / "logs",
            time.time() - 1,
            "seed",
        )
    )
    assert result.status == "inconclusive"
    assert result.error_code == "TimeoutError"
    assert calls == []


def test_local_head_is_frozen_while_training_can_update_original_export(tmp_path, pod, monkeypatch):
    import struct

    from shared.checkpoints import sha256_file

    head, snapshot, calls = pod
    original_hash = sha256_file(head / "model.safetensors")
    module = __import__("evaluation.measurement", fromlist=["launch_server"])
    original = module.launch_server
    loaded_hashes = []

    @asynccontextmanager
    async def training(config, draft, *args, **kwargs):
        if draft is not None:
            loaded_hashes.append(sha256_file(draft / "model.safetensors"))
            tensor = head / "model.safetensors"
            data = tensor.read_bytes()
            tensor.write_bytes(data[:-8] + struct.pack("<ff", 1, 1))
        async with original(config, draft, *args, **kwargs) as engine:
            yield engine

    monkeypatch.setattr(module, "launch_server", training)
    report = asyncio.run(evaluate(target(), head, tmp_path / "logs"))
    assert report.status == "complete"
    assert loaded_hashes == [original_hash] * 5
