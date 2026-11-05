import asyncio
import os
import sys
import time

import pytest

from evaluation.cache import stage_target, stage_target_async
from tests.factories import target


def test_target_is_staged_once_from_immutable_snapshot_with_links_resolved(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    weight = shared / "weight"
    weight.write_bytes(b"target weights")
    snapshot = shared / "snapshot"
    snapshot.mkdir()
    (snapshot / "model.safetensors").symlink_to(weight)
    calls = []

    def resolve(model_id, revision):
        calls.append((model_id, revision))
        return str(snapshot)

    local = stage_target(target(), tmp_path / "local", resolver=resolve)
    assert not (local / "model.safetensors").is_symlink()
    assert (local / "model.safetensors").read_bytes() == b"target weights"
    assert stage_target(target(), tmp_path / "local", resolver=resolve) == local
    assert calls == [("org/model1", "a" * 40)]


def test_target_staging_deadline_kills_process_without_executor_shutdown_delay(
    tmp_path, monkeypatch
):
    original = asyncio.create_subprocess_exec
    children = []

    async def start(*args, **kwargs):
        child = await original(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr("evaluation.cache.asyncio.create_subprocess_exec", start)
    began = time.perf_counter()
    with pytest.raises(TimeoutError):
        asyncio.run(
            stage_target_async(
                target(), tmp_path / "cache", time.time() + 0.1, tmp_path / "staging.log"
            )
        )
    assert time.perf_counter() - began < 3
    assert len(children) == 1 and children[0].returncode is not None
    with pytest.raises(ProcessLookupError):
        os.kill(children[0].pid, 0)
