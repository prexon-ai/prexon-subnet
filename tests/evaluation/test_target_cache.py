import asyncio
import os
import sys
import time

import pytest

from evaluation.cache import stage_target, stage_target_async
from tests.factories import target


def fake_download(calls):
    def download(model_id, revision, destination):
        calls.append((model_id, revision))
        destination.mkdir(parents=True)
        (destination / "config.json").write_text("{}")
        (destination / "model.safetensors").write_bytes(b"target weights")
        cache = destination / ".cache/huggingface"
        cache.mkdir(parents=True)
        (cache / "download.metadata").write_text("bookkeeping")

    return download


def test_target_is_downloaded_once_straight_into_local_staging(tmp_path):
    calls = []
    local = stage_target(target(), tmp_path / "local", download=fake_download(calls))
    assert (local / "model.safetensors").read_bytes() == b"target weights"
    assert not (local / ".cache").exists()
    assert stage_target(target(), tmp_path / "local", download=fake_download(calls)) == local
    assert calls == [("org/model1", "a" * 40)]


def test_download_is_saved_to_the_volume_for_later_rentals(tmp_path):
    calls = []
    volume = tmp_path / "volume"
    stage_target(target(), tmp_path / "first", download=fake_download(calls), volume=volume)

    def no_download(*args):
        pytest.fail("Downloaded although the volume holds a complete copy")

    local = stage_target(target(), tmp_path / "second", download=no_download, volume=volume)
    assert (local / "model.safetensors").read_bytes() == b"target weights"
    assert (local / "config.json").read_text() == "{}"
    assert not (local / ".cache").exists()
    assert calls == [("org/model1", "a" * 40)]


def test_volume_copy_that_does_not_match_its_record_is_downloaded_again(tmp_path):
    calls = []
    volume = tmp_path / "volume"
    stage_target(target(), tmp_path / "first", download=fake_download(calls), volume=volume)
    (copy,) = [p for p in volume.iterdir() if p.is_dir()]
    (copy / "model.safetensors").write_bytes(b"cut short")
    local = stage_target(
        target(), tmp_path / "second", download=fake_download(calls), volume=volume
    )
    assert (local / "model.safetensors").read_bytes() == b"target weights"
    assert len(calls) == 2


def test_volume_that_cannot_be_written_does_not_stop_the_evaluation(tmp_path):
    volume = tmp_path / "volume"
    volume.write_text("not a directory")
    local = stage_target(target(), tmp_path / "local", download=fake_download([]), volume=volume)
    assert (local / "model.safetensors").read_bytes() == b"target weights"


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
