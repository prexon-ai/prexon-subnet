import asyncio
import json
import subprocess
import sys

import httpx
import pytest

from evaluation.sglang import SGLangClient, server_command
from tests.factories import job, target


def test_launch_pins_target_and_only_enables_draft_for_speculative_arm(tmp_path):
    t = target()
    normal = server_command(t, 30000, None)
    draft = server_command(t, 30000, tmp_path / "draft")
    assert normal[normal.index("--revision") + 1] == "a" * 40
    assert "--speculative-algorithm" not in normal
    assert draft[draft.index("--speculative-algorithm") + 1] == "DFLASH"
    assert "--trust-remote-code" not in draft
    assert "--disable-radix-cache" in draft


def test_real_client_counts_tokens_and_rejects_short_output():
    short = False
    calls = []

    def handle(request):
        nonlocal short
        calls.append(request.url.path)
        if request.url.path == "/flush_cache":
            return httpx.Response(200, json={"success": True})
        body = json.loads(request.content)
        assert body["sampling_params"]["ignore_eos"] is True
        assert body["sampling_params"]["temperature"] == 0
        n = 3 if short else 8
        return httpx.Response(
            200, json={"output_ids": list(range(n)), "meta_info": {"completion_tokens": n}}
        )

    async def run():
        nonlocal short
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handle), base_url="http://engine"
        ) as client:
            engine = SGLangClient(client)
            result, tokens = await engine.measure(target().workloads[0], 0, "candidate")
            assert result.output_tokens == 16
            assert result.elapsed_seconds > 0
            assert len(tokens) == 2
            short = True
            with pytest.raises(ValueError, match="output"):
                await engine.measure(target().workloads[0], 1, "candidate")

    asyncio.run(run())
    assert calls.count("/flush_cache") == 2


def test_worker_protocol_returns_inconclusive_without_qualification(tmp_path):
    from tests.factories import job
    from validator.jobs import run_job

    j = job().model_copy(update={"qualification_url": None})
    result = asyncio.run(run_job(j, tmp_path))
    assert result.status == "inconclusive"
    assert result.error_code == "qualification_missing"


@pytest.mark.parametrize("bad_after", [False, True])
def test_worker_collects_complete_evidence_and_separate_operational_timings(
    tmp_path, monkeypatch, bad_after
):
    from contextlib import asynccontextmanager

    from tests.factories import evidence
    from tests.integration.test_controller import setup_pipeline
    from validator.jobs import run_job

    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    j = provider.jobs[0].model_copy(update={"deadline": 9999999999})
    hardware = evidence(j).hardware
    monkeypatch.setattr("evaluation.measurement.hardware_evidence", lambda _: hardware)

    async def stage(*args, **kwargs):
        return tmp_path / "target"

    monkeypatch.setattr("evaluation.measurement.stage_target_async", stage)

    is_after = False

    def handle(request):
        if request.url.path == "/flush_cache":
            return httpx.Response(200, json={"success": True})
        body = json.loads(request.content)
        n = body["sampling_params"]["max_new_tokens"]
        ids = list(range(1, n + 1)) if bad_after and is_after else list(range(n))
        return httpx.Response(200, json={"output_ids": ids, "meta_info": {"completion_tokens": n}})

    @asynccontextmanager
    async def server(*args, **kwargs):
        nonlocal is_after
        is_after = args[2].name == "reference-after.log"
        async with httpx.AsyncClient(
            base_url="http://engine", transport=httpx.MockTransport(handle)
        ) as client:
            yield SGLangClient(client)

    monkeypatch.setattr("evaluation.measurement.launch_server", server)
    result = asyncio.run(run_job(j, tmp_path / "worker", allow_file=True))
    assert result.status == "complete", result.error_code
    assert result.correctness_passed
    assert result.reference_correctness_passed is (not bad_after)
    assert len(result.measurements) == 20
    assert len(result.reference_before) == len(result.reference_after) == 2
    assert result.timings.total_seconds > 0
    assert result.timings.engine_startup_seconds > 0
    assert result.timings.target_staging_seconds > 0
    assert result.timings.measured_seconds > 0


@pytest.mark.parametrize("healthy", [True, False])
def test_engine_startup_timeout_always_cleans_its_process_group(tmp_path, monkeypatch, healthy):
    import signal
    import time

    from evaluation.sglang import launch_server

    class Process:
        pid = 123456789
        returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

    async def start(*args, **kwargs):
        return Process()

    monkeypatch.setattr("evaluation.sglang.asyncio.create_subprocess_exec", start)
    groups = []
    monkeypatch.setattr("evaluation.process.os.killpg", lambda pid, sig: groups.append((pid, sig)))
    original_client = httpx.AsyncClient

    def client(**kwargs):
        return original_client(
            **kwargs, transport=httpx.MockTransport(lambda _: httpx.Response(200))
        )

    monkeypatch.setattr("evaluation.sglang.httpx.AsyncClient", client)

    async def run():
        async with launch_server(
            target(), None, tmp_path / "engine.log", time.time() + (10 if healthy else -1)
        ) as engine:
            assert isinstance(engine, SGLangClient)

    if healthy:
        asyncio.run(run())
    else:
        with pytest.raises(TimeoutError):
            asyncio.run(run())
    assert groups == [(123456789, signal.SIGTERM)]


def test_engine_refuses_a_port_owned_by_an_existing_pod_server(tmp_path, monkeypatch):
    import socket
    import time

    from evaluation.sglang import launch_server

    async def must_not_start(*args, **kwargs):
        raise AssertionError("A busy port must be detected before process launch")

    monkeypatch.setattr("evaluation.sglang.asyncio.create_subprocess_exec", must_not_start)
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]

        async def run():
            async with launch_server(
                target(), None, tmp_path / "engine.log", time.time() + 10, port=port
            ):
                raise AssertionError("The existing server must never be used")

        with pytest.raises(OSError):
            asyncio.run(run())


def test_cancellation_during_engine_teardown_still_kills_and_reaps(tmp_path, monkeypatch):
    import signal
    import time

    from evaluation.sglang import launch_server

    async def run():
        entered, killed = asyncio.Event(), asyncio.Event()
        calls = []

        class Process:
            pid = 123456789
            returncode = None

            async def wait(self):
                entered.set()
                await killed.wait()
                self.returncode = 0
                return 0

        process = Process()

        async def start(*args, **kwargs):
            return process

        def kill(pid, sig):
            calls.append(sig)
            if sig == signal.SIGKILL:
                killed.set()

        original_wait = asyncio.wait_for

        async def wait(awaitable, timeout):
            return await original_wait(awaitable, min(timeout, 0.05))

        original_client = httpx.AsyncClient
        monkeypatch.setattr("evaluation.sglang.asyncio.create_subprocess_exec", start)
        monkeypatch.setattr("evaluation.process.os.killpg", kill)
        monkeypatch.setattr("evaluation.process.asyncio.wait_for", wait)
        monkeypatch.setattr(
            "evaluation.sglang.httpx.AsyncClient",
            lambda **kwargs: original_client(
                **kwargs, transport=httpx.MockTransport(lambda _: httpx.Response(200))
            ),
        )

        async def context():
            async with launch_server(target(), None, tmp_path / "log", time.time() + 5):
                pass

        task = asyncio.create_task(context())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert calls == [signal.SIGTERM, signal.SIGKILL]
        assert process.returncode == 0

    asyncio.run(run())


def test_engine_port_preflight_allows_reuse_after_previous_server_closed(tmp_path, monkeypatch):
    import socket
    import time

    from evaluation.sglang import launch_server

    class Process:
        pid = 123456789
        returncode = None

        async def wait(self):
            return 0

    async def start(*args, **kwargs):
        return Process()

    monkeypatch.setattr("evaluation.sglang.asyncio.create_subprocess_exec", start)
    monkeypatch.setattr("evaluation.process.os.killpg", lambda *_: None)
    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        "evaluation.sglang.httpx.AsyncClient",
        lambda **kwargs: original_client(
            **kwargs, transport=httpx.MockTransport(lambda _: httpx.Response(200))
        ),
    )
    with socket.socket() as previous, socket.socket() as client:
        previous.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        previous.bind(("127.0.0.1", 0))
        previous.listen()
        port = previous.getsockname()[1]
        client.connect(("127.0.0.1", port))
        connection, _ = previous.accept()
        connection.close()
        assert client.recv(1) == b""

    async def run():
        async with launch_server(target(), None, tmp_path / "log", time.time() + 5, port=port):
            pass

    asyncio.run(run())


def test_worker_termination_cancels_evaluation_and_writes_recoverable_result(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(job().model_copy(update={"deadline": 9999999999}).model_dump_json())
    result = tmp_path / "result.json"
    code = """
import asyncio, os, signal, sys, time
from shared.contracts import EvaluationResult
from validator import worker
async def interrupted(job, directory, **kwargs):
    asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)
    try:
        await asyncio.sleep(60)
    except asyncio.CancelledError:
        return EvaluationResult(job_id=job.job_id, session_id=job.session_id,
            competition_id=job.target.identity, submission_id=job.commitment.identity,
            status="inconclusive", started_at=time.time(), finished_at=time.time(),
            error_code="CancelledError")
worker.run_job=interrupted
sys.argv=["prexon-worker","--job",sys.argv[1],"--result",sys.argv[2],"--work-dir",sys.argv[3]]
worker.main()
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(path), str(result), str(tmp_path / "work")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert process.returncode == 0, process.stderr
    assert '"status":"inconclusive"' in result.read_text()
