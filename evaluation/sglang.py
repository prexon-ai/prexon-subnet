"""Pinned SGLang process lifecycle and independently timed native generation."""

from __future__ import annotations

import asyncio
import math
import socket
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import httpx

from evaluation.process import terminate_process_group
from shared.config import TargetConfig, Workload
from shared.contracts import Measurement

Arm = Literal["target", "candidate", "public", "incumbent"]


def server_command(
    target: TargetConfig, port: int, draft: Path | None, target_path: Path | None = None
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model-path",
        str(target_path) if target_path else target.model_id,
        "--revision",
        target.revision,
        "--dtype",
        target.dtype,
        "--tp-size",
        str(target.tensor_parallel_size),
        "--context-length",
        str(target.context_length),
        "--mem-fraction-static",
        str(target.memory_fraction),
        "--disable-radix-cache",
        "--random-seed",
        "42",
    ]
    if draft is not None:
        command += [
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            str(draft),
            "--speculative-num-draft-tokens",
            str(target.draft_tokens),
        ]
    return command


class SGLangClient:
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client

    async def generate(
        self, prompt: str | tuple[int, ...], tokens: int
    ) -> tuple[tuple[int, ...], dict[str, Any]]:
        response = await self.client.post(
            "/generate",
            json={
                **({"text": prompt} if isinstance(prompt, str) else {"input_ids": prompt}),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": tokens,
                    "min_new_tokens": tokens,
                    "ignore_eos": True,
                },
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()
        ids = data.get("output_ids")
        metadata = data.get("meta_info", {})
        if (
            not isinstance(ids, list)
            or len(ids) != tokens
            or not all(type(i) is int and i >= 0 for i in ids)
            or metadata.get("completion_tokens") != tokens
        ):
            raise ValueError("Incomplete or inconsistent output token budget")
        return tuple(ids), metadata

    async def warmup(self, target: TargetConfig) -> None:
        for index in range(target.policy.warmup_requests):
            w = target.workloads[index % len(target.workloads)]
            await self.generate((w.input_ids or w.prompts)[0], w.output_tokens)

    async def measure(
        self,
        workload: Workload,
        repeat: int,
        arm: Arm,
    ) -> tuple[Measurement, tuple[tuple[int, ...], ...]]:
        flush = await self.client.post("/flush_cache")
        flush.raise_for_status()
        semaphore = asyncio.Semaphore(workload.concurrency)
        latencies: list[float] = []
        prompts = workload.input_ids or workload.prompts
        if not prompts or workload.input_ids and len(prompts) != workload.requests:
            raise ValueError("Tokenized workloads require one frozen input per request")

        async def request(index: int) -> tuple[tuple[int, ...], dict[str, Any]]:
            async with semaphore:
                started = time.perf_counter()
                result = await self.generate(prompts[index % len(prompts)], workload.output_tokens)
                latencies.append(time.perf_counter() - started)
                return result

        started = time.perf_counter()
        results = await asyncio.gather(*(request(i) for i in range(workload.requests)))
        elapsed = time.perf_counter() - started
        latency = sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)]
        # Runtime versions may omit speculative counters; retain null rather than inventing them.
        metadata = [m for _, m in results]

        def counter(name: str) -> int | None:
            values: list[int] = []
            for item in metadata:
                value = item.get(name)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    return None
                values.append(value)
            return sum(values)

        steps = counter("spec_verify_ct")
        measurement = Measurement(
            workload=workload.name,
            repeat=repeat,
            arm=arm,
            output_tokens=sum(len(ids) for ids, _ in results),
            expected_tokens=workload.requests * workload.output_tokens,
            elapsed_seconds=elapsed,
            p95_latency_seconds=latency,
            accepted_tokens=counter("spec_num_correct_drafts"),
            drafted_tokens=counter("spec_num_proposed_drafts"),
            verify_steps=steps if steps else None,
        )
        return measurement, tuple(ids for ids, _ in results)


@asynccontextmanager
async def launch_server(
    target: TargetConfig,
    draft: Path | None,
    log_path: Path,
    deadline: float,
    *,
    port: int = 30000,
    target_path: Path | None = None,
) -> AsyncIterator[SGLangClient]:
    # An existing pod may already have an inference/capture server. Never benchmark it
    # accidentally while a new engine fails to claim the same port.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            *server_command(target, port, draft, target_path),
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
        try:
            timeout = httpx.Timeout(min(target.policy.max_job_seconds, 1800), connect=10)
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", timeout=timeout, trust_env=False
            ) as client:
                ready_deadline = min(deadline, time.time() + 1800)
                while True:
                    if process.returncode is not None:
                        raise RuntimeError(
                            "SGLang exited before becoming healthy; see private worker log"
                        )
                    if time.time() >= ready_deadline:
                        raise TimeoutError("SGLang startup deadline")
                    try:
                        response = await client.get("/health", timeout=2)
                        if response.is_success:
                            break
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.5)
                yield SGLangClient(client)
        finally:
            await terminate_process_group(process)
