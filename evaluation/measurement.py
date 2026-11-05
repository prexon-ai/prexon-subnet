"""Measurement lifecycle shared by standalone evaluation and production workers."""

from __future__ import annotations

import asyncio
import importlib
import importlib.metadata
import os
import random
import subprocess
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from evaluation.cache import stage_target_async
from evaluation.datasets.sampling import materialize, select_round
from evaluation.sglang import Arm, SGLangClient, launch_server
from shared.config import TargetConfig
from shared.contracts import BenchmarkEvidence, ExecutionTimings, HardwareEvidence, Measurement
from shared.datasets import DatasetEvidence, RoundPlan


def hardware_evidence(tensor_parallel_size: int) -> HardwareEvidence:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    rows = [row.split(",") for row in result.stdout.strip().splitlines()]
    topology = subprocess.run(
        ["nvidia-smi", "topo", "-m"], check=True, capture_output=True, text=True, timeout=30
    ).stdout
    return HardwareEvidence(
        gpu_names=tuple(row[0].strip() for row in rows),
        gpu_memory_gb=tuple(float(row[1]) * 1024**2 / 1e9 for row in rows),
        engine_version=importlib.metadata.version("sglang"),
        cuda_version=str(importlib.import_module("torch").version.cuda),
        topology=topology,
        tensor_parallel_size=tensor_parallel_size,
    )


def hardware_matches(target: TargetConfig, hardware: HardwareEvidence) -> bool:
    return (
        hardware.engine_version == target.engine_version
        and len(hardware.gpu_names) == target.hardware.gpu_count
        and len(hardware.gpu_memory_gb) == target.hardware.gpu_count
        and hardware.tensor_parallel_size == target.tensor_parallel_size
        and all(
            target.hardware.gpu_type.lower().replace(" ", "") in name.lower().replace(" ", "")
            for name in hardware.gpu_names
        )
        and all(v >= target.hardware.min_vram_gb for v in hardware.gpu_memory_gb)
        and (not target.hardware.require_nvlink or "NV" in hardware.topology)
    )


async def measure_configurations(
    target: TargetConfig,
    checkpoints: Mapping[Arm, Path | None],
    log_dir: Path,
    deadline: float,
    seed: str,
    *,
    target_path: Path | None = None,
    port: int = 30000,
    mode: Literal["head", "full"] = "full",
    dataset_cache: Path | None = None,
    prepared_round: RoundPlan | None = None,
) -> BenchmarkEvidence:
    started, monotonic_started = time.time(), time.perf_counter()
    measurements: list[Measurement] = []
    inventory: HardwareEvidence | None = None
    timings: dict[str, float] = {}
    before: list[float] = []
    after: list[float] = []
    selection: DatasetEvidence | None = None
    measured_target = target

    def recorded_timings() -> ExecutionTimings:
        return ExecutionTimings(
            **timings,
            measured_seconds=sum(m.elapsed_seconds for m in measurements),
            total_seconds=time.perf_counter() - monotonic_started,
        )

    async def collect() -> BenchmarkEvidence:
        nonlocal inventory, target_path, selection, measured_target
        if mode == "head":
            if set(checkpoints) != {"candidate"} or checkpoints["candidate"] is None:
                raise ValueError("Head mode requires exactly one candidate checkpoint")
        elif mode != "full" or (
            set(checkpoints) - {"target", "candidate", "public", "incumbent"}
            or checkpoints.get("target", False) is not None
            or checkpoints.get("candidate") is None
        ):
            raise ValueError("Full mode requires target-only and a candidate checkpoint")
        if any(path is None for arm, path in checkpoints.items() if arm != "target"):
            raise ValueError("Missing reference checkpoint")
        if not 1 <= port <= 65535:
            raise ValueError("Invalid serving port")
        if deadline <= time.time():
            raise TimeoutError("Evaluation deadline expired")
        log_dir.mkdir(parents=True, exist_ok=True)
        if target.datasets is not None:
            plan = prepared_round or select_round(
                target,
                dataset_cache or Path(os.environ.get("PREXON_DATASET_CACHE", "runtime/datasets")),
                seed,
            )
            measured_target = materialize(target, plan, seed)
            selection = plan.evidence()
            (log_dir / "dataset-plan.json").write_text(plan.model_dump_json() + "\n")
        elif prepared_round is not None:
            raise ValueError("Unexpected dataset plan for inline workload")
        inventory = hardware_evidence(target.tensor_parallel_size)
        if not hardware_matches(target, inventory):
            raise ValueError("Hardware/runtime profile mismatch")
        staging_started = time.perf_counter()
        if target_path is None:
            target_path = await stage_target_async(
                target,
                Path(os.environ.get("PREXON_TARGET_CACHE", "/tmp/prexon-targets")),
                deadline,
                log_dir / "target-staging.log",
            )
        elif not target_path.is_dir() or not (target_path / "config.json").is_file():
            raise ValueError("Local target snapshot needs a config.json")
        timings["target_staging_seconds"] = time.perf_counter() - staging_started

        @asynccontextmanager
        async def prepared(draft: Path | None, label: str) -> AsyncIterator[SGLangClient]:
            launched = time.perf_counter()
            async with launch_server(
                target, draft, log_dir / label, deadline, target_path=target_path, port=port
            ) as engine:
                timings["engine_startup_seconds"] = (
                    timings.get("engine_startup_seconds", 0) + time.perf_counter() - launched
                )
                warmup = time.perf_counter()
                await engine.warmup(measured_target)
                timings["warmup_seconds"] = (
                    timings.get("warmup_seconds", 0) + time.perf_counter() - warmup
                )
                yield engine

        if mode == "head":
            async with prepared(checkpoints["candidate"], "candidate.log") as engine:
                for repeat in range(target.policy.repeats):
                    for workload in measured_target.workloads:
                        measure, _ = await engine.measure(workload, repeat, "candidate")
                        measurements.append(measure)
            return BenchmarkEvidence(
                status="complete",
                measurements=tuple(measurements),
                hardware=inventory,
                started_at=started,
                finished_at=time.time(),
                timings=recorded_timings(),
                dataset_selection=selection,
            )

        canonical: dict[str, tuple[tuple[int, ...], ...]] = {}
        correct = references_correct = True
        async with prepared(None, "reference-before.log") as engine:
            for w in measured_target.workloads:
                measure, tokens = await engine.measure(w, 0, "target")
                before.append(measure.throughput)
                timings["reference_anchor_seconds"] = (
                    timings.get("reference_anchor_seconds", 0) + measure.elapsed_seconds
                )
                canonical[w.name] = tokens
        rng = random.Random(seed)
        for repeat in range(target.policy.repeats):
            arms = list(checkpoints)
            rng.shuffle(arms)
            for arm in arms:
                async with prepared(checkpoints[arm], f"{repeat}-{arm}.log") as engine:
                    for w in measured_target.workloads:
                        measure, tokens = await engine.measure(w, repeat, arm)
                        measurements.append(measure)
                        if arm == "candidate":
                            correct = correct and tokens == canonical[w.name]
                        else:
                            references_correct = references_correct and tokens == canonical[w.name]
        async with prepared(None, "reference-after.log") as engine:
            for w in measured_target.workloads:
                measure, tokens = await engine.measure(w, 0, "target")
                references_correct = references_correct and tokens == canonical[w.name]
                after.append(measure.throughput)
                timings["reference_anchor_seconds"] = (
                    timings.get("reference_anchor_seconds", 0) + measure.elapsed_seconds
                )
        return BenchmarkEvidence(
            status="complete",
            measurements=tuple(measurements),
            correctness_passed=correct,
            reference_correctness_passed=references_correct,
            hardware=inventory,
            reference_before=tuple(before),
            reference_after=tuple(after),
            started_at=started,
            finished_at=time.time(),
            timings=recorded_timings(),
            dataset_selection=selection,
        )

    try:
        async with asyncio.timeout(max(0.001, deadline - time.time())):
            return await collect()
    except (Exception, asyncio.CancelledError) as exc:
        return BenchmarkEvidence(
            status="inconclusive",
            measurements=tuple(measurements),
            hardware=inventory,
            reference_before=tuple(before),
            reference_after=tuple(after),
            started_at=started,
            finished_at=time.time(),
            timings=recorded_timings(),
            error_code=type(exc).__name__,
            dataset_selection=selection,
        )
