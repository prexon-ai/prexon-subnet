"""Deterministic checkpoints, profiles, and evidence shared by the test suite."""

import json
import struct

from shared.config import TargetConfig
from shared.contracts import (
    Commitment,
    EvaluationJob,
    EvaluationResult,
    HardwareEvidence,
    Measurement,
    Provenance,
)


def make_checkpoint(path):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlash2DraftModel"],
                "hidden_size": 4,
                "vocab_size": 8,
                "num_hidden_layers": 1,
                "dflash_config": {
                    "block_size": 4,
                    "target_layer_ids": [0, 1],
                    "mask_token_id": 7,
                    "conv_kernel_size": 2,
                    "conv_group_size": 2,
                    "selector_rank": 2,
                    "selector_top_k": 4,
                },
            }
        )
    )
    header = json.dumps({"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    return path


def provenance():
    return Provenance(
        training_description="public recipe",
        data_sources=("licensed corpus",),
        license="Apache-2.0",
        recipe_sha256="b" * 64,
        deployment_notes="Use the pinned DFLASH engine",
    )


def target(slot=1, allocation=100):
    return TargetConfig(
        slot=slot,
        model_id=f"org/model{slot}",
        revision="a" * 40,
        allocation=allocation,
        engine_version="0.5.23",
        worker_image="image@sha256:" + "a" * 64,
        hardware={"gpu_type": "H100", "min_vram_gb": 80},
        workloads=[
            {
                "name": "low",
                "concurrency": 1,
                "requests": 2,
                "output_tokens": 8,
                "weight": 0.2,
                "prompts": ["hidden"],
            },
            {
                "name": "high",
                "concurrency": 2,
                "requests": 2,
                "output_tokens": 8,
                "weight": 0.8,
                "prompts": ["secret"],
            },
        ],
        qualification_artifact="d" * 64,
    )


def job(t=None, incumbent=None):
    t = t or target()
    return EvaluationJob(
        job_id="a" * 32,
        session_id="b" * 32,
        target=t,
        commitment=Commitment(
            netuid=1, block=10, block_hash="hash", hotkey="miner", uid=1, artifact="c" * 64
        ),
        candidate_url="https://private/object?secret=key",
        deadline=10000,
        incumbent_artifact=incumbent,
    )


def evidence(j, candidate=2, incumbent=None):
    measures = []
    for repeat in range(j.target.policy.repeats):
        for w in j.target.workloads:
            for arm, speed in [("target", 1), ("candidate", candidate)]:
                measures.append(
                    Measurement(
                        workload=w.name,
                        repeat=repeat,
                        arm=arm,
                        output_tokens=16,
                        expected_tokens=16,
                        elapsed_seconds=16 / speed,
                        p95_latency_seconds=1,
                    )
                )
            if incumbent:
                measures.append(
                    Measurement(
                        workload=w.name,
                        repeat=repeat,
                        arm="incumbent",
                        output_tokens=16,
                        expected_tokens=16,
                        elapsed_seconds=16 / incumbent,
                        p95_latency_seconds=1,
                    )
                )
    return EvaluationResult(
        job_id=j.job_id,
        session_id=j.session_id,
        competition_id=j.target.identity,
        submission_id=j.commitment.identity,
        incumbent_artifact=j.incumbent_artifact,
        status="complete",
        measurements=measures,
        correctness_passed=True,
        reference_correctness_passed=True,
        qualification_passed=True,
        hardware=HardwareEvidence(
            gpu_names=("NVIDIA H100",),
            gpu_memory_gb=(80,),
            engine_version="0.5.23",
            cuda_version="12.9",
            tensor_parallel_size=1,
        ),
        reference_before=(1, 1),
        reference_after=(1, 1),
        started_at=100,
        finished_at=200,
    )


def write_profile(path, head, **updates):
    data = {
        "target": target().model_dump(mode="json"),
        "draft_head": str(head),
        "backend": "pod",
        "output_dir": "reports",
        **updates,
    }
    path.write_text(json.dumps(data))
    return path
