import asyncio
import os
from pathlib import Path

import pytest

from shared.contracts import EvaluationJob
from validator.jobs import run_job


@pytest.mark.gpu
def test_configured_real_gpu_job_completes_with_preserved_greedy_outputs(tmp_path):
    path = os.environ.get("PREXON_GPU_JOB_PATH")
    if not path:
        pytest.skip("Set PREXON_GPU_JOB_PATH to an explicitly configured GPU pilot job")
    job = EvaluationJob.model_validate_json(Path(path).read_bytes())
    result = asyncio.run(run_job(job, tmp_path, allow_file=True))
    assert result.status == "complete", result.error_code
    assert result.correctness_passed
    assert result.reference_correctness_passed
    assert len(result.measurements) >= job.target.policy.repeats * len(job.target.workloads) * 2
