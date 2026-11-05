from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import signal
import tempfile
import time
from pathlib import Path

import httpx

from shared.artifacts import download_url
from shared.contracts import EvaluationJob, EvaluationResult
from validator.jobs import run_job


def main() -> None:
    parser = argparse.ArgumentParser(description="Prexon temporary GPU evidence worker")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--job", type=Path)
    source.add_argument("--job-url")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--result", type=Path)
    destination.add_argument("--result-url")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/prexon-worker"))
    args = parser.parse_args()
    os.umask(0o077)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.work_dir) as temp:
        path = args.job or Path(temp) / "job.json"
        if args.job_url:
            download_url(args.job_url, path, 16 * 1024**2)
        job = EvaluationJob.model_validate_json(path.read_bytes())
        # Remote relaunches of the same job cannot run GPU work concurrently.
        with (args.work_dir / f"{job.job_id}.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            cached = args.work_dir / f"{job.job_id}.result.json"
            if cached.exists():
                data = cached.read_bytes()
            else:

                async def execute() -> EvaluationResult:
                    running = asyncio.create_task(
                        run_job(job, Path(temp) / "evaluation", allow_file=bool(args.job))
                    )
                    loop = asyncio.get_running_loop()
                    loop.add_signal_handler(signal.SIGTERM, running.cancel)
                    try:
                        return await asyncio.wait_for(
                            running, max(0.001, job.deadline - time.time())
                        )
                    finally:
                        loop.remove_signal_handler(signal.SIGTERM)

                result = asyncio.run(execute())
                data = result.model_dump_json().encode()
                cached.write_bytes(data)
            if args.result_url:
                response = httpx.put(args.result_url, content=data, timeout=120)
                response.raise_for_status()
            else:
                args.result.parent.mkdir(parents=True, exist_ok=True)
                args.result.write_bytes(data)


if __name__ == "__main__":
    main()
