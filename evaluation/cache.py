from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from evaluation.process import terminate_process_group
from shared.config import TargetConfig


def stage_target(
    target: TargetConfig,
    local_root: Path,
    *,
    resolver: Callable[[str, str], str] | None = None,
    staging_root: Path | None = None,
) -> Path:
    """Reuse shared downloads, then serve heavy reads from local disk across a rental's jobs."""
    key = hashlib.sha256(f"{target.model_id}@{target.revision}".encode()).hexdigest()
    local_root.mkdir(parents=True, exist_ok=True)
    destination = local_root / key
    with (local_root / f"{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_dir():
            return destination
        if resolver is None:
            from huggingface_hub import snapshot_download

            def resolver(model_id: str, revision: str) -> str:
                return str(snapshot_download(repo_id=model_id, revision=revision))

        source = Path(resolver(target.model_id, target.revision))
        with tempfile.TemporaryDirectory(dir=staging_root or local_root) as temp:
            staged = Path(temp) / "snapshot"
            shutil.copytree(source, staged, symlinks=False)
            staged.replace(destination)
    return destination


async def stage_target_async(
    target: TargetConfig,
    local_root: Path,
    deadline: float,
    log_path: Path,
) -> Path:
    """Run cache lock/download/copy in a process that a deadline can actually stop."""
    local_root = local_root.resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local_root, prefix=".prexon-stage-") as scratch:
        with log_path.open("ab") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "evaluation.cache",
                "--root",
                str(local_root),
                "--scratch",
                scratch,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                start_new_session=True,
            )
            try:
                async with asyncio.timeout(max(0.001, deadline - time.time())):
                    stdout, _ = await process.communicate(target.model_dump_json().encode())
                if process.returncode != 0:
                    raise RuntimeError("Target staging failed; see target-staging.log")
                result = Path(json.loads(stdout))
                key = hashlib.sha256(f"{target.model_id}@{target.revision}".encode()).hexdigest()
                if result != local_root / key or not result.is_dir():
                    raise ValueError("Invalid staged target path")
                return result
            finally:
                await terminate_process_group(process)


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal pinned target-cache staging")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    data = sys.stdin.buffer.read(16 * 1024**2 + 1)
    if len(data) > 16 * 1024**2:
        raise ValueError("Target profile exceeds staging input limit")
    target = TargetConfig.model_validate_json(data)
    result = stage_target(target, args.root, staging_root=args.scratch)
    print(json.dumps(str(result.resolve())))


if __name__ == "__main__":
    main()
