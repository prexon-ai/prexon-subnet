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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluation.process import terminate_process_group
from shared.targets import TargetConfig

# Several files at a time: a network volume reads and writes each file slowly.
COPY_THREADS = 8
# Written last when saving to a volume, listing every file and its size.
VOLUME_RECORD = "prexon-complete.json"


def _download(model_id: str, revision: str, destination: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id, revision=revision, local_dir=destination)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _copy(files: dict[str, int], source: Path, destination: Path) -> None:
    def copy(name: str) -> None:
        (destination / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination / name)

    destination.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(COPY_THREADS) as pool:
        # Largest first so the slowest files are not left to run alone at the end.
        list(pool.map(copy, sorted(files, key=files.__getitem__, reverse=True)))


def _recorded(copy: Path) -> dict[str, int] | None:
    """The volume copy's files, when a finished save listed them and each is still whole."""
    try:
        files = json.loads((copy / VOLUME_RECORD).read_text())
        if (
            isinstance(files, dict)
            and files
            and all(
                isinstance(size, int)
                and not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and (copy / name).stat().st_size == size
                for name, size in files.items()
            )
        ):
            return files
    except (OSError, ValueError):
        pass
    return None


def _save(staged: Path, copy: Path) -> None:
    copy.mkdir(parents=True, exist_ok=True)
    (copy / VOLUME_RECORD).unlink(missing_ok=True)
    files = {
        str(path.relative_to(staged)): path.stat().st_size
        for path in staged.rglob("*")
        if path.is_file()
    }
    _copy(files, staged, copy)
    (copy / VOLUME_RECORD).write_text(json.dumps(files, sort_keys=True))


def stage_target(
    target: TargetConfig,
    local_root: Path,
    *,
    download: Callable[[str, str, Path], None] | None = None,
    staging_root: Path | None = None,
    volume: Path | None = None,
) -> Path:
    """Put the pinned target on local disk once per rental. With a volume, copy a complete
    earlier copy from it; otherwise download, then leave a copy there for later rentals."""
    key = hashlib.sha256(f"{target.model_id}@{target.revision}".encode()).hexdigest()
    local_root.mkdir(parents=True, exist_ok=True)
    destination = local_root / key
    with (local_root / f"{key}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_dir():
            return destination
        copy = volume / key if volume else None
        with tempfile.TemporaryDirectory(dir=staging_root or local_root) as temp:
            staged = Path(temp) / "snapshot"
            started = time.monotonic()
            if copy and (files := _recorded(copy)) is not None:
                _copy(files, copy, staged)
                _log(f"Copied the target from the volume in {time.monotonic() - started:.0f} s")
            else:
                (download or _download)(target.model_id, target.revision, staged)
                # Download bookkeeping only; nothing serves from it.
                shutil.rmtree(staged / ".cache", ignore_errors=True)
                _log(f"Downloaded the target in {time.monotonic() - started:.0f} s")
                if copy:
                    started = time.monotonic()
                    try:
                        _save(staged, copy)
                        seconds = time.monotonic() - started
                        _log(f"Saved the target to the volume in {seconds:.0f} s")
                    except OSError as exc:
                        # The evaluation already has its local copy; only later rentals lose out.
                        _log(f"Could not save the target to the volume: {exc}")
            staged.replace(destination)
    return destination


async def stage_target_async(
    target: TargetConfig,
    local_root: Path,
    deadline: float,
    log_path: Path,
    volume: Path | None = None,
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
                *(["--volume", str(volume)] if volume else []),
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
    parser.add_argument("--volume", type=Path)
    args = parser.parse_args()
    data = sys.stdin.buffer.read(16 * 1024**2 + 1)
    if len(data) > 16 * 1024**2:
        raise ValueError("Target profile exceeds staging input limit")
    target = TargetConfig.model_validate_json(data)
    result = stage_target(target, args.root, staging_root=args.scratch, volume=args.volume)
    print(json.dumps(str(result.resolve())))


if __name__ == "__main__":
    main()
