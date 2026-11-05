"""Fixed SSH protocol for development evaluation in a rented pod; never receives API keys."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO

from evaluation.datasets.sampling import validate_plan
from evaluation.profile import EvaluationProfile
from shared.checkpoints import validate_checkpoint
from shared.config import TargetConfig
from shared.contracts import FileDigest
from shared.datasets import RoundPlan


def install(root: Path, archive: BinaryIO, expected_digest: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        staging = Path(directory)
        total = 0
        names: set[str] = set()
        with tarfile.open(fileobj=archive, mode="r|") as stream:
            for member in stream:
                parts = Path(member.name).parts
                if (
                    not member.isfile()
                    or member.name in names
                    or member.size < 0
                    or len(parts) not in {1, 2}
                    or any(p in {".", ".."} for p in parts)
                    or Path(member.name).is_absolute()
                ):
                    raise ValueError("Unsafe evaluation archive")
                names.add(member.name)
                total += member.size
                if total > 3 * 4 * 1024**3 + 64 * 1024**2 or len(names) > 386:
                    raise ValueError("Oversized evaluation archive")
                path = staging / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                incoming = stream.extractfile(member)
                assert incoming is not None
                with incoming, path.open("wb") as stream_out:
                    shutil.copyfileobj(incoming, stream_out)
        request_path = staging / "request.json"
        if hashlib.sha256(request_path.read_bytes()).hexdigest() != expected_digest:
            raise ValueError("Evaluation request digest mismatch")
        request = json.loads(request_path.read_bytes())
        target = TargetConfig.model_validate(request["target"])
        if target.datasets is not None:
            validate_plan(
                target, RoundPlan.model_validate(request["dataset_plan"]), request["seed"]
            )
        elif request.get("dataset_plan") is not None:
            raise ValueError("Unexpected transferred dataset plan")
        expected_names = {"request.json", "target.json"}
        for arm, digests in request["checkpoints"].items():
            if arm not in {"candidate", "public", "incumbent"}:
                raise ValueError("Unexpected checkpoint arm")
            expected = tuple(FileDigest.model_validate(d) for d in digests)
            if validate_checkpoint(staging / arm, target.policy.max_checkpoint_bytes) != expected:
                raise ValueError("Transferred checkpoint digest mismatch")
            expected_names.update(f"{arm}/{f.path}" for f in expected)
        if names != expected_names or "candidate" not in request["checkpoints"]:
            raise ValueError("Unexpected files in evaluation archive")
        if TargetConfig.model_validate_json((staging / "target.json").read_bytes()) != target:
            raise ValueError("Transferred profile mismatch")
        with (root / "install.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            destination = root / "input"
            if destination.exists():
                if (destination / "request.json").read_bytes() != request_path.read_bytes():
                    raise ValueError("Remote evaluation identity already has different inputs")
                return  # Never overwrite weights that a resumed run may be reading.
            staging.rename(destination)


def inline_profile(root: Path) -> EvaluationProfile:
    request = json.loads((root / "input/request.json").read_bytes())
    return EvaluationProfile(
        target=TargetConfig.model_validate(request["target"]),
        draft_head=root / "input/candidate",
        backend="pod",
        mode=request["mode"],
        baseline_head=root / "input/public" if "public" in request["checkpoints"] else None,
        incumbent_head=root / "input/incumbent" if "incumbent" in request["checkpoints"] else None,
        output_dir=root / "logs",
        target_path=request["target_path"],
        seed=request["seed"],
        port=request["port"],
    )


def run(root: Path) -> None:
    with (root / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        if (root / "report.json").exists() or (root / "exit.json").exists():
            return
        with (root / "launcher.log").open("ab") as log:
            result = subprocess.run(
                [sys.executable, "-m", "evaluation.remote", "execute", "--root", str(root)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                check=False,
            )
        (root / "exit.json").write_text(json.dumps({"returncode": result.returncode}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["upload", "start", "run", "execute", "report", "logs"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--digest", default="")
    args = parser.parse_args()
    root = args.root
    if args.action == "upload":
        install(root, sys.stdin.buffer, args.digest)
    elif args.action == "start":
        with (root / "launcher.log").open("ab") as log:
            subprocess.Popen(
                [sys.executable, "-m", "evaluation.remote", "run", "--root", str(root)],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
    elif args.action == "run":
        run(root)
    elif args.action == "execute":
        from evaluation.cli import run_inline
        from evaluation.lium import write_json

        request = json.loads((root / "input/request.json").read_bytes())
        plan = (
            RoundPlan.model_validate(request["dataset_plan"])
            if request.get("dataset_plan")
            else None
        )
        report = run_inline(inline_profile(root), request["execution_id"], prepared_round=plan)
        write_json(root / "report.json", report.model_dump(mode="json"))
        if report.assessment not in {"head_measured", "passes_local_checks"}:
            raise SystemExit(2 if report.assessment == "inconclusive" else 1)
    elif args.action == "report":
        report = root / "report.json"
        if not report.exists():
            raise SystemExit(45 if (root / "exit.json").exists() else 44)
        sys.stdout.buffer.write(report.read_bytes())
    elif args.action == "logs":
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|gz") as archive:
            for name in ("logs", "launcher.log", "exit.json"):
                if (root / name).exists():
                    archive.add(root / name, arcname=name)


if __name__ == "__main__":
    main()
