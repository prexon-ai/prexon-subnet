"""Run and export a pinned public SpecForge training recipe."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

from miner.config import MinerTarget
from shared.checkpoints import sha256_file, validate_checkpoint
from shared.config import TargetConfig


def train_and_export(
    target: TargetConfig | MinerTarget,
    recipe_path: Path,
    run_dir: Path,
    checkpoint: Path,
    output: Path,
    *,
    specforge_version: str = "0.2.0",
    snapshot_resolver: Callable[[str, str], str] | None = None,
) -> None:
    import yaml

    from miner.training.dflash2 import resolve_recipe

    target = MinerTarget.from_profile(target) if isinstance(target, TargetConfig) else target
    recipe = yaml.safe_load(recipe_path.read_text())
    if not isinstance(recipe, dict):
        raise ValueError("Expected a training recipe mapping")
    if recipe.get("model", {}).get("trust_remote_code"):
        raise ValueError("Training recipes must not enable remote code")
    if importlib.metadata.version("specforge") != specforge_version:
        raise ValueError("Installed SpecForge does not match the requested version")
    if output.exists():
        raise ValueError("Export directory already exists")
    if snapshot_resolver is None:
        from huggingface_hub import snapshot_download

        def snapshot_resolver(model: str, revision: str) -> str:
            return str(snapshot_download(repo_id=model, revision=revision))

    snapshot = snapshot_resolver(target.model_id, target.revision)
    recipe = resolve_recipe(recipe, recipe_path.resolve().parent, snapshot)
    run_dir.mkdir(parents=True, exist_ok=True)
    recipe["output_dir"] = str((run_dir / "training").resolve())
    resolved = run_dir / "recipe.yaml"
    resolved.write_text(yaml.safe_dump(recipe, sort_keys=False))
    subprocess.run(
        ["specforge", "train", "--config", str(resolved.resolve())],
        check=True,
        cwd=recipe_path.resolve().parent,
    )
    if not checkpoint.is_dir():
        raise ValueError("Requested training checkpoint does not exist after training")
    subprocess.run(
        [
            "specforge",
            "export",
            "--to",
            "hf",
            "--checkpoint",
            str(checkpoint.resolve()),
            "--draft-config",
            recipe["model"]["draft_model_config"],
            "--embedding-source",
            snapshot,
            "--output-dir",
            str(output.resolve()),
        ],
        check=True,
    )
    validate_checkpoint(output, target.max_checkpoint_bytes)
    (run_dir / "training-record.json").write_text(
        json.dumps(
            {
                "target_id": target.model_id,
                "target_revision": target.revision,
                "competition_id": target.competition_id,
                "recipe_sha256": sha256_file(resolved),
                "specforge_version": specforge_version,
                "export": str(output.resolve()),
            },
            indent=2,
        )
        + "\n"
    )
