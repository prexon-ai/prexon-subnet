"""Validate the public SpecForge DFlash2 recipe boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_recipe(recipe: dict[str, Any], parent: Path, snapshot: str) -> dict[str, Any]:
    model = recipe.setdefault("model", {})
    if model.get("trust_remote_code"):
        raise ValueError("Training recipes must not enable remote code")
    if recipe.get("training", {}).get("strategy") != "dflash":
        raise ValueError("DFlash2 training requires training.strategy=dflash")
    config = model.get("draft_model_config")
    if not config:
        raise ValueError("An explicit DFlash2 draft_model_config is required")
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = parent / config_path
    data = json.loads(config_path.read_text())
    if data.get("architectures") != ["DFlash2DraftModel"] or "auto_map" in data:
        raise ValueError("Expected a data-only DFlash2DraftModel config")
    model.update(
        target_model_path=snapshot,
        trust_remote_code=False,
        draft_model_config=str(config_path.resolve()),
    )
    for key in ("train_data_path", "prompts_path", "hidden_states_path", "cache_dir"):
        value = recipe.get("data", {}).get(key)
        if value and not Path(value).is_absolute():
            recipe["data"][key] = str((parent / value).resolve())
    return recipe
