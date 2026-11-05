"""A standalone, secret-free profile for one developer evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from shared.config import FrozenModel, Positive, TargetConfig

EvaluationMode = Literal["head", "full"]


class EvaluationProfile(FrozenModel):
    target: TargetConfig
    draft_head: Path
    baseline_head: Path | None = None
    incumbent_head: Path | None = None
    mode: EvaluationMode = "head"
    backend: Literal["lium", "pod"] = "lium"
    output_dir: Path = Path("runtime/evaluations")
    target_path: Path | None = None
    seed: str = "42"
    port: int = Field(default=30000, ge=1, le=65535)
    session_seconds: int | None = Field(default=None, ge=60, le=720 * 3600)
    session_max_cost: Positive | None = None

    @model_validator(mode="after")
    def remote_target_is_absolute(self) -> EvaluationProfile:
        if (
            self.backend == "lium"
            and self.target_path is not None
            and not self.target_path.is_absolute()
        ):
            raise ValueError("The target_path inside a Lium pod must be absolute")
        return self


def load_profile(path: Path) -> EvaluationProfile:
    profile = EvaluationProfile.model_validate_json(path.read_bytes())
    base = path.resolve().parent
    updates: dict[str, Path] = {}
    for field in ("draft_head", "baseline_head", "incumbent_head", "output_dir", "target_path"):
        value = getattr(profile, field)
        if value is not None and not (field == "target_path" and profile.backend == "lium"):
            updates[field] = value if value.is_absolute() else base / value
    return profile.model_copy(update=updates)
