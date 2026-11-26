"""The identity and local resource limit needed for miner training and packaging."""

from __future__ import annotations

from pydantic import Field

from shared.targets import FrozenModel, Policy, Revision, TargetConfig


class MinerTarget(FrozenModel):
    model_id: str = Field(min_length=1)
    revision: Revision
    competition_id: str | None = None
    max_checkpoint_bytes: int = Field(default=Policy().max_checkpoint_bytes, ge=1024)

    @classmethod
    def from_profile(cls, profile: TargetConfig) -> MinerTarget:
        return cls(
            model_id=profile.model_id,
            revision=profile.revision,
            competition_id=profile.competition_id,
            max_checkpoint_bytes=profile.policy.max_checkpoint_bytes,
        )
