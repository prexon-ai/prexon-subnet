"""Portable dataset plans and provenance; no dataset or engine dependencies."""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import Field, model_validator

from shared.config import Digest, FrozenModel, Name, Revision, identity


class RoundContext(FrozenModel):
    cutoff_block: int = Field(ge=0)
    seed_block: int = Field(ge=1)
    block_hash: str = Field(pattern=r"^0x[0-9a-f]{64}$")
    chain: str = Field(min_length=1)
    netuid: int = Field(ge=0)

    @model_validator(mode="after")
    def future_entropy(self) -> RoundContext:
        if self.seed_block <= self.cutoff_block:
            raise ValueError("Sampling entropy must follow the submission cutoff")
        return self

    def seed(self, competition_id: str) -> str:
        return hashlib.sha256(
            ("prexon-datasets-v1:" + competition_id + ":" + identity(self)).encode()
        ).hexdigest()


class Prompt(FrozenModel):
    category: Name
    bucket: Name
    input_ids: tuple[Annotated[int, Field(ge=0)], ...] = Field(min_length=1)

    @property
    def identity(self) -> str:
        return hashlib.sha256(
            json.dumps(self.input_ids, separators=(",", ":")).encode()
        ).hexdigest()


class SelectedWorkload(FrozenModel):
    name: Name
    prompt_ids: tuple[Digest, ...] = Field(min_length=1)


class DatasetEvidence(FrozenModel):
    version: Literal[1] = 1
    manifest: Digest
    plan_digest: Digest
    seed: Digest
    workloads: tuple[SelectedWorkload, ...]
    category_counts: dict[str, dict[str, int]]
    bucket_counts: dict[str, dict[str, int]]


class RoundPlan(FrozenModel):
    version: Literal[1] = 1
    manifest: Digest
    model_id: str
    revision: Revision
    seed: Digest
    prompts: tuple[Prompt, ...]
    workloads: tuple[SelectedWorkload, ...]

    @property
    def identity(self) -> str:
        return identity(self)

    def evidence(self) -> DatasetEvidence:
        from collections import Counter

        lookup = {p.identity: p for p in self.prompts}
        return DatasetEvidence(
            manifest=self.manifest,
            plan_digest=self.identity,
            seed=self.seed,
            workloads=self.workloads,
            category_counts={
                w.name: dict(Counter(lookup[p].category for p in w.prompt_ids))
                for w in self.workloads
            },
            bucket_counts={
                w.name: dict(Counter(lookup[p].bucket for p in w.prompt_ids))
                for w in self.workloads
            },
        )
