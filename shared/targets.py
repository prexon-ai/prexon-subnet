"""Target competition profiles: the models that define each competition and its ID."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
# A competition is named by its target model, for example "Qwen/Qwen3-8B".
CompetitionId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][\w.-]*(/[\w.-]+)?$", max_length=200)]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def identity(value: BaseModel, *, exclude: set[str] | None = None) -> str:
    data = value.model_dump(mode="json", exclude=exclude)
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


class Hardware(FrozenModel):
    gpu_type: str = Field(min_length=1)
    gpu_count: int = Field(default=1, ge=1)
    min_vram_gb: Positive
    min_cpus: int = Field(default=4, ge=1)
    min_ram_gb: Positive = 32
    min_disk_gb: Positive = 100
    max_price_per_gpu_hour: Positive = 3
    require_nvlink: bool = False


class LengthBucket(FrozenModel):
    name: Name
    min_tokens: int = Field(ge=1)
    max_tokens: int = Field(ge=1)
    weight: Positive

    @model_validator(mode="after")
    def valid_range(self) -> LengthBucket:
        if self.max_tokens < self.min_tokens:
            raise ValueError("Invalid input length range")
        return self


DEFAULT_BUCKETS = (
    LengthBucket(name="short", min_tokens=1, max_tokens=512, weight=0.3),
    LengthBucket(name="medium", min_tokens=513, max_tokens=2048, weight=0.4),
    LengthBucket(name="long", min_tokens=2049, max_tokens=6144, weight=0.3),
)


def validate_buckets(buckets: tuple[LengthBucket, ...]) -> None:
    if not buckets or len({b.name for b in buckets}) != len(buckets):
        raise ValueError("Specify unique input length buckets")
    ordered = sorted(buckets, key=lambda b: b.min_tokens)
    if any(a.max_tokens >= b.min_tokens for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("Input length buckets must not overlap")
    if abs(sum(b.weight for b in buckets) - 1) > 1e-9:
        raise ValueError("Input length weights must sum to 1")


class DatasetConfig(FrozenModel):
    manifest: Digest
    category_weights: dict[Name, Positive] = {"chat": 0.4, "code": 0.35, "math": 0.25}
    buckets: tuple[LengthBucket, ...] = DEFAULT_BUCKETS
    min_unique_prompts: int = Field(default=512, ge=1)
    seed_block_delay: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def valid_mix(self) -> DatasetConfig:
        validate_buckets(self.buckets)
        if not self.category_weights or abs(sum(self.category_weights.values()) - 1) > 1e-9:
            raise ValueError("Dataset category weights must sum to 1")
        return self


class Workload(FrozenModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")
    concurrency: int = Field(ge=1)
    requests: int = Field(ge=1)
    output_tokens: int = Field(ge=1)
    weight: Positive
    prompts: tuple[str, ...] = ()
    input_ids: tuple[tuple[Annotated[int, Field(ge=0)], ...], ...] = ()
    max_p95_latency_seconds: Positive | None = None

    @model_validator(mode="after")
    def sufficient_requests(self) -> Workload:
        if self.requests < self.concurrency or any(not p.strip() for p in self.prompts):
            raise ValueError("Each workload needs requests >= concurrency and nonempty prompts")
        if self.prompts and self.input_ids or any(not ids for ids in self.input_ids):
            raise ValueError("Specify text or tokenized inputs, with no empty inputs")
        return self


class Policy(FrozenModel):
    improvement_margin: float = Field(default=0.015, ge=0, lt=1)
    confidence: float = Field(default=0.95, gt=0.5, lt=1)
    max_regression: float = Field(default=0.05, ge=0, lt=1)
    max_reference_drift: float = Field(default=0.05, ge=0, lt=1)
    repeats: int = Field(default=5, ge=3, le=100)
    warmup_requests: int = Field(default=2, ge=1)
    max_checkpoint_bytes: int = Field(default=4 * 1024**3, ge=1024)
    max_job_seconds: int = Field(default=7200, ge=60)


class PublicReference(FrozenModel):
    artifact: Digest
    method: Literal["dflash2"] = "dflash2"


class SimilarityPolicy(FrozenModel):
    version: Literal[1] = 1
    reject_near_copies: bool = False
    min_cosine: float = Field(default=0.999999, ge=0, le=1)
    max_relative_distance: float = Field(default=0.000001, ge=0, lt=1)
    min_copied_fraction: float = Field(default=0.99, gt=0, le=1)
    calibration_artifact: Digest | None = None

    @model_validator(mode="after")
    def calibrated_rejection(self) -> SimilarityPolicy:
        if self.reject_near_copies and self.calibration_artifact is None:
            raise ValueError("Near-copy rejection requires published calibration evidence")
        return self


class TargetConfig(FrozenModel):
    """One competition: the target model and the exact benchmark its drafters are measured on."""

    # The competition's public ID is its model ID.
    model_id: CompetitionId
    revision: Revision
    # The validator evaluates only active competitions.
    active: bool
    allocation: float = Field(gt=0, le=100)
    methods: tuple[Literal["dflash2"], ...] = ("dflash2",)
    engine_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:[a-z0-9.+-]*)$")
    worker_image: str = Field(pattern=r"^\S+@sha256:[0-9a-f]{64}$")
    lium_template_id: str = ""
    hardware: Hardware
    workloads: tuple[Workload, ...] = Field(min_length=1)
    policy: Policy = Policy()
    public_reference: PublicReference | None = None
    datasets: DatasetConfig | None = None
    similarity: SimilarityPolicy | None = None
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    context_length: int = Field(default=8192, ge=128)
    memory_fraction: float = Field(default=0.8, gt=0, lt=1)
    draft_tokens: int = Field(default=8, ge=2, le=64)
    # Operator qualification of the exact runtime/profile is separate from finite output tests.
    qualification_artifact: Digest | None = None
    tensor_parallel_size: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def valid_profile(self) -> TargetConfig:
        if abs(sum(w.weight for w in self.workloads) - 1) > 1e-9:
            raise ValueError("Workload weights must sum to 1")
        if len({w.name for w in self.workloads}) != len(self.workloads):
            raise ValueError("Workload names must be unique")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ValueError("Specify unique supported speculative methods")
        if self.tensor_parallel_size != self.hardware.gpu_count:
            raise ValueError("tensor_parallel_size must match the total GPU allocation")
        if any(w.output_tokens >= self.context_length for w in self.workloads):
            raise ValueError("Output budget exceeds context length")
        if self.datasets is None:
            if any(not w.prompts and not w.input_ids for w in self.workloads):
                raise ValueError("Each workload requires inputs or a dataset configuration")
        else:
            for w in self.workloads:
                if w.prompts or w.input_ids:
                    raise ValueError("Dataset workloads cannot mix inline inputs")
                if w.requests < self.datasets.min_unique_prompts:
                    raise ValueError("Dataset workloads need the minimum distinct prompt count")
                if any(
                    b.max_tokens + w.output_tokens > self.context_length
                    for b in self.datasets.buckets
                ):
                    raise ValueError("Input and output token budgets exceed context length")
        return self

    @property
    def competition_id(self) -> str:
        return self.model_id

    @property
    def identity(self) -> str:
        """Internal profile hash: what is measured, not whether or how much it is rewarded."""
        return identity(self, exclude={"active", "allocation"})

    @property
    def qualification_identity(self) -> str:
        """What a qualification record names: the profile hash without that record."""
        return identity(self, exclude={"active", "allocation", "qualification_artifact"})


# competitions/ at the root of the checkout, found from the code, not the working folder:
# shared.json holds the settings every competition shares, and each other file one
# competition's own settings, laid over them.
COMPETITIONS_DIR = Path(__file__).resolve().parent.parent / "competitions"


def merge(shared: dict[str, Any], own: dict[str, Any]) -> dict[str, Any]:
    """Lay one competition's settings over the shared ones; nested objects merge."""
    merged = dict(shared)
    for key, value in own.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_competitions() -> tuple[TargetConfig, ...]:
    """Every competition in competitions/, active or not, in file name order."""
    shared = json.loads((COMPETITIONS_DIR / "shared.json").read_text())
    return tuple(
        TargetConfig.model_validate(merge(shared, json.loads(path.read_text())))
        for path in sorted(COMPETITIONS_DIR.glob("*.json"))
        if path.name != "shared.json"
    )


def find_competition(model_id: str) -> TargetConfig:
    for competition in load_competitions():
        if competition.model_id == model_id:
            return competition
    raise ValueError(f"No competition for {model_id} in competitions/")
