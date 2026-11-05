"""Role-scoped dotenv configuration and immutable competition profiles."""

from __future__ import annotations

import hashlib
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, model_validator

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Name = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


def identity(value: BaseModel, *, exclude: set[str] | None = None) -> str:
    data = value.model_dump(mode="json", exclude=exclude)
    if isinstance(value, TargetConfig) and value.similarity is None:
        data.pop("similarity", None)
    if isinstance(value, TargetConfig) and value.datasets is None:
        # Optional dataset fields must not invalidate queued pre-dataset profiles/qualifications.
        data.pop("datasets")
        for workload in data["workloads"]:
            if not workload["input_ids"]:
                workload.pop("input_ids")
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
    award_seconds: Positive = 5 * 24 * 3600
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
    slot: int = Field(ge=1, le=2)
    model_id: str = Field(min_length=1)
    revision: Revision
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
    def identity(self) -> str:
        return identity(self)

    @property
    def qualification_identity(self) -> str:
        return identity(self.model_copy(update={"qualification_artifact": None}))


class Settings(FrozenModel):
    role: Literal["validator", "worker", "miner", "api", "evaluation"]
    targets: tuple[TargetConfig, ...] = ()
    state_path: Path = Path("runtime/validator.sqlite3")
    artifact_root: str = "runtime/artifacts"
    upload_staging_root: str = ""
    upload_api_url: str = "https://prexon.ai/api/v1"
    uploads_enabled: bool = False
    upload_state_path: Path = Path("runtime/api/uploads.sqlite3")
    upload_policy_path: Path = Path("runtime/intake-policy/upload-policy.json")
    upload_policy_max_age_seconds: int = Field(default=300, ge=30, le=3600)
    upload_url_ttl_seconds: int = Field(default=900, ge=60, le=3600)
    upload_request_max_age_seconds: int = Field(default=60, ge=15, le=300)
    upload_max_requests_per_hour: int = Field(default=4, ge=1, le=100)
    upload_max_bytes: int = Field(default=4 * 1024**3 + 2 * 1024**2, ge=1024)
    public_artifact_root: str = "runtime/public/artifacts"
    public_artifact_base_url: str = ""
    dataset_cache: Path = Path("runtime/datasets")
    dashboard_path: Path = Path("runtime/public/dashboard.json")
    poll_seconds: Positive = 30
    schedule_seconds: Positive = 6 * 3600
    early_queue_depth: int = Field(default=10, ge=1)
    session_seconds: int = Field(default=4 * 3600, ge=60)
    session_max_cost: Positive = 100
    publish_seconds: Positive = 600
    provider: Literal["lium", "local"] = "lium"
    chain: Literal["bittensor", "file"] = "bittensor"
    chain_file: Path = Path("runtime/chain.json")
    network: str = "finney"
    netuid: int = Field(default=0, ge=0)
    burn_uid: int = Field(default=0, ge=0)
    wallet_name: str = "default"
    wallet_hotkey: str = "default"
    wallet_path: str = "~/.bittensor/wallets"
    weights_version: int = Field(default=1, ge=0)
    lium_base_url: str = "https://lium.io/api"
    lium_template_id: str = ""
    lium_volume_id: str = ""
    ssh_private_key: Path = Path("~/.ssh/id_ed25519")
    ssh_public_key: Path = Path("~/.ssh/id_ed25519.pub")
    ssh_known_hosts: Path = Path("runtime/known_hosts")
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    env_file: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def isolated_storage(self) -> Settings:
        if self.role not in {"validator", "api"}:
            return self
        if self.public_artifact_base_url:
            base = urlsplit(self.public_artifact_base_url)
            if (
                base.scheme != "https"
                or not base.hostname
                or base.username
                or base.password
                or base.query
                or base.fragment
                or "\\" in self.public_artifact_base_url
            ):
                raise ValueError("Public release base must be stable HTTPS without credentials")
        roots = [self.public_artifact_root]
        if self.role == "validator":
            roots.append(self.artifact_root)
        if self.upload_staging_root:
            roots.append(self.upload_staging_root)
        addresses: list[tuple[str | None, Path | None]] = []
        for root in roots:
            parsed = urlsplit(root)
            if parsed.scheme:
                if (
                    parsed.scheme != "s3"
                    or not parsed.netloc
                    or parsed.username
                    or parsed.password
                    or parsed.query
                    or parsed.fragment
                    or ".." in parsed.path
                    or "\\" in root
                    or "%" in root
                ):
                    raise ValueError("Invalid storage root; use a directory or s3://bucket/prefix")
                addresses.append((parsed.netloc.lower(), None))
            else:
                addresses.append((None, Path(root).resolve()))
        for (a_bucket, a_path), (b_bucket, b_path) in combinations(addresses, 2):
            shared_bucket = a_bucket is not None and a_bucket == b_bucket
            overlapping_paths = (
                a_path is not None
                and b_path is not None
                and (a_path.is_relative_to(b_path) or b_path.is_relative_to(a_path))
            )
            if shared_bucket or overlapping_paths:
                raise ValueError(
                    "Upload, canonical and public storage must use separate namespaces"
                )
        return self


def read_environment(env_file: str | Path | None = ".env") -> dict[str, str]:
    values = (
        {key: value for key, value in dotenv_values(env_file).items() if value is not None}
        if env_file and Path(env_file).exists()
        else {}
    )
    return {**values, **os.environ}


def load_settings(
    role: Literal["validator", "worker", "miner", "api", "evaluation"],
    env_file: str | Path | None = ".env",
    *,
    load_targets: bool = True,
    overrides: dict[str, str | int] | None = None,
) -> Settings:
    values = read_environment(env_file)
    fields = Settings.model_fields
    if role == "api":
        fields = {
            name: field
            for name, field in fields.items()
            if name
            in {"api_host", "api_port", "dashboard_path", "uploads_enabled", "public_artifact_root"}
            or name.startswith("upload_")
            and name != "upload_api_url"
        }
    elif role == "evaluation":
        fields = {
            name: field
            for name, field in fields.items()
            if name
            in {
                "session_seconds",
                "session_max_cost",
                "lium_base_url",
                "lium_template_id",
                "lium_volume_id",
                "ssh_private_key",
                "ssh_public_key",
                "ssh_known_hosts",
                "dataset_cache",
            }
        }
    selected: dict[str, object] = {
        name: values[f"PREXON_{name.upper()}"]
        for name in fields
        if name not in {"role", "targets", "env_file"} and values.get(f"PREXON_{name.upper()}")
    }
    selected.update(overrides or {})
    targets: list[TargetConfig] = []
    if load_targets and role in {"validator", "miner"}:
        for slot in (1, 2):
            prefix = f"PREXON_MODEL_{slot}_"
            model_id = (values.get(prefix + "ID") or "").strip()
            if not model_id:
                continue
            profile_path = values.get(prefix + "PROFILE")
            if not profile_path:
                raise ValueError(f"{prefix}PROFILE is required for an enabled target")
            data = json.loads(Path(profile_path).read_text())
            data.update(slot=slot, model_id=model_id, revision=values.get(prefix + "REVISION", ""))
            data["allocation"] = float(values.get(prefix + "ALLOCATION") or 100)
            if values.get(prefix + "METHODS"):
                data["methods"] = [s.strip() for s in values[prefix + "METHODS"].split(",")]
            targets.append(TargetConfig.model_validate(data))
        if targets and abs(sum(t.allocation for t in targets) - 100) > 1e-9:
            raise ValueError("Enabled target emission allocations must sum to 100")
        if len({t.model_id for t in targets}) != len(targets):
            raise ValueError("Slots must use distinct target models")
    return Settings.model_validate(
        {"role": role, "targets": tuple(targets), "env_file": env_file, **selected}
    )
