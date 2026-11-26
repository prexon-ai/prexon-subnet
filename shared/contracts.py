"""Versioned, explicit boundaries between miners, controller, worker, and API."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field

from shared.datasets import DatasetEvidence, RoundContext
from shared.releases import WinnerRelease
from shared.targets import CompetitionId, Digest, FrozenModel, Positive, Revision, TargetConfig


class FileDigest(FrozenModel):
    path: str = Field(min_length=1)
    sha256: Digest
    size: int = Field(ge=1)


class TensorSpec(FrozenModel):
    dtype: str = Field(min_length=1)
    shape: tuple[int, ...]


class CheckpointSchema(FrozenModel):
    config_sha256: Digest
    tensors: dict[str, TensorSpec] = Field(min_length=1)


class Provenance(FrozenModel):
    training_description: str = Field(min_length=1, max_length=10000)
    data_sources: tuple[str, ...] = Field(min_length=1)
    license: str = Field(min_length=1)
    parent_artifacts: tuple[Digest, ...] = ()
    recipe_sha256: Digest
    deployment_notes: str = Field(min_length=1, max_length=10000)


class SubmissionManifest(FrozenModel):
    schema_version: Literal[1] = 1
    target_id: str = Field(min_length=1)
    target_revision: Revision
    competition_id: CompetitionId
    method: Literal["dflash2"] = "dflash2"
    files: tuple[FileDigest, ...] = Field(min_length=2)
    provenance: Provenance


class Commitment(FrozenModel):
    schema_version: Literal[1] = 1
    netuid: int = Field(ge=0)
    block: int = Field(ge=0)
    block_hash: str = Field(min_length=1)
    extrinsic_index: int = Field(default=0, ge=0)
    hotkey: str = Field(min_length=1)
    uid: int = Field(ge=0)
    artifact: Digest
    submitted_at: float | None = Field(default=None, ge=0)

    @property
    def identity(self) -> str:
        from shared.targets import identity

        return identity(self, exclude={"submitted_at"})


class EvaluationJob(FrozenModel):
    schema_version: Literal[1] = 1
    job_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    session_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    target: TargetConfig
    commitment: Commitment
    incumbent_artifact: Digest | None = None
    candidate_url: str = Field(min_length=1)
    public_url: str | None = None
    incumbent_url: str | None = None
    qualification_url: str | None = None
    deadline: Positive
    round_context: RoundContext | None = None
    dataset_plan_digest: Digest | None = None
    dataset_plan_url: str | None = None
    expected_dataset_selection: DatasetEvidence | None = None


class Measurement(FrozenModel):
    workload: str
    repeat: int = Field(ge=0)
    arm: Literal["target", "candidate", "public", "incumbent"]
    output_tokens: int = Field(ge=1)
    expected_tokens: int = Field(ge=1)
    elapsed_seconds: Positive
    p95_latency_seconds: Positive
    accepted_tokens: int | None = Field(default=None, ge=0)
    drafted_tokens: int | None = Field(default=None, ge=0)
    verify_steps: int | None = Field(default=None, ge=1)

    @property
    def throughput(self) -> float:
        return self.output_tokens / self.elapsed_seconds


class HardwareEvidence(FrozenModel):
    gpu_names: tuple[str, ...]
    gpu_memory_gb: tuple[Positive, ...]
    engine_version: str
    cuda_version: str
    topology: str = ""
    tensor_parallel_size: int = Field(ge=1)


class ExecutionTimings(FrozenModel):
    artifact_staging_seconds: float = Field(default=0, ge=0)
    target_staging_seconds: float = Field(default=0, ge=0)
    engine_startup_seconds: float = Field(default=0, ge=0)
    warmup_seconds: float = Field(default=0, ge=0)
    reference_anchor_seconds: float = Field(default=0, ge=0)
    measured_seconds: float = Field(default=0, ge=0)
    total_seconds: float = Field(default=0, ge=0)


class BenchmarkEvidence(FrozenModel):
    schema_version: Literal[1] = 1
    status: Literal["complete", "inconclusive"]
    measurements: tuple[Measurement, ...] = ()
    correctness_passed: bool = False
    reference_correctness_passed: bool = False
    qualification_passed: bool = False
    hardware: HardwareEvidence | None = None
    timings: ExecutionTimings | None = None
    reference_before: tuple[Positive, ...] = ()
    reference_after: tuple[Positive, ...] = ()
    started_at: Positive
    finished_at: Positive
    error_code: str | None = None
    dataset_selection: DatasetEvidence | None = None


class EvaluationResult(BenchmarkEvidence):
    job_id: str
    session_id: str
    # The measured profile's internal hash (TargetConfig.identity).
    profile_id: Digest
    submission_id: Digest
    incumbent_artifact: Digest | None = None
    round_context: RoundContext | None = None


class Decision(FrozenModel):
    outcome: Literal["won", "lost", "inconclusive"]
    reason: str
    score: Positive | None = None
    threshold: Positive | None = None
    lower_gain: Positive | None = None


class Award(FrozenModel):
    submission_id: Digest
    # The profile the award was won under (TargetConfig.identity).
    profile_id: Digest
    hotkey: str
    uid: int
    promoted_at: Positive
    artifact: Digest


class Qualification(FrozenModel):
    schema_version: Literal[1] = 1
    profile_id: Digest
    engine_version: str
    method: Literal["dflash2"] = "dflash2"
    sampling: Literal["greedy"] = "greedy"
    approved_by: str = Field(min_length=1)
    evidence_description: str = Field(min_length=1)
    target_preserving_verification_reviewed: Literal[True]
    checkpoint_schemas: tuple[CheckpointSchema, ...] = Field(min_length=1)


class PublicTarget(FrozenModel):
    competition_id: CompetitionId
    model_id: str
    revision: str
    methods: tuple[str, ...]
    allocation: float
    gpu_type: str
    gpu_count: int
    ready_submissions: int = 0
    session_state: str | None = None


class PublicSubmission(FrozenModel):
    submission_id: Digest
    competition_id: CompetitionId
    artifact: Digest
    hotkey: str
    uid: int
    block: int
    status: str
    score: float | None = None
    reason: str | None = None


class PublicEvaluation(FrozenModel):
    job_id: str
    submission_id: Digest
    competition_id: CompetitionId
    model_id: str
    revision: str
    outcome: str
    reason: str
    score: float | None = None
    threshold: float | None = None
    lower_gain: float | None = None
    # Aggregated metrics only: no prompts, URLs, credentials, or full job configuration.
    measurements: tuple[Measurement, ...] = ()
    timings: ExecutionTimings | None = None


class PublicWinner(FrozenModel):
    submission_id: Digest
    competition_id: CompetitionId
    artifact: Digest
    hotkey: str
    uid: int
    promoted_at: float
    # Share of its target's emissions this winner currently receives.
    share: Annotated[float, Field(gt=0, le=1)]
    checkpoint_url: str | None = None
    published_at: float | None = None


class QueuedSubmission(FrozenModel):
    """A head waiting for evaluation. Its checkpoint stays private."""

    submission_id: Digest
    hotkey: str
    uid: int
    submitted_at: float
    status: Literal["evaluating", "waiting"]


class EvaluatedSubmission(FrozenModel):
    """The public outcome of one evaluation; only a crowned head has a checkpoint."""

    submission_id: Digest
    hotkey: str
    uid: int
    submitted_at: float
    evaluated_at: float
    crowned: bool
    # Throughput relative to target-only decoding; absent when no score was measured.
    speedup: float | None = None
    # The speedup it had to beat, measured in the same evaluation: the faster of the
    # incumbent and the public reference, or 1 when neither beat target-only decoding.
    # Absent, like the speedup, when no score was measured.
    previous_best: float | None = None
    # Share of the model's emissions it receives now.
    share: Annotated[float, Field(ge=0, le=1)] = 0
    checkpoint_url: str | None = None


class ModelSummary(FrozenModel):
    # Stable and URL-safe.
    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=96)
    model_id: str
    revision: str
    active_from: float
    # None while the model is still active.
    active_to: float | None = None
    allocation: float
    methods: tuple[str, ...]
    gpu_type: str
    gpu_count: int
    queued: int = Field(ge=0)
    evaluated: int = Field(ge=0)
    winners: int = Field(ge=0)
    best_speedup: float | None = None
    # Current split of the model's emissions, largest first.
    reward_shares: tuple[float, ...] = ()


class ModelHistory(ModelSummary):
    schema_version: Literal[1] = 1
    # In the order the heads will run.
    queue: tuple[QueuedSubmission, ...] = ()
    # Newest result first.
    submissions: tuple[EvaluatedSubmission, ...] = ()


class ModelIndex(FrozenModel):
    schema_version: Literal[1] = 1
    updated_at: Positive
    finalized_block: int = Field(ge=0)
    # Active models first, then past ones, most recently active first.
    models: tuple[ModelSummary, ...] = ()


class Emission(FrozenModel):
    uid: int
    share: Annotated[float, Field(ge=0, le=1)]


class DashboardSnapshot(FrozenModel):
    schema_version: Literal[1] = 1
    updated_at: Positive
    finalized_block: int = Field(ge=0)
    targets: tuple[PublicTarget, ...] = ()
    submissions: tuple[PublicSubmission, ...] = ()
    evaluations: tuple[PublicEvaluation, ...] = ()
    winners: tuple[PublicWinner, ...] = ()
    emissions: tuple[Emission, ...] = ()
    releases: tuple[WinnerRelease, ...] = ()
