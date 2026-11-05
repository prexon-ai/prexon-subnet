"""Validator job preparation around the shared inline measurement lifecycle."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from evaluation.datasets.sampling import validate_plan
from evaluation.measurement import measure_configurations
from evaluation.sglang import Arm
from shared.artifacts import download_url, extract_package
from shared.checkpoints import sha256_file, validate_checkpoint
from shared.contracts import EvaluationJob, EvaluationResult, Qualification
from shared.datasets import RoundPlan


async def run_job(
    job: EvaluationJob,
    directory: Path,
    *,
    allow_file: bool = False,
) -> EvaluationResult:
    started, monotonic_started = time.time(), time.perf_counter()
    base: dict[str, Any] = dict(
        job_id=job.job_id,
        session_id=job.session_id,
        competition_id=job.target.identity,
        submission_id=job.commitment.identity,
        incumbent_artifact=job.incumbent_artifact,
        started_at=started,
        round_context=job.round_context,
    )
    qualification_passed = False
    try:
        if not job.qualification_url or not job.target.qualification_artifact:
            return EvaluationResult(
                **base,
                status="inconclusive",
                finished_at=time.time(),
                error_code="qualification_missing",
            )
        directory.mkdir(parents=True, exist_ok=True)
        qualification_path = directory / "qualification.json"
        download_url(job.qualification_url, qualification_path, 1024**2, allow_file=allow_file)
        if sha256_file(qualification_path) != job.target.qualification_artifact:
            raise ValueError("Qualification digest mismatch")
        qualification = Qualification.model_validate_json(qualification_path.read_bytes())
        if (
            qualification.profile_id != job.target.qualification_identity
            or qualification.engine_version != job.target.engine_version
        ):
            raise ValueError("Qualification does not cover this profile")
        qualification_passed = True
        plan = None
        sample_seed = job.job_id
        if job.target.datasets is not None:
            if not job.round_context or not job.dataset_plan_digest or not job.dataset_plan_url:
                raise ValueError("Dataset job needs an immutable round plan")
            sample_seed = job.round_context.seed(job.target.identity)
            plan_path = directory / "dataset-plan.json"
            download_url(job.dataset_plan_url, plan_path, 64 * 1024**2, allow_file=allow_file)
            if sha256_file(plan_path) != job.dataset_plan_digest:
                raise ValueError("Dataset plan digest mismatch")
            plan = RoundPlan.model_validate_json(plan_path.read_bytes())
            if plan.identity != job.dataset_plan_digest:
                raise ValueError("Dataset plan identity mismatch")
            validate_plan(job.target, plan, sample_seed)
            if plan.evidence() != job.expected_dataset_selection:
                raise ValueError("Dataset plan selection does not match job")
        paths: dict[Arm, Path | None] = {"target": None}
        artifacts: list[tuple[Arm, str | None, str | None]] = [
            ("candidate", job.candidate_url, job.commitment.artifact),
            (
                "public",
                job.public_url,
                job.target.public_reference.artifact if job.target.public_reference else None,
            ),
            ("incumbent", job.incumbent_url, job.incumbent_artifact),
        ]
        staging_started = time.perf_counter()
        for arm, url, digest in artifacts:
            if digest is None:
                continue
            if url is None:
                raise ValueError("Reference artifact URL missing")
            archive = directory / f"{arm}.tar.gz"
            download_url(
                url,
                archive,
                job.target.policy.max_checkpoint_bytes + 2 * 1024**2,
                allow_file=allow_file,
            )
            path = directory / arm
            manifest = extract_package(
                archive, path, digest, job.target.policy.max_checkpoint_bytes
            )
            validate_checkpoint(
                path,
                job.target.policy.max_checkpoint_bytes,
                schemas=qualification.checkpoint_schemas,
            )
            if (
                manifest.target_id != job.target.model_id
                or manifest.target_revision != job.target.revision
                or manifest.method not in job.target.methods
                or (arm == "candidate" and manifest.competition_id != job.target.identity)
            ):
                raise ValueError("Artifact target or method mismatch")
            paths[arm] = path
        staging_seconds = time.perf_counter() - staging_started
        evidence = await measure_configurations(
            job.target, paths, directory, job.deadline, sample_seed, prepared_round=plan
        )
        data = evidence.model_dump()
        data.update(base, qualification_passed=True)
        if evidence.timings is not None:
            data["timings"] = evidence.timings.model_copy(
                update={
                    "artifact_staging_seconds": staging_seconds,
                    "total_seconds": time.perf_counter() - monotonic_started,
                }
            )
        return EvaluationResult.model_validate(data)
    except (Exception, asyncio.CancelledError) as exc:
        return EvaluationResult(
            **base,
            status="inconclusive",
            finished_at=time.time(),
            qualification_passed=qualification_passed,
            error_code=type(exc).__name__,
        )
