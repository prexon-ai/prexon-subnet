"""Versioned sampling without replacement and immutable request plans."""

from __future__ import annotations

import hashlib
from collections import Counter
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from evaluation.datasets.pool import INDEX, PoolShard, load_manifest
from shared.checkpoints import sha256_file
from shared.config import TargetConfig
from shared.datasets import DatasetEvidence, Prompt, RoundPlan, SelectedWorkload


def seed_digest(target: TargetConfig, seed: str) -> str:
    return hashlib.sha256(
        ("prexon-local-datasets-v1:" + target.identity + ":" + seed).encode()
    ).hexdigest()


def quotas(count: int, weights: dict[str, float]) -> dict[str, int]:
    exact = {k: count * Fraction(str(v)) for k, v in weights.items()}
    allocated = {k: int(value) for k, value in exact.items()}
    remaining = count - sum(allocated.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - allocated[k]), k))
    for key in order[:remaining]:
        allocated[key] += 1
    return allocated


def indices(size: int, count: int, seed: str) -> list[int]:
    if count > size:
        raise ValueError("Insufficient distinct prompts in dataset stratum")
    # SHA-256 counter stream + partial Fisher-Yates: stable across Python RNG versions.
    swaps: dict[int, int] = {}
    result = []
    for i in range(count):
        remaining = size - i
        counter = 0
        limit = (1 << 256) - ((1 << 256) % remaining)
        while True:
            number = int.from_bytes(hashlib.sha256(f"{seed}:{i}:{counter}".encode()).digest())
            if number < limit:
                break
            counter += 1
        selected = number % remaining
        result.append(swaps.get(selected, selected))
        swaps[selected] = swaps.get(remaining - 1, remaining - 1)
    return result


@lru_cache(maxsize=128)
def _verified_index(path: str, size: int, mtime: int, digest: str) -> None:
    if sha256_file(Path(path)) != digest:
        raise ValueError("Dataset index digest mismatch")


def read_prompts(root: Path, shard: PoolShard, selected: list[int]) -> list[Prompt]:
    index_path, data_path = root / shard.index_file, root / shard.data_file
    stat = index_path.stat()
    if stat.st_size != shard.count * INDEX.size or data_path.stat().st_size != shard.data_bytes:
        raise ValueError("Dataset file size/digest mismatch")
    _verified_index(str(index_path.resolve()), stat.st_size, stat.st_mtime_ns, shard.index_sha256)
    prompts = []
    with index_path.open("rb") as index, data_path.open("rb") as data:
        for number in selected:
            index.seek(number * INDEX.size)
            offset, length, digest = INDEX.unpack(index.read(INDEX.size))
            data.seek(offset)
            encoded = data.read(length)
            if hashlib.sha256(encoded).digest() != digest:
                raise ValueError("Dataset record digest mismatch")
            prompt = Prompt.model_validate_json(encoded)
            if prompt.category != shard.category or prompt.bucket != shard.bucket:
                raise ValueError("Dataset stratum mismatch")
            prompts.append(prompt)
    return prompts


def select_round(target: TargetConfig, cache: Path, seed: str) -> RoundPlan:
    policy = target.datasets
    if policy is None:
        raise ValueError("Target has no dataset configuration")
    manifest = load_manifest(cache, policy.manifest)
    if (manifest.config.model_id, manifest.config.revision) != (target.model_id, target.revision):
        raise ValueError("Dataset tokenizer does not match target revision")
    expected_buckets = {(b.name, b.min_tokens, b.max_tokens) for b in policy.buckets}
    if expected_buckets != {(b.name, b.min_tokens, b.max_tokens) for b in manifest.config.buckets}:
        raise ValueError("Dataset length buckets do not match preparation")
    shards = {(s.category, s.bucket): s for s in manifest.shards}
    resolved_seed = seed_digest(target, seed)
    prompts: dict[str, Prompt] = {}
    workloads = []
    # Same request count shares the same sample at every concurrency/output budget.
    for workload in target.workloads:
        selected = []
        for category, category_count in sorted(
            quotas(workload.requests, policy.category_weights).items()
        ):
            for bucket, count in sorted(
                quotas(category_count, {b.name: b.weight for b in policy.buckets}).items()
            ):
                shard = shards.get((category, bucket))
                if shard is None:
                    raise ValueError(f"Missing dataset stratum {category}/{bucket}")
                domain = f"{resolved_seed}:{workload.requests}:{category}:{bucket}"
                for prompt in read_prompts(
                    cache / policy.manifest, shard, indices(shard.count, count, domain)
                ):
                    prompts[prompt.identity] = prompt
                    selected.append(prompt.identity)
        order = indices(len(selected), len(selected), f"{resolved_seed}:order:{workload.requests}")
        workloads.append(
            SelectedWorkload(name=workload.name, prompt_ids=tuple(selected[i] for i in order))
        )
    plan = RoundPlan(
        manifest=policy.manifest,
        model_id=target.model_id,
        revision=target.revision,
        seed=resolved_seed,
        prompts=tuple(prompts[k] for k in sorted(prompts)),
        workloads=tuple(workloads),
    )
    validate_plan(target, plan, seed)
    return plan


def validate_plan(target: TargetConfig, plan: RoundPlan, seed: str) -> None:
    policy = target.datasets
    if policy is None or (plan.manifest, plan.model_id, plan.revision, plan.seed) != (
        policy.manifest,
        target.model_id,
        target.revision,
        seed_digest(target, seed),
    ):
        raise ValueError("Dataset plan identity/seed mismatch")
    lookup = {p.identity: p for p in plan.prompts}
    if len(lookup) != len(plan.prompts) or tuple(w.name for w in plan.workloads) != tuple(
        w.name for w in target.workloads
    ):
        raise ValueError("Duplicate prompts or unexpected dataset workloads")
    used: set[str] = set()
    for workload, selected in zip(target.workloads, plan.workloads, strict=True):
        if (
            len(selected.prompt_ids) != workload.requests
            or len(set(selected.prompt_ids)) != workload.requests
        ):
            raise ValueError("Dataset requests require distinct prompts without replacement")
        if set(selected.prompt_ids) - lookup.keys():
            raise ValueError("Missing dataset prompt")
        records = [lookup[p] for p in selected.prompt_ids]
        categories = quotas(workload.requests, policy.category_weights)
        if Counter(p.category for p in records) != {k: v for k, v in categories.items() if v}:
            raise ValueError("Dataset category quota mismatch")
        for category, count in categories.items():
            expected = quotas(count, {b.name: b.weight for b in policy.buckets})
            if Counter(p.bucket for p in records if p.category == category) != {
                k: v for k, v in expected.items() if v
            }:
                raise ValueError("Dataset length quota mismatch")
        for record in records:
            bucket = next((b for b in policy.buckets if b.name == record.bucket), None)
            if (
                bucket is None
                or not bucket.min_tokens <= len(record.input_ids) <= bucket.max_tokens
            ):
                raise ValueError("Dataset prompt length mismatch")
            if len(record.input_ids) + workload.output_tokens > target.context_length:
                raise ValueError("Dataset prompt exceeds input/output context budget")
        used.update(selected.prompt_ids)
    if used != lookup.keys():
        raise ValueError("Unexpected unused dataset prompts")


def materialize(target: TargetConfig, plan: RoundPlan, seed: str) -> TargetConfig:
    validate_plan(target, plan, seed)
    lookup = {p.identity: p.input_ids for p in plan.prompts}
    return target.model_copy(
        update={
            "workloads": tuple(
                workload.model_copy(
                    update={"input_ids": tuple(lookup[p] for p in selected.prompt_ids)}
                )
                for workload, selected in zip(target.workloads, plan.workloads, strict=True)
            )
        }
    )


def validate_selection(
    target: TargetConfig,
    selection: DatasetEvidence | None,
    *,
    seed: str | None = None,
) -> None:
    policy = target.datasets
    if policy is None:
        if selection is not None:
            raise ValueError("Unexpected dataset evidence")
        return
    if selection is None or selection.manifest != policy.manifest:
        raise ValueError("Missing or mismatched dataset evidence")
    if seed is not None and selection.seed != seed_digest(target, seed):
        raise ValueError("Dataset evidence seed mismatch")
    if tuple(w.name for w in selection.workloads) != tuple(w.name for w in target.workloads):
        raise ValueError("Dataset evidence workloads mismatch")
    if set(selection.category_counts) != {w.name for w in target.workloads} or set(
        selection.bucket_counts
    ) != {w.name for w in target.workloads}:
        raise ValueError("Missing dataset stratum counts")
    for workload, selected in zip(target.workloads, selection.workloads, strict=True):
        if (
            len(selected.prompt_ids) != workload.requests
            or len(set(selected.prompt_ids)) != workload.requests
        ):
            raise ValueError("Incomplete or duplicated dataset evidence")
        categories = quotas(workload.requests, policy.category_weights)
        expected_categories = {k: v for k, v in categories.items() if v}
        lengths: Counter[str] = Counter()
        for count in categories.values():
            lengths.update(quotas(count, {b.name: b.weight for b in policy.buckets}))
        if selection.category_counts[
            workload.name
        ] != expected_categories or selection.bucket_counts[workload.name] != {
            k: v for k, v in lengths.items() if v
        }:
            raise ValueError("Dataset evidence stratum counts mismatch")
