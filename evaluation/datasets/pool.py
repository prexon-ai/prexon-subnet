"""Content-addressed JSONL pools with fixed-width random-access indexes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import sqlite3
import struct
import tempfile
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from evaluation.datasets.sources import DatasetSource, PreparationConfig, normalize, source_rows
from shared.checkpoints import sha256_file
from shared.datasets import Prompt
from shared.targets import Digest, FrozenModel, LengthBucket, Name, identity

INDEX = struct.Struct(">QQ32s")  # Offset, byte count, SHA-256 of the complete record.
# Conversations per task sent to a tokenizing worker.
BATCH = 256

Messages = list[dict[str, str]]
# One finished candidate: its category, its length bucket (None when it fits none), and
# the prompt's identity and stored JSON line.
Prepared = tuple[str, str | None, str, bytes]
_worker: tuple[Any, dict[str, Any], tuple[LengthBucket, ...]] | None = None


def _start_worker(
    tokenizer: Any, options: dict[str, Any], buckets: tuple[LengthBucket, ...]
) -> None:
    global _worker
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    _worker = (tokenizer, options, buckets)


def _prepare_batch(batch: list[tuple[str, Messages]]) -> list[Prepared]:
    """Template, tokenize, shelve and encode each candidate: the per-prompt work."""
    assert _worker is not None
    tokenizer, options, buckets = _worker
    prepared: list[Prepared] = []
    for category, messages in batch:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False, **options
        )
        bucket = next((b for b in buckets if b.min_tokens <= len(ids) <= b.max_tokens), None)
        if bucket is None:
            prepared.append((category, None, "", b""))
            continue
        prompt = Prompt(category=category, bucket=bucket.name, input_ids=tuple(ids))
        prepared.append(
            (category, bucket.name, prompt.identity, (prompt.model_dump_json() + "\n").encode())
        )
    return prepared


def _prepared(
    items: Iterable[tuple[str, Messages]],
    tokenizer: Any,
    options: dict[str, Any],
    buckets: tuple[LengthBucket, ...],
    workers: int,
) -> Iterator[Prepared]:
    """Each candidate prepared, in input order; workers > 1 prepare batches in parallel.

    Results are taken back strictly in submission order, so the prepared files are the same
    byte for byte whatever the number of workers. Only a bounded window of batches is in
    flight, so a multi-million-row source never sits in memory.
    """

    def batches() -> Iterator[list[tuple[str, Messages]]]:
        batch: list[tuple[str, Messages]] = []
        for item in items:
            batch.append(item)
            if len(batch) == BATCH:
                yield batch
                batch = []
        if batch:
            yield batch

    if workers <= 1:
        _start_worker(tokenizer, options, buckets)
        for batch in batches():
            yield from _prepare_batch(batch)
        return
    # Fresh worker processes, each with its own copy of the tokenizer: forking a process that
    # already runs tokenizer or download threads can deadlock the children.
    with multiprocessing.get_context("spawn").Pool(
        workers, initializer=_start_worker, initargs=(tokenizer, options, buckets)
    ) as pool:
        pending: deque[Any] = deque()
        for batch in batches():
            pending.append(pool.apply_async(_prepare_batch, (batch,)))
            if len(pending) >= workers * 4:
                yield from pending.popleft().get()
        while pending:
            yield from pending.popleft().get()


class PoolShard(FrozenModel):
    category: Name
    bucket: Name
    count: int = Field(ge=1)
    data_file: str = Field(pattern=r"^[a-z0-9_-]+\.jsonl$")
    index_file: str = Field(pattern=r"^[a-z0-9_-]+\.idx$")
    data_sha256: Digest
    index_sha256: Digest
    data_bytes: int = Field(ge=1)


class PoolManifest(FrozenModel):
    version: Literal[1] = 1
    config: PreparationConfig
    versions: dict[str, str]
    shards: tuple[PoolShard, ...]
    skipped_rows: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)

    @property
    def identity(self) -> str:
        return identity(self)


def prepare_pool(
    config: PreparationConfig,
    cache: Path,
    tokenizer: Any,
    rows: Callable[[DatasetSource], Iterable[dict[str, Any]]] = source_rows,
    *,
    versions: dict[str, str] | None = None,
    workers: int = 1,
) -> PoolManifest:
    cache.mkdir(parents=True, exist_ok=True)
    if versions is None:
        versions = {
            name: importlib.metadata.version(name)
            for name in ("datasets", "transformers", "tokenizers")
        }
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=cache) as temporary:
        root = Path(temporary)
        shards: list[PoolShard] = []
        counts: dict[tuple[str, str], int] = {}
        skipped = duplicates = 0
        with ExitStack() as stack:
            unique = sqlite3.connect(root / "unique.sqlite")
            stack.callback(unique.close)
            unique.execute("CREATE TABLE seen (id TEXT PRIMARY KEY)")
            unique.execute("CREATE TABLE questions (id TEXT PRIMARY KEY)")
            files = {}
            for category in sorted({s.category for s in config.sources}):
                for configured in config.buckets:
                    stem = f"{category}--{configured.name}"
                    files[category, configured.name] = (
                        stack.enter_context((root / f"{stem}.jsonl").open("wb")),
                        stack.enter_context((root / f"{stem}.idx").open("wb")),
                    )
                    counts[category, configured.name] = 0

            def candidates() -> Iterator[tuple[str, Messages]]:
                # Normalizing and dropping repeated questions are cheap: they stay in order here.
                nonlocal skipped, duplicates
                for source in config.sources:
                    for row in rows(source):
                        messages = normalize(source.kind, row)
                        if messages is None:
                            skipped += 1
                            continue
                        question_id = hashlib.sha256(
                            json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()
                        ).hexdigest()
                        if not unique.execute(
                            "INSERT OR IGNORE INTO questions VALUES(?)", (question_id,)
                        ).rowcount:
                            duplicates += 1
                            continue
                        yield source.category, messages

            options = dict(config.chat_template_kwargs)
            for category, bucket_name, prompt_id, encoded in _prepared(
                candidates(), tokenizer, options, config.buckets, workers
            ):
                if bucket_name is None:
                    skipped += 1
                    continue
                if not unique.execute(
                    "INSERT OR IGNORE INTO seen VALUES(?)", (prompt_id,)
                ).rowcount:
                    duplicates += 1
                    continue
                data, index = files[category, bucket_name]
                index.write(INDEX.pack(data.tell(), len(encoded), hashlib.sha256(encoded).digest()))
                data.write(encoded)
                counts[category, bucket_name] += 1
        # All streams are closed before hashing or publishing the immutable snapshot.
        for (category, bucket_name), count in counts.items():
            stem = f"{category}--{bucket_name}"
            if count == 0:
                # No prompt of this category reaches this length (coding and maths questions
                # are rarely long): the shelf is left out, and a mix must not ask for it.
                (root / f"{stem}.jsonl").unlink()
                (root / f"{stem}.idx").unlink()
                continue
            data_file, index_file = f"{stem}.jsonl", f"{stem}.idx"
            shards.append(
                PoolShard(
                    category=category,
                    bucket=bucket_name,
                    count=count,
                    data_file=data_file,
                    index_file=index_file,
                    data_sha256=sha256_file(root / data_file),
                    index_sha256=sha256_file(root / index_file),
                    data_bytes=(root / data_file).stat().st_size,
                )
            )
        if not shards:
            raise ValueError("No prompt fits any length bucket")
        manifest = PoolManifest(
            config=config,
            versions=versions,
            shards=tuple(shards),
            skipped_rows=skipped,
            duplicate_rows=duplicates,
        )
        (root / "unique.sqlite").unlink()
        (root / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
        destination = cache / manifest.identity
        if not destination.exists():
            # Move only completed files, retaining the temporary directory for safe cleanup.
            root.rename(destination)
            root.mkdir()
        return manifest


def load_manifest(cache: Path, digest: str) -> PoolManifest:
    manifest = PoolManifest.model_validate_json((cache / digest / "manifest.json").read_bytes())
    if manifest.identity != digest:
        raise ValueError("Dataset manifest digest mismatch")
    keys = [(s.category, s.bucket) for s in manifest.shards]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate dataset shard")
    return manifest
