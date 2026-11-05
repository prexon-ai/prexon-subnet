"""Content-addressed JSONL pools with fixed-width random-access indexes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sqlite3
import struct
import tempfile
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from evaluation.datasets.sources import DatasetSource, PreparationConfig, normalize, source_rows
from shared.checkpoints import sha256_file
from shared.config import Digest, FrozenModel, Name, identity
from shared.datasets import Prompt

INDEX = struct.Struct(">QQ32s")  # Offset, byte count, SHA-256 of the complete record.


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
                    ids = tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=False,
                        **config.chat_template_kwargs,
                    )
                    bucket = next(
                        (b for b in config.buckets if b.min_tokens <= len(ids) <= b.max_tokens),
                        None,
                    )
                    if bucket is None:
                        skipped += 1
                        continue
                    prompt = Prompt(category=source.category, bucket=bucket.name, input_ids=ids)
                    if not unique.execute(
                        "INSERT OR IGNORE INTO seen VALUES(?)", (prompt.identity,)
                    ).rowcount:
                        duplicates += 1
                        continue
                    data, index = files[source.category, bucket.name]
                    encoded = (prompt.model_dump_json() + "\n").encode()
                    index.write(
                        INDEX.pack(data.tell(), len(encoded), hashlib.sha256(encoded).digest())
                    )
                    data.write(encoded)
                    counts[source.category, bucket.name] += 1
        # All streams are closed before hashing or publishing the immutable snapshot.
        for (category, bucket_name), count in counts.items():
            if count == 0:
                raise ValueError(f"Empty prepared pool for {category}/{bucket_name}")
            stem = f"{category}--{bucket_name}"
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
