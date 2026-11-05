"""CPU-only dataset preparation; no rental, inference or chain operations."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.datasets.pool import prepare_pool
from evaluation.datasets.sources import PreparationConfig


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("runtime/datasets"))
    args = parser.parse_args(arguments)
    try:
        from transformers import AutoTokenizer

        config = PreparationConfig.model_validate_json(args.config.read_bytes())
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
        )
        manifest = prepare_pool(config, args.cache_dir, tokenizer)
    except ImportError:
        parser.error("Dataset preparation requires: uv sync --extra datasets")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Dataset manifest: {manifest.identity}")
    print(f"Prepared prompts: {sum(s.count for s in manifest.shards)}")
    print(f"Skipped rows: {manifest.skipped_rows}; duplicates: {manifest.duplicate_rows}")
    print(f"Cache: {(args.cache_dir / manifest.identity).resolve()}")
