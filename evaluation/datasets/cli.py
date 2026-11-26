"""CPU-only dataset preparation; no rental, inference or chain operations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from evaluation.config import DATASET_CACHE
from evaluation.datasets.pool import prepare_pool
from evaluation.datasets.sources import recipe
from shared.targets import find_competition


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--competition", required=True, metavar="MODEL_ID", help="A model in competitions/"
    )
    parser.add_argument("--cache-dir", type=Path, default=DATASET_CACHE)
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Tokenizing processes (default: every CPU); the bank is the same for any number",
    )
    args = parser.parse_args(arguments)
    try:
        from transformers import AutoTokenizer

        config = recipe(find_competition(args.competition))
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            trust_remote_code=False,
        )
        manifest = prepare_pool(config, args.cache_dir, tokenizer, workers=args.workers)
    except ImportError:
        parser.error("Dataset preparation requires: uv sync --extra datasets")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Dataset manifest: {manifest.identity}")
    print(f"Prepared prompts: {sum(s.count for s in manifest.shards)}")
    print(f"Skipped rows: {manifest.skipped_rows}; duplicates: {manifest.duplicate_rows}")
    print(f"Cache: {(args.cache_dir / manifest.identity).resolve()}")
