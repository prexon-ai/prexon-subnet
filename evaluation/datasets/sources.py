"""Normalize public source questions without their stored completions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import Field, model_validator

from shared.targets import (
    DEFAULT_BUCKETS,
    DatasetSource,
    FrozenModel,
    LengthBucket,
    Revision,
    TargetConfig,
    validate_buckets,
)


class PreparationConfig(FrozenModel):
    version: Literal[1] = 1
    normalization: Literal["prexon-prompts-v1"] = "prexon-prompts-v1"
    model_id: str = Field(min_length=1)
    revision: Revision
    sources: tuple[DatasetSource, ...] = Field(min_length=1)
    buckets: tuple[LengthBucket, ...] = DEFAULT_BUCKETS
    chat_template_kwargs: dict[str, str | bool | int] = {}

    @model_validator(mode="after")
    def valid_preparation(self) -> PreparationConfig:
        validate_buckets(self.buckets)
        if {"tokenize", "add_generation_prompt", "return_dict", "return_tensors"} & (
            self.chat_template_kwargs.keys()
        ):
            raise ValueError("Chat template options cannot override tokenization output")
        for source in self.sources:
            if not source.splits or len(set(source.splits)) != len(source.splits):
                raise ValueError("Specify unique dataset splits")
        return self


def recipe(competition: TargetConfig) -> PreparationConfig:
    """The prompt-bank recipe of a competition: its model and its datasets settings."""
    datasets = competition.datasets
    if datasets is None or not datasets.sources:
        raise ValueError(f"{competition.model_id} has no dataset sources to prepare")
    return PreparationConfig(
        model_id=competition.model_id,
        revision=competition.revision,
        sources=datasets.sources,
        buckets=datasets.buckets,
        chat_template_kwargs=datasets.chat_template_kwargs,
    )


# The only column each kind of source contributes to a prompt.
COLUMNS = {"wildchat": "conversation", "code": "input", "math": "problem", "document": "article"}


def normalize(kind: str, row: dict[str, Any]) -> list[dict[str, str]] | None:
    if kind == "wildchat":
        conversation = row.get("conversation")
        if not isinstance(conversation, list):
            return None
        users = [
            i
            for i, turn in enumerate(conversation)
            if isinstance(turn, dict) and turn.get("role") == "user"
        ]
        if not users:
            return None
        messages = []
        for turn in conversation[: users[-1] + 1]:
            if not isinstance(turn, dict):
                return None
            role, content = turn.get("role"), turn.get("content")
            if role not in {"user", "assistant", "system"} or not isinstance(content, str):
                return None
            if not content.strip():
                return None
            messages.append({"role": role, "content": content.strip()})
        return messages
    value = row.get(COLUMNS[kind])
    if not isinstance(value, str) or not value.strip() or value.strip() == "-":
        return None
    text = value.strip()
    if kind == "document":
        text = "Summarize the following article:\n\n" + text
    return [{"role": "user", "content": text}]


def source_rows(source: DatasetSource) -> Iterable[dict[str, Any]]:
    # These dependencies are CPU preparation tools, never worker/API import requirements.
    from datasets import load_dataset

    for split in source.splits:
        yield from load_dataset(
            source.repo_id,
            name=source.subset,
            split=split,
            revision=source.revision,
            streaming=True,
            # Read only the column a prompt uses, not stored answers or solutions.
            columns=[COLUMNS[source.kind]],
        )
