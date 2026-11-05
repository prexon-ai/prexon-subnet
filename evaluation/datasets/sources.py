"""Normalize public source questions without their stored completions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import Field, model_validator

from shared.config import (
    DEFAULT_BUCKETS,
    FrozenModel,
    LengthBucket,
    Name,
    Revision,
    validate_buckets,
)


class DatasetSource(FrozenModel):
    kind: Literal["wildchat", "code", "math", "document"]
    category: Name
    repo_id: str = Field(min_length=1)
    revision: Revision
    subset: str | None = None
    splits: tuple[str, ...] = ("train",)


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
    field = {"code": "input", "math": "problem", "document": "article"}[kind]
    value = row.get(field)
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
        )
