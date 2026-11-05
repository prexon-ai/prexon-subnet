"""Inspect data-only DFlash2 exports before passing them to the serving engine."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from pathlib import Path
from typing import Any

from shared.contracts import CheckpointSchema, FileDigest, TensorSpec

_NAME = re.compile(
    r"^(config\.json|model(?:-[0-9]+-of-[0-9]+)?\.safetensors|"
    r"model\.safetensors\.index\.json)$"
)
_DTYPE_BYTES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(data: str | bytes) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(data, object_pairs_hook=unique)


def _finite_values(path: Path, start: int, length: int, dtype: str) -> None:
    if dtype not in {"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2"}:
        return
    import numpy as np

    with path.open("rb") as stream:
        stream.seek(start)
        while length:
            chunk = stream.read(min(length, 1024**2))
            length -= len(chunk)
            if not chunk:
                raise ValueError("Truncated tensor data")
            if dtype == "BF16":
                valid = not np.any((np.frombuffer(chunk, dtype="<u2") & 0x7F80) == 0x7F80)
            elif dtype.startswith("F8"):
                values = np.frombuffer(chunk, dtype="u1")
                mask = 0x7F if dtype == "F8_E4M3" else 0x7C
                valid = not np.any((values & mask) == mask)
            else:
                valid = bool(
                    np.all(
                        np.isfinite(
                            np.frombuffer(
                                chunk, dtype={"F64": "<f8", "F32": "<f4", "F16": "<f2"}[dtype]
                            )
                        )
                    )
                )
            if not valid:
                raise ValueError("Checkpoint contains nonfinite weights")


def _tensor_keys(path: Path) -> dict[str, TensorSpec]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated tensor header")
        size = struct.unpack("<Q", prefix)[0]
        if size < 2 or size > 16 * 1024**2 or size + 8 > path.stat().st_size:
            raise ValueError("Invalid tensor header size")
        header = strict_json(stream.read(size))
    if not isinstance(header, dict):
        raise ValueError("Invalid tensor header")
    ranges: list[tuple[int, int]] = []
    finite_checks: list[tuple[int, int, str]] = []
    names: dict[str, TensorSpec] = {}
    for name, tensor in header.items():
        if name == "__metadata__":
            if not isinstance(tensor, dict):
                raise ValueError("Invalid tensor metadata")
            continue
        if not isinstance(tensor, dict) or tensor.get("dtype") not in _DTYPE_BYTES:
            raise ValueError("Unsupported tensor dtype")
        shape = tensor.get("shape")
        offsets = tensor.get("data_offsets")
        if (
            not isinstance(shape, list)
            or len(shape) > 8
            or not all(type(d) is int and 0 <= d <= 2**31 for d in shape)
        ):
            raise ValueError("Invalid tensor shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(d) is int for d in offsets)
        ):
            raise ValueError("Invalid tensor offsets")
        start, end = offsets
        expected = math.prod(shape) * _DTYPE_BYTES[tensor["dtype"]]
        if start < 0 or end < start or end - start != expected:
            raise ValueError("Invalid tensor storage length")
        if end > path.stat().st_size - 8 - size:
            raise ValueError("Invalid tensor data range")
        finite_checks.append((8 + size + start, end - start, tensor["dtype"]))
        ranges.append((start, end))
        names[name] = TensorSpec(dtype=tensor["dtype"], shape=tuple(shape))
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor:
            raise ValueError("Noncontiguous or overlapping tensor data")
        cursor = end
    if not names or cursor != path.stat().st_size - 8 - size:
        raise ValueError("Truncated or trailing tensor data")
    for start, length, dtype in finite_checks:
        _finite_values(path, start, length, dtype)
    return names


def validate_checkpoint(
    directory: Path, max_bytes: int, *, schemas: tuple[CheckpointSchema, ...] = ()
) -> tuple[FileDigest, ...]:
    files = sorted(directory.iterdir())
    if not files or len(files) > 128:
        raise ValueError("Invalid checkpoint file count")
    if any(p.is_symlink() or not p.is_file() or not _NAME.fullmatch(p.name) for p in files):
        raise ValueError("Only data-only config and safetensors checkpoint files are permitted")
    if sum(p.stat().st_size for p in files) > max_bytes:
        raise ValueError("Checkpoint exceeds size limit")
    config_path = directory / "config.json"
    if not config_path.exists() or config_path.stat().st_size > 1024**2:
        raise ValueError("Missing or oversized config.json")
    config = strict_json(config_path.read_bytes())
    if not isinstance(config, dict):
        raise ValueError("Expected a checkpoint configuration object")
    if "auto_map" in config or config.get("trust_remote_code"):
        raise ValueError("Checkpoint remote code is prohibited")
    if config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("Expected a DFlash2DraftModel export")
    dflash = config.get("dflash_config", {})
    try:
        hidden, vocab = config["hidden_size"], config["vocab_size"]
        group, block = dflash["conv_group_size"], dflash["block_size"]
        valid = (
            all(type(x) is int and x > 0 for x in (hidden, vocab, group, block))
            and hidden % group == 0
            and 1 <= dflash["conv_kernel_size"] <= block
            and dflash["selector_rank"] > 0
            and 1 <= dflash["selector_top_k"] <= vocab
            and 0 <= dflash["mask_token_id"] < vocab
            and dflash["target_layer_ids"]
            and all(type(x) is int and x >= 0 for x in dflash["target_layer_ids"])
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid DFlash2 architecture configuration") from exc
    if not valid:
        raise ValueError("Invalid DFlash2 architecture configuration")
    tensors = [p for p in files if p.suffix == ".safetensors"]
    if not tensors:
        raise ValueError("Missing safetensors weights")
    locations: dict[str, str] = {}
    tensor_specs: dict[str, TensorSpec] = {}
    for path in tensors:
        for key, spec in _tensor_keys(path).items():
            if key in locations:
                raise ValueError("Duplicate tensor across shards")
            locations[key] = path.name
            tensor_specs[key] = spec
    index = directory / "model.safetensors.index.json"
    if len(tensors) > 1 or index.exists():
        if not index.exists() or index.stat().st_size > 16 * 1024**2:
            raise ValueError("Missing or oversized tensor index")
        index_data = strict_json(index.read_bytes())
        if not isinstance(index_data, dict) or index_data.get("weight_map") != locations:
            raise ValueError("Tensor index does not match checkpoint shards")
    if schemas:
        actual_schema = CheckpointSchema(
            config_sha256=sha256_file(config_path), tensors=tensor_specs
        )
        if actual_schema not in schemas:
            raise ValueError("Checkpoint architecture/tensors do not match a qualified schema")
    return tuple(
        FileDigest(path=p.name, size=p.stat().st_size, sha256=sha256_file(p)) for p in files
    )


def checkpoint_schema(directory: Path, max_bytes: int = 4 * 1024**3) -> CheckpointSchema:
    validate_checkpoint(directory, max_bytes)
    tensors = {
        name: spec
        for path in sorted(directory.glob("*.safetensors"))
        for name, spec in _tensor_keys(path).items()
    }
    return CheckpointSchema(config_sha256=sha256_file(directory / "config.json"), tensors=tensors)
