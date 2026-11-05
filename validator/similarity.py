"""Private, versioned CPU numerical screening for compatible drafter checkpoints.

Full tensor hashes cover every value, independent of safetensors packaging. Near
metrics use exact streamed norms and at most 256 deterministic coordinates per
eligible tensor. Shared target embeddings/output heads contribute only to full
identity. These estimates are numerical evidence, not authorship determinations:
independent heads can be very close, known samples can be evaded, and neither
permutation nor rotation invariance is claimed. Near rejection needs a separately
calibrated, versioned policy; this module makes no admission decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from shared.checkpoints import strict_json
from shared.config import Digest, FrozenModel, TargetConfig

_CHUNK_BYTES = 1024**2
_MAX_HEADER_BYTES = 16 * 1024**2
_SAMPLE_COUNT = 256
_DTYPES = {
    "F64": "<f8",
    "F32": "<f4",
    "F16": "<f2",
    "BF16": "<u2",
    "I64": "<i8",
    "I32": "<i4",
    "I16": "<i2",
    "I8": "i1",
    "U64": "<u8",
    "U32": "<u4",
    "U16": "<u2",
    "U8": "u1",
    "BOOL": "?",
}
# DFlash2 target-owned parameters, at any module prefix. They remain fully hashed.
_SHARED_MODULES = frozenset({"embed_tokens", "word_embeddings", "lm_head"})


class TensorFingerprint(FrozenModel):
    name: str = Field(min_length=1)
    dtype: str
    shape: tuple[int, ...]
    digest: Digest
    l2_norm: float = Field(ge=0)
    near_eligible: bool
    samples: tuple[float, ...] = Field(max_length=_SAMPLE_COUNT)
    # Preserve exact equality for integer values above float64's exact range.
    sample_bits: tuple[str, ...] = Field(max_length=_SAMPLE_COUNT)

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @model_validator(mode="after")
    def consistent_samples(self) -> TensorFingerprint:
        if self.dtype not in _DTYPES or any(d < 0 for d in self.shape):
            raise ValueError("Invalid fingerprint tensor schema")
        expected = min(self.elements, _SAMPLE_COUNT) if self.near_eligible else 0
        if len(self.samples) != expected or len(self.sample_bits) != expected:
            raise ValueError("Invalid fingerprint sample count")
        return self


class Fingerprint(FrozenModel):
    version: Literal[1] = 1
    scope: Digest
    tensor_digest: Digest
    config_digest: Digest
    tensors: tuple[TensorFingerprint, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def canonical_order(self) -> Fingerprint:
        names = tuple(t.name for t in self.tensors)
        if names != tuple(sorted(set(names))):
            raise ValueError("Fingerprint tensors must have unique sorted names")
        return self


class TensorComparison(FrozenModel):
    """Full-value evidence for one eligible tensor, without a near-copy verdict."""

    name: str = Field(min_length=1)
    elements: int = Field(gt=0)
    candidate_norm: float = Field(ge=0)
    previous_norm: float = Field(ge=0)
    exact_match: bool
    cosine: float = Field(ge=-1, le=1)
    relative_distance: float = Field(ge=0)


class SimilarityComparison(FrozenModel):
    version: Literal[1] = 1
    comparable: bool
    exact_duplicate: bool
    cosine: float | None = Field(default=None, ge=-1, le=1)
    relative_distance: float | None = Field(default=None, ge=0)
    copied_fraction: float | None = Field(default=None, ge=0, le=1)
    full_comparison: bool = False
    tensor_comparisons: tuple[TensorComparison, ...] = ()


def comparison_shortlist(
    comparisons: Mapping[str, SimilarityComparison], parents: Iterable[str]
) -> list[str]:
    """Require the three closest compatible checkpoints and all compatible parents."""
    comparable = {artifact: result for artifact, result in comparisons.items() if result.comparable}

    def priority(artifact: str) -> tuple[float, float, str]:
        result = comparable[artifact]
        return (
            -result.cosine if result.cosine is not None else float("inf"),
            result.relative_distance if result.relative_distance is not None else float("inf"),
            artifact,
        )

    selected = set(sorted(comparable, key=priority)[:3])
    selected.update(artifact for artifact in parents if artifact in comparable)
    return sorted(selected)


@dataclass(frozen=True)
class _TensorLocation:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    start: int
    length: int

    @property
    def elements(self) -> int:
        return math.prod(self.shape)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _tensor_locations(directory: Path) -> list[_TensorLocation]:
    paths = sorted(directory.glob("*.safetensors"))
    if not paths or len(paths) > 128:
        raise ValueError("Missing or excessive safetensors shards")
    locations: dict[str, _TensorLocation] = {}
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("Safetensors shards must be regular files")
        size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Truncated safetensors header")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= _MAX_HEADER_BYTES or header_size + 8 > size:
                raise ValueError("Invalid safetensors header size")
            header = strict_json(stream.read(header_size))
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors header")
        data_start = 8 + header_size
        ranges = []
        for name, entry in header.items():
            if name == "__metadata__":
                if not isinstance(entry, dict):
                    raise ValueError("Invalid safetensors metadata")
                continue
            if (
                not name
                or not isinstance(entry, dict)
                or not isinstance(entry.get("dtype"), str)
                or entry["dtype"] not in _DTYPES
            ):
                raise ValueError("Unsupported tensor dtype or invalid tensor header")
            if name in locations:
                raise ValueError("Duplicate tensor across safetensors shards")
            dtype, shape, offsets = entry["dtype"], entry.get("shape"), entry.get("data_offsets")
            if (
                not isinstance(shape, list)
                or len(shape) > 8
                or any(type(d) is not int or not 0 <= d <= 2**31 for d in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(d) is not int for d in offsets)
            ):
                raise ValueError("Invalid tensor shape or offsets")
            start, end = offsets
            expected = math.prod(shape) * np.dtype(_DTYPES[dtype]).itemsize
            if start < 0 or end - start != expected or end > size - data_start:
                raise ValueError("Invalid tensor storage range")
            ranges.append((start, end))
            locations[name] = _TensorLocation(
                name,
                dtype,
                tuple(shape),
                path,
                data_start + start,
                expected,
            )
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise ValueError("Noncontiguous or overlapping tensor storage")
            cursor = end
        if not ranges or cursor != size - data_start:
            raise ValueError("Truncated or trailing tensor storage")
    return [locations[name] for name in sorted(locations)]


def _near_eligible(name: str) -> bool:
    return not _SHARED_MODULES.intersection(name.split("."))


def _sample_indices(name: str, elements: int) -> list[int]:
    count = min(elements, _SAMPLE_COUNT)
    # One hashed position per disjoint stratum, without RNG/library-version state.
    return [
        i * elements // count
        + int.from_bytes(hashlib.sha256(f"prexon-v1:{name}:{i}".encode()).digest()[:8], "little")
        % ((i + 1) * elements // count - i * elements // count)
        for i in range(count)
    ]


def _inspect(location: _TensorLocation) -> TensorFingerprint:
    eligible = _near_eligible(location.name)
    indices = _sample_indices(location.name, location.elements) if eligible else []
    cursor, sample_cursor, norm = 0, 0, 0.0
    samples: list[float] = []
    sample_bits: list[str] = []
    digest = hashlib.sha256()
    dtype = np.dtype(_DTYPES[location.dtype])
    remaining = location.length
    with location.path.open("rb") as stream:
        stream.seek(location.start)
        while remaining:
            chunk = stream.read(min(remaining, _CHUNK_BYTES))
            if not chunk or len(chunk) % dtype.itemsize:
                raise ValueError("Truncated tensor storage")
            remaining -= len(chunk)
            raw = np.frombuffer(chunk, dtype=dtype).copy()
            if location.dtype == "BF16":
                values = (raw.astype("<u4") << 16).view("<f4").astype(np.float64)
            else:
                values = raw.astype(np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Checkpoint weights must be finite")
            if location.dtype.startswith("F") or location.dtype == "BF16":
                raw[values == 0] = 0  # Canonicalize positive and negative zero.
            elif location.dtype == "BOOL" and any(byte > 1 for byte in chunk):
                raise ValueError("Invalid BOOL tensor representation")
            digest.update(raw.tobytes())
            maximum = float(np.max(np.abs(values), initial=0))
            if maximum:
                values_scaled = values / maximum
                chunk_norm = maximum * math.sqrt(float(np.dot(values_scaled, values_scaled)))
                norm = math.hypot(norm, chunk_norm)
                if not math.isfinite(norm):
                    raise ValueError("Tensor norm exceeds finite float64 statistics")
            while sample_cursor < len(indices) and indices[sample_cursor] < cursor + raw.size:
                offset = indices[sample_cursor] - cursor
                samples.append(float(values[offset]))
                sample_bits.append(raw[offset : offset + 1].tobytes().hex())
                sample_cursor += 1
            cursor += raw.size
    return TensorFingerprint(
        name=location.name,
        dtype=location.dtype,
        shape=location.shape,
        digest=digest.hexdigest(),
        l2_norm=norm,
        near_eligible=eligible,
        samples=tuple(samples),
        sample_bits=tuple(sample_bits),
    )


def fingerprint(directory: Path, target: TargetConfig) -> Fingerprint:
    """Inspect an already frozen checkpoint using CPU and bounded tensor chunks.

    Scope covers target ID/revision, DFlash2, and the exact tensor schema; changing
    benchmark profiles does not erase prior comparisons. Config identity includes
    every config.json field, canonicalizing JSON object order and whitespace.
    Safetensors metadata, index layout and shard names are packaging only.
    """
    config_path = directory / "config.json"
    if (
        config_path.is_symlink()
        or not config_path.is_file()
        or config_path.stat().st_size > 1024**2
    ):
        raise ValueError("Missing or oversized checkpoint config.json")
    config = strict_json(config_path.read_bytes())
    if not isinstance(config, dict) or config.get("architectures") != ["DFlash2DraftModel"]:
        raise ValueError("CPU similarity supports DFlash2DraftModel checkpoints")
    if "dflash2" not in target.methods:
        raise ValueError("Target does not permit the checkpoint method")
    config_digest = _digest_json(config)
    tensors = tuple(_inspect(location) for location in _tensor_locations(directory))
    schema = [(t.name, t.dtype, t.shape) for t in tensors]
    return Fingerprint(
        scope=_digest_json(
            {
                "version": 1,
                "model_id": target.model_id,
                "revision": target.revision,
                "method": "dflash2",
                "tensors": schema,
            }
        ),
        tensor_digest=_digest_json([(t.name, t.dtype, t.shape, t.digest) for t in tensors]),
        config_digest=config_digest,
        tensors=tensors,
    )


def _unit_samples(tensor: TensorFingerprint) -> NDArray[np.float64] | None:
    values = np.asarray(tensor.samples, dtype=np.float64)
    maximum = float(np.max(np.abs(values), initial=0))
    if maximum == 0:
        return values if tensor.l2_norm == 0 else None
    values /= maximum
    values /= math.sqrt(float(np.dot(values, values)))
    return values


def compare(candidate: Fingerprint, previous: Fingerprint) -> SimilarityComparison:
    """Compare aligned schemas, never padding or matching arbitrary architectures.

    Exact identity requires all tensor contents and canonical config to match.
    Near cosine uses sampled tensor directions weighted by exact tensor norms.
    Relative distance uses those same vectors divided by the larger full eligible
    norm (symmetric, zero-safe). Copied fraction is element-weighted exact tensor
    matches plus exact sample matches, not an authorship probability. If samples
    miss every nonzero element of a nonzero tensor, direction metrics are absent.
    """

    def schema(fp: Fingerprint) -> tuple[tuple[str, str, tuple[int, ...], bool], ...]:
        return tuple((t.name, t.dtype, t.shape, t.near_eligible) for t in fp.tensors)

    if candidate.version != previous.version or candidate.scope != previous.scope:
        return SimilarityComparison(comparable=False, exact_duplicate=False)
    if schema(candidate) != schema(previous):
        return SimilarityComparison(comparable=False, exact_duplicate=False)
    exact = (
        candidate.tensor_digest == previous.tensor_digest
        and candidate.config_digest == previous.config_digest
    )
    pairs = [
        (a, b)
        for a, b in zip(candidate.tensors, previous.tensors, strict=True)
        if a.near_eligible and a.elements
    ]
    if not pairs:
        return SimilarityComparison(comparable=True, exact_duplicate=exact)
    a_norm = math.hypot(*(a.l2_norm for a, _ in pairs))
    b_norm = math.hypot(*(b.l2_norm for _, b in pairs))
    scale = max(a_norm, b_norm)
    if not math.isfinite(scale):
        raise ValueError("Checkpoint norm exceeds finite float64 statistics")
    copied, dots, distances = [], [], []
    missing_direction = False
    for a, b in pairs:
        if a.digest == b.digest:
            copied.append(float(a.elements))
            dots.append((a.l2_norm / scale) ** 2 if scale else 0)
            continue
        equal = sum(x == y for x, y in zip(a.sample_bits, b.sample_bits, strict=True))
        copied.append(a.elements * equal / len(a.samples))
        if not scale:
            continue
        a_unit, b_unit = _unit_samples(a), _unit_samples(b)
        if a_unit is None or b_unit is None:
            missing_direction = True
            continue
        a_vector, b_vector = a_unit * (a.l2_norm / scale), b_unit * (b.l2_norm / scale)
        dots.append(float(np.dot(a_vector, b_vector)))
        difference = a_vector - b_vector
        distances.append(float(np.dot(difference, difference)))
    copied_fraction = math.fsum(copied) / sum(a.elements for a, _ in pairs)
    if missing_direction:
        cosine = distance = None
    elif scale == 0:
        cosine, distance = 1.0, 0.0
    else:
        denominator = (a_norm / scale) * (b_norm / scale)
        cosine = min(1.0, max(-1.0, math.fsum(dots) / denominator)) if denominator else 0.0
        distance = math.sqrt(math.fsum(distances))
    return SimilarityComparison(
        comparable=True,
        exact_duplicate=exact,
        cosine=cosine,
        relative_distance=distance,
        copied_fraction=copied_fraction,
    )


def _full_tensor_comparison(
    candidate: _TensorLocation,
    previous: _TensorLocation,
    a: TensorFingerprint,
    b: TensorFingerprint,
) -> TensorComparison:
    scale = max(a.l2_norm, b.l2_norm)
    exact = a.digest == b.digest
    if exact or scale == 0:
        cosine, distance = 1.0, 0.0
    else:
        dtype = np.dtype(_DTYPES[candidate.dtype])
        remaining = candidate.length
        dot, dot_correction, distance_squared = 0.0, 0.0, 0.0
        with candidate.path.open("rb") as a_stream, previous.path.open("rb") as b_stream:
            a_stream.seek(candidate.start)
            b_stream.seek(previous.start)
            while remaining:
                length = min(remaining, _CHUNK_BYTES)
                a_chunk, b_chunk = a_stream.read(length), b_stream.read(length)
                if len(a_chunk) != length or len(b_chunk) != length:
                    raise ValueError("Truncated tensor storage during full comparison")
                remaining -= length
                a_raw, b_raw = (
                    np.frombuffer(a_chunk, dtype=dtype),
                    np.frombuffer(b_chunk, dtype=dtype),
                )
                if candidate.dtype == "BF16":
                    a_values = (a_raw.astype("<u4") << 16).view("<f4").astype(np.float64)
                    b_values = (b_raw.astype("<u4") << 16).view("<f4").astype(np.float64)
                else:
                    a_values, b_values = a_raw.astype(np.float64), b_raw.astype(np.float64)
                if not np.isfinite(a_values).all() or not np.isfinite(b_values).all():
                    raise ValueError("Checkpoint weights must be finite")
                # Scaling before products/subtraction avoids float64 overflow.
                a_values /= scale
                b_values /= scale
                term = float(np.dot(a_values, b_values)) - dot_correction
                updated = dot + term
                dot_correction = (updated - dot) - term
                dot = updated
                difference = a_values - b_values
                distance_squared = math.fsum(
                    (
                        distance_squared,
                        float(np.dot(difference, difference)),
                    )
                )
        denominator = (a.l2_norm / scale) * (b.l2_norm / scale)
        cosine = min(1.0, max(-1.0, dot / denominator)) if denominator else 0.0
        distance = math.sqrt(distance_squared)
    return TensorComparison(
        name=a.name,
        elements=a.elements,
        candidate_norm=a.l2_norm,
        previous_norm=b.l2_norm,
        exact_match=exact,
        cosine=cosine,
        relative_distance=distance,
    )


def compare_full(
    candidate_dir: Path,
    previous_dir: Path,
    target: TargetConfig,
) -> SimilarityComparison:
    """Confirm a shortlist against every value of immutable CPU checkpoints.

    Both inputs are fully fingerprinted first, covering shared tensors and config
    in exact identity. Only aligned eligible tensors contribute numerical evidence;
    their full values are then read in bounded chunks. No sample is used for these
    metrics. Per-tensor and aggregate relative distances divide L2 difference by
    the larger corresponding candidate/previous L2 norm, with zero-safe behavior.

    Unlike the shortlist's sampled equality fraction, copied_fraction here is the
    fraction of eligible parameters belonging to EXACTLY matching whole tensors.
    It is not a copying verdict. tensor_comparisons supplies full cosine/distance
    for each eligible tensor so a published calibrated policy can count very-near
    matches with its own thresholds, rather than a hidden detector constant.
    """
    candidate = fingerprint(candidate_dir, target)
    previous = fingerprint(previous_dir, target)
    screened = compare(candidate, previous)
    if not screened.comparable:
        return SimilarityComparison(
            comparable=False,
            exact_duplicate=False,
            full_comparison=True,
        )
    candidate_locations = {t.name: t for t in _tensor_locations(candidate_dir)}
    previous_locations = {t.name: t for t in _tensor_locations(previous_dir)}
    tensors = tuple(
        _full_tensor_comparison(candidate_locations[a.name], previous_locations[b.name], a, b)
        for a, b in zip(candidate.tensors, previous.tensors, strict=True)
        if a.near_eligible and a.elements
    )
    if not tensors:
        return SimilarityComparison(
            comparable=True,
            exact_duplicate=screened.exact_duplicate,
            full_comparison=True,
        )
    a_norm = math.hypot(*(t.candidate_norm for t in tensors))
    b_norm = math.hypot(*(t.previous_norm for t in tensors))
    scale = max(a_norm, b_norm)
    if scale == 0:
        cosine, distance = 1.0, 0.0
    else:
        cosine = (
            math.fsum(
                t.cosine * (t.candidate_norm / a_norm) * (t.previous_norm / b_norm) for t in tensors
            )
            if a_norm and b_norm
            else 0.0
        )
        cosine = min(1.0, max(-1.0, cosine))
        distance = math.hypot(
            *(
                t.relative_distance * (max(t.candidate_norm, t.previous_norm) / scale)
                for t in tensors
            )
        )
    copied_fraction = sum(t.elements for t in tensors if t.exact_match) / sum(
        t.elements for t in tensors
    )
    return SimilarityComparison(
        comparable=True,
        exact_duplicate=screened.exact_duplicate,
        full_comparison=True,
        cosine=cosine,
        relative_distance=distance,
        copied_fraction=copied_fraction,
        tensor_comparisons=tensors,
    )
