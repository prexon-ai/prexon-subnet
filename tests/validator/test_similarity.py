"""CPU screening contracts, using real safetensors without torch or target loading."""

import json
import math
import struct
import tracemalloc

import numpy as np
import pytest

from shared.config import Hardware, TargetConfig, Workload
from tests.factories import make_checkpoint


def target(**updates):
    value = TargetConfig(
        slot=1,
        model_id="org/target",
        revision="a" * 40,
        allocation=100,
        engine_version="0.5.3",
        worker_image="engine@sha256:" + "b" * 64,
        hardware=Hardware(gpu_type="test", min_vram_gb=1),
        workloads=(
            Workload(
                name="test",
                concurrency=1,
                requests=1,
                output_tokens=1,
                weight=1,
                prompts=("test",),
            ),
        ),
    )
    return value.model_copy(update=updates)


def write_shard(path, tensors, metadata=None):
    header = {}
    data = bytearray()
    for name, dtype, shape, values in tensors:
        start = len(data)
        data.extend(values)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(data)]}
    if metadata is not None:
        header["__metadata__"] = metadata
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def head(tmp_path, name, values, *, dtype="F32", tensor_name="draft.weight", shape=None):
    directory = make_checkpoint(tmp_path / name)
    formats = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I64": "<i8", "I8": "i1"}
    if dtype == "BF16":
        data = np.asarray(values, dtype="<f4").view("<u4")
        data = (data >> 16).astype("<u2").tobytes()
    else:
        data = np.asarray(values, dtype=formats[dtype]).tobytes()
    write_shard(
        directory / "model.safetensors",
        [(tensor_name, dtype, shape or [len(values)], data)],
    )
    return directory


def test_repackaging_and_resharding_preserve_canonical_identity(tmp_path):
    from validator.similarity import Fingerprint, compare, fingerprint

    original = head(tmp_path, "original", [1, 2, 3, 4])
    tensors = [
        ("draft.weight", "F32", [4], struct.pack("<4f", 1, 2, 3, 4)),
        ("draft.bias", "F16", [2], struct.pack("<2e", -1, 0)),
    ]
    write_shard(original / "model.safetensors", tensors, {"format": "pt"})
    repacked = make_checkpoint(tmp_path / "repacked")
    (repacked / "model.safetensors").unlink()
    config = json.loads((original / "config.json").read_text())
    (repacked / "config.json").write_text(json.dumps(config, sort_keys=True, indent=4))
    write_shard(repacked / "model-2-of-2.safetensors", tensors[:1], {"producer": "other"})
    write_shard(repacked / "model-1-of-2.safetensors", tensors[1:])
    (repacked / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {t[0]: "ignored packaging mapping" for t in tensors}})
    )
    a, b = fingerprint(original, target()), fingerprint(repacked, target())
    assert a.tensor_digest == b.tensor_digest
    assert a.config_digest == b.config_digest
    assert a.scope == b.scope
    assert compare(a, b).exact_duplicate
    assert Fingerprint.model_validate_json(a.model_dump_json()) == a
    assert a.version == 1


def test_changed_weights_have_distinct_identity_and_hand_calculated_metrics(tmp_path):
    from validator.similarity import compare, fingerprint

    previous = fingerprint(head(tmp_path, "before", [1, 2, 3, 4]), target())
    candidate = fingerprint(head(tmp_path, "after", [1, 2, 3, 5]), target())
    result = compare(candidate, previous)
    assert result.comparable and not result.exact_duplicate
    assert candidate.tensor_digest != previous.tensor_digest
    assert result.cosine == pytest.approx(34 / math.sqrt(30 * 39))
    assert result.relative_distance == pytest.approx(1 / math.sqrt(39))
    assert result.copied_fraction == pytest.approx(0.75)


def test_changed_inference_config_is_part_of_exact_identity(tmp_path):
    from validator.similarity import compare, fingerprint

    directory = head(tmp_path, "head", [1, 2])
    before = fingerprint(directory, target())
    config = json.loads((directory / "config.json").read_text())
    config["dflash_config"]["block_size"] = 8
    (directory / "config.json").write_text(json.dumps(config))
    after = fingerprint(directory, target())
    assert after.config_digest != before.config_digest
    assert after.tensor_digest == before.tensor_digest
    assert not compare(after, before).exact_duplicate


@pytest.mark.parametrize(
    "name",
    ["model.embed_tokens.weight", "lm_head.weight", "word_embeddings.weight"],
)
def test_shared_target_tensors_do_not_dominate_near_evidence(tmp_path, name):
    from validator.similarity import compare, fingerprint

    a = head(tmp_path, "a", [1, 2])
    b = head(tmp_path, "b", [-1, -2])
    shared = (name, "F32", [2048], np.full(2048, 10000, dtype="<f4").tobytes())
    write_shard(
        a / "model.safetensors",
        [shared, ("draft.weight", "F32", [2], struct.pack("<2f", 1, 2))],
    )
    write_shard(
        b / "model.safetensors",
        [shared, ("draft.weight", "F32", [2], struct.pack("<2f", -1, -2))],
    )
    result = compare(fingerprint(a, target()), fingerprint(b, target()))
    assert not result.exact_duplicate
    assert result.cosine == pytest.approx(-1)
    assert result.relative_distance == pytest.approx(2)
    assert result.copied_fraction == 0


def test_shared_target_tensor_change_still_changes_exact_identity(tmp_path):
    from validator.similarity import compare, fingerprint

    a = head(tmp_path, "a", [1, 2], tensor_name="lm_head.weight")
    b = head(tmp_path, "b", [1, 3], tensor_name="lm_head.weight")
    result = compare(fingerprint(a, target()), fingerprint(b, target()))
    assert result.comparable and not result.exact_duplicate
    assert result.cosine is None
    assert result.relative_distance is None
    assert result.copied_fraction is None


def test_scope_survives_benchmark_changes_but_not_target_changes(tmp_path):
    from validator.similarity import compare, fingerprint

    directory = head(tmp_path, "head", [1, 2])
    before = fingerprint(directory, target())
    changed_profile = target(context_length=4096, engine_version="0.5.4", allocation=50, slot=2)
    assert fingerprint(directory, changed_profile).scope == before.scope
    for change in ({"model_id": "different/model"}, {"revision": "c" * 40}):
        other = fingerprint(directory, target(**change))
        result = compare(other, before)
        assert other.scope != before.scope
        assert not result.comparable and not result.exact_duplicate
        assert result.cosine is result.relative_distance is result.copied_fraction is None


@pytest.mark.parametrize(
    "updates",
    [{"shape": [1, 4]}, {"tensor_name": "different.weight"}, {"dtype": "F16"}],
)
def test_incompatible_tensor_schemas_are_not_compared(tmp_path, updates):
    from validator.similarity import compare, fingerprint

    before = fingerprint(head(tmp_path, "a", [1, 2, 3, 4]), target())
    after = fingerprint(head(tmp_path, "b", [1, 2, 3, 4], **updates), target())
    assert not compare(after, before).comparable


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16", "F64", "I64", "I8"])
def test_supported_dtypes_decode_for_correct_norm_and_similarity(tmp_path, dtype):
    from validator.similarity import compare, fingerprint

    before = fingerprint(head(tmp_path, "a", [1, 2, -2, 0], dtype=dtype), target())
    after = fingerprint(head(tmp_path, "b", [1, 2, -1, 0], dtype=dtype), target())
    assert before.tensors[0].l2_norm == pytest.approx(3)
    assert before.tensors[0].samples == (1, 2, -2, 0)
    result = compare(after, before)
    assert result.cosine == pytest.approx(7 / math.sqrt(54))
    assert result.relative_distance == pytest.approx(1 / 3)
    assert result.copied_fraction == 0.75


@pytest.mark.parametrize(
    "values,other,cosine,distance,copied",
    [
        ([0, 0], [0, 0], 1, 0, 1),
        ([0, 0], [1, 1], 0, 1, 0),
        ([2, 2], [2, 2], 1, 0, 1),
    ],
)
def test_zero_and_constant_tensors_have_finite_metrics(
    tmp_path,
    values,
    other,
    cosine,
    distance,
    copied,
):
    from validator.similarity import compare, fingerprint

    a = fingerprint(head(tmp_path, "a", values), target())
    b = fingerprint(head(tmp_path, "b", other), target())
    result = compare(a, b)
    assert result.cosine == pytest.approx(cosine)
    assert result.relative_distance == pytest.approx(distance)
    assert result.copied_fraction == pytest.approx(copied)
    assert "NaN" not in result.model_dump_json()


def test_signed_zero_is_canonical_numerical_identity(tmp_path):
    from validator.similarity import compare, fingerprint

    a = fingerprint(head(tmp_path, "a", [0.0, -0.0]), target())
    b = fingerprint(head(tmp_path, "b", [-0.0, 0.0]), target())
    assert a.tensor_digest == b.tensor_digest
    assert compare(a, b).exact_duplicate


def test_nonfinite_and_unsupported_weights_fail_clearly(tmp_path):
    from validator.similarity import fingerprint

    directory = head(tmp_path, "head", [float("nan")])
    with pytest.raises(ValueError, match="finite"):
        fingerprint(directory, target())
    write_shard(directory / "model.safetensors", [("draft.weight", "F8_E4M3", [1], b"\x00")])
    with pytest.raises(ValueError, match="Unsupported.*dtype"):
        fingerprint(directory, target())


def test_fingerprinting_memory_and_serialized_samples_are_bounded(tmp_path):
    from validator.similarity import fingerprint

    directory = make_checkpoint(tmp_path / "large")
    count = 8 * 1024**2
    header = json.dumps(
        {
            "draft.weight": {"dtype": "F32", "shape": [count], "data_offsets": [0, 4 * count]},
        }
    ).encode()
    with (directory / "model.safetensors").open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)) + header)
        chunk = np.ones(256 * 1024, dtype="<f4").tobytes()
        for _ in range(count // (256 * 1024)):
            stream.write(chunk)
    tracemalloc.start()
    try:
        value = fingerprint(directory, target())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert value.tensors[0].l2_norm == pytest.approx(math.sqrt(count))
    assert len(value.tensors[0].samples) <= 4096
    assert len(value.model_dump_json()) < 100000
    assert peak < 16 * 1024**2


def test_empty_tensors_have_no_near_evidence_or_nan(tmp_path):
    from validator.similarity import compare, fingerprint

    directory = head(tmp_path, "empty", [], shape=[0])
    value = fingerprint(directory, target())
    result = compare(value, value)
    assert result.exact_duplicate and result.comparable
    assert result.cosine is result.relative_distance is result.copied_fraction is None


def test_large_integer_copy_fraction_uses_exact_integer_values(tmp_path):
    from validator.similarity import compare, fingerprint

    a = fingerprint(head(tmp_path, "a", [2**60, 2**60], dtype="I64"), target())
    b = fingerprint(head(tmp_path, "b", [2**60, 2**60 + 1], dtype="I64"), target())
    result = compare(a, b)
    assert not result.exact_duplicate
    assert result.copied_fraction == 0.5


def test_sparse_unobserved_directions_are_unknown_not_nan_or_certain_copy(tmp_path):
    from validator.similarity import compare, fingerprint

    # 1024 positions exceed the sample budget. Pigeonhole principle guarantees
    # an unsampled nonzero support for a deterministic single-coordinate vector.
    values = np.zeros(1024, dtype="<f4")
    values[10] = 1
    a_path = head(tmp_path, "a", values)
    a = fingerprint(a_path, target())
    if any(a.tensors[0].samples):
        values[10] = 0
        # Use the serialized canonical sample bits to independently observe when
        # a support is missed, without duplicating the sampling algorithm.
        for index in range(1024):
            values[index] = 1
            write_shard(
                a_path / "model.safetensors",
                [
                    ("draft.weight", "F32", [1024], values.tobytes()),
                ],
            )
            a = fingerprint(a_path, target())
            if not any(a.tensors[0].samples):
                break
            values[index] = 0
    assert a.tensors[0].l2_norm == 1 and not any(a.tensors[0].samples)
    b = fingerprint(head(tmp_path, "b", values * 2), target())
    result = compare(a, b)
    assert not result.exact_duplicate
    assert result.cosine is None and result.relative_distance is None


@pytest.mark.parametrize("mode", ["duplicate", "overlap", "trailing", "truncated", "bad-dtype"])
def test_malformed_tensor_storage_is_rejected(tmp_path, mode):
    from validator.similarity import fingerprint

    directory = head(tmp_path, "head", [1, 2])
    tensor = directory / "model.safetensors"
    if mode == "duplicate":
        write_shard(
            directory / "second.safetensors",
            [
                ("draft.weight", "F32", [2], struct.pack("<2f", 1, 2)),
            ],
        )
    elif mode == "overlap":
        entries = {
            "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]},
            "b": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
        encoded = json.dumps(entries).encode()
        tensor.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 8)
    elif mode == "bad-dtype":
        entries = {"a": {"dtype": [], "shape": [1], "data_offsets": [0, 1]}}
        encoded = json.dumps(entries).encode()
        tensor.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00")
    else:
        data = tensor.read_bytes()
        tensor.write_bytes(data + b"\x00" if mode == "trailing" else data[:-1])
    with pytest.raises(ValueError):
        fingerprint(directory, target())


def test_full_comparison_detects_changes_outside_every_sample(tmp_path):
    from validator.similarity import compare, compare_full, fingerprint

    values = np.arange(1, 1025, dtype="<f4")
    previous_dir = head(tmp_path, "previous", values)
    previous = fingerprint(previous_dir, target())
    observed = {int(value) - 1 for value in previous.tensors[0].samples}
    changed_index = max(set(range(1024)) - observed)
    changed_value = float(values[changed_index])
    values[changed_index] *= -1  # Same full norm; every public sample stays equal.
    candidate_dir = head(tmp_path, "candidate", values)
    sampled = compare(fingerprint(candidate_dir, target()), previous)
    assert sampled.cosine == 1 and sampled.relative_distance == 0
    assert sampled.copied_fraction == 1
    result = compare_full(candidate_dir, previous_dir, target())
    norm_squared = 1024 * 1025 * 2049 / 6
    assert result.full_comparison and result.comparable and not result.exact_duplicate
    assert result.cosine == pytest.approx(1 - 2 * changed_value**2 / norm_squared)
    assert result.relative_distance == pytest.approx(2 * changed_value / math.sqrt(norm_squared))
    assert result.copied_fraction == 0  # Whole-tensor exact-match fraction.
    assert len(result.tensor_comparisons) == 1
    tensor = result.tensor_comparisons[0]
    assert tensor.name == "draft.weight" and tensor.elements == 1024
    assert not tensor.exact_match
    assert tensor.relative_distance == result.relative_distance


@pytest.mark.parametrize(
    "candidate_values,previous_values,cosine,distance,fraction",
    [
        ([0, 0], [0, 0], 1, 0, 1),
        ([1, 1], [0, 0], 0, 1, 0),
        ([0, 0], [1, 1], 0, 1, 0),
        ([-1, -2], [1, 2], -1, 2, 0),
    ],
)
def test_full_comparison_zero_vectors_and_sign_changes(
    tmp_path,
    candidate_values,
    previous_values,
    cosine,
    distance,
    fraction,
):
    from validator.similarity import compare_full

    candidate_dir = head(tmp_path, "candidate", candidate_values)
    previous_dir = head(tmp_path, "previous", previous_values)
    result = compare_full(candidate_dir, previous_dir, target())
    assert result.cosine == pytest.approx(cosine)
    assert result.relative_distance == pytest.approx(distance)
    assert result.copied_fraction == fraction
    assert result.tensor_comparisons[0].cosine == pytest.approx(cosine)
    assert result.tensor_comparisons[0].relative_distance == pytest.approx(distance)
    assert "NaN" not in result.model_dump_json()


@pytest.mark.parametrize("dtype", ["F32", "F16", "BF16", "F64", "I64", "I8"])
def test_full_comparison_supported_dtypes_use_all_numerical_values(tmp_path, dtype):
    from validator.similarity import compare_full

    candidate_dir = head(tmp_path, "candidate", [1, 2, 3, 5], dtype=dtype)
    previous_dir = head(tmp_path, "previous", [1, 2, 3, 4], dtype=dtype)
    result = compare_full(candidate_dir, previous_dir, target())
    assert result.cosine == pytest.approx(34 / math.sqrt(1170))
    assert result.relative_distance == pytest.approx(1 / math.sqrt(39))
    assert result.copied_fraction == 0
    tensor = result.tensor_comparisons[0]
    assert tensor.candidate_norm == pytest.approx(math.sqrt(39))
    assert tensor.previous_norm == pytest.approx(math.sqrt(30))


def test_full_comparison_reports_parameter_weighted_tensor_matches(tmp_path):
    from validator.similarity import compare_full

    candidate_dir = head(tmp_path, "candidate", [1, 2])
    previous_dir = head(tmp_path, "previous", [1, 2])
    shared = ("lm_head.weight", "F32", [2048], np.full(2048, 10000, dtype="<f4").tobytes())
    for directory, changed_values in [(candidate_dir, [2] * 6), (previous_dir, [1] * 6)]:
        write_shard(
            directory / "model.safetensors",
            [
                shared,
                ("draft.weight", "F32", [2], struct.pack("<2f", 1, 2)),
                ("draft.bias", "F32", [6], struct.pack("<6f", *changed_values)),
            ],
        )
    result = compare_full(candidate_dir, previous_dir, target())
    assert result.copied_fraction == 0.25
    tensors = {tensor.name: tensor for tensor in result.tensor_comparisons}
    assert set(tensors) == {"draft.weight", "draft.bias"}
    assert tensors["draft.weight"].exact_match
    assert not tensors["draft.bias"].exact_match
    assert tensors["draft.bias"].cosine == pytest.approx(1)
    assert tensors["draft.bias"].relative_distance == pytest.approx(0.5)
    assert result.cosine == pytest.approx(17 / math.sqrt(11 * 29))
    assert result.relative_distance == pytest.approx(math.sqrt(6 / 29))


@pytest.mark.parametrize("kind", ["empty", "shared"])
def test_full_comparison_absent_eligible_values_produces_no_near_metrics(tmp_path, kind):
    from validator.similarity import compare_full

    directory = head(
        tmp_path,
        "head",
        [] if kind == "empty" else [1, 2],
        shape=[0] if kind == "empty" else [2],
        tensor_name="draft.weight" if kind == "empty" else "lm_head.weight",
    )
    result = compare_full(directory, directory, target())
    assert result.full_comparison and result.exact_duplicate and result.comparable
    assert result.cosine is result.relative_distance is result.copied_fraction is None
    assert result.tensor_comparisons == ()


def test_full_comparison_preserves_packaging_identity_and_config_checks(tmp_path):
    from validator.similarity import compare_full

    previous_dir = head(tmp_path, "previous", [1, 2, 3])
    candidate_dir = head(tmp_path, "candidate", [1, 2, 3])
    config = json.loads((candidate_dir / "config.json").read_text())
    (candidate_dir / "config.json").write_text(json.dumps(config, indent=4, sort_keys=True))
    tensor = candidate_dir / "model.safetensors"
    tensor.rename(candidate_dir / "other-shard.safetensors")
    assert compare_full(candidate_dir, previous_dir, target()).exact_duplicate
    config["dflash_config"]["block_size"] = 8
    (candidate_dir / "config.json").write_text(json.dumps(config))
    assert not compare_full(candidate_dir, previous_dir, target()).exact_duplicate


def test_full_comparison_rejects_incompatible_tensor_shapes(tmp_path):
    from validator.similarity import compare_full

    candidate_dir = head(tmp_path, "candidate", [1, 2, 3, 4], shape=[2, 2])
    previous_dir = head(tmp_path, "previous", [1, 2, 3, 4])
    result = compare_full(candidate_dir, previous_dir, target())
    assert not result.comparable and not result.exact_duplicate
    assert result.cosine is result.relative_distance is result.copied_fraction is None
    assert result.tensor_comparisons == ()


def test_full_comparison_memory_is_bounded_by_chunks(tmp_path):
    from validator.similarity import compare_full

    count = 8 * 1024**2
    directories = []
    for name, fill in [("candidate", 2), ("previous", 1)]:
        directory = make_checkpoint(tmp_path / name)
        directories.append(directory)
        header = json.dumps(
            {
                "draft.weight": {"dtype": "F32", "shape": [count], "data_offsets": [0, 4 * count]},
            }
        ).encode()
        with (directory / "model.safetensors").open("wb") as stream:
            stream.write(struct.pack("<Q", len(header)) + header)
            chunk = np.full(256 * 1024, fill, dtype="<f4").tobytes()
            for _ in range(count // (256 * 1024)):
                stream.write(chunk)
    tracemalloc.start()
    try:
        result = compare_full(directories[0], directories[1], target())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.cosine == pytest.approx(1)
    assert result.relative_distance == pytest.approx(0.5)
    assert result.copied_fraction == 0
    assert peak < 24 * 1024**2
