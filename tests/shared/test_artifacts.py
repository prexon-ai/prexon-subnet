import hashlib
import io
import json
import logging
import struct
import subprocess
import sys
import tarfile
import tracemalloc
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from shared.artifacts import ArtifactStore, download_url, extract_package, package_checkpoint
from shared.checkpoints import checkpoint_schema, validate_checkpoint
from tests.factories import make_checkpoint, provenance


def test_package_roundtrip_is_deterministic_and_content_addressed(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "head")
    a, b = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    manifest = package_checkpoint(
        checkpoint,
        a,
        target_id="org/model",
        target_revision="c" * 40,
        competition_id="a" * 64,
        provenance=provenance(),
    )
    package_checkpoint(
        checkpoint,
        b,
        target_id="org/model",
        target_revision="c" * 40,
        competition_id="a" * 64,
        provenance=provenance(),
    )
    assert a.read_bytes() == b.read_bytes()
    store = ArtifactStore(str(tmp_path / "store"))
    digest = store.put_file(a)
    assert digest == hashlib.sha256(a.read_bytes()).hexdigest()
    destination = tmp_path / "restored"
    assert extract_package(a, destination, digest, 1024**2) == manifest
    assert (destination / "model.safetensors").read_bytes().endswith(b"\0" * 8)
    a.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digest"):
        extract_package(a, tmp_path / "bad", digest, 1024**2)


@pytest.mark.parametrize(
    "name,kind", [("../escape", "file"), ("/absolute", "file"), ("model.safetensors", "symlink")]
)
def test_untrusted_archive_cannot_escape_or_link(tmp_path, name, kind):
    path = tmp_path / "attack.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = 1
        if kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
        archive.addfile(member, io.BytesIO(b"x"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        extract_package(path, tmp_path / "extract", digest, 1024)
    assert not (tmp_path / "escape").exists()


def test_checkpoint_rejects_remote_code_and_invalid_tensor_offsets(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "head")
    validate_checkpoint(checkpoint, 1024**2)
    config = json.loads((checkpoint / "config.json").read_text())
    config["auto_map"] = {"AutoModel": "evil.Model"}
    (checkpoint / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="remote"):
        validate_checkpoint(checkpoint, 1024**2)
    config.pop("auto_map")
    (checkpoint / "config.json").write_text(json.dumps(config))
    tensor = checkpoint / "model.safetensors"
    tensor.write_bytes(tensor.read_bytes()[:-1])
    with pytest.raises(ValueError, match="tensor"):
        validate_checkpoint(checkpoint, 1024**2)


def test_size_limits_apply_before_extraction(tmp_path):
    checkpoint = make_checkpoint(tmp_path / "head")
    with pytest.raises(ValueError, match="size"):
        validate_checkpoint(checkpoint, 64)


def test_pax_metadata_is_rejected_without_unbounded_parser_allocation(tmp_path):
    archive = tmp_path / "metadata.tar.gz"
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as tar:
        member = tarfile.TarInfo("config.json")
        member.size = 1
        member.pax_headers = {"comment": "x" * (8 * 1024**2)}
        tar.addfile(member, io.BytesIO(b"x"))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    tracemalloc.start()
    try:
        with pytest.raises(ValueError):
            extract_package(archive, tmp_path / "extracted", digest, 1024)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024**2


def test_qualified_schema_rejects_wrong_tensor_names_even_with_valid_storage(tmp_path):
    head = make_checkpoint(tmp_path / "head")
    schema = checkpoint_schema(head)
    tensor = head / "model.safetensors"
    header = json.dumps(
        {"unexpected": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    ).encode()
    tensor.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
    with pytest.raises(ValueError, match="qualified"):
        validate_checkpoint(head, 1024**2, schemas=(schema,))


def test_nonfinite_weights_are_rejected_before_engine_loading(tmp_path):
    head = make_checkpoint(tmp_path / "head")
    tensor = head / "model.safetensors"
    data = tensor.read_bytes()
    tensor.write_bytes(data[:-8] + struct.pack("<ff", float("nan"), 0))
    with pytest.raises(ValueError, match="finite"):
        validate_checkpoint(head, 1024**2)


def test_optimized_python_still_rejects_invalid_checkpoint_architecture(tmp_path):
    head = make_checkpoint(tmp_path / "head")
    config = head / "config.json"
    data = json.loads(config.read_text())
    data["hidden_size"] = 0
    config.write_text(json.dumps(data))
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "from pathlib import Path\n"
            "import sys\n"
            "from shared.checkpoints import validate_checkpoint\n"
            "try:\n"
            "    validate_checkpoint(Path(sys.argv[1]), 1024**2)\n"
            "except ValueError:\n"
            "    sys.exit(0)\n"
            "sys.exit('Invalid checkpoint architecture was accepted')\n",
            str(head),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_checkpoint_rejects_a_nonobject_tensor_index(tmp_path):
    head = make_checkpoint(tmp_path / "head")
    (head / "model.safetensors.index.json").write_text("[]")
    with pytest.raises(ValueError, match="index"):
        validate_checkpoint(head, 1024**2)


@pytest.mark.parametrize("status", [200, 403])
def test_private_download_capabilities_do_not_leak_to_logs_or_errors(
    tmp_path, monkeypatch, caplog, status
):
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(status, content=b"head"))
    )
    monkeypatch.setattr("shared.artifacts.store.httpx.stream", client.stream)
    caplog.set_level(logging.DEBUG)
    url = "https://s3.hippius.com/private/head?X-Amz-Signature=private-capability"
    if status == 200:
        download_url(url, tmp_path / "head", 1024)
        assert (tmp_path / "head").read_bytes() == b"head"
    else:
        with pytest.raises(RuntimeError, match="download failed") as error:
            download_url(url, tmp_path / "head", 1024)
        assert "private-capability" not in str(error.value)
    assert "private-capability" not in caplog.text
    client.get("https://public.example/normal-request")
    assert "normal-request" in caplog.text
    client.close()


def test_storage_does_not_take_credentials_from_aws_environment():
    with pytest.raises(ValueError, match="S3_ACCESS_KEY_ID"):
        ArtifactStore(
            "s3://private",
            environment={
                "AWS_ACCESS_KEY_ID": "unrelated-aws-account",
                "AWS_SECRET_ACCESS_KEY": "aws-secret",
            },
        )


def test_presigned_download_uses_hippius_region_and_path_style():
    environment = {"S3_ACCESS_KEY_ID": "hip_test_access", "S3_SECRET_ACCESS_KEY": "test-secret"}
    store = ArtifactStore("s3://private-artifacts/prefix", environment=environment)
    url = urlsplit(store.url("objects/" + "a" * 64, expires=300))
    assert url.scheme == "https" and url.hostname == "s3.hippius.com"
    assert url.path == "/private-artifacts/prefix/objects/" + "a" * 64
    query = parse_qs(url.query)
    assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
    assert query["X-Amz-Credential"][0].endswith("/decentralized/s3/aws4_request")
    assert query["X-Amz-Expires"] == ["300"]
    assert "test-secret" not in url.geturl()
