from __future__ import annotations

import gzip
import io
import tarfile
from pathlib import Path, PurePosixPath

from shared.checkpoints import sha256_file, validate_checkpoint
from shared.contracts import Provenance, SubmissionManifest


def package_checkpoint(
    directory: Path,
    output: Path,
    *,
    target_id: str,
    target_revision: str,
    competition_id: str,
    provenance: Provenance,
    max_bytes: int = 4 * 1024**3,
) -> SubmissionManifest:
    files = validate_checkpoint(directory, max_bytes)
    manifest = SubmissionManifest(
        target_id=target_id,
        target_revision=target_revision,
        competition_id=competition_id,
        files=files,
        provenance=provenance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("wb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz,
    ):
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            data = manifest.model_dump_json().encode()
            member = tarfile.TarInfo("manifest.json")
            member.size = len(data)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))
            for file in files:
                member = tarfile.TarInfo(file.path)
                member.size = file.size
                member.mode = 0o644
                with (directory / file.path).open("rb") as stream:
                    archive.addfile(member, stream)
    return manifest


def extract_package(
    package: Path,
    destination: Path,
    expected_digest: str,
    max_bytes: int,
) -> SubmissionManifest:
    if sha256_file(package) != expected_digest:
        raise ValueError("Artifact digest mismatch")
    if package.stat().st_size > max_bytes + 2 * 1024**2:
        raise ValueError("Compressed artifact exceeds size limit")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Extraction requires an empty destination")
    destination.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    total = 0
    # Read fixed-size USTAR headers ourselves. TarFile would allocate extension metadata
    # (PAX/GNU/sparse) before yielding a member and bypass the limits below.
    with gzip.open(package, "rb") as archive:
        while True:
            header = archive.read(512)
            if len(header) != 512:
                raise ValueError("Truncated archive header")
            if header == b"\0" * 512:
                if archive.read(512) != b"\0" * 512:
                    raise ValueError("Invalid archive terminator")
                tail = archive.read(10241)
                if len(tail) > 10240 or tail.strip(b"\0"):
                    raise ValueError("Unexpected trailing archive data")
                break
            try:
                member = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
            except (tarfile.TarError, UnicodeError) as exc:
                raise ValueError("Invalid archive header") from exc
            if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                raise ValueError("Extended or nonregular archive headers are prohibited")
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or len(path.parts) != 1
                or member.name in {".", ".."}
                or "\\" in member.name
                or member.name in seen
                or len(seen) >= 129
            ):
                raise ValueError("Unsafe or duplicate archive member")
            total += member.size
            if total > max_bytes + 1024**2 or member.size < 0:
                raise ValueError("Artifact exceeds expanded size limit")
            if member.name == "manifest.json" and member.size > 1024**2:
                raise ValueError("Oversized manifest")
            seen.add(member.name)
            with (destination / member.name).open("xb") as output:
                remaining = member.size
                while remaining:
                    chunk = archive.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Truncated archive data")
                    output.write(chunk)
                    remaining -= len(chunk)
            padding = (-member.size) % 512
            if archive.read(padding) != b"\0" * padding:
                raise ValueError("Invalid archive padding")
    manifest_path = destination / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Missing submission manifest")
    manifest = SubmissionManifest.model_validate_json(manifest_path.read_bytes())
    manifest_path.unlink()
    actual = validate_checkpoint(destination, max_bytes)
    if tuple(sorted(manifest.files, key=lambda f: f.path)) != actual:
        raise ValueError("Checkpoint does not match manifest digests")
    return manifest
