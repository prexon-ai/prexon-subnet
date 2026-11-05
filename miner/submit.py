"""Freeze and validate a package, upload with a capability, then commit its digest."""

from __future__ import annotations

import ipaddress
import secrets
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from shared.artifacts import extract_package
from shared.chain import commit_artifact, wallet
from shared.checkpoints import sha256_file
from shared.config import Policy, Settings
from shared.http_logging import private_http
from shared.uploads import UploadGrant, UploadRequest


def _upload_endpoint(base: str) -> str:
    try:
        parsed = urlsplit(base)
        hostname = parsed.hostname or ""
        loopback = hostname == "localhost"
        try:
            loopback = loopback or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            pass
        # Explicit localhost/loopback configuration permits a local development API.
        if (
            not hostname
            or not (parsed.scheme == "https" or parsed.scheme == "http" and loopback)
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Invalid upload API URL")
        _ = parsed.port  # Reject malformed ports before signing or opening a connection.
    except ValueError:
        raise ValueError(
            "Configure an HTTPS upload API URL (HTTP is allowed on loopback)"
        ) from None
    return base.rstrip("/") + "/uploads"


def _freeze(package: Path, destination: Path, max_bytes: int) -> None:
    size = 0
    with package.open("rb") as source, destination.open("xb") as output:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > max_bytes:
                raise ValueError("Compressed package exceeds upload size limit")
            output.write(chunk)


def submit(
    settings: Settings,
    package: Path,
    *,
    publish: bool = True,
    max_checkpoint_bytes: int | None = None,
    client: httpx.Client | None = None,
) -> str:
    endpoint = _upload_endpoint(settings.upload_api_url)
    if publish and settings.chain != "bittensor":
        raise ValueError("File chain mode cannot publish commitments; use --upload-only")
    if max_checkpoint_bytes is not None and (
        isinstance(max_checkpoint_bytes, bool)
        or not isinstance(max_checkpoint_bytes, int)
        or max_checkpoint_bytes < 1024
    ):
        raise ValueError("Maximum checkpoint bytes must be an integer of at least 1024")
    checkpoint_limit = (
        max_checkpoint_bytes
        if max_checkpoint_bytes is not None
        else max(
            (target.policy.max_checkpoint_bytes for target in settings.targets),
            default=Policy().max_checkpoint_bytes,
        )
    )
    with tempfile.TemporaryDirectory() as directory:
        frozen = Path(directory) / "submission.tar.gz"
        _freeze(package, frozen, settings.upload_max_bytes)
        digest = sha256_file(frozen)
        manifest = extract_package(
            frozen,
            Path(directory) / "checkpoint",
            digest,
            checkpoint_limit,
        )
        target = next(
            (
                target
                for target in settings.targets
                if target.model_id == manifest.target_id
                and target.revision == manifest.target_revision
                and manifest.method in target.methods
            ),
            None,
        )
        if settings.targets and target is None:
            raise ValueError("Package target/method does not match a configured target")
        if (
            target
            and sum(file.size for file in manifest.files) > target.policy.max_checkpoint_bytes
        ):
            raise ValueError("Package checkpoint exceeds configured size limit")
        size = frozen.stat().st_size
        hotkey = wallet(settings).hotkey
        authorization = UploadRequest(
            hotkey=hotkey.ss58_address,
            network=settings.network,
            netuid=settings.netuid,
            competition_id=manifest.competition_id,
            artifact=digest,
            size=size,
            timestamp=int(time.time()),
            nonce=secrets.token_hex(16),
            signature="0" * 128,
        )
        authorization = authorization.model_copy(
            update={"signature": bytes(hotkey.sign(authorization.signing_bytes())).hex()}
        )
        session_context = (
            nullcontext(client)
            if client is not None
            else httpx.Client(timeout=httpx.Timeout(120, connect=20), follow_redirects=False)
        )
        with session_context as session:
            try:
                response = session.post(
                    endpoint,
                    json=authorization.model_dump(mode="json"),
                    follow_redirects=False,
                )
                response.raise_for_status()
                grant = UploadGrant.model_validate(response.json())
                required_headers = {
                    "content-type": "application/gzip",
                    "content-length": str(size),
                    "x-amz-meta-prexon-competition-id": manifest.competition_id,
                }
                if (
                    grant.artifact != digest
                    or grant.expires_at <= time.time()
                    or {key.lower(): value for key, value in grant.headers.items()}
                    != required_headers
                ):
                    raise ValueError("Invalid upload grant")
            except (httpx.HTTPError, ValidationError, ValueError):
                raise RuntimeError("Private upload authorization failed") from None
            with private_http():
                try:
                    with frozen.open("rb") as stream:
                        response = session.put(
                            grant.url,
                            headers=grant.headers,
                            content=stream,
                            follow_redirects=False,
                        )
                    response.raise_for_status()
                except (httpx.HTTPError, httpx.InvalidURL):
                    # HTTP errors include request URLs; capability URLs must stay private.
                    raise RuntimeError(
                        "Private artifact upload failed; no commitment was sent"
                    ) from None
        if publish:
            commit_artifact(settings, digest)
        return digest
