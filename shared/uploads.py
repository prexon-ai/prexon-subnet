"""Portable contracts for private, hotkey-authorized submission uploads."""

from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlsplit

from pydantic import Field, field_validator

from shared.config import Digest, FrozenModel, Positive


class UploadTarget(FrozenModel):
    competition_id: Digest
    max_package_bytes: int = Field(gt=0, strict=True)


class UploadPolicy(FrozenModel):
    updated_at: Positive
    network: str = Field(min_length=1, max_length=512)
    netuid: int = Field(ge=0, strict=True)
    hotkeys: tuple[str, ...]
    targets: tuple[UploadTarget, ...]


class UploadRequest(FrozenModel):
    hotkey: str = Field(min_length=1, max_length=128)
    network: str = Field(min_length=1, max_length=512)
    netuid: int = Field(ge=0, strict=True)
    competition_id: Digest
    artifact: Digest
    size: int = Field(gt=0, strict=True)
    timestamp: int = Field(gt=0, strict=True)
    nonce: str = Field(pattern=r"^[0-9a-f]{32}$")
    signature: str = Field(pattern=r"^[0-9a-f]{128}$")

    @field_validator("signature", mode="before")
    @classmethod
    def normalize_signature(cls, value: object) -> object:
        return value.removeprefix("0x") if isinstance(value, str) else value

    def signing_bytes(self) -> bytes:
        payload = self.model_dump(mode="json", exclude={"signature"})
        return b"prexon-upload-v1\n" + json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


class UploadGrant(FrozenModel):
    url: str = Field(min_length=1)
    headers: dict[str, str]
    expires_at: Positive
    artifact: Digest

    @field_validator("url")
    @classmethod
    def require_https(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Upload capability URLs must use HTTPS")
        if parsed.fragment:
            raise ValueError("Upload capability URLs cannot contain fragments")
        return value


def submission_key(network: str, netuid: int, hotkey: str, artifact: str) -> str:
    """Derive a fixed private object key without trusting miner-provided path text."""
    if not network or not hotkey:
        raise ValueError("Network and hotkey are required")
    if isinstance(netuid, bool) or not isinstance(netuid, int) or netuid < 0:
        raise ValueError("Invalid subnet identifier")
    if not re.fullmatch(r"[0-9a-f]{64}", artifact):
        raise ValueError("Expected a SHA-256 artifact digest")
    network_hash = hashlib.sha256(network.encode()).hexdigest()[:16]
    hotkey_hash = hashlib.sha256(hotkey.encode()).hexdigest()
    return f"submissions/{network_hash}/{netuid}/{hotkey_hash}/{artifact}"
