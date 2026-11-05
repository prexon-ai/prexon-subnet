"""Public, immutable metadata for validator-promoted checkpoint releases."""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import model_validator

from shared.config import Digest, FrozenModel, Positive


def public_release_key(source_artifact: str) -> str:
    return f"winners/{source_artifact}.tar.gz"


class WinnerRelease(FrozenModel):
    submission_id: Digest
    source_artifact: Digest
    artifact: Digest
    checkpoint_url: str
    published_at: Positive

    @model_validator(mode="after")
    def verified_identity(self) -> WinnerRelease:
        if self.artifact != self.source_artifact:
            raise ValueError("Published package digest must equal the evaluated source digest")
        parsed = urlparse(self.checkpoint_url)
        local_path = f"/v1/winners/{self.submission_id}/checkpoint"
        if self.checkpoint_url in {f"/api{local_path}", local_path}:
            return self
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or "\\" in self.checkpoint_url
            or not parsed.path.endswith(f"/{public_release_key(self.source_artifact)}")
            or ".." in parsed.path.split("/")
        ):
            raise ValueError("Release URL must be a public HTTPS object or its local winner route")
        return self
