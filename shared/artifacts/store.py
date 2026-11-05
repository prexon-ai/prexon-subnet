"""Local or Hippius S3 content storage; workers use scoped HTTP URLs only."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from shared.artifacts.hippius import hippius_environment
from shared.checkpoints import sha256_file
from shared.http_logging import private_http


def download_url(url: str, output: Path, max_bytes: int, *, allow_file: bool = False) -> None:
    parsed = urlparse(url)
    output.parent.mkdir(parents=True, exist_ok=True)
    if parsed.scheme == "file" and allow_file and not parsed.netloc:
        source = Path(unquote(parsed.path))
        if source.stat().st_size > max_bytes:
            raise ValueError("Artifact exceeds download size limit")
        shutil.copyfile(source, output)
        return
    if parsed.scheme != "https":
        raise ValueError("Artifact URLs must use HTTPS")
    size = 0
    with private_http():
        try:
            with httpx.stream("GET", url, timeout=httpx.Timeout(120, connect=20)) as response:
                response.raise_for_status()
                with output.open("wb") as stream:
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise ValueError("Artifact exceeds download size limit")
                        stream.write(chunk)
        except httpx.HTTPError:
            raise RuntimeError("Private artifact download failed") from None


class ArtifactStore:
    def __init__(
        self, root: str, *, s3_client: Any = None, environment: Mapping[str, str] | None = None
    ) -> None:
        self.root = root
        parsed = urlparse(root)
        self.bucket = parsed.netloc if parsed.scheme == "s3" else None
        self.prefix = parsed.path.strip("/") if self.bucket else ""
        self.client = s3_client
        if self.bucket and self.client is None:
            import boto3
            from botocore.config import Config

            env = hippius_environment(environment if environment is not None else os.environ)
            kwargs = {
                name: env[key]
                for name, key in {
                    "aws_access_key_id": "S3_ACCESS_KEY_ID",
                    "aws_secret_access_key": "S3_SECRET_ACCESS_KEY",
                    "aws_session_token": "S3_SESSION_TOKEN",
                    "region_name": "S3_REGION",
                }.items()
                if env.get(key)
            }
            self.client = boto3.client(
                "s3",
                endpoint_url=env["S3_ENDPOINT_URL"],
                config=Config(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                ),
                **kwargs,
            )
        if parsed.scheme and parsed.scheme not in {"s3"}:
            raise ValueError("Artifact root must be a local directory or s3:// URI")

    def _key(self, key: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9/_\-.]+", key) or ".." in key or key.startswith("/"):
            raise ValueError("Unsafe storage key")
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_file(self, path: Path) -> str:
        digest = sha256_file(path)
        self.write_file(f"objects/{digest}", path)
        return digest

    def put_bytes(self, data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        self.write_bytes(f"objects/{digest}", data)
        return digest

    def write_file(self, key: str, source: Path) -> None:
        full = self._key(key)
        if self.bucket:
            self.client.upload_file(str(source), self.bucket, full)
        else:
            output = Path(self.root) / full
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temp:
                temp_path = Path(temp.name)
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, temp)
            temp_path.replace(output)

    def write_bytes(self, key: str, data: bytes) -> None:
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(data)
            temp.flush()
            self.write_file(key, Path(temp.name))

    def read_bytes(self, key: str, max_bytes: int = 32 * 1024**2) -> bytes:
        full = self._key(key)
        if self.bucket:
            response = self.client.get_object(Bucket=self.bucket, Key=full)
            stream = response["Body"]
            try:
                if response["ContentLength"] > max_bytes:
                    raise ValueError("Object exceeds size limit")
                data: bytes = stream.read(max_bytes + 1)
            finally:
                stream.close()
        else:
            path = Path(self.root) / full
            if path.stat().st_size > max_bytes:
                raise ValueError("Object exceeds size limit")
            data = path.read_bytes()
        if len(data) > max_bytes:
            raise ValueError("Object exceeds size limit")
        return data

    def exists(self, key: str) -> bool:
        full = self._key(key)
        if not self.bucket:
            return (Path(self.root) / full).is_file()
        try:
            self.client.head_object(Bucket=self.bucket, Key=full)
            return True
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in {"404", "NoSuchKey"}:
                return False
            raise

    def url(self, key: str, *, write: bool = False, expires: int = 3600) -> str:
        full = self._key(key)
        if self.bucket:
            return str(
                self.client.generate_presigned_url(
                    "put_object" if write else "get_object",
                    Params={"Bucket": self.bucket, "Key": full},
                    ExpiresIn=expires,
                )
            )
        return (Path(self.root) / full).resolve().as_uri()
