"""Authorize private Hippius PUT capabilities without admitting validator jobs."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from shared.artifacts import ArtifactStore
from shared.uploads import UploadGrant, UploadPolicy, UploadRequest, submission_key
from validator.config import (
    S3_ACCESS_KEY_ID,
    S3_SECRET_ACCESS_KEY,
    UPLOAD_MAX_BYTES,
    UPLOAD_MAX_REQUESTS_PER_HOUR,
    UPLOAD_POLICY_MAX_AGE_SECONDS,
    UPLOAD_POLICY_PATH,
    UPLOAD_REQUEST_MAX_AGE_SECONDS,
    UPLOAD_STAGING_ROOT,
    UPLOAD_STATE_PATH,
    UPLOAD_URL_TTL_SECONDS,
    UPLOADS_ENABLED,
)


class UploadService:
    def __init__(
        self,
        *,
        store: ArtifactStore | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.clock = clock

    @staticmethod
    def unavailable() -> HTTPException:
        return HTTPException(
            503, "Private uploads temporarily unavailable", headers={"Retry-After": "30"}
        )

    def _check_backend(self) -> None:
        parsed = urlsplit(UPLOAD_STAGING_ROOT)
        if (
            not UPLOADS_ENABLED
            or parsed.scheme != "s3"
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or ".." in parsed.path
        ):
            raise self.unavailable()

    def _storage(self) -> ArtifactStore:
        if self.store is None:
            self.store = ArtifactStore(
                UPLOAD_STAGING_ROOT,
                access_key=S3_ACCESS_KEY_ID,
                secret_key=S3_SECRET_ACCESS_KEY,
            )
        if not self.store.bucket:
            raise self.unavailable()
        return self.store

    def _policy(self, now: float) -> UploadPolicy:
        try:
            with UPLOAD_POLICY_PATH.open("rb") as stream:
                raw = stream.read(4 * 1024**2 + 1)
            if len(raw) > 4 * 1024**2:
                raise ValueError("Policy exceeds size limit")
            policy = UploadPolicy.model_validate_json(raw)
        except (OSError, ValueError):
            raise self.unavailable() from None
        if not 0 <= now - policy.updated_at <= UPLOAD_POLICY_MAX_AGE_SECONDS:
            raise self.unavailable()
        return policy

    def issue(self, request: UploadRequest) -> UploadGrant:
        self._check_backend()
        now = self.clock()
        policy = self._policy(now)
        if request.network != policy.network or request.netuid != policy.netuid:
            raise HTTPException(403, "Upload request does not match this subnet")
        if request.hotkey not in policy.hotkeys:
            raise HTTPException(403, "Hotkey is not registered on this subnet")
        target = next(
            (
                target
                for target in policy.targets
                if target.competition_id == request.competition_id
            ),
            None,
        )
        if target is None:
            raise HTTPException(403, "Competition does not accept uploads")
        if request.revision != target.revision:
            raise HTTPException(
                403, "Package is for another revision of this competition; git pull competitions/"
            )
        if request.size > min(target.max_package_bytes, UPLOAD_MAX_BYTES):
            raise HTTPException(413, "Package exceeds upload size limit")
        if not 0 <= now - request.timestamp <= UPLOAD_REQUEST_MAX_AGE_SECONDS:
            raise HTTPException(401, "Upload authorization timestamp is stale or in the future")
        try:
            from bittensor_wallet import Keypair

            verified = Keypair(ss58_address=request.hotkey).verify(
                request.signing_bytes(), bytes.fromhex(request.signature)
            )
        except (ValueError, TypeError):
            verified = False
        if not verified:
            raise HTTPException(401, "Invalid hotkey signature")

        # BEGIN IMMEDIATE serializes quota and replay checks across workers/restarts.
        # A failed local presign rolls the reservation back, allowing an honest retry.
        connection: sqlite3.Connection | None = None
        try:
            UPLOAD_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(UPLOAD_STATE_PATH, timeout=10)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS upload_requests ("
                "hotkey TEXT NOT NULL, nonce TEXT NOT NULL, issued_at REAL NOT NULL, "
                "PRIMARY KEY (hotkey, nonce))"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS upload_requests_window "
                "ON upload_requests (hotkey, issued_at)"
            )
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM upload_requests WHERE hotkey=? AND nonce=?",
                (request.hotkey, request.nonce),
            ).fetchone():
                raise HTTPException(409, "Upload nonce has already been used")
            count, oldest = connection.execute(
                "SELECT COUNT(*), MIN(issued_at) FROM upload_requests "
                "WHERE hotkey=? AND issued_at>?",
                (request.hotkey, now - 3600),
            ).fetchone()
            if count >= UPLOAD_MAX_REQUESTS_PER_HOUR:
                retry = max(1, int(oldest + 3600 - now) + 1)
                raise HTTPException(
                    429, "Hourly upload quota exceeded", headers={"Retry-After": str(retry)}
                )
            store = self._storage()
            key = store._key(
                submission_key(request.network, request.netuid, request.hotkey, request.artifact)
            )
            ttl = UPLOAD_URL_TTL_SECONDS
            url = store.client.generate_presigned_url(
                "put_object",
                Params={
                    "Bucket": store.bucket,
                    "Key": key,
                    "ContentType": "application/gzip",
                    "ContentLength": request.size,
                    "Metadata": {"prexon-competition-id": request.competition_id},
                },
                ExpiresIn=ttl,
                HttpMethod="PUT",
            )
            grant = UploadGrant(
                url=url,
                headers={
                    "Content-Type": "application/gzip",
                    "Content-Length": str(request.size),
                    "x-amz-meta-prexon-competition-id": request.competition_id,
                },
                expires_at=now + ttl,
                artifact=request.artifact,
            )
            connection.execute(
                "INSERT INTO upload_requests(hotkey, nonce, issued_at) VALUES (?, ?, ?)",
                (request.hotkey, request.nonce, now),
            )
            connection.commit()
            return grant
        except HTTPException:
            raise
        except Exception:
            # Never expose endpoint credentials, signatures or capability URLs in errors.
            raise self.unavailable() from None
        finally:
            if connection is not None:
                connection.close()


def upload_routes(service: UploadService) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/uploads", response_model=UploadGrant)
    def authorize_upload(body: dict[str, Any]) -> UploadGrant:
        try:
            request = UploadRequest.model_validate(body)
        except ValidationError:
            raise HTTPException(400, "Malformed upload authorization") from None
        return service.issue(request)

    return router
