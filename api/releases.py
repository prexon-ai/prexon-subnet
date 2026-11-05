"""Winner downloads use only the public release manifest and public artifact directory."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.responses import FileResponse, RedirectResponse, Response

from api.queries import PublishedViews
from shared.checkpoints import sha256_file
from shared.releases import public_release_key


def release_routes(views: PublishedViews, artifact_root: Path | None) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.get("/winners/{submission_id}/checkpoint")
    def checkpoint(submission_id: str) -> Response:
        try:
            release = views.release(submission_id)
        except (OSError, ValueError) as exc:
            raise HTTPException(503, "Published release view is unavailable") from exc
        if release is None:
            raise HTTPException(404, "Published winner checkpoint not found")
        if release.checkpoint_url.startswith("https://"):
            return RedirectResponse(release.checkpoint_url, status_code=307)
        if artifact_root is None:
            raise HTTPException(404, "Published winner checkpoint not found")
        root = artifact_root.resolve()
        path = (root / public_release_key(release.source_artifact)).resolve()
        try:
            verified = (
                path.is_relative_to(root)
                and path.is_file()
                and sha256_file(path) == release.artifact
            )
        except OSError:
            verified = False
        if not verified:
            raise HTTPException(404, "Published winner checkpoint not found")
        return FileResponse(
            path,
            media_type="application/gzip",
            filename=f"{release.artifact}.tar.gz",
            headers={"X-Artifact-SHA256": release.artifact},
        )

    return router
