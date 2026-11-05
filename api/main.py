from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI

from api.queries import PublishedViews
from api.releases import release_routes
from api.routes import dashboard_routes
from api.uploads import UploadService, upload_routes
from shared.config import Settings, load_settings


def create_app(
    snapshot_path: Path | None = None,
    *,
    settings: Settings | None = None,
    upload_service: UploadService | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Prexon dashboard API",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    settings = settings or load_settings("api")
    path = snapshot_path or settings.dashboard_path
    views = PublishedViews(path)
    public_root = (
        None
        if urlsplit(settings.public_artifact_root).scheme
        else Path(settings.public_artifact_root)
    )
    for router in (
        dashboard_routes(views),
        release_routes(views, public_root),
        upload_routes(upload_service or UploadService(settings)),
    ):
        app.include_router(router, prefix="/api")
        app.include_router(router, include_in_schema=False)

    @app.get("/api/health")
    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Prexon dashboard and private upload API")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    settings = load_settings("api", args.env_file)
    uvicorn.run(create_app(settings=settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
