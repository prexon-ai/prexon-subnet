"""Upload authorization for miners. Public data is read from public storage, not from here."""

from __future__ import annotations

import argparse

from fastapi import FastAPI

from api.uploads import UploadService, upload_routes
from shared.config import Settings, load_settings


def create_app(
    *, settings: Settings | None = None, upload_service: UploadService | None = None
) -> FastAPI:
    app = FastAPI(
        title="Prexon upload API",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    settings = settings or load_settings("api")
    app.include_router(upload_routes(upload_service or UploadService(settings)), prefix="/api")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Prexon private upload API")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    settings = load_settings("api", args.env_file)
    uvicorn.run(create_app(settings=settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
