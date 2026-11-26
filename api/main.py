"""Upload authorization for miners. Public data is read from public storage, not from here."""

from __future__ import annotations

import argparse

from fastapi import FastAPI

from api.uploads import UploadService, upload_routes
from validator.config import API_HOST, API_PORT


def create_app(*, upload_service: UploadService | None = None) -> FastAPI:
    app = FastAPI(
        title="Prexon upload API",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )
    app.include_router(upload_routes(upload_service or UploadService()), prefix="/api")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Prexon private upload API")
    parser.add_argument("--host", help="Override API_HOST, e.g. 0.0.0.0 in a container")
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host or API_HOST, port=API_PORT)


if __name__ == "__main__":
    main()
