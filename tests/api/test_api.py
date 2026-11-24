from fastapi.testclient import TestClient

from api.main import create_app
from shared.config import Settings


def test_api_offers_upload_authorization_and_nothing_else(tmp_path):
    """Public data is read from public storage; this service only authorizes uploads."""
    settings = Settings(role="api", upload_state_path=tmp_path / "uploads.sqlite3")
    client = TestClient(create_app(settings=settings))
    assert client.get("/api/health").json() == {"status": "ok"}
    assert set(client.get("/api/openapi.json").json()["paths"]) == {
        "/api/health",
        "/api/v1/uploads",
    }
    for path in (
        "/api/v1/snapshot",
        "/api/v1/status",
        "/api/v1/targets",
        "/api/v1/submissions",
        "/api/v1/evaluations",
        "/api/v1/winners",
        "/api/v1/emissions",
        "/api/v1/models",
        f"/api/v1/winners/{'a' * 64}/checkpoint",
        # Every address starts with /api: there is no second, shorter spelling.
        "/health",
    ):
        assert client.get(path).status_code == 404, path
    # Uploads are off until they are configured, and say so in a retryable way.
    assert client.post("/api/v1/uploads", json={}).status_code == 400
    assert client.post("/v1/uploads", json={}).status_code == 404
