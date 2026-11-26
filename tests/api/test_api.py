import sys

from fastapi.testclient import TestClient

from api.main import create_app, main


def test_api_offers_upload_authorization_and_nothing_else(tmp_path):
    """Public data is read from public storage; this service only authorizes uploads."""
    client = TestClient(create_app())
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


def test_main_uses_config_port_and_host_override(tmp_path, monkeypatch, set_config):
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda app, *, host, port: calls.append((host, port)))
    set_config(API_HOST="127.0.0.2", API_PORT=9001)
    monkeypatch.setattr(sys, "argv", ["prexon-api"])
    main()
    monkeypatch.setattr(sys, "argv", ["prexon-api", "--host", "0.0.0.0"])
    main()
    assert calls == [("127.0.0.2", 9001), ("0.0.0.0", 9001)]
