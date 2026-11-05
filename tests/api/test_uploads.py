"""Signed upload authorization uses real keys and replaces only storage boundaries."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from bittensor_wallet import Keypair
from fastapi.testclient import TestClient

from api.main import create_app
from shared.artifacts import ArtifactStore
from shared.config import Settings

NOW = 1_800_000_000
ALICE = Keypair.create_from_uri("//Alice")
BOB = Keypair.create_from_uri("//Bob")
COMPETITION = "a" * 64
ARTIFACT = "b" * 64


class Presigner:
    def __init__(self):
        self.calls = []

    def generate_presigned_url(self, operation, **kwargs):
        self.calls.append((operation, kwargs))
        return "https://s3.hippius.com/private/staging?secret=capability"


def request_body(*, key=ALICE, nonce="0" * 32, **changes):
    from shared.uploads import UploadRequest

    request = UploadRequest(
        hotkey=key.ss58_address,
        network="finney",
        netuid=17,
        competition_id=COMPETITION,
        artifact=ARTIFACT,
        size=4096,
        timestamp=NOW,
        nonce=nonce,
        signature="0" * 128,
    ).model_copy(update=changes)
    return {
        **request.model_dump(mode="json", warnings=False),
        "signature": key.sign(request.signing_bytes()).hex(),
    }


def write_policy(settings, **changes):
    from shared.uploads import UploadPolicy, UploadTarget

    policy = UploadPolicy(
        updated_at=NOW,
        network="finney",
        netuid=17,
        hotkeys=(ALICE.ss58_address,),
        targets=(UploadTarget(competition_id=COMPETITION, max_package_bytes=8192),),
    ).model_copy(update=changes)
    settings.upload_policy_path.parent.mkdir(parents=True, exist_ok=True)
    settings.upload_policy_path.write_text(policy.model_dump_json())


@pytest.fixture
def upload_env(tmp_path):
    from api.uploads import UploadService

    settings = Settings(
        role="api",
        uploads_enabled=True,
        upload_staging_root="s3://private-staging/prefix",
        upload_state_path=tmp_path / "api.sqlite3",
        upload_policy_path=tmp_path / "private/policy.json",
        dashboard_path=tmp_path / "public/dashboard.json",
    )
    write_policy(settings)
    presigner = Presigner()
    store = ArtifactStore(settings.upload_staging_root, s3_client=presigner)
    clock = [NOW]

    def app():
        service = UploadService(settings, store=store, clock=lambda: clock[0])
        return TestClient(create_app(settings=settings, upload_service=service))

    return SimpleNamespace(
        settings=settings, presigner=presigner, store=store, clock=clock, app=app
    )


def test_upload_contract_canonical_signing_and_safe_key():
    from shared.uploads import UploadRequest, submission_key

    body = request_body()
    request = UploadRequest.model_validate(body)
    expected = {key: value for key, value in body.items() if key != "signature"}
    assert (
        request.signing_bytes()
        == b"prexon-upload-v1\n"
        + json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    )
    assert Keypair(ss58_address=request.hotkey).verify(
        request.signing_bytes(), bytes.fromhex(request.signature)
    )
    normalized = UploadRequest.model_validate({**body, "signature": "0x" + body["signature"]})
    assert normalized.signature == request.signature
    key = submission_key("finney", 17, ALICE.ss58_address, ARTIFACT)
    assert key == (
        f"submissions/{hashlib.sha256(b'finney').hexdigest()[:16]}/17/"
        f"{hashlib.sha256(ALICE.ss58_address.encode()).hexdigest()}/{ARTIFACT}"
    )
    assert ".." not in submission_key("../evil", 17, "../evil", ARTIFACT)
    with pytest.raises(ValueError):
        submission_key("finney", -1, ALICE.ss58_address, ARTIFACT)
    with pytest.raises(ValueError):
        submission_key("finney", 17, ALICE.ss58_address, "../bad")


def test_signed_upload_grants_exact_scoped_put_headers(upload_env):
    from shared.uploads import submission_key

    response = upload_env.app().post("/api/v1/uploads", json=request_body())
    assert response.status_code == 200, response.text
    grant = response.json()
    assert grant == {
        "url": "https://s3.hippius.com/private/staging?secret=capability",
        "headers": {
            "Content-Type": "application/gzip",
            "Content-Length": "4096",
            "x-amz-meta-prexon-competition-id": COMPETITION,
        },
        "expires_at": NOW + upload_env.settings.upload_url_ttl_seconds,
        "artifact": ARTIFACT,
    }
    assert upload_env.presigner.calls == [
        (
            "put_object",
            {
                "Params": {
                    "Bucket": "private-staging",
                    "Key": "prefix/" + submission_key("finney", 17, ALICE.ss58_address, ARTIFACT),
                    "ContentType": "application/gzip",
                    "ContentLength": 4096,
                    "Metadata": {"prexon-competition-id": COMPETITION},
                },
                "ExpiresIn": upload_env.settings.upload_url_ttl_seconds,
                "HttpMethod": "PUT",
            },
        )
    ]
    assert not (upload_env.settings.upload_state_path.parent / "validator.sqlite3").exists()


@pytest.mark.parametrize("policy_change", [{"updated_at": NOW - 301}, {"updated_at": NOW + 301}])
def test_stale_or_future_policy_is_retryable_without_storage(upload_env, policy_change):
    write_policy(upload_env.settings, **policy_change)
    response = upload_env.app().post("/api/v1/uploads", json=request_body())
    assert response.status_code == 503
    assert response.headers["retry-after"]
    assert upload_env.presigner.calls == []


@pytest.mark.parametrize("policy_bytes", [None, b"not json"])
def test_missing_or_invalid_private_policy_is_retryable(upload_env, policy_bytes):
    if policy_bytes is None:
        upload_env.settings.upload_policy_path.unlink()
    else:
        upload_env.settings.upload_policy_path.write_bytes(policy_bytes)
    response = upload_env.app().post("/api/v1/uploads", json=request_body())
    assert response.status_code == 503
    assert upload_env.presigner.calls == []


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"network": "test"}, 403),
        ({"netuid": 18}, 403),
        ({"competition_id": "c" * 64}, 403),
        ({"timestamp": NOW - 61}, 401),
        ({"timestamp": NOW + 61}, 401),
        ({"size": 8193}, 413),
        ({"size": 0}, 400),
        ({"size": -1}, 400),
    ],
)
def test_invalid_scope_freshness_or_size_never_presigns(upload_env, change, status):
    response = upload_env.app().post("/api/v1/uploads", json=request_body(**change))
    assert response.status_code == status, response.text
    assert upload_env.presigner.calls == []


def test_upload_requires_registered_hotkey_and_ownership(upload_env):
    client = upload_env.app()
    response = client.post("/api/v1/uploads", json=request_body(key=BOB))
    assert response.status_code == 403
    body = request_body()
    body["signature"] = BOB.sign(b"unrelated").hex()
    response = client.post("/api/v1/uploads", json=body)
    assert response.status_code == 401
    assert upload_env.presigner.calls == []


@pytest.mark.parametrize("field", ["size", "artifact", "timestamp", "nonce"])
def test_every_signed_field_is_bound(upload_env, field):
    body = request_body()
    body[field] = {"size": 4097, "artifact": "c" * 64, "timestamp": NOW - 1, "nonce": "1" * 32}[
        field
    ]
    response = upload_env.app().post("/api/v1/uploads", json=body)
    assert response.status_code == 401
    assert upload_env.presigner.calls == []


def test_replay_survives_api_restart_and_is_hotkey_scoped(upload_env):
    body = request_body()
    assert upload_env.app().post("/api/v1/uploads", json=body).status_code == 200
    assert upload_env.app().post("/api/v1/uploads", json=body).status_code == 409
    write_policy(upload_env.settings, hotkeys=(ALICE.ss58_address, BOB.ss58_address))
    assert upload_env.app().post("/api/v1/uploads", json=request_body(key=BOB)).status_code == 200
    assert len(upload_env.presigner.calls) == 2


def test_hourly_quota_is_durable_and_recovers_after_window(upload_env):
    for nonce in range(4):
        response = upload_env.app().post(
            "/api/v1/uploads", json=request_body(nonce=f"{nonce:032x}")
        )
        assert response.status_code == 200
    response = upload_env.app().post("/api/v1/uploads", json=request_body(nonce="5" * 32))
    assert response.status_code == 429
    assert response.headers["retry-after"]
    assert len(upload_env.presigner.calls) == 4
    upload_env.clock[0] += 3601
    write_policy(upload_env.settings, updated_at=upload_env.clock[0])
    response = upload_env.app().post(
        "/api/v1/uploads", json=request_body(nonce="6" * 32, timestamp=upload_env.clock[0])
    )
    assert response.status_code == 200
    # Old nonces remain consumed even after the hourly rate-limit window.
    response = upload_env.app().post(
        "/api/v1/uploads", json=request_body(nonce="0" * 32, timestamp=upload_env.clock[0])
    )
    assert response.status_code == 409


def test_global_byte_limit_caps_target_policy(upload_env):
    from api.uploads import UploadService

    settings = upload_env.settings.model_copy(update={"upload_max_bytes": 2048})
    service = UploadService(settings, store=upload_env.store, clock=lambda: NOW)
    response = TestClient(create_app(settings=settings, upload_service=service)).post(
        "/api/v1/uploads", json=request_body()
    )
    assert response.status_code == 413
    assert upload_env.presigner.calls == []


def test_real_local_hippius_presign_binds_exact_size_type_and_competition(upload_env, monkeypatch):
    from urllib.parse import parse_qs, urlsplit

    from api.uploads import UploadService

    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.hippius.com")
    monkeypatch.setenv("S3_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("S3_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("S3_REGION", "decentralized")
    service = UploadService(upload_env.settings, clock=lambda: NOW)
    client = TestClient(create_app(settings=upload_env.settings, upload_service=service))
    response = client.post("/api/v1/uploads", json=request_body())
    assert response.status_code == 200, response.text
    capability = urlsplit(response.json()["url"])
    assert capability.hostname == "s3.hippius.com"
    assert capability.path.startswith("/private-staging/prefix/submissions/")
    query = parse_qs(capability.query)
    assert query["X-Amz-SignedHeaders"] == [
        "content-length;content-type;host;x-amz-meta-prexon-competition-id"
    ]
    assert response.json()["headers"]["x-amz-meta-prexon-competition-id"] == COMPETITION
    assert query["X-Amz-Expires"] == ["900"]
    assert query["X-Amz-Credential"][0].endswith("/decentralized/s3/aws4_request")
    assert "test-secret-key" not in response.text


def test_failed_presign_does_not_consume_nonce_or_quota(upload_env, monkeypatch):
    original = upload_env.presigner.generate_presigned_url

    def failed(*args, **kwargs):
        raise RuntimeError("backend credentials must never be returned")

    monkeypatch.setattr(upload_env.presigner, "generate_presigned_url", failed)
    client = upload_env.app()
    response = client.post("/api/v1/uploads", json=request_body())
    assert response.status_code == 503
    assert "credentials" not in response.text
    monkeypatch.setattr(upload_env.presigner, "generate_presigned_url", original)
    assert client.post("/api/v1/uploads", json=request_body()).status_code == 200


def test_parallel_replays_have_only_one_grant(upload_env):
    from concurrent.futures import ThreadPoolExecutor

    client = upload_env.app()
    body = request_body()
    with ThreadPoolExecutor(max_workers=6) as executor:
        statuses = list(
            executor.map(lambda _: client.post("/api/v1/uploads", json=body).status_code, range(6))
        )
    assert sorted(statuses) == [200, 409, 409, 409, 409, 409]
    assert len(upload_env.presigner.calls) == 1


def test_parallel_requests_cannot_exceed_hotkey_quota(upload_env):
    from concurrent.futures import ThreadPoolExecutor

    client = upload_env.app()

    def issue(nonce):
        return client.post("/api/v1/uploads", json=request_body(nonce=f"{nonce:032x}")).status_code

    with ThreadPoolExecutor(max_workers=6) as executor:
        statuses = list(executor.map(issue, range(6)))
    assert sorted(statuses) == [200, 200, 200, 200, 429, 429]
    assert len(upload_env.presigner.calls) == 4


def test_policy_refresh_applies_to_existing_api_process(upload_env):
    client = upload_env.app()
    assert client.post("/api/v1/uploads", json=request_body()).status_code == 200
    write_policy(upload_env.settings, hotkeys=())
    response = client.post("/api/v1/uploads", json=request_body(nonce="1" * 32))
    assert response.status_code == 403
    assert len(upload_env.presigner.calls) == 1


def test_tampering_with_registered_hotkey_or_allowed_competition_invalidates_signature(upload_env):
    from shared.uploads import UploadTarget

    write_policy(
        upload_env.settings,
        hotkeys=(ALICE.ss58_address, BOB.ss58_address),
        targets=(
            UploadTarget(competition_id=COMPETITION, max_package_bytes=8192),
            UploadTarget(competition_id="c" * 64, max_package_bytes=8192),
        ),
    )
    client = upload_env.app()
    for field, value in [("hotkey", BOB.ss58_address), ("competition_id", "c" * 64)]:
        body = request_body()
        body[field] = value
        assert client.post("/api/v1/uploads", json=body).status_code == 401
    assert upload_env.presigner.calls == []
