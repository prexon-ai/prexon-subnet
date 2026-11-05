import json

import pytest

from shared.config import Settings, load_settings


def profile(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "engine_version": "0.5.23",
                "worker_image": "example/worker@sha256:" + "a" * 64,
                "hardware": {"gpu_type": "H100", "gpu_count": 1, "min_vram_gb": 80},
                "workloads": [
                    {
                        "name": "low",
                        "concurrency": 1,
                        "requests": 2,
                        "output_tokens": 8,
                        "weight": 1,
                        "prompts": ["Hello"],
                    }
                ],
            }
        )
    )
    return path


def test_two_allocations_and_profile_identity(tmp_path, monkeypatch):
    p = profile(tmp_path)
    for slot, allocation in [(1, 20), (2, 80)]:
        for name, value in {
            "ID": f"org/target{slot}",
            "REVISION": "a" * 40,
            "PROFILE": str(p),
            "ALLOCATION": str(allocation),
        }.items():
            monkeypatch.setenv(f"PREXON_MODEL_{slot}_{name}", value)
    settings = load_settings("validator", None)
    assert [t.allocation for t in settings.targets] == [20, 80]
    assert settings.targets[0].identity != settings.targets[1].identity
    original = settings.targets[0].identity
    monkeypatch.setenv("LIUM_API_KEY", "private")
    assert load_settings("validator", None).targets[0].identity == original
    monkeypatch.setenv("PREXON_MODEL_2_ALLOCATION", "70")
    with pytest.raises(ValueError, match="100"):
        load_settings("validator", None)


def test_api_loads_upload_settings_without_validator_or_target_configuration(monkeypatch):
    monkeypatch.setenv("PREXON_UPLOADS_ENABLED", "true")
    monkeypatch.setenv("PREXON_UPLOAD_STAGING_ROOT", "s3://private-staging/prexon")
    monkeypatch.setenv("PREXON_UPLOAD_POLICY_PATH", "runtime/private/upload-policy.json")
    monkeypatch.setenv("PREXON_MODEL_1_ID", "not-installed")
    monkeypatch.setenv("PREXON_SESSION_SECONDS", "invalid-for-validator")
    monkeypatch.setenv("PREXON_POLL_SECONDS", "invalid-for-validator")
    monkeypatch.setenv("PREXON_MODEL_1_PROFILE", "/missing")
    settings = load_settings("api", None)
    assert settings.uploads_enabled
    assert settings.upload_staging_root == "s3://private-staging/prexon"
    assert settings.upload_policy_path.as_posix() == "runtime/private/upload-policy.json"
    assert settings.targets == ()


@pytest.mark.parametrize(
    ("staging", "canonical", "public"),
    [
        ("s3://same/staging", "s3://same/canonical", "s3://winners"),
        ("s3://same/staging", "s3://private", "s3://same/winners"),
        ("s3://staging", "s3://same/private", "s3://same/winners"),
    ],
)
def test_private_stores_and_public_releases_require_separate_buckets(staging, canonical, public):
    with pytest.raises(ValueError, match="separate"):
        Settings(
            role="validator",
            upload_staging_root=staging,
            artifact_root=canonical,
            public_artifact_root=public,
        )


def test_local_staging_cannot_overlap_canonical_store(tmp_path):
    with pytest.raises(ValueError, match="separate"):
        Settings(
            role="validator",
            artifact_root=str(tmp_path / "private"),
            upload_staging_root=str(tmp_path / "private/staging"),
            public_artifact_root=str(tmp_path / "public"),
        )


def test_api_cannot_presign_uploads_into_public_release_bucket():
    with pytest.raises(ValueError, match="separate"):
        Settings(
            role="api",
            upload_staging_root="s3://public/staging",
            public_artifact_root="s3://public/winners",
        )


@pytest.mark.parametrize(
    "root",
    [
        "s3://bucket/path?secret=x",
        "s3://user:password@bucket",
        "s3:///missing",
        "s3://bucket/../public",
    ],
)
def test_storage_roots_reject_ambiguous_object_addresses(root):
    with pytest.raises(ValueError, match="storage"):
        Settings(role="validator", upload_staging_root=root)


@pytest.mark.parametrize(
    "url",
    [
        "http://checkpoints.example",
        "https://user:secret@example",
        "https://checkpoints.example?token=secret",
    ],
)
def test_public_release_base_must_be_stable_https(url):
    with pytest.raises(ValueError, match="release base"):
        Settings(role="validator", public_artifact_base_url=url)


def test_evaluation_configuration_ignores_unrelated_validator_settings(monkeypatch):
    from shared.config import load_settings

    monkeypatch.setenv("PREXON_NETUID", "invalid-validator-setting")
    monkeypatch.setenv("PREXON_POLL_SECONDS", "invalid-validator-setting")
    monkeypatch.setenv("PREXON_SESSION_SECONDS", "60")
    settings = load_settings("evaluation", None)
    assert settings.role == "evaluation" and settings.session_seconds == 60
    assert settings.targets == ()
