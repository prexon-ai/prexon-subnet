import hashlib
import time
from types import SimpleNamespace

import httpx
import pytest
from bittensor_wallet import Keypair

from miner.submit import submit
from shared.artifacts import package_checkpoint
from shared.config import Settings
from shared.uploads import UploadRequest
from tests.factories import make_checkpoint, provenance, target


@pytest.fixture
def submission(tmp_path, monkeypatch):
    configured = target()
    head = make_checkpoint(tmp_path / "head")
    package = tmp_path / "head.tar.gz"
    package_checkpoint(
        head,
        package,
        target_id=configured.model_id,
        target_revision=configured.revision,
        competition_id=configured.identity,
        provenance=provenance(),
    )
    settings = Settings(
        role="miner",
        targets=(configured,),
        upload_api_url="https://uploads.example.org/api/v1",
        network="finney",
        netuid=17,
    )
    signer = Keypair.create_from_uri("//Alice")
    # Wallet access is the signing boundary; signatures themselves are real.
    monkeypatch.setattr(
        "miner.submit.wallet", lambda _: SimpleNamespace(hotkey=signer), raising=False
    )
    calls = []
    requests = []
    grant_changes = {}
    state = SimpleNamespace(
        package=package,
        settings=settings,
        signer=signer,
        calls=calls,
        requests=requests,
        original=package.read_bytes(),
        put_status=200,
        grant_changes=grant_changes,
        mutate_on_authorization=False,
        reject_commit=False,
        authorize_url="https://uploads.example.org/api/v1/uploads",
    )

    def transport(request):
        requests.append(request)
        if request.method == "POST":
            calls.append("authorize")
            assert str(request.url) == state.authorize_url
            authorization = UploadRequest.model_validate_json(request.content)
            assert authorization.hotkey == signer.ss58_address
            assert authorization.network == settings.network
            assert authorization.netuid == settings.netuid
            assert authorization.competition_id == configured.identity
            assert authorization.size == len(state.original)
            assert authorization.artifact == hashlib.sha256(state.original).hexdigest()
            assert Keypair(ss58_address=authorization.hotkey).verify(
                authorization.signing_bytes(), bytes.fromhex(authorization.signature)
            )
            if state.mutate_on_authorization:
                package.write_bytes(b"training overwrote the original file")
            return httpx.Response(
                200,
                json={
                    "url": "https://private.example.org/object?secret=capability",
                    "headers": {
                        "Content-Type": "application/gzip",
                        "Content-Length": str(len(state.original)),
                        "x-amz-meta-prexon-competition-id": configured.identity,
                    },
                    "expires_at": time.time() + 900,
                    "artifact": authorization.artifact,
                    **grant_changes,
                },
            )
        assert request.method == "PUT"
        calls.append("upload")
        assert request.headers["Content-Type"] == "application/gzip"
        assert request.headers["Content-Length"] == str(len(state.original))
        assert request.headers["x-amz-meta-prexon-competition-id"] == configured.identity
        state.uploaded = request.read()
        assert state.uploaded == state.original
        return httpx.Response(state.put_status)

    def commit(settings, digest):
        calls.append("commit")
        assert state.uploaded == state.original
        assert digest == hashlib.sha256(state.uploaded).hexdigest()
        if state.reject_commit:
            raise RuntimeError("chain rejected")

    monkeypatch.setattr("miner.submit.commit_artifact", commit)
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        state.client = client
        yield state


def test_upload_is_signed_and_completed_before_chain_commit(submission):
    digest = submit(submission.settings, submission.package, client=submission.client)
    assert digest == hashlib.sha256(submission.original).hexdigest()
    assert submission.calls == ["authorize", "upload", "commit"]


def test_upload_only_signs_authorization_without_chain_transaction(submission):
    submit(submission.settings, submission.package, publish=False, client=submission.client)
    assert submission.calls == ["authorize", "upload"]


def test_manifest_is_sufficient_to_submit_without_configured_targets(submission):
    settings = submission.settings.model_copy(update={"targets": ()})
    digest = submit(settings, submission.package, client=submission.client)
    assert digest == hashlib.sha256(submission.original).hexdigest()
    assert submission.calls == ["authorize", "upload", "commit"]


def test_frozen_validated_bytes_survive_training_overwrite(submission):
    submission.mutate_on_authorization = True
    digest = submit(submission.settings, submission.package, client=submission.client)
    assert digest == hashlib.sha256(submission.original).hexdigest()
    assert submission.package.read_bytes() != submission.uploaded
    assert submission.calls == ["authorize", "upload", "commit"]


@pytest.mark.parametrize("status", [403, 500, 307])
def test_failed_or_redirected_put_never_commits_or_exposes_capability(submission, status):
    submission.put_status = status
    with pytest.raises(RuntimeError) as exc:
        submit(submission.settings, submission.package, client=submission.client)
    assert "capability" not in str(exc.value)
    assert "private.example.org" not in str(exc.value)
    assert submission.calls == ["authorize", "upload"]


def test_failed_chain_submission_leaves_uploaded_artifact_available_for_retry(submission):
    submission.reject_commit = True
    with pytest.raises(RuntimeError, match="chain rejected"):
        submit(submission.settings, submission.package, client=submission.client)
    assert submission.calls == ["authorize", "upload", "commit"]
    assert submission.uploaded == submission.package.read_bytes()


@pytest.mark.parametrize(
    "url",
    [
        "",
        "http://remote.example.org/api/v1",
        "ftp://localhost/api/v1",
        "https://user:secret@uploads.org/api/v1",
        "https://uploads.org:bad/api/v1",
        "https://uploads.org/api/v1?secret=value",
        "https://uploads.org/api/v1#fragment",
    ],
)
def test_missing_or_insecure_remote_api_is_rejected_before_network(submission, url):
    settings = submission.settings.model_copy(update={"upload_api_url": url})
    with pytest.raises(ValueError, match="upload API"):
        submit(settings, submission.package, client=submission.client)
    assert submission.calls == []


@pytest.mark.parametrize(
    "grant_change",
    [
        {"artifact": "c" * 64},
        {"url": "http://private.example.org/object"},
        {"expires_at": 1},
        {"headers": {"Content-Type": "application/gzip", "Content-Length": "1"}},
    ],
)
def test_invalid_grant_never_uploads_or_commits(submission, grant_change):
    submission.grant_changes.update(grant_change)
    with pytest.raises(RuntimeError, match="upload"):
        submit(submission.settings, submission.package, client=submission.client)
    assert submission.calls == ["authorize"]


@pytest.mark.parametrize("competition", [None, "f" * 64])
def test_missing_or_different_competition_binding_never_uploads_or_commits(submission, competition):
    headers = {
        "Content-Type": "application/gzip",
        "Content-Length": str(len(submission.original)),
    }
    if competition is not None:
        headers["x-amz-meta-prexon-competition-id"] = competition
    submission.grant_changes["headers"] = headers
    with pytest.raises(RuntimeError, match="upload"):
        submit(submission.settings, submission.package, client=submission.client)
    assert submission.calls == ["authorize"]


@pytest.mark.parametrize("configured_targets", [True, False])
def test_invalid_package_is_rejected_before_wallet_or_authorization(
    submission, monkeypatch, configured_targets
):
    submission.package.write_bytes(b"invalid tarball")

    def forbidden_wallet(settings):
        raise AssertionError("Invalid archives must not open the wallet")

    monkeypatch.setattr("miner.submit.wallet", forbidden_wallet)
    settings = submission.settings
    if not configured_targets:
        settings = settings.model_copy(update={"targets": ()})
    with pytest.raises((ValueError, OSError)):
        submit(settings, submission.package, client=submission.client)
    assert submission.calls == []


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
def test_explicit_loopback_api_supports_local_development(submission, host):
    settings = submission.settings.model_copy(
        update={"upload_api_url": f"http://{host}:8000/api/v1/"}
    )

    def transport(request):
        assert str(request.url) == f"http://{host}:8000/api/v1/uploads"
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(RuntimeError, match="upload"):
            submit(settings, submission.package, publish=False, client=client)
    assert submission.calls == []


def test_explicit_checkpoint_bound_rejects_oversized_valid_checkpoint(submission):
    head = make_checkpoint(submission.package.parent / "larger-head")
    # Valid config data can be larger than the smallest permitted extraction bound.
    config = head / "config.json"
    config.write_text(config.read_text().rstrip() + " " * 2048)
    package_checkpoint(
        head,
        submission.package,
        target_id=submission.settings.targets[0].model_id,
        target_revision=submission.settings.targets[0].revision,
        competition_id=submission.settings.targets[0].identity,
        provenance=provenance(),
    )
    settings = submission.settings.model_copy(update={"targets": ()})
    with pytest.raises(ValueError, match="size limit"):
        submit(settings, submission.package, max_checkpoint_bytes=1024, client=submission.client)
    assert submission.calls == []


def test_configured_profile_match_still_required_with_explicit_checkpoint_bound(submission):
    configured = submission.settings.targets[0].model_copy(update={"revision": "another-revision"})
    settings = submission.settings.model_copy(update={"targets": (configured,)})
    with pytest.raises(ValueError, match="configured target"):
        submit(settings, submission.package, max_checkpoint_bytes=1024, client=submission.client)
    assert submission.calls == []


def test_http_client_logging_does_not_expose_upload_capability(submission, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="httpx"):
        submit(submission.settings, submission.package, client=submission.client)
        logging.getLogger("httpx").info("normal requests remain visible")
    assert "normal requests remain visible" in caplog.text
    assert "capability" not in caplog.text
    assert "private.example.org" not in caplog.text
