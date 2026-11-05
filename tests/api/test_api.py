import hashlib
import json

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from shared.artifacts import ArtifactStore
from shared.config import Settings
from shared.contracts import DashboardSnapshot, PublicEvaluation, PublicSubmission, PublicWinner
from tests.factories import evidence
from tests.integration.test_controller import setup_pipeline
from validator.dashboard import publish_snapshot
from validator.publication import publish_winners


def publish_pipeline_winners(controller, state, store, chain, target, now):
    publish_winners(
        state,
        store,
        ArtifactStore(controller.settings.public_artifact_root),
        controller.settings.public_artifact_base_url,
        now,
    )
    publish_snapshot(
        state,
        controller.settings.dashboard_path,
        (target,),
        chain.current_owners(),
        {},
        now,
        100,
    )


def test_end_to_end_public_views_never_include_private_job_configuration(tmp_path):
    controller, state, store, chain, provider, target = setup_pipeline(tmp_path)
    controller.tick(21600)
    job = provider.jobs[0]
    result = evidence(job).model_copy(update={"started_at": 21601, "finished_at": 21602})
    store.write_bytes(f"jobs/{job.job_id}/result.json", result.model_dump_json().encode())
    controller.tick(21603)
    publish_pipeline_winners(controller, state, store, chain, target, 21603)
    client = TestClient(create_app(controller.settings.dashboard_path))
    targets = client.get("/api/v1/targets").json()
    assert targets["items"][0]["model_id"] == target.model_id
    assert targets["updated_at"] == 21603
    assert client.get("/api/v1/winners").json()["items"][0]["eligible"] is True
    report = client.get(f"/api/v1/evaluations/{job.job_id}")
    assert report.status_code == 200
    assert report.json()["score"] == 2
    public = json.dumps(client.get("/api/v1/snapshot").json())
    for private in (
        "candidate_url",
        "qualification_url",
        "prompts",
        "hidden",
        "secret",
        "wallet",
        "job.json",
    ):
        assert private not in public
    assert client.get("/api/v1/evaluations/unknown").status_code == 404


def test_published_reports_remain_available_outside_recent_snapshot(tmp_path):
    controller, state, store, chain, provider, target = setup_pipeline(tmp_path)
    controller.tick(21600)
    job = provider.jobs[0]
    result = evidence(job).model_copy(update={"started_at": 21601, "finished_at": 21602})
    store.write_bytes(f"jobs/{job.job_id}/result.json", result.model_dump_json().encode())
    controller.tick(21603)
    publish_pipeline_winners(controller, state, store, chain, target, 21603)
    path = controller.settings.dashboard_path
    snapshot = DashboardSnapshot.model_validate_json(path.read_bytes())
    path.write_text(snapshot.model_copy(update={"evaluations": ()}).model_dump_json())
    client = TestClient(create_app(path))
    assert client.get(f"/api/v1/evaluations/{job.job_id}").status_code == 200


def test_legacy_snapshot_details_without_verified_releases_are_hidden(tmp_path):
    submission = PublicSubmission(
        submission_id="a" * 64,
        competition_id="b" * 64,
        artifact="c" * 64,
        hotkey="private-pending-miner",
        uid=1,
        block=9,
        status="rejected",
        reason="private-similarity-evidence",
    )
    evaluation = PublicEvaluation(
        job_id="d" * 32,
        submission_id=submission.submission_id,
        competition_id=submission.competition_id,
        outcome="lost",
        reason="private-failure",
    )
    winner = PublicWinner(
        submission_id=submission.submission_id,
        competition_id=submission.competition_id,
        artifact=submission.artifact,
        hotkey=submission.hotkey,
        uid=1,
        promoted_at=1,
        expires_at=2,
        eligible=False,
    )
    path = tmp_path / "dashboard.json"
    path.write_text(
        DashboardSnapshot(
            updated_at=100,
            finalized_block=9,
            submissions=(submission,),
            evaluations=(evaluation,),
            winners=(winner,),
        ).model_dump_json()
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / f"{evaluation.job_id}.json").write_text(evaluation.model_dump_json())
    client = TestClient(create_app(path))
    public = client.get("/api/v1/snapshot").json()
    assert public["submissions"] == []
    assert public["evaluations"] == []
    assert public["winners"] == []
    assert client.get(f"/api/v1/evaluations/{evaluation.job_id}").status_code == 404


def test_legacy_snapshot_with_a_winner_still_hides_nonwinner_rows_and_private_urls(tmp_path):
    from tests.validator.test_publication import winning_state

    state, private, public, profile, job, package = winning_state(tmp_path)
    publish_winners(state, private, public, "", 202)
    path = tmp_path / "dashboard" / "snapshot.json"
    snapshot = publish_snapshot(state, path, (profile,), {1: "winner"}, {}, 203, 10)
    loser = PublicSubmission(
        submission_id="e" * 64,
        competition_id=profile.identity,
        artifact="f" * 64,
        hotkey="private-losing-miner",
        uid=2,
        block=11,
        status="lost",
        reason="private-similarity-details",
    )
    losing_report = PublicEvaluation(
        job_id="d" * 32,
        submission_id=loser.submission_id,
        competition_id=profile.identity,
        outcome="lost",
        reason="private-storage-error",
    )
    leaked_url = "https://private.example/winner?credential=secret"
    legacy = snapshot.model_copy(
        update={
            "submissions": (*snapshot.submissions, loser),
            "evaluations": (*snapshot.evaluations, losing_report),
            "winners": (snapshot.winners[0].model_copy(update={"checkpoint_url": leaked_url}),),
        }
    )
    path.write_text(legacy.model_dump_json())
    (path.parent / "reports" / f"{losing_report.job_id}.json").write_text(
        losing_report.model_dump_json()
    )
    client = TestClient(create_app(path))
    response = client.get("/api/v1/snapshot")
    data = response.json()
    assert {s["hotkey"] for s in data["submissions"]} == {"winner"}
    assert {e["submission_id"] for e in data["evaluations"]} == {job.commitment.identity}
    assert data["winners"][0]["checkpoint_url"] == snapshot.releases[0].checkpoint_url
    assert "private" not in response.text
    assert client.get(f"/api/v1/evaluations/{losing_report.job_id}").status_code == 404


def release_client(tmp_path, *, checkpoint_url=None):
    from shared.releases import WinnerRelease

    contents = b"the exact verified winner package"
    digest = hashlib.sha256(contents).hexdigest()
    release = WinnerRelease(
        submission_id="a" * 64,
        source_artifact=digest,
        artifact=digest,
        checkpoint_url=checkpoint_url or f"/api/v1/winners/{'a' * 64}/checkpoint",
        published_at=1,
    )
    path = tmp_path / "dashboard.json"
    path.write_text(
        DashboardSnapshot(updated_at=2, finalized_block=1, releases=(release,)).model_dump_json()
    )
    root = tmp_path / "public-artifacts"
    artifact = root / "winners" / f"{digest}.tar.gz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(contents)
    app = create_app(
        settings=Settings(role="api", dashboard_path=path, public_artifact_root=str(root))
    )
    return TestClient(app), path, artifact, contents


@pytest.mark.parametrize("prefix", ["/api/v1", "/v1"])
def test_archived_winner_download_is_exact_and_unknown_files_are_not_served(tmp_path, prefix):
    client, path, artifact, contents = release_client(tmp_path)
    response = client.get(f"{prefix}/winners/{'a' * 64}/checkpoint")
    assert response.status_code == 200
    assert response.content == contents
    assert response.headers["x-artifact-sha256"] == hashlib.sha256(contents).hexdigest()
    assert client.get(f"/api/v1/winners/{'b' * 64}/checkpoint").status_code == 404


def test_persisted_legacy_winner_url_stays_downloadable_on_both_routes(tmp_path):
    url = f"/v1/winners/{'a' * 64}/checkpoint"
    client, path, artifact, contents = release_client(tmp_path, checkpoint_url=url)
    for endpoint in (url, f"/api{url}"):
        response = client.get(endpoint)
        assert response.status_code == 200
        assert response.content == contents


def test_corrupted_public_checkpoint_is_denied(tmp_path):
    client, path, artifact, contents = release_client(tmp_path)
    artifact.write_bytes(b"changed after release")
    assert client.get(f"/api/v1/winners/{'a' * 64}/checkpoint").status_code == 404


def test_public_checkpoint_symlink_cannot_escape_public_directory(tmp_path):
    client, path, artifact, contents = release_client(tmp_path)
    private = tmp_path / "private-package"
    private.write_bytes(contents)
    artifact.unlink()
    artifact.symlink_to(private)
    assert client.get(f"/api/v1/winners/{'a' * 64}/checkpoint").status_code == 404


def test_public_https_release_redirects_without_storage_access(tmp_path):
    digest = hashlib.sha256(b"the exact verified winner package").hexdigest()
    url = f"https://checkpoints.example/public/winners/{digest}.tar.gz"
    client, path, artifact, contents = release_client(tmp_path, checkpoint_url=url)
    artifact.unlink()
    response = client.get(f"/api/v1/winners/{'a' * 64}/checkpoint", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == url


@pytest.mark.parametrize("field,value", [("artifact", "b" * 64), ("source_artifact", "c" * 64)])
def test_tampered_public_release_digest_is_unavailable(tmp_path, field, value):
    client, path, artifact, contents = release_client(tmp_path)
    data = json.loads(path.read_text())
    data["releases"][0][field] = value
    path.write_text(json.dumps(data))
    assert client.get(f"/api/v1/winners/{'a' * 64}/checkpoint").status_code == 503
