"""Exercise the role boundaries with real signing, packages and local persistence."""

import time
from types import SimpleNamespace

import httpx
from bittensor_wallet import Keypair
from fastapi.testclient import TestClient

from api.main import create_app
from api.uploads import UploadService
from miner.submit import submit
from shared.artifacts import ArtifactStore
from shared.config import Settings
from shared.contracts import Commitment
from shared.uploads import submission_key
from tests.factories import evidence
from tests.integration.test_controller import setup_pipeline
from validator.chain import ChainSnapshot
from validator.controller import Controller


def test_signed_upload_chain_similarity_queue_and_winner_release(tmp_path, monkeypatch):
    controller, state, canonical, chain, provider, target = setup_pipeline(tmp_path)
    hotkey = Keypair.create_from_uri("//Alice")
    now = int(time.time())
    owners = {1: hotkey.ss58_address, 0: "burn"}
    chain.commitments = ()
    chain.current_owners = lambda: owners
    chain.scan = lambda *_: ChainSnapshot(block=100, owners=owners, commitments=chain.commitments)
    staging = ArtifactStore(str(tmp_path / "staging"))
    settings = controller.settings.model_copy(update={"upload_staging_root": staging.root})
    controller = Controller(settings, state, canonical, chain, provider)
    controller.tick(now)
    # The candidate must arrive through staging; canonical storage has no candidate yet.
    for object_path in (tmp_path / "artifacts/objects").iterdir():
        if object_path.name != target.qualification_artifact:
            object_path.unlink()

    class Presigner:
        def generate_presigned_url(self, operation, **kwargs):
            assert operation == "put_object"
            return "https://s3.hippius.com/staging/" + kwargs["Params"]["Key"]

    api_settings = Settings(
        role="api",
        uploads_enabled=True,
        upload_staging_root="s3://staging",
        upload_state_path=tmp_path / "api/uploads.sqlite3",
        upload_policy_path=settings.upload_policy_path,
        dashboard_path=settings.dashboard_path,
        public_artifact_root=settings.public_artifact_root,
    )
    api = TestClient(
        create_app(
            settings=api_settings,
            upload_service=UploadService(
                api_settings,
                store=ArtifactStore("s3://staging", s3_client=Presigner()),
                clock=time.time,
            ),
        )
    )
    uploaded = []

    def transport(request):
        if request.method == "POST":
            response = api.post(
                request.url.path,
                content=request.content,
                headers={"Content-Type": "application/json"},
            )
            assert response.status_code == 200, response.text
            return httpx.Response(response.status_code, json=response.json())
        assert request.method == "PUT"
        data = request.read()
        assert int(request.headers["Content-Length"]) == len(data)
        staging.write_bytes(request.url.path.removeprefix("/staging/"), data)
        uploaded.append(True)
        return httpx.Response(200)

    def commit(_settings, artifact):
        assert uploaded == [True]
        chain.commitments = (
            Commitment(
                netuid=settings.netuid,
                block=99,
                block_hash="finalized",
                hotkey=hotkey.ss58_address,
                uid=1,
                artifact=artifact,
                submitted_at=now,
            ),
        )

    monkeypatch.setattr("miner.submit.wallet", lambda _: SimpleNamespace(hotkey=hotkey))
    monkeypatch.setattr("miner.submit.commit_artifact", commit)
    miner_settings = settings.model_copy(
        update={
            "role": "miner",
            "chain": "bittensor",
            "upload_api_url": "https://prexon.ai/api/v1",
        }
    )
    digest = submit(
        miner_settings,
        tmp_path / "head.tar.gz",
        client=httpx.Client(transport=httpx.MockTransport(transport)),
    )
    assert staging.exists(
        submission_key(settings.network, settings.netuid, hotkey.ss58_address, digest)
    )
    assert not canonical.exists(f"objects/{digest}")
    controller.tick(now + 1)
    assert state.ready_count(target.identity) == 1
    assert state.similarity_verdict(chain.commitments[0].identity)["outcome"] == "pass"
    assert canonical.exists(f"objects/{digest}")
    assert api.get("/api/v1/submissions").json()["items"] == []
    assert provider.rents == []
    controller.tick(now + 21600)
    job = provider.jobs[0]
    result = evidence(job).model_copy(
        update={"started_at": now + 21601, "finished_at": now + 21602}
    )
    canonical.write_bytes(f"jobs/{job.job_id}/result.json", result.model_dump_json().encode())
    controller.tick(now + 21603)
    assert len(state.awards()) == 1
    winner = api.get("/api/v1/winners").json()["items"][0]
    release = api.get(winner["checkpoint_url"])
    assert release.status_code == 200
    assert release.content == (tmp_path / "head.tar.gz").read_bytes()
    assert "s3.hippius.com" not in api.get("/api/v1/snapshot").text
    state.close()
