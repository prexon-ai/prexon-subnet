import json

from shared.artifacts import ArtifactStore, package_checkpoint
from shared.checkpoints import checkpoint_schema
from shared.config import Settings
from shared.contracts import Commitment, Qualification
from tests.factories import evidence, make_checkpoint, provenance, target
from validator.chain import ChainSnapshot
from validator.controller import Controller
from validator.state import State


class TestChain:
    __test__ = False

    def __init__(self, commitments):
        self.commitments = tuple(commitments)
        self.published = []

    def scan(self, start_block=None):
        return ChainSnapshot(
            block=100, owners={1: "miner", 0: "burn"}, commitments=self.commitments
        )

    def publish(self, weights):
        self.published.append(weights)

    def current_owners(self):
        return {1: "miner", 0: "burn"}


class RentalBoundary:
    def __init__(self, store):
        self.store = store
        self.rents = []
        self.deleted = []
        self.jobs = []

    def build_request(self, session, target):
        return {"termination_hours": 1, "id": session["id"]}

    def rent(self, session_id, request):
        self.rents.append(session_id)
        return "pod-" + session_id

    def status(self, pod_id):
        return {"status": "RUNNING", "spend_to_date": 0}

    def launch(self, pod_id, job, job_url, result_url):
        self.jobs.append(job)

    def delete(self, pod_id):
        self.deleted.append(pod_id)


def setup_pipeline(tmp_path, t=None):
    store = ArtifactStore(str(tmp_path / "artifacts"))
    t = t or target()
    head = make_checkpoint(tmp_path / "head")
    q = Qualification(
        profile_id=t.qualification_identity,
        engine_version=t.engine_version,
        approved_by="operator",
        evidence_description="Reviewed exact runtime verification",
        target_preserving_verification_reviewed=True,
        checkpoint_schemas=(checkpoint_schema(head),),
    )
    q_hash = store.put_bytes(q.model_dump_json().encode())
    t = t.model_copy(update={"qualification_artifact": q_hash})
    package = tmp_path / "head.tar.gz"
    package_checkpoint(
        head,
        package,
        target_id=t.model_id,
        target_revision=t.revision,
        competition_id=t.identity,
        provenance=provenance(),
    )
    digest = store.put_file(package)
    commitment = Commitment(
        netuid=1,
        block=99,
        block_hash="finalized",
        hotkey="miner",
        uid=1,
        artifact=digest,
        submitted_at=0,
    )
    settings = Settings(
        role="validator",
        targets=(t,),
        state_path=tmp_path / "state.sqlite",
        artifact_root=store.root,
        dashboard_path=tmp_path / "public/dashboard.json",
        chain="file",
        provider="local",
        schedule_seconds=21600,
        dataset_cache=tmp_path / "datasets",
        public_artifact_root=str(tmp_path / "public/artifacts"),
        upload_policy_path=tmp_path / "private/upload-policy.json",
    )
    state = State(settings.state_path)
    chain = TestChain([commitment])
    provider = RentalBoundary(store)
    return Controller(settings, state, store, chain, provider), state, store, chain, provider, t


def test_chain_to_winner_and_recovery_is_idempotent(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    assert len(provider.rents) == 1
    assert len(provider.jobs) == 1
    j = provider.jobs[0]
    result = evidence(j).model_copy(update={"started_at": 21601, "finished_at": 21602})
    store.write_bytes(f"jobs/{j.job_id}/result.json", result.model_dump_json().encode())
    controller.tick(21603)
    assert state.incumbent(t) == j.commitment.artifact
    assert len(state.awards()) == 1
    controller.tick(22201)
    assert len(state.awards()) == 1
    assert len(provider.rents) == 1
    assert provider.deleted
    assert chain.published[-1][1] == 0.2
    assert state.awards()[0].promoted_at == 21603


def test_expired_session_deletes_pod_and_keeps_unfinished_queue(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    controller.tick(21600 + controller.settings.session_seconds + 1)
    assert len(provider.deleted) == 1
    # Its original six-hour deadline remains due after an infrastructure retry.
    assert state.ready_count(t.identity) == 0
    assert len(provider.rents) == 2
    assert state.incumbent(t) is None
    assert len(state.active_sessions()) == 1
    assert state.submissions()[0]["submitted_at"] == 0


def test_incompatible_checkpoint_is_terminally_rejected_before_gpu_rental(tmp_path):
    controller, state, store, chain, provider, target = setup_pipeline(tmp_path)
    # A valid generic DFlash2 archive, but not one of the qualified architectures.
    config_path = tmp_path / "head/config.json"
    config = json.loads(config_path.read_text())
    config["vocab_size"] = 16
    config_path.write_text(json.dumps(config))
    package = tmp_path / "incompatible.tar.gz"
    package_checkpoint(
        tmp_path / "head",
        package,
        target_id=target.model_id,
        target_revision=target.revision,
        competition_id=target.identity,
        provenance=provenance(),
    )
    digest = store.put_file(package)
    c = Commitment(netuid=1, block=99, block_hash="hash", hotkey="miner", uid=1, artifact=digest)
    chain.commitments = (c,)
    controller.tick(21600)
    assert provider.rents == []
    assert state.ready_count(target.identity) == 0
    assert state.pending_admissions() == []
