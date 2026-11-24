import gzip
import hashlib
import io
import json
import struct
import tarfile

import pytest

from shared.artifacts import ArtifactStore, package_checkpoint
from shared.checkpoints import checkpoint_schema
from shared.config import Settings
from shared.contracts import Commitment, Qualification, SubmissionManifest
from tests.factories import PrunedChainNode, evidence, make_checkpoint, provenance, target
from validator.chain import BittensorChain, ChainSnapshot
from validator.controller import Controller
from validator.state import State


class TestChain:
    __test__ = False

    def __init__(self, commitments):
        self.commitments = tuple(commitments)
        self.published = []

    def scan(self, start_block=None, known=()):
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
    assert chain.published[-1] == {1: 1.0}
    assert state.awards()[0].promoted_at == 21603


def test_expired_session_deletes_pod_and_keeps_unfinished_queue(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    expired = 21600 + controller.settings.session_seconds + 1
    controller.tick(expired)
    assert len(provider.deleted) == 1
    # A session that reached no verdict is not followed by another rental at once.
    assert len(provider.rents) == 1
    assert state.ready_count(t.identity) == 1
    controller.tick(expired + controller.settings.session_retry_seconds)
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


def _raw_package(path, target, files, extra=()):
    """Archive arbitrary bytes under a manifest that matches them exactly."""
    manifest = SubmissionManifest(
        target_id=target.model_id,
        target_revision=target.revision,
        competition_id=target.identity,
        files=[
            {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            for name, data in sorted(files)
        ],
        provenance=provenance(),
    )
    members = [("manifest.json", manifest.model_dump_json().encode()), *extra, *files]
    with path.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            for name, data in members:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("defect", ["nested_config", "aliased_member", "unhashable_dtype"])
def test_malformed_package_is_terminally_rejected_instead_of_retried(tmp_path, defect):
    controller, state, store, chain, provider, target = setup_pipeline(tmp_path)
    config = (tmp_path / "head/config.json").read_bytes()
    weights = (tmp_path / "head/model.safetensors").read_bytes()
    extra = ()
    if defect == "nested_config":
        config = b"[" * 100_000
    elif defect == "aliased_member":
        extra = (("config.json/", config),)
    else:
        header = json.dumps(
            {"weight": {"dtype": ["F32"], "shape": [2], "data_offsets": [0, 8]}}
        ).encode()
        weights = struct.pack("<Q", len(header)) + header + b"\0" * 8
    package = tmp_path / "malformed.tar.gz"
    _raw_package(package, target, [("config.json", config), ("model.safetensors", weights)], extra)
    digest = store.put_file(package)
    c = Commitment(netuid=1, block=99, block_hash="hash", hotkey="miner", uid=1, artifact=digest)
    chain.commitments = (c,)
    controller.tick(21600)
    # A package that can never be valid must not be downloaded again on every round.
    assert state.pending_admissions() == []
    assert provider.rents == []


def test_scanning_and_weights_continue_after_a_reveal_outlives_node_history(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    digest = chain.commitments[0].artifact
    node = PrunedChainNode(head=100, reveals={"miner": ((99, f"px1:{digest}"),)})
    live = Controller(
        controller.settings,
        state,
        store,
        BittensorChain(controller.settings, client=node, signer=object()),
        provider,
    )
    live.tick(1000)
    assert state.ready_count(t.identity) == 1
    for head, now in ((140, 2000), (180, 3000)):  # Block 99 leaves the node's 50-block history.
        node.head = head
        live.tick(now)
    assert state.metadata("finalized_block") == "180"
    assert len(node.weights) == 3


def test_unusable_evaluations_stop_after_the_attempt_limit(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    now = 21600.0
    for _ in range(40):  # every evaluation of this head comes back unusable
        controller.tick(now)
        running = state.active_sessions() and state.running_job(state.active_sessions()[0]["id"])
        if running:
            unusable = evidence(running).model_copy(
                update={"status": "inconclusive", "error_code": "RuntimeError"}
            )
            store.write_bytes(
                f"jobs/{running.job_id}/result.json", unusable.model_dump_json().encode()
            )
        now += 120
    assert len(provider.rents) == controller.settings.max_evaluation_attempts == 3
    assert len(provider.deleted) == 3
    assert state.submissions()[0]["status"] == "stalled"
    assert not state.active_sessions()
    assert state.awards() == []


def test_job_that_never_reports_is_given_up_after_the_attempt_limit(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    now = 21600.0
    while now < 21600 + 24 * 3600:  # the worker never writes a result
        controller.tick(now)
        now += 60
    assert len(provider.rents) == 3
    row = state.submissions()[0]
    assert (row["status"], row["reason"]) == ("stalled", "evaluation_timeout")
    assert not state.active_sessions()


def test_nothing_private_about_a_job_reaches_public_storage(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    j = provider.jobs[0]
    result = evidence(j).model_copy(update={"started_at": 21601, "finished_at": 21602})
    store.write_bytes(f"jobs/{j.job_id}/result.json", result.model_dump_json().encode())
    controller.tick(21603)
    published = sorted((tmp_path / "public").rglob("*.json"))
    assert {path.relative_to(tmp_path / "public/artifacts").as_posix() for path in published} == {
        "dashboard/snapshot.json",
        "dashboard/index.json",
        "dashboard/models/model1-aaaaaaa.json",
    }
    snapshot = json.loads((tmp_path / "public/artifacts/dashboard/snapshot.json").read_text())
    assert snapshot["targets"][0]["model_id"] == t.model_id
    assert [e["score"] for e in snapshot["evaluations"]] == [2]
    for path in published:
        text = path.read_text()
        for private in (
            "candidate_url",
            "qualification_url",
            "prompts",
            "hidden",
            "secret",
            "wallet",
            "job.json",
        ):
            assert private not in text, (path.name, private)


def test_tick_publishes_model_history_and_keeps_it_while_the_chain_is_unreachable(tmp_path):
    controller, state, store, chain, provider, t = setup_pipeline(tmp_path)
    controller.tick(21600)
    j = provider.jobs[0]
    result = evidence(j).model_copy(update={"started_at": 21601, "finished_at": 21602})
    store.write_bytes(f"jobs/{j.job_id}/result.json", result.model_dump_json().encode())
    controller.tick(21603)
    # The website reads these from public storage, never from this machine.
    public = ArtifactStore(controller.settings.public_artifact_root)
    index = json.loads(public.read_bytes("dashboard/index.json"))
    (model,) = index["models"]
    assert (model["winners"], model["reward_shares"], model["active_from"]) == (1, [1.0], 21600)
    history = json.loads(public.read_bytes(f"dashboard/models/{model['id']}.json"))
    assert [(s["hotkey"], s["crowned"], s["share"]) for s in history["submissions"]] == [
        ("miner", True, 1.0)
    ]
    snapshot = json.loads(public.read_bytes("dashboard/snapshot.json"))
    assert [w["share"] for w in snapshot["winners"]] == [1.0]

    def offline(*args, **kwargs):
        raise TimeoutError("rpc unreachable")

    chain.scan = offline
    controller.tick(21700)
    # Who owns each UID is unknown while the chain is unreachable, so the last
    # published files stay instead of showing every reward as gone.
    assert json.loads(public.read_bytes("dashboard/index.json")) == index
    assert json.loads(public.read_bytes("dashboard/snapshot.json")) == snapshot
