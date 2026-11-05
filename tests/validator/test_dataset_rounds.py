import pytest

from evaluation.datasets.sampling import select_round
from shared.artifacts import ArtifactStore
from shared.config import Settings
from shared.contracts import Decision
from shared.datasets import RoundContext
from tests.evaluation.test_datasets import dataset_target, prepared_pool
from tests.factories import evidence
from tests.integration.test_controller import RentalBoundary
from tests.validator.test_state import admit
from validator.chain import ChainSnapshot
from validator.controller import Controller
from validator.state import State


class Chain:
    height = 110
    finalized_block = None

    def scan(self, start_block=None):
        return ChainSnapshot(
            block=self.height,
            finalized_block=self.finalized_block,
            commitments=(),
            owners={1: "miner1", 2: "miner2"},
        )

    def block_hash(self, block):
        assert block <= self.height
        return "0x" + f"{block:064x}"

    def finalized_height(self):
        return self.height

    def publish(self, weights):
        pass

    def current_owners(self):
        return {1: "miner1", 2: "miner2"}


def test_round_waits_for_future_entropy_freezes_members_and_survives_restart(tmp_path, monkeypatch):
    _, manifest = prepared_pool(tmp_path / "datasets")
    target = dataset_target(manifest)
    state = State(tmp_path / "state.sqlite")
    first, second = admit(state, target, 1), admit(state, target, 2)
    settings = Settings(
        role="validator",
        targets=(target,),
        chain="file",
        dataset_cache=tmp_path / "datasets",
        dashboard_path=tmp_path / "dashboard.json",
    )
    store, chain = ArtifactStore(str(tmp_path / "artifacts")), Chain()
    provider = RentalBoundary(store)
    monkeypatch.setattr(Controller, "_qualified", lambda *args: True)
    controller = Controller(settings, state, store, chain, provider)
    chain.finalized_block = 150  # Discovery is replaying history; future hashes are already known.
    controller.tick(21600)
    assert state.active_sessions() == []
    chain.finalized_block = None
    controller.tick(21600)
    assert provider.rents == []  # cutoff hash is already known, so cannot seed this round.
    session = state.active_sessions()[0]
    assert session["cutoff_block"] == 110
    late = admit(state, target, 0)  # An older late admission must not block frozen members.
    chain.height = 117  # Polling can skip blocks; entropy must still be exactly block 111.
    controller = Controller(settings, State(tmp_path / "state.sqlite"), store, chain, provider)
    controller.tick(21601)
    assert len(provider.rents) == len(provider.jobs) == 1
    job = provider.jobs[0]
    assert job.commitment == first
    assert job.round_context.seed_block == 111
    assert job.round_context.cutoff_block == 110
    assert job.dataset_plan_digest is not None
    original = state.session(session["id"])
    state.complete_job(job, evidence(job), Decision(outcome="lost", reason="test"), 21602)
    controller.tick(21603)
    next_job = provider.jobs[-1]
    assert next_job.commitment == second
    assert next_job.dataset_plan_digest == job.dataset_plan_digest
    assert next_job.round_context == job.round_context
    state.complete_job(next_job, evidence(next_job), Decision(outcome="lost", reason="test"), 21604)
    assert state.next_submission(session["id"]) is None
    assert state.known(late.identity)
    assert state.ready_count(target.identity) == 1
    with pytest.raises(ValueError, match="immutable"):
        state.set_sampling(session["id"], job.round_context, "f" * 64)
    assert state.session(session["id"])["sampling"] == original["sampling"]


def test_entropy_anchor_is_observed_after_membership_freezes(tmp_path, monkeypatch):
    _, manifest = prepared_pool(tmp_path / "datasets")
    target = dataset_target(manifest)
    state = State(tmp_path / "state.sqlite")
    admit(state, target, 1)
    settings = Settings(
        role="validator",
        targets=(target,),
        chain="file",
        dataset_cache=tmp_path / "datasets",
        dashboard_path=tmp_path / "dashboard.json",
    )
    store, chain = ArtifactStore(str(tmp_path / "artifacts")), Chain()
    provider = RentalBoundary(store)

    def qualify(*args):
        chain.height = 117  # Slow qualification/admission occurred after scan at height 110.
        return True

    monkeypatch.setattr(Controller, "_qualified", qualify)
    controller = Controller(settings, state, store, chain, provider)
    controller.tick(21600)
    session = state.active_sessions()[0]
    assert session["cutoff_block"] == 117
    assert provider.rents == []
    monkeypatch.setattr(Controller, "_qualified", lambda *args: True)
    chain.height = 118
    controller.tick(21601)
    assert provider.jobs[0].round_context.seed_block == 118


def test_production_scoring_rejects_evidence_from_another_sample(tmp_path):
    from tests.factories import job
    from validator.competition import score_evidence

    _, manifest = prepared_pool(tmp_path)
    target = dataset_target(manifest)
    context = RoundContext(
        cutoff_block=10, seed_block=11, block_hash="0x" + "b" * 64, chain="finney", netuid=1
    )
    plan = select_round(target, tmp_path, context.seed(target.identity))
    j = job(target).model_copy(
        update={
            "round_context": context,
            "dataset_plan_digest": plan.identity,
            "dataset_plan_url": "https://example.com/plan",
            "expected_dataset_selection": plan.evidence(),
        }
    )
    result = evidence(j).model_copy(
        update={
            "round_context": context,
            "dataset_selection": select_round(target, tmp_path, "another").evidence(),
        }
    )
    assert score_evidence(j, result).reason == "dataset_selection_mismatch"


def test_production_rejects_forged_prompt_ids_even_with_the_correct_digest(tmp_path):
    from tests.factories import job
    from validator.competition import score_evidence

    _, manifest = prepared_pool(tmp_path)
    target = dataset_target(manifest)
    context = RoundContext(
        cutoff_block=10, seed_block=11, block_hash="0x" + "b" * 64, chain="finney", netuid=1
    )
    plan = select_round(target, tmp_path, context.seed(target.identity))
    j = job(target).model_copy(
        update={
            "round_context": context,
            "dataset_plan_digest": plan.identity,
            "expected_dataset_selection": plan.evidence(),
        }
    )
    result = evidence(j)
    result = result.model_copy(
        update={
            "round_context": context,
            "dataset_selection": plan.evidence(),
            "measurements": tuple(
                m.model_copy(update={"output_tokens": 160, "expected_tokens": 160})
                for m in result.measurements
            ),
        }
    )
    assert score_evidence(j, result).outcome == "won"
    forged = plan.evidence().model_copy(
        update={
            "workloads": tuple(
                w.model_copy(update={"prompt_ids": tuple(f"{i:064x}" for i in range(20))})
                for w in plan.workloads
            )
        }
    )
    assert score_evidence(j, result.model_copy(update={"dataset_selection": forged})).reason == (
        "dataset_selection_mismatch"
    )


def test_no_dataset_session_without_a_known_finalized_cutoff(tmp_path):
    from tests.validator.test_state import admit
    from validator.state import State

    _, manifest = prepared_pool(tmp_path / "datasets")
    target = dataset_target(manifest)
    state = State(tmp_path / "state.sqlite")
    admit(state, target, 1)
    assert state.claim_session(target, 21600, 21600, 10, 3600) is None
    assert state.active_sessions() == []


def test_qualified_production_worker_consumes_the_controller_plan(tmp_path, pod, monkeypatch):
    import asyncio
    import time

    from tests.integration.test_controller import setup_pipeline
    from validator.chain import ChainSnapshot
    from validator.competition import score_evidence
    from validator.jobs import run_job

    root = tmp_path / "controller"
    _, manifest = prepared_pool(root / "datasets")
    controller, state, store, chain, provider, config = setup_pipeline(
        root,
        dataset_target(manifest),
    )
    monkeypatch.setattr(chain, "finalized_height", lambda: 100, raising=False)
    monkeypatch.setattr(chain, "block_hash", lambda _: "0x" + "a" * 64, raising=False)
    controller.tick(21600)
    assert provider.jobs == []
    monkeypatch.setattr(
        chain,
        "scan",
        lambda *args: ChainSnapshot(
            block=101,
            commitments=(),
            owners={1: "miner"},
        ),
    )
    controller.tick(21601)
    job = provider.jobs[0].model_copy(update={"deadline": time.time() + 1800})
    result = asyncio.run(run_job(job, tmp_path / "worker", allow_file=True))
    assert result.status == "complete", result.error_code
    assert result.dataset_selection == job.expected_dataset_selection
    assert result.round_context == job.round_context
    assert score_evidence(job, result).outcome == "won"
