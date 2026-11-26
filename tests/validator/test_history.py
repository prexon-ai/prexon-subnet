import json
import struct

from shared.artifacts import ArtifactStore, package_checkpoint
from shared.contracts import Commitment, DashboardSnapshot, Decision, ModelHistory, ModelIndex
from tests.factories import evidence, make_checkpoint, provenance, target
from tests.validator.test_publication import add_next_winner, winning_state
from validator.dashboard import publish_snapshot
from validator.history import build_history, model_slug, publish_history
from validator.public_files import REFRESH_SECONDS
from validator.publication import publish_winners
from validator.similarity import fingerprint


def add_submission(state, private, profile, tmp_path, *, hotkey, uid, block, now):
    head = make_checkpoint(tmp_path / f"head-{hotkey}")
    tensor = head / "model.safetensors"
    tensor.write_bytes(tensor.read_bytes()[:-8] + struct.pack("<ff", uid, block))
    package = tmp_path / f"{hotkey}.tar.gz"
    manifest = package_checkpoint(
        head,
        package,
        target_id=profile.model_id,
        target_revision=profile.revision,
        competition_id=profile.competition_id,
        provenance=provenance(),
    )
    digest = private.put_file(package)
    commitment = Commitment(
        netuid=1,
        block=block,
        block_hash="hash",
        uid=uid,
        hotkey=hotkey,
        artifact=digest,
        submitted_at=now - 50,
    )
    state.admit(commitment, manifest, profile, now, fingerprint=fingerprint(head, profile))
    return commitment


def competition(tmp_path):
    """Two winners, one loser, one head being evaluated and one waiting."""
    state, private, public, profile, first_job, _ = winning_state(tmp_path)
    add_next_winner(state, private, profile, first_job, tmp_path)
    add_submission(state, private, profile, tmp_path, hotkey="loser", uid=3, block=12, now=205)
    job = state.create_job(first_job.session_id, candidate_url="file:///loser", now=206)
    lost = Decision(outcome="lost", reason="private failure details", score=2.9, threshold=3)
    state.complete_job(job, evidence(job), lost, 207)
    for index, hotkey in enumerate(("queued-a", "queued-b")):
        add_submission(
            state,
            private,
            profile,
            tmp_path,
            hotkey=hotkey,
            uid=4 + index,
            block=13 + index,
            now=208,
        )
    state.create_job(first_job.session_id, candidate_url="file:///queued-a", now=210)
    publish_winners(state, private, public, "", 211)
    state.touch_models((profile,), 100)
    owners = {1: "winner", 2: "next-winner", 3: "loser", 4: "queued-a", 5: "queued-b"}
    return state, private, public, profile, owners


def test_history_lists_the_queue_and_every_result_without_private_details(tmp_path):
    state, private, public, profile, owners = competition(tmp_path)
    index, models = build_history(state, (profile,), owners, 300, 20)
    summary = index.models[0]
    assert summary.id == model_slug(profile.model_id, profile.revision) == "model1-aaaaaaa"
    assert (summary.queued, summary.evaluated, summary.winners) == (2, 3, 2)
    assert summary.best_speedup == 3
    assert summary.reward_shares == (0.7, 0.3)
    assert (summary.active_from, summary.active_to) == (100, None)
    model = models[summary.id]
    # The head on the GPU comes first, then the rest in the order they will run.
    assert [(q.hotkey, q.uid, q.status) for q in model.queue] == [
        ("queued-a", 4, "evaluating"),
        ("queued-b", 5, "waiting"),
    ]
    assert [(s.hotkey, s.crowned, s.speedup, s.share) for s in model.submissions] == [
        ("loser", False, 2.9, 0),
        ("next-winner", True, 3, 0.7),
        ("winner", True, 2, 0.3),
    ]
    loser, newest, _ = model.submissions
    assert (loser.previous_best, loser.evaluated_at, loser.submitted_at) == (3, 207, 155)
    assert loser.checkpoint_url is None  # only a crowned head is published
    # These winners were released to a local directory, which nobody can download from.
    assert newest.checkpoint_url is None
    assert "private" not in model.model_dump_json()


def test_crowned_head_links_to_its_package_once_it_is_in_public_storage(tmp_path):
    state, private, public, profile, job, _ = winning_state(tmp_path)
    publish_winners(state, private, public, "https://cdn.example/prexon-public", 202)
    state.touch_models((profile,), 100)
    _, models = build_history(state, (profile,), {1: "winner"}, 300, 20)
    (crowned,) = models["model1-aaaaaaa"].submissions
    assert crowned.checkpoint_url == (
        f"https://cdn.example/prexon-public/winners/{job.commitment.artifact}.tar.gz"
    )


def test_retired_model_keeps_its_results_but_has_no_queue_or_rewards(tmp_path):
    state, private, public, profile, owners = competition(tmp_path)
    state.touch_models((profile,), 500)
    replacement = target(2)
    state.touch_models((replacement,), 600)
    index, models = build_history(state, (replacement,), owners, 700, 30)
    # Active models come first; a retired model stays in the list.
    assert [m.model_id for m in index.models] == [replacement.model_id, profile.model_id]
    active, past = index.models
    assert (active.active_from, active.active_to, active.evaluated) == (600, None, 0)
    assert (past.active_from, past.active_to) == (100, 500)
    assert (past.queued, past.reward_shares, past.winners, past.evaluated) == (0, (), 2, 3)
    history = models[past.id]
    assert history.queue == ()
    assert all(s.share == 0 for s in history.submissions)
    assert [s.hotkey for s in history.submissions] == ["loser", "next-winner", "winner"]


def test_model_retired_before_periods_were_recorded_is_dated_by_its_submissions(tmp_path):
    state, private, public, profile, owners = competition(tmp_path)
    # A database from before the validator recorded when each model was active.
    state.connection.execute("DELETE FROM metadata WHERE key LIKE 'model_period:%'")
    replacement = target(2)
    state.touch_models((replacement,), 600)
    index, _ = build_history(state, (replacement,), owners, 700, 30)
    active, past = index.models
    assert past.model_id == profile.model_id
    # Its first head was submitted at 1 and its last verdict was reached at 207.
    assert (past.active_from, past.active_to) == (1, 207)
    assert (past.allocation, past.evaluated) == (100, 3)


class Recording(ArtifactStore):
    """Public storage that notes each upload."""

    def __init__(self, root):
        super().__init__(root)
        self.uploaded = []

    def write_bytes(self, key, data, **options):
        self.uploaded.append(key)
        super().write_bytes(key, data, **options)


def test_history_is_uploaded_when_it_changes_and_its_list_is_kept_fresh(tmp_path):
    state, private, public, profile, owners = competition(tmp_path)
    store = Recording(public.root)
    index = publish_history(state, store, (profile,), owners, 300, 20)
    model_file = f"dashboard/models/{index.models[0].id}.json"
    assert sorted(store.uploaded) == ["dashboard/index.json", model_file]
    assert ModelIndex.model_validate_json(store.read_bytes("dashboard/index.json")) == index
    stored = ModelHistory.model_validate_json(store.read_bytes(model_file))
    assert [s.hotkey for s in stored.submissions] == ["loser", "next-winner", "winner"]

    store.uploaded.clear()
    publish_history(state, store, (profile,), owners, 330, 21)
    assert store.uploaded == []  # only the clock moved

    # Readers tell a quiet subnet from a stopped validator by the list's time, so the
    # list is uploaded again once its copy is old. The unchanged model file is not.
    later = 300 + REFRESH_SECONDS
    publish_history(state, store, (profile,), owners, later, 30)
    assert store.uploaded == ["dashboard/index.json"]
    assert json.loads(store.read_bytes("dashboard/index.json"))["updated_at"] == later

    store.uploaded.clear()
    add_submission(state, private, profile, tmp_path, hotkey="late", uid=6, block=15, now=later)
    publish_history(state, store, (profile,), {**owners, 6: "late"}, later + 30, 31)
    assert sorted(store.uploaded) == ["dashboard/index.json", model_file]


def test_snapshot_is_uploaded_when_it_changes_and_kept_fresh(tmp_path):
    state, private, public, profile, owners = competition(tmp_path)
    store = Recording(public.root)

    def publish(now, block):
        return publish_snapshot(state, store, (profile,), owners, {1: 0.3, 2: 0.7}, now, block)

    snapshot = publish(300, 20)
    assert store.uploaded == ["dashboard/snapshot.json"]
    stored = DashboardSnapshot.model_validate_json(store.read_bytes("dashboard/snapshot.json"))
    assert stored == snapshot

    publish(330, 21)  # only the clock moved
    assert len(store.uploaded) == 1
    publish(300 + REFRESH_SECONDS, 30)  # the copy is old: refreshed although nothing changed
    assert len(store.uploaded) == 2
    stored = json.loads(store.read_bytes("dashboard/snapshot.json"))
    assert stored["updated_at"] == 300 + REFRESH_SECONDS

    # A change in what the snapshot shows is uploaded at once.
    add_submission(state, private, profile, tmp_path, hotkey="late", uid=6, block=15, now=700)
    publish(300 + REFRESH_SECONDS + 30, 31)
    assert len(store.uploaded) == 3


def test_snapshot_evaluations_name_their_model_so_a_profile_change_keeps_them(tmp_path):
    state, private, public, profile, job, _ = winning_state(tmp_path)
    publish_winners(state, private, public, "", 202)
    changed = profile.model_copy(update={"context_length": 4096})
    snapshot = publish_snapshot(state, public, (changed,), {1: "winner"}, {1: 1.0}, 300, 20)
    (evaluation,) = snapshot.evaluations
    # Scored under the earlier profile, but still an evaluation of the current target.
    assert evaluation.competition_id == profile.competition_id == snapshot.targets[0].competition_id
    assert (evaluation.model_id, evaluation.revision) == ("org/model1", "a" * 40)


def test_failed_upload_is_retried(tmp_path):
    state, private, public, profile, owners = competition(tmp_path)
    attempts = []

    class Unavailable(ArtifactStore):
        def write_bytes(self, key, data, **options):
            attempts.append(key)
            raise OSError("public storage unavailable")

    publish_history(state, Unavailable(public.root), (profile,), owners, 300, 20)
    assert len(attempts) == 2
    publish_history(state, Unavailable(public.root), (profile,), owners, 330, 21)
    assert len(attempts) == 4  # still owed, so tried again
