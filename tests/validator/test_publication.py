import hashlib
import struct
import threading
import time
from pathlib import Path

import pytest

from shared.artifacts import ArtifactStore, package_checkpoint
from shared.contracts import Commitment, Decision
from tests.factories import evidence, make_checkpoint, provenance, target
from validator.similarity import fingerprint
from validator.state import State


def winning_state(tmp_path):
    state = State(tmp_path / "state.sqlite")
    profile = target()
    head = make_checkpoint(tmp_path / "head")
    package = tmp_path / "head.tar.gz"
    manifest = package_checkpoint(
        head,
        package,
        target_id=profile.model_id,
        target_revision=profile.revision,
        competition_id=profile.identity,
        provenance=provenance(),
    )
    private = ArtifactStore(str(tmp_path / "private"))
    digest = private.put_file(package)
    commitment = Commitment(
        netuid=1, block=10, block_hash="hash", uid=1, hotkey="winner", artifact=digest
    )
    state.admit(commitment, manifest, profile, 1, fingerprint=fingerprint(head, profile))
    session = state.claim_session(profile, 100, 0, 0, 1000)
    assert session is not None
    job = state.create_job(session["id"], candidate_url=private.url(f"objects/{digest}"), now=101)
    state.complete_job(
        job,
        evidence(job),
        Decision(outcome="won", reason="private diagnostic URL", score=2),
        201,
    )
    return state, private, ArtifactStore(str(tmp_path / "public")), profile, job, package


def test_winner_release_copies_original_verified_package_and_retries_once(tmp_path, monkeypatch):
    from validator.publication import publish_winners

    state, private, public, profile, job, package = winning_state(tmp_path)

    def no_buffered_heads(self, key, max_bytes=32 * 1024**2):
        raise AssertionError("Publication must stream checkpoint objects")

    monkeypatch.setattr(ArtifactStore, "read_bytes", no_buffered_heads)
    publish_winners(state, private, public, "", 202)
    release = state.publications()[job.commitment.identity]
    output = Path(public.root) / f"winners/{job.commitment.artifact}.tar.gz"
    assert output.read_bytes() == package.read_bytes()
    assert release.source_artifact == job.commitment.artifact
    assert release.artifact == hashlib.sha256(output.read_bytes()).hexdigest()
    assert release.checkpoint_url == f"/api/v1/winners/{job.commitment.identity}/checkpoint"
    publish_winners(state, private, public, "", 999)
    assert state.publications()[job.commitment.identity].published_at == 202
    assert len(state.awards()) == 1


def test_tampered_source_is_never_written_publicly(tmp_path):
    from validator.publication import publish_winners

    state, private, public, profile, job, package = winning_state(tmp_path)
    (Path(private.root) / f"objects/{job.commitment.artifact}").write_bytes(b"tampered")
    publish_winners(state, private, public, "", 202)
    assert not list(Path(public.root).rglob("*.tar.gz"))
    assert state.publications() == {}
    assert len(state.awards()) == 1


def test_frozen_admission_manifest_must_match_published_package(tmp_path):
    from validator.publication import publish_winners

    state, private, public, profile, job, package = winning_state(tmp_path)
    state.connection.execute(
        "UPDATE submissions SET manifest=replace(manifest, 'public recipe', 'changed recipe')"
    )
    publish_winners(state, private, public, "", 202)
    assert state.publications() == {}
    assert not list(Path(public.root).rglob("*.tar.gz"))


def test_release_rejects_digest_changes_and_credential_urls():
    from shared.releases import WinnerRelease

    values = dict(
        submission_id="a" * 64,
        source_artifact="b" * 64,
        artifact="b" * 64,
        checkpoint_url=f"https://checkpoints.example/winners/{'b' * 64}.tar.gz",
        published_at=1,
    )
    assert WinnerRelease(**values).artifact == "b" * 64
    with pytest.raises(ValueError, match="digest"):
        WinnerRelease(**{**values, "artifact": "c" * 64})
    for url in (
        "file:///private/checkpoint",
        "https://private.example/head?X-Amz-Signature=secret",
        f"https://user:secret@example/winners/{'b' * 64}.tar.gz",
        "/v1/winners/another/checkpoint",
    ):
        with pytest.raises(ValueError):
            WinnerRelease(**{**values, "checkpoint_url": url})


def test_snapshot_hides_unpublished_winners_and_cleans_stale_public_reports(tmp_path):
    from validator.dashboard import publish_snapshot

    state, private, public, profile, job, package = winning_state(tmp_path)
    path = tmp_path / "dashboard" / "snapshot.json"
    reports = path.parent / "reports"
    reports.mkdir(parents=True)
    stale = reports / f"{'e' * 32}.json"
    stale.write_text('{"reason":"private loser details"}')
    stale_temporary = stale.with_suffix(".tmp")
    stale_temporary.write_text('{"reason":"interrupted private loser report"}')
    snapshot = publish_snapshot(state, path, (profile,), {1: "winner"}, {0: 1}, 202, 10)
    assert snapshot.submissions == ()
    assert snapshot.evaluations == ()
    assert snapshot.winners == ()
    assert not stale.exists()
    assert not stale_temporary.exists()
    assert not list(reports.glob("*.json"))


def test_only_winners_publish_and_expired_releases_keep_archived_reports(tmp_path):
    from api.queries import PublishedViews
    from validator.dashboard import publish_snapshot
    from validator.publication import publish_winners

    state, private, public, profile, winner_job, package = winning_state(tmp_path)
    for block, hotkey in ((11, "loser"), (12, "pending"), (13, "rejected")):
        head = make_checkpoint(tmp_path / hotkey)
        tensor = head / "model.safetensors"
        data = tensor.read_bytes()
        # The rejected head is an exact duplicate of the pending one.
        tensor.write_bytes(data[:-8] + struct.pack("<ff", min(block, 12), 1))
        candidate = tmp_path / f"{hotkey}.tar.gz"
        manifest = package_checkpoint(
            head,
            candidate,
            target_id=profile.model_id,
            target_revision=profile.revision,
            competition_id=profile.identity,
            provenance=provenance(),
        )
        digest = private.put_file(candidate)
        commitment = Commitment(
            netuid=1, block=block, block_hash="hash", uid=block, hotkey=hotkey, artifact=digest
        )
        state.admit(commitment, manifest, profile, 202, fingerprint=fingerprint(head, profile))
        if hotkey == "loser":
            job = state.create_job(
                winner_job.session_id, candidate_url=private.url(f"objects/{digest}"), now=203
            )
            state.complete_job(
                job,
                evidence(job),
                Decision(outcome="lost", reason="private failure details", score=1),
                204,
            )
    publish_winners(state, private, public, "", 205)
    path = tmp_path / "dashboard" / "snapshot.json"
    snapshot = publish_snapshot(
        state, path, (profile,), {1: "winner"}, {0: 1}, profile.policy.award_seconds + 202, 20
    )
    assert len(snapshot.releases) == 1
    assert {s.hotkey for s in snapshot.submissions} == {"winner"}
    assert {e.submission_id for e in snapshot.evaluations} == {winner_job.commitment.identity}
    assert snapshot.winners[0].eligible is False
    assert "private" not in path.read_text()
    assert len(list(Path(public.root).rglob("*.tar.gz"))) == 1
    path.write_text(
        snapshot.model_copy(update={"evaluations": (), "winners": ()}).model_dump_json()
    )
    archived = PublishedViews(path).report(winner_job.job_id)
    assert archived is not None
    assert archived.score == 2


def test_public_write_failure_is_retried_without_changing_award(tmp_path, monkeypatch):
    from validator.publication import publish_winners

    state, private, public, profile, job, package = winning_state(tmp_path)
    original_write = public.write_file

    def unavailable(key, source):
        raise OSError("public storage unavailable")

    monkeypatch.setattr(public, "write_file", unavailable)
    publish_winners(state, private, public, "", 202)
    assert state.publications() == {}
    assert len(state.awards()) == 1
    assert state.awards()[0].promoted_at == 201
    monkeypatch.setattr(public, "write_file", original_write)
    publish_winners(state, private, public, "", 203)
    assert state.publications()[job.commitment.identity].published_at == 203
    assert state.awards()[0].promoted_at == 201


@pytest.mark.parametrize("public_path", ["private", "private/public", "."])
def test_publication_rejects_overlapping_private_and_public_namespaces(tmp_path, public_path):
    from validator.publication import publish_winners

    state, private, public, profile, job, package = winning_state(tmp_path)
    overlapping = ArtifactStore(str(tmp_path / public_path))
    with pytest.raises(ValueError, match="namespace"):
        publish_winners(state, private, overlapping, "", 202)
    assert not list(Path(private.root).rglob("*.tar.gz"))


def test_one_missing_winner_source_does_not_starve_new_winners(tmp_path):
    from validator.publication import publish_winners

    state, private, public, profile, first_job, package = winning_state(tmp_path)
    job = add_next_winner(state, private, profile, first_job, tmp_path)
    (Path(private.root) / f"objects/{first_job.commitment.artifact}").unlink()
    publish_winners(state, private, public, "", 205)
    assert set(state.publications()) == {job.commitment.identity}
    assert len(state.awards()) == 2


def add_next_winner(state, private, profile, first_job, tmp_path):
    head = make_checkpoint(tmp_path / "next-winner")
    tensor = head / "model.safetensors"
    tensor.write_bytes(tensor.read_bytes()[:-8] + struct.pack("<ff", 1, 2))
    next_package = tmp_path / "next.tar.gz"
    manifest = package_checkpoint(
        head,
        next_package,
        target_id=profile.model_id,
        target_revision=profile.revision,
        competition_id=profile.identity,
        provenance=provenance(),
    )
    digest = private.put_file(next_package)
    commitment = Commitment(
        netuid=1, block=11, block_hash="hash", uid=2, hotkey="next-winner", artifact=digest
    )
    state.admit(commitment, manifest, profile, 202, fingerprint=fingerprint(head, profile))
    job = state.create_job(
        first_job.session_id, candidate_url=private.url(f"objects/{digest}"), now=203
    )
    state.complete_job(job, evidence(job), Decision(outcome="won", reason="winner", score=3), 204)
    return job


def pump_until_published(service, state, private, public, submission_id, *, now=202):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        service.pump(state, private, public, "", now)
        if submission_id in state.publications():
            return
        threading.Event().wait(0.01)
    pytest.fail("Publication service did not commit the completed winner release")


def test_publication_service_does_not_block_polling_or_use_sqlite_on_worker(tmp_path, monkeypatch):
    import validator.publication as publication

    state, private, public, profile, job, package = winning_state(tmp_path)
    started, unblock, timed_out = threading.Event(), threading.Event(), threading.Event()
    original_download = publication.download_url
    owner = threading.get_ident()
    original_record = state.record_publication
    original_submit = publication.ThreadPoolExecutor.submit
    submitted = []

    def counted_submit(executor, function, *args, **kwargs):
        submitted.append(function)
        return original_submit(executor, function, *args, **kwargs)

    def guarded_record(release):
        assert threading.get_ident() == owner
        original_record(release)

    def blocked_download(*args, **kwargs):
        assert threading.get_ident() != owner
        started.set()
        if not unblock.wait(3):
            timed_out.set()
            raise TimeoutError("test transfer was not released")
        original_download(*args, **kwargs)

    monkeypatch.setattr(state, "record_publication", guarded_record)
    monkeypatch.setattr(publication, "download_url", blocked_download)
    monkeypatch.setattr(publication.ThreadPoolExecutor, "submit", counted_submit)
    service = publication.PublicationService()
    try:
        service.pump(state, private, public, "", 202)
        assert started.wait(1)
        assert not timed_out.is_set()
        assert state.publications() == {}
        # A still-running transfer cannot consume a second worker or duplicate the award.
        service.pump(state, private, public, "", 203)
        assert len(submitted) == 1
        assert len(state.awards()) == 1
        unblock.set()
        pump_until_published(service, state, private, public, job.commitment.identity)
        release = state.publications()[job.commitment.identity]
        assert release.published_at == 202
        assert (Path(public.root) / f"winners/{release.artifact}.tar.gz").read_bytes() == (
            package.read_bytes()
        )
        service.pump(state, private, public, "", 999)
        assert state.publications()[job.commitment.identity] == release
        assert len(state.awards()) == 1
    finally:
        unblock.set()
        service.close()


def test_publication_service_retries_failed_source_without_new_award(tmp_path, monkeypatch):
    import validator.publication as publication

    state, private, public, profile, job, package = winning_state(tmp_path)
    source = Path(private.root) / f"objects/{job.commitment.artifact}"
    original = source.read_bytes()
    source.write_bytes(b"corrupted source")
    first_finished = threading.Event()
    original_publish = publication._publish_snapshot

    def observed_publish(*args, **kwargs):
        try:
            return original_publish(*args, **kwargs)
        finally:
            first_finished.set()

    monkeypatch.setattr(publication, "_publish_snapshot", observed_publish)
    service = publication.PublicationService()
    try:
        service.pump(state, private, public, "", 202)
        assert first_finished.wait(3)
        assert state.publications() == {}
        assert not list(Path(public.root).rglob("*.tar.gz"))
        source.write_bytes(original)
        pump_until_published(service, state, private, public, job.commitment.identity, now=203)
        assert state.publications()[job.commitment.identity].published_at == 203
        assert len(state.awards()) == 1
        assert state.awards()[0].promoted_at == 201
    finally:
        service.close()


def test_publication_service_retains_completed_release_when_sqlite_commit_retries(
    tmp_path, monkeypatch
):
    from validator.publication import PublicationService

    state, private, public, profile, job, package = winning_state(tmp_path)
    original_record = state.record_publication
    commit_failed = threading.Event()

    def unavailable(release):
        commit_failed.set()
        raise OSError("temporary authoritative state failure")

    monkeypatch.setattr(state, "record_publication", unavailable)
    service = PublicationService()
    try:
        deadline = time.monotonic() + 5
        while not commit_failed.is_set() and time.monotonic() < deadline:
            service.pump(state, private, public, "", 202)
            threading.Event().wait(0.01)
        assert commit_failed.is_set()
        assert state.publications() == {}
        # A cached, already verified release must not require the private source again.
        (Path(private.root) / f"objects/{job.commitment.artifact}").unlink()
        monkeypatch.setattr(state, "record_publication", original_record)
        service.pump(state, private, public, "", 203)
        assert state.publications()[job.commitment.identity].published_at == 202
        assert len(state.awards()) == 1
    finally:
        service.close()


def test_publication_service_gives_each_unpublished_winner_a_fair_retry_turn(tmp_path):
    from validator.publication import PublicationService

    state, private, public, profile, first_job, package = winning_state(tmp_path)
    next_job = add_next_winner(state, private, profile, first_job, tmp_path)
    source = Path(private.root) / f"objects/{first_job.commitment.artifact}"
    source.unlink()
    service = PublicationService()
    try:
        pump_until_published(service, state, private, public, next_job.commitment.identity)
        assert set(state.publications()) == {next_job.commitment.identity}
        private.write_file(f"objects/{first_job.commitment.artifact}", package)
        pump_until_published(service, state, private, public, first_job.commitment.identity)
        assert len(state.publications()) == 2
        assert len(state.awards()) == 2
    finally:
        service.close()


def test_publication_service_close_waits_and_final_pump_only_collects(tmp_path):
    from validator.publication import PublicationService

    state, private, public, profile, job, package = winning_state(tmp_path)
    service = PublicationService()
    service.pump(state, private, public, "", 202)
    service.close()
    # close does not own authoritative state; a final main-thread pump persists its result.
    assert state.publications() == {}
    assert len(list(Path(public.root).rglob("*.tar.gz"))) == 1
    service.pump(state, private, public, "", 203)
    assert state.publications()[job.commitment.identity].published_at == 202
    service.pump(state, private, public, "", 204)
    assert len(state.publications()) == 1
