import hashlib
import json
import sys

import pytest

from shared.checkpoints import checkpoint_schema
from shared.contracts import DashboardSnapshot, Qualification
from tests.factories import make_checkpoint, provenance, target, write_competitions


def run_validator(set_config, monkeypatch, tmp_path, *args, **constants):
    """Run prexon-validator in-process with every runtime path of the config files in tmp."""
    from validator.main import main

    paths = {
        "STATE_PATH": tmp_path / "state.sqlite",
        "ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "PUBLIC_ARTIFACT_ROOT": str(tmp_path / "public"),
        "UPLOAD_POLICY_PATH": tmp_path / "intake-policy/upload-policy.json",
        "DATASET_CACHE": tmp_path / "datasets",
        "SSH_KNOWN_HOSTS_DIR": tmp_path / "known_hosts",
        "UPLOAD_STATE_PATH": tmp_path / "api/uploads.sqlite3",
    }
    # Pin the values the owner sets in the committed file, so tests stay offline and idle
    # and never reach the owner's buckets.
    pinned = {
        "BURN_UID": 0,
        "PROVIDER": "local",
        "UPLOAD_STAGING_ROOT": "",
        "PUBLIC_ARTIFACT_BASE_URL": "",
    }
    set_config(**{**paths, **pinned, **constants})
    monkeypatch.setattr(sys, "argv", ["prexon-validator", *args])
    main()


def test_idle_controller_runs_without_external_credentials(set_config, tmp_path, monkeypatch):
    def connect(settings):
        raise AssertionError("an idle validator never connects to the chain")

    monkeypatch.setattr("validator.chain.connect", connect)
    run_validator(set_config, monkeypatch, tmp_path, "--once")
    snapshot = DashboardSnapshot.model_validate_json(
        (tmp_path / "public/dashboard/snapshot.json").read_bytes()
    )
    assert snapshot.targets == ()
    assert snapshot.emissions[0].uid == 0
    assert snapshot.emissions[0].share == 1


def active_competition():
    t = target()
    write_competitions(t)
    return t


def test_qualification_cli_validates_uploads_and_prints_immutable_id(
    set_config, tmp_path, monkeypatch, capsys
):
    t = active_competition()
    run_validator(set_config, monkeypatch, tmp_path, "--qualification-profile", t.model_id)
    assert capsys.readouterr().out.strip() == t.qualification_identity
    q = Qualification(
        profile_id=t.qualification_identity,
        engine_version=t.engine_version,
        approved_by="reviewer",
        evidence_description="Recorded verification review and pilot",
        target_preserving_verification_reviewed=True,
        checkpoint_schemas=(checkpoint_schema(make_checkpoint(tmp_path / "head")),),
    )
    path = tmp_path / "qualification.json"
    path.write_text(q.model_dump_json())
    run_validator(set_config, monkeypatch, tmp_path, "--publish-qualification", str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert capsys.readouterr().out.strip() == digest
    assert (tmp_path / "artifacts/objects" / digest).read_bytes() == path.read_bytes()


def test_reference_cli_stores_only_a_package_built_for_a_competition(
    set_config, tmp_path, monkeypatch, capsys
):
    from shared.artifacts import package_checkpoint

    t = active_competition()

    def package(name, model_id):
        path = tmp_path / f"{name}.tar.gz"
        package_checkpoint(
            make_checkpoint(tmp_path / name),
            path,
            target_id=model_id,
            target_revision=t.revision,
            competition_id=model_id,
            provenance=provenance(),
        )
        return path, hashlib.sha256(path.read_bytes()).hexdigest()

    reference, digest = package("reference", t.model_id)
    run_validator(set_config, monkeypatch, tmp_path, "--publish-reference", str(reference))
    assert capsys.readouterr().out.strip() == digest
    assert (tmp_path / "artifacts/objects" / digest).read_bytes() == reference.read_bytes()
    stranger, other = package("stranger", "org/unlisted")
    with pytest.raises(SystemExit):
        run_validator(set_config, monkeypatch, tmp_path, "--publish-reference", str(stranger))
    assert not (tmp_path / "artifacts/objects" / other).exists()


def test_initialized_empty_controller_keeps_bittensor_for_clearing_weights(
    tmp_path, monkeypatch, set_config
):
    from validator.chain import ChainSnapshot
    from validator.main import main
    from validator.state import State

    state_path = tmp_path / "state.sqlite"
    state = State(state_path)
    state.set_metadata("chain_context", json.dumps({"network": "finney", "netuid": 1}))
    state.set_metadata("chain_initialized", "1")
    state.close()
    published = []

    class Boundary:
        def scan(self, start_block=None, known=()):
            return ChainSnapshot(block=100, commitments=(), owners={0: "burn"})

        def current_owners(self):
            return {0: "burn"}

        def publish(self, weights):
            published.append(weights)

    set_config(NETUID=1, STATE_PATH=state_path)
    monkeypatch.setattr("validator.main.BittensorChain", lambda *_: Boundary())
    monkeypatch.setattr(sys, "argv", ["prexon-validator", "--once"])
    main()
    assert published == [{0: 1.0}]


def test_requeue_command_returns_set_aside_heads_to_the_queue(
    set_config, tmp_path, monkeypatch, capsys
):
    from shared.contracts import Decision
    from tests.factories import evidence
    from tests.validator.test_state import admit
    from validator.state import State

    state_path = tmp_path / "state.sqlite"
    state = State(state_path)
    t = target()
    head = admit(state, t, 1)
    session = state.claim_session(t, 21600, 21600, 10, 3600)
    j = state.create_job(session["id"], candidate_url="file:///candidate", now=21601)
    unusable = Decision(outcome="inconclusive", reason="reference_drift")
    state.complete_job(j, evidence(j), unusable, 21602, max_attempts=1)
    state.finish_session(session["id"], 21603)
    state.close()
    run_validator(set_config, monkeypatch, tmp_path, "--requeue-stalled", STATE_PATH=state_path)
    assert capsys.readouterr().out.strip() == "Requeued 1 stalled submission(s)"
    row = next(r for r in State(state_path).submissions() if r["id"] == head.identity)
    assert row["status"] == "ready"


def test_check_config_runs_without_env_files_from_any_directory(
    set_config, tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    run_validator(set_config, monkeypatch, tmp_path, "--check-config")
    assert json.loads(capsys.readouterr().out) == {"competitions": []}
    assert list(tmp_path.iterdir()) == []


def test_enabled_targets_need_a_netuid_and_hippius_storage(set_config, tmp_path, monkeypatch):
    active_competition()
    with pytest.raises(ValueError, match="Set PREXON_NETUID in .env before enabling models"):
        run_validator(set_config, monkeypatch, tmp_path, "--once")
    # No local development chain: enabled targets always mean the live subnet.
    with pytest.raises(ValueError, match="Live targets require three separate Hippius S3 stores"):
        run_validator(set_config, monkeypatch, tmp_path, "--once", NETUID=5)


def test_inactive_competitions_are_not_evaluated(set_config, tmp_path, monkeypatch):
    # With PREXON_NETUID unset an active competition would stop startup; this one is idle.
    write_competitions(target(active=False))
    run_validator(set_config, monkeypatch, tmp_path, "--once")
    snapshot = DashboardSnapshot.model_validate_json(
        (tmp_path / "public/dashboard/snapshot.json").read_bytes()
    )
    assert snapshot.targets == ()
