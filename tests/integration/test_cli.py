import hashlib
import json
import subprocess
import sys

from shared.checkpoints import checkpoint_schema
from shared.contracts import DashboardSnapshot, Qualification
from tests.factories import make_checkpoint, target


def test_idle_controller_runs_without_external_credentials(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        f"PREXON_STATE_PATH={tmp_path}/state.sqlite\n"
        f"PREXON_DASHBOARD_PATH={tmp_path}/public/dashboard.json\n"
        "PREXON_MODEL_1_ID=\nPREXON_MODEL_2_ID=\n"
    )
    result = subprocess.run(
        [sys.executable, "-m", "validator.main", "--env-file", str(env), "--once"],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0
    snapshot = DashboardSnapshot.model_validate_json(
        (tmp_path / "public/dashboard.json").read_bytes()
    )
    assert snapshot.targets == ()
    assert snapshot.emissions[0].uid == 0
    assert snapshot.emissions[0].share == 1


def test_qualification_cli_validates_uploads_and_prints_immutable_id(tmp_path):
    t = target()
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(t.model_dump(exclude={"slot", "model_id", "revision", "allocation"}))
    )
    env = tmp_path / ".env"
    env.write_text(
        f"PREXON_MODEL_1_ID={t.model_id}\nPREXON_MODEL_1_REVISION={t.revision}\n"
        f"PREXON_MODEL_1_PROFILE={profile}\nPREXON_ARTIFACT_ROOT={tmp_path}/artifacts\n"
    )
    base = [sys.executable, "-m", "validator.main", "--env-file", str(env)]
    output = subprocess.run(
        [*base, "--qualification-profile", "1"], check=True, capture_output=True, text=True
    )
    assert output.stdout.strip() == t.qualification_identity
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
    output = subprocess.run(
        [*base, "--publish-qualification", str(path)], check=True, capture_output=True, text=True
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert output.stdout.strip() == digest
    assert (tmp_path / "artifacts/objects" / digest).read_bytes() == path.read_bytes()


def test_initialized_empty_controller_keeps_bittensor_for_clearing_weights(tmp_path, monkeypatch):
    from shared.config import Settings
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
        def scan(self, start_block=None):
            return ChainSnapshot(block=100, commitments=(), owners={0: "burn"})

        def current_owners(self):
            return {0: "burn"}

        def publish(self, weights):
            published.append(weights)

    settings = Settings(
        role="validator",
        netuid=1,
        state_path=state_path,
        dashboard_path=tmp_path / "public/dashboard.json",
        artifact_root=str(tmp_path / "artifacts"),
    )
    monkeypatch.setattr("validator.main.load_settings", lambda *_: settings)
    monkeypatch.setattr("validator.main.BittensorChain", lambda *_: Boundary())
    monkeypatch.setattr(sys, "argv", ["prexon-validator", "--once"])
    main()
    assert published == [{0: 1.0}]
