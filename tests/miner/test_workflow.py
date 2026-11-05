import json
import subprocess
import sys

import pytest

from miner.cli import main
from miner.train import train_and_export
from shared.artifacts import extract_package
from shared.checkpoints import sha256_file
from tests.factories import make_checkpoint, provenance, target


@pytest.mark.parametrize("direct", [False, True])
def test_training_pins_snapshot_and_exports_hf_checkpoint(tmp_path, monkeypatch, direct):
    import yaml

    from miner.config import MinerTarget

    recipe = tmp_path / "recipe.yaml"
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps({"architectures": ["DFlash2DraftModel"]}))
    recipe.write_text(
        yaml.safe_dump(
            {
                "model": {"draft_model_config": str(draft)},
                "training": {"strategy": "dflash"},
                "data": {"train_data_path": "licensed.jsonl"},
            }
        )
    )
    snapshot = tmp_path / "target-snapshot"
    snapshot.mkdir()
    exported = tmp_path / "exported"
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "train":
            config = yaml.safe_load((tmp_path / "run" / "recipe.yaml").read_text())
            assert config["model"]["target_model_path"] == str(snapshot)
            assert config["model"]["trust_remote_code"] is False
            (tmp_path / "training-checkpoint").mkdir()
        if command[1] == "export":
            make_checkpoint(exported)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("miner.train.subprocess.run", run)
    monkeypatch.setattr("miner.train.importlib.metadata.version", lambda _: "0.2.0")
    configured = MinerTarget(model_id="org/model", revision="a" * 40) if direct else target()
    train_and_export(
        configured,
        recipe,
        tmp_path / "run",
        tmp_path / "training-checkpoint",
        exported,
        snapshot_resolver=lambda model, revision: str(snapshot),
    )
    assert [c[1] for c in calls] == ["train", "export"]
    assert calls[1][calls[1].index("--to") + 1] == "hf"
    assert (exported / "model.safetensors").exists()
    record = json.loads((tmp_path / "run/training-record.json").read_text())
    assert record["target_id"] == configured.model_id
    assert record["competition_id"] == (None if direct else configured.identity)


def test_training_rejects_method_or_remote_code_before_launch(tmp_path):
    import yaml

    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump({"model": {"trust_remote_code": True}, "training": {"strategy": "dflash"}})
    )
    with pytest.raises(ValueError, match="remote"):
        train_and_export(
            target(), recipe, tmp_path / "run", tmp_path / "checkpoint", tmp_path / "export"
        )


def test_failed_training_never_exports(tmp_path, monkeypatch):
    import yaml

    recipe = tmp_path / "recipe.yaml"
    draft = tmp_path / "draft.json"
    draft.write_text('{"architectures":["DFlash2DraftModel"]}')
    recipe.write_text(
        yaml.safe_dump(
            {"model": {"draft_model_config": str(draft)}, "training": {"strategy": "dflash"}}
        )
    )

    def run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr("miner.train.subprocess.run", run)
    monkeypatch.setattr("miner.train.importlib.metadata.version", lambda _: "0.2.0")
    with pytest.raises(subprocess.CalledProcessError):
        train_and_export(
            target(),
            recipe,
            tmp_path / "run",
            tmp_path / "checkpoint",
            tmp_path / "export",
            snapshot_resolver=lambda *_: str(tmp_path),
        )
    assert not (tmp_path / "export").exists()


def test_package_accepts_identity_flags_without_loading_server_dotenv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("PREXON_MODEL_1_ID=server\nPREXON_MODEL_1_PROFILE=/missing\n")
    head = make_checkpoint(tmp_path / "head")
    record = tmp_path / "provenance.json"
    record.write_text(provenance().model_dump_json())
    output = tmp_path / "head.tar.gz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prexon-miner",
            "package",
            "--model-id",
            "org/model",
            "--revision",
            "a" * 40,
            "--competition-id",
            "b" * 64,
            "--checkpoint",
            str(head),
            "--provenance",
            str(record),
            "--output",
            str(output),
        ],
    )
    main()
    manifest = extract_package(output, tmp_path / "restored", sha256_file(output), 1024**2)
    assert manifest.target_id == "org/model"
    assert manifest.target_revision == "a" * 40
    assert manifest.competition_id == "b" * 64


@pytest.mark.parametrize("before_command", [True, False])
def test_submit_flags_override_an_optional_miner_env(tmp_path, monkeypatch, before_command):
    env_file = tmp_path / "miner.env"
    env_file.write_text("PREXON_NETUID=invalid-but-overridden\n")
    calls = []
    monkeypatch.setattr(
        "miner.submit.submit", lambda *args, **kwargs: calls.append((args, kwargs)) or "a" * 64
    )
    options = ["--env-file", str(env_file), "--netuid", "17"]
    command = ["submit", str(tmp_path / "head.tar.gz"), "--upload-only"]
    monkeypatch.setattr(
        sys, "argv", ["prexon-miner", *(options + command if before_command else command + options)]
    )
    main()
    settings, package = calls[0][0]
    assert settings.netuid == 17
    assert calls[0][1] == {"publish": False}
