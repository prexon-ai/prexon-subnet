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

    from miner.target import MinerTarget

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
def test_submit_flags_work_before_or_after_the_command(tmp_path, monkeypatch, before_command):
    calls = []
    monkeypatch.setattr(
        "miner.submit.submit", lambda *args, **kwargs: calls.append((args, kwargs)) or "a" * 64
    )
    options = ["--netuid", "17", "--network", "test"]
    command = ["submit", str(tmp_path / "head.tar.gz"), "--upload-only"]
    monkeypatch.setattr(
        sys, "argv", ["prexon-miner", *(options + command if before_command else command + options)]
    )
    main()
    (package,), kwargs = calls[0]
    assert package == tmp_path / "head.tar.gz"
    assert (kwargs["netuid"], kwargs["network"], kwargs["publish"]) == (17, "test", False)


def test_miner_reads_no_env_file_or_prexon_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PREXON_NETUID", "99")
    monkeypatch.setenv("PREXON_NETWORK", "test")
    calls = []
    monkeypatch.setattr(
        "miner.submit.submit", lambda *args, **kwargs: calls.append((args, kwargs)) or "a" * 64
    )
    monkeypatch.setattr(sys, "argv", ["prexon-miner", "submit", str(tmp_path / "head.tar.gz")])
    main()
    assert (calls[0][1]["netuid"], calls[0][1]["network"]) == (0, "finney")
    monkeypatch.setattr(sys, "argv", ["prexon-miner", "targets", "--env-file", ".env"])
    with pytest.raises(SystemExit) as stop:
        main()
    assert stop.value.code == 2
    assert "unrecognized arguments: --env-file" in capsys.readouterr().err


def test_miner_network_is_finney_or_test(tmp_path, monkeypatch, capsys):
    command = ["prexon-miner", "submit", str(tmp_path / "head.tar.gz"), "--upload-only"]
    monkeypatch.setattr(sys, "argv", [*command, "--network", "local"])
    with pytest.raises(SystemExit) as stop:
        main()
    assert stop.value.code == 2
    assert "argument --network: invalid choice: 'local'" in capsys.readouterr().err
