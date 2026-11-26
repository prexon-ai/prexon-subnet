import json
import subprocess
import sys

import pytest

from miner.cli import main
from miner.train import train_and_export
from shared.artifacts import extract_package
from shared.checkpoints import sha256_file
from tests.factories import make_checkpoint, provenance, target, write_competitions


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
    assert record["competition_id"] == (None if direct else configured.competition_id)


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


def test_package_reads_its_competition_from_competitions_json(tmp_path, monkeypatch, capsys):
    configured = target()
    write_competitions(configured)
    head = make_checkpoint(tmp_path / "head")
    record = tmp_path / "provenance.json"
    record.write_text(provenance().model_dump_json())
    output = tmp_path / "head.tar.gz"
    files = ["--checkpoint", str(head), "--provenance", str(record), "--output", str(output)]
    monkeypatch.setattr(
        sys, "argv", ["prexon-miner", "package", "--competition", configured.model_id, *files]
    )
    main()
    manifest = extract_package(output, tmp_path / "restored", sha256_file(output), 1024**2)
    assert manifest.target_id == manifest.competition_id == configured.model_id
    assert manifest.target_revision == configured.revision
    monkeypatch.setattr(
        sys, "argv", ["prexon-miner", "package", "--competition", "org/unknown", *files]
    )
    with pytest.raises(SystemExit):
        main()
    assert "No competition for org/unknown in competitions/" in capsys.readouterr().err


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


def test_miner_takes_everything_from_flags_never_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PREXON_NETUID", "99")
    monkeypatch.setenv("PREXON_NETWORK", "test")
    calls = []
    monkeypatch.setattr(
        "miner.submit.submit", lambda *args, **kwargs: calls.append((args, kwargs)) or "a" * 64
    )
    package = str(tmp_path / "head.tar.gz")
    # There is no default subnet: submit needs --netuid, and PREXON_NETUID is ignored.
    monkeypatch.setattr(sys, "argv", ["prexon-miner", "submit", package])
    with pytest.raises(SystemExit) as stop:
        main()
    assert stop.value.code == 2 and "--netuid" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["prexon-miner", "submit", package, "--netuid", "17"])
    main()
    assert calls[0][1] == {
        "netuid": 17,
        "network": "finney",
        "api_url": "https://prexonai.com/api/v1",
        "wallet_name": "default",
        "wallet_hotkey": "default",
        "wallet_path": "~/.bittensor/wallets",
        "targets": (),
        "max_upload_bytes": 4 * 1024**3 + 2 * 1024**2,
        "publish": True,
    }
    monkeypatch.setattr(sys, "argv", ["prexon-miner", "competitions", "--env-file", ".env"])
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
