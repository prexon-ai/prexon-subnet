import json
import subprocess
import sys

import pytest

from evaluation.cli import main
from tests.factories import evaluate_argv, target


def test_cli_starts_without_role_or_gpu_dependencies(tmp_path):
    code = """
import builtins
import sys

original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split('.')[0] in {
        'validator', 'miner', 'api', 'bittensor', 'boto3', 'torch', 'transformers', 'datasets'
    }:
        raise AssertionError(f'Standalone CLI imported {name}')
    return original(name, globals, locals, fromlist, level)

builtins.__import__ = guarded
sys.argv = ['prexon-evaluate', '--help']
from evaluation.cli import main
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert "--competition" in result.stdout


def test_cli_needs_a_listed_competition_and_a_head_except_to_clean_up(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prexon-evaluate", "--head", "head"])
    with pytest.raises(SystemExit):
        main()
    assert "--competition and --head" in capsys.readouterr().err
    argv = ["prexon-evaluate", "--competition", "org/unknown", "--head", "head"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        main()
    assert "No competition for org/unknown in competitions/" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", ["prexon-evaluate", "--cleanup-only"])
    main()
    assert "Rental reconciliation complete" in capsys.readouterr().out


def test_public_cli_only_needs_competition_and_head_for_head_mode(
    tmp_path, monkeypatch, capsys, pod
):
    head, snapshot, calls = pod
    argv = evaluate_argv(
        head,
        "--target-path",
        str(snapshot),
        "--public-head",
        "missing-public-head-is-not-used-in-head-mode",
    )
    monkeypatch.setattr(sys, "argv", argv)
    main()
    reports = list((tmp_path / "reports/head").glob("*.json"))
    assert len(reports) == 1
    data = json.loads(reports[0].read_bytes())
    assert data["mode"] == "head"
    assert data["assessment"] == "head_measured"
    assert data["score"] is None and data["decision"] is None
    assert data["head_metrics"]["low"]["throughput_tps"] == pytest.approx(20)
    assert all(draft is not None for draft, *_ in calls)
    assert len(calls) == 1
    console = capsys.readouterr().out
    assert "20.000" in console
    assert "speedup" not in console.lower()
    # Repeating a completed development command measures again and preserves both reports.
    main()
    assert len(list((tmp_path / "reports/head").glob("*.json"))) == 2


def test_unfinished_lium_rental_cannot_be_resumed_with_inline_backend(tmp_path, monkeypatch, pod):
    import sys

    from evaluation.cli import main

    head, _, _ = pod
    argv = evaluate_argv(head)
    directory = tmp_path / "reports/head"
    directory.mkdir(parents=True)
    (directory / "unfinished.lium.json").write_text(
        json.dumps(
            {
                "phase": "active",
                "cleanup": "not_applicable",
                "output": str(directory / "report.json"),
            }
        )
    )
    monkeypatch.setattr(
        "evaluation.cli.execute",
        lambda *args: pytest.fail("Started work while a Lium rental needs recovery"),
    )
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2


def test_cli_runs_single_submission_round_without_production_side_effects(
    tmp_path, monkeypatch, capsys, pod
):
    import builtins

    head, snapshot, calls = pod
    config = target().model_copy(update={"qualification_artifact": None})
    argv = evaluate_argv(
        head,
        *("--target-path", str(snapshot), "--mode", "full", "--seed", "7"),
        *("--public-head", str(head), "--incumbent-head", str(head)),
        competition=config.model_dump(mode="json"),
    )
    monkeypatch.setattr(sys, "argv", argv)
    original_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name.split(".")[0] in {"validator", "miner", "api", "bittensor", "boto3"}:
            raise AssertionError(f"Local evaluation imported {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)
    main()
    data = json.loads(next((tmp_path / "reports/full").glob("*.json")).read_bytes())
    assert data["authoritative"] is False
    assert data["purpose"] == "development" and data["submission_count"] == 1
    assert set(data["checkpoints"]) == {"candidate", "public", "incumbent"}
    assert data["decision"]["outcome"] == "won"
    assert data["score"] == pytest.approx(2)
    assert data["scores"]["public"] == pytest.approx(1.8)
    assert {row["arm"] for row in data["measurements"]} == {
        "candidate",
        "target",
        "public",
        "incumbent",
    }
    assert any(draft is None for draft, *_ in calls)
    assert data["seed"] == "7" and data["qualification_passed"] is False
    assert "2.000x" in capsys.readouterr().out
    assert all(call[2]["port"] == 30000 for call in calls)
    assert not list(tmp_path.rglob("*.sqlite*"))


def test_cli_writes_inconclusive_report_and_returns_nonzero_on_runtime_failure(
    tmp_path, monkeypatch, pod
):
    from contextlib import asynccontextmanager

    head, snapshot, _ = pod

    @asynccontextmanager
    async def broken(*args, **kwargs):
        raise RuntimeError("engine unavailable")
        yield  # pragma: no cover

    monkeypatch.setattr("evaluation.measurement.launch_server", broken)
    monkeypatch.setattr(sys, "argv", evaluate_argv(head, "--target-path", str(snapshot)))
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 2
    report = json.loads(next((tmp_path / "reports/head").glob("*.json")).read_bytes())
    assert report["score"] is None and report["status"] == "inconclusive"
    assert report["decision"] is None
