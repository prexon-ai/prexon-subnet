import hashlib
import io
import json
import tarfile

import pytest

from evaluation.remote import install, run
from tests.evaluation.test_datasets import dataset_target, prepared_pool
from tests.factories import make_checkpoint


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "candidate/../../outside"])
def test_upload_rejects_archive_path_escape(tmp_path, name):
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as stream:
        info = tarfile.TarInfo(name)
        info.size = 1
        stream.addfile(info, io.BytesIO(b"x"))
    archive.seek(0)
    with pytest.raises(ValueError, match="Unsafe"):
        install(tmp_path / "run", archive, hashlib.sha256(b"{}").hexdigest())


def test_remote_run_lock_prevents_duplicate_engine_processes(tmp_path, monkeypatch):
    import fcntl

    with (tmp_path / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        monkeypatch.setattr(
            "evaluation.remote.subprocess.run",
            lambda *a, **k: pytest.fail("Second engine launched"),
        )
        run(tmp_path)


def test_remote_install_validates_and_preserves_selected_dataset_plan(tmp_path):
    from evaluation.lium import inputs
    from evaluation.remote import install

    _, manifest = prepared_pool(tmp_path / "datasets")
    target = dataset_target(manifest)
    request, paths = inputs(
        target,
        make_checkpoint(tmp_path / "head"),
        None,
        None,
        "local",
        30000,
        None,
        "head",
        dataset_cache=tmp_path / "datasets",
    )

    def archive(request):
        payload = json.dumps(request).encode()
        data = io.BytesIO()
        with tarfile.open(fileobj=data, mode="w") as stream:
            for name, value in (
                ("request.json", payload),
                ("target.json", target.model_dump_json().encode()),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(value)
                stream.addfile(info, io.BytesIO(value))
            for arm, path in paths.items():
                for digest in request["checkpoints"][arm]:
                    stream.add(path / digest["path"], arcname=f"{arm}/{digest['path']}")
        data.seek(0)
        return data, hashlib.sha256(payload).hexdigest()

    data, digest = archive(request)
    install(tmp_path / "remote", data, digest)
    stored = json.loads((tmp_path / "remote/input/request.json").read_bytes())
    assert stored["dataset_plan"] == request["dataset_plan"]
    wrong = {**request, "seed": "different"}
    data, digest = archive(wrong)
    with pytest.raises(ValueError, match="seed"):
        install(tmp_path / "wrong-remote", data, digest)
