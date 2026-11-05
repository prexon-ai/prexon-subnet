"""Local GPU execution for development; it still runs the real evaluation worker."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from typing import Any
from urllib.parse import unquote, urlparse

from shared.config import Settings, TargetConfig
from shared.contracts import EvaluationJob


class LocalProvider:
    def __init__(self, settings: Settings) -> None:
        self.directory = settings.state_path.parent / "local-workers"
        self.directory.mkdir(parents=True, exist_ok=True)

    def build_request(self, session: dict[str, Any], target: TargetConfig) -> dict[str, Any]:
        return {"session_id": session["id"], "target_id": target.identity}

    def rent(self, session_id: str, request: dict[str, Any]) -> str:
        del request
        return session_id

    def status(self, pod_id: str) -> dict[str, Any]:
        del pod_id
        return {"status": "running", "spend_to_date": 0}

    def launch(self, pod_id: str, job: EvaluationJob, job_url: str, result_url: str) -> None:
        marker = self.directory / f"{pod_id}.json"
        if marker.exists():
            prior = json.loads(marker.read_text())
            if prior["job_id"] == job.job_id:
                try:
                    os.kill(prior["pid"], 0)
                    return
                except ProcessLookupError:
                    pass
        urls = [urlparse(u) for u in (job_url, result_url)]
        if any(u.scheme != "file" or u.netloc for u in urls):
            raise ValueError("Local provider requires local artifact storage")
        with (self.directory / f"{job.job_id}.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    "timeout",
                    "--signal=TERM",
                    "--kill-after=30",
                    str(max(1, int(job.deadline - time.time()))),
                    sys.executable,
                    "-m",
                    "validator.worker",
                    "--job",
                    unquote(urls[0].path),
                    "--result",
                    unquote(urls[1].path),
                    "--work-dir",
                    str(self.directory / pod_id),
                ],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
        marker.write_text(json.dumps({"job_id": job.job_id, "pid": process.pid}))

    def delete(self, pod_id: str) -> None:
        marker = self.directory / f"{pod_id}.json"
        if marker.exists():
            try:
                os.killpg(json.loads(marker.read_text())["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            marker.unlink()
