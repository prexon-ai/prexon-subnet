"""Validator-specific launch on top of the shared Lium rental client."""

from __future__ import annotations

import json
import shlex
import subprocess

from evaluation.config import LIUM_VOLUMES
from evaluation.lium_api import LiumClient, worker_settings
from evaluation.ssh import ssh_arguments
from shared.contracts import EvaluationJob

_LAUNCH = """import json,os,subprocess,sys
data=json.load(sys.stdin)
os.makedirs('/tmp/prexon-worker',exist_ok=True)
with open('/tmp/prexon-worker/launcher.log','ab') as log:
 subprocess.Popen([sys.executable,'-m','validator.worker','--job-url',data['job_url'],
                   '--result-url',data['result_url']],stdin=subprocess.DEVNULL,
                  stdout=log,stderr=log,start_new_session=True,
                  env={**os.environ,**data['env']})
"""


class LiumProvider(LiumClient):
    def launch(self, pod_id: str, job: EvaluationJob, job_url: str, result_url: str) -> None:
        pod = self.status(pod_id)
        # The same volume decision as build_request made when renting this pod.
        settings = worker_settings(job.target.model_id in LIUM_VOLUMES)
        subprocess.run(
            [
                *ssh_arguments(pod["ssh_connect_cmd"], pod_id),
                "/opt/prexon/.venv/bin/python -c " + shlex.quote(_LAUNCH),
            ],
            input=json.dumps({"job_url": job_url, "result_url": result_url, "env": settings}),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=True,
        )
