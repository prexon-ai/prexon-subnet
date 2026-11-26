"""Validator-specific launch on top of the shared Lium rental client."""

from __future__ import annotations

import json
import shlex
import subprocess

from evaluation.lium_api import LiumClient
from evaluation.ssh import ssh_arguments
from shared.contracts import EvaluationJob

_LAUNCH = """import json,os,subprocess,sys
data=json.load(sys.stdin)
os.makedirs('/tmp/prexon-worker',exist_ok=True)
with open('/tmp/prexon-worker/launcher.log','ab') as log:
 subprocess.Popen([sys.executable,'-m','validator.worker','--job-url',data['job_url'],
                   '--result-url',data['result_url']],stdin=subprocess.DEVNULL,
                  stdout=log,stderr=log,start_new_session=True)
"""


class LiumProvider(LiumClient):
    def launch(self, pod_id: str, job: EvaluationJob, job_url: str, result_url: str) -> None:
        del job
        pod = self.status(pod_id)
        subprocess.run(
            [
                *ssh_arguments(pod["ssh_connect_cmd"]),
                "/opt/prexon/.venv/bin/python -c " + shlex.quote(_LAUNCH),
            ],
            input=json.dumps({"job_url": job_url, "result_url": result_url}),
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=True,
        )
