"""Strict OpenSSH descriptors and monitored subprocess transport."""

from __future__ import annotations

import re
import shlex
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from evaluation.config import SSH_KNOWN_HOSTS_DIR, SSH_PRIVATE_KEY


def ssh_arguments(descriptor: str, pod_id: str) -> list[str]:
    """ssh's arguments for one pod: its host key is trusted on first connect, for that pod only."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", pod_id):
        raise ValueError("Unsafe pod identity")
    parts = shlex.split(descriptor)
    if len(parts) == 4 and parts[0:2] == ["ssh", "-p"]:
        port, host = parts[2], parts[3]
    elif len(parts) == 4 and parts[0] == "ssh" and parts[2] == "-p":
        port, host = parts[3], parts[1]
    elif len(parts) == 2 and parts[0] == "ssh":
        port, host = "22", parts[1]
    else:
        raise ValueError("Unsupported SSH connection descriptor")
    if (
        not port.isdigit()
        or not 1 <= int(port) <= 65535
        or not re.fullmatch(r"[a-zA-Z0-9_.@:-]+", host)
        or host.startswith("-")
    ):
        raise ValueError("Unsafe SSH connection descriptor")
    known_hosts = SSH_KNOWN_HOSTS_DIR.expanduser() / pod_id
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-i",
        str(SSH_PRIVATE_KEY.expanduser()),
        "-p",
        port,
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ForwardAgent=no",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        host,
    ]


class SshTransport:
    def __init__(self, descriptor: str, pod_id: str, tick: Callable[[], None]) -> None:
        self.arguments = ssh_arguments(descriptor, pod_id)
        self.tick = tick

    def run(
        self,
        argv: Sequence[str],
        *,
        source: Path | None = None,
        destination: Path | None = None,
        limit: int = 32 * 1024**2,
        timeout_seconds: float | None = None,
        data: bytes | None = None,
    ) -> int:
        """Stream files without shell evaluation locally; monitor cancellation and spend."""
        with tempfile.TemporaryFile() as discard:
            started = time.monotonic()
            incoming = (
                source.open("rb")
                if source
                else subprocess.PIPE
                if data is not None
                else subprocess.DEVNULL
            )
            outgoing = destination.open("wb") if destination else discard
            process = None
            try:
                process = subprocess.Popen(
                    [*self.arguments, shlex.join(argv)],
                    stdin=incoming,
                    stdout=outgoing,
                    stderr=discard,
                    start_new_session=True,
                )
                if data is not None and process.stdin:
                    # Small settings only: the pipe takes them without waiting for a reader.
                    process.stdin.write(data)
                    process.stdin.close()
                while process.poll() is None:
                    self.tick()
                    if (
                        timeout_seconds is not None
                        and time.monotonic() - started >= timeout_seconds
                    ):
                        raise TimeoutError("SSH operation timed out")
                    if destination and outgoing.tell() > limit:
                        raise ValueError("Remote response exceeds transfer limit")
                    time.sleep(0.2)
                if destination and outgoing.tell() > limit:
                    raise ValueError("Remote response exceeds transfer limit")
                return process.returncode
            finally:
                if process is not None and process.poll() is None:
                    import os

                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                if source:
                    incoming.close()
                if destination:
                    outgoing.close()
