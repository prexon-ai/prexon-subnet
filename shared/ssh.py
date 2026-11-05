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

from shared.config import Settings


def ssh_arguments(settings: Settings, descriptor: str) -> list[str]:
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
    known_hosts = settings.ssh_known_hosts.expanduser()
    known_hosts.parent.mkdir(parents=True, exist_ok=True)
    return [
        "ssh",
        "-F",
        "/dev/null",
        "-i",
        str(settings.ssh_private_key.expanduser()),
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
    def __init__(self, settings: Settings, descriptor: str, tick: Callable[[], None]) -> None:
        self.arguments = ssh_arguments(settings, descriptor)
        self.tick = tick

    def run(
        self,
        argv: Sequence[str],
        *,
        source: Path | None = None,
        destination: Path | None = None,
        limit: int = 32 * 1024**2,
        timeout_seconds: float | None = None,
    ) -> int:
        """Stream files without shell evaluation locally; monitor cancellation and spend."""
        with tempfile.TemporaryFile() as discard:
            started = time.monotonic()
            incoming = source.open("rb") if source else subprocess.DEVNULL
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
                if source and not isinstance(incoming, int):
                    incoming.close()
                if destination:
                    outgoing.close()
