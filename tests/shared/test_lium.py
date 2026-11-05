import httpx
import pytest

from shared.config import Settings
from shared.lium import LiumClient
from shared.ssh import ssh_arguments


def test_quote_is_not_a_rental_and_keeps_the_configured_request():
    requests = []

    def handle(request):
        requests.append(request)
        assert b'"dry_run":true' in request.content
        assert "Idempotency-Key" not in request.headers
        return httpx.Response(200, json={"success": True, "dry_run": True, "price_per_hour": 2.5})

    client = httpx.Client(base_url="https://lium.io/api", transport=httpx.MockTransport(handle))
    lium = LiumClient(Settings(role="miner"), "private", client=client)
    request = {"gpu_type": "H100", "pod_name": "test"}
    assert lium.quote(request)["price_per_hour"] == 2.5
    assert "dry_run" not in request
    assert len(requests) == 1


def test_ssh_descriptor_is_parsed_without_evaluating_provider_shell(tmp_path):
    settings = Settings(role="miner", ssh_known_hosts=tmp_path / "known_hosts")
    args = ssh_arguments(settings, "ssh root@example -p 2222")
    assert args[-1] == "root@example"
    assert "ForwardAgent=no" in args
    assert "ClearAllForwardings=yes" in args
    assert args[args.index("-p") + 1] == "2222"
    with pytest.raises(ValueError):
        ssh_arguments(settings, "ssh -o ProxyCommand=evil root@example")
    with pytest.raises(ValueError):
        ssh_arguments(settings, "ssh root@example; touch /tmp/unsafe")


def test_ssh_transport_cancellation_reaps_the_process(tmp_path, monkeypatch):
    import subprocess

    from shared.ssh import SshTransport

    processes = []
    original = subprocess.Popen

    def start(*args, **kwargs):
        process = original(["sleep", "60"], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("shared.ssh.subprocess.Popen", start)

    def cancel():
        raise KeyboardInterrupt

    transport = SshTransport(
        Settings(role="miner", ssh_known_hosts=tmp_path / "hosts"), "ssh root@example", cancel
    )
    with pytest.raises(KeyboardInterrupt):
        transport.run(["a command", "$(not-a-local-shell)"])
    assert processes[0].poll() is not None
