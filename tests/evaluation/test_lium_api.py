import httpx
import pytest

from evaluation.lium_api import LiumClient
from evaluation.ssh import ssh_arguments


def test_quote_is_not_a_rental_and_keeps_the_configured_request():
    requests = []

    def handle(request):
        requests.append(request)
        assert b'"dry_run":true' in request.content
        assert "Idempotency-Key" not in request.headers
        return httpx.Response(200, json={"success": True, "dry_run": True, "price_per_hour": 2.5})

    client = httpx.Client(base_url="https://lium.io/api", transport=httpx.MockTransport(handle))
    lium = LiumClient("private", 100, client=client)
    request = {"gpu_type": "H100", "pod_name": "test"}
    assert lium.quote(request)["price_per_hour"] == 2.5
    assert "dry_run" not in request
    assert len(requests) == 1


def test_ssh_descriptor_is_parsed_without_evaluating_provider_shell(tmp_path, set_config):
    set_config(SSH_KNOWN_HOSTS_DIR=tmp_path / "known_hosts")
    args = ssh_arguments("ssh root@example -p 2222", "pod-1")
    assert args[-1] == "root@example"
    assert "ForwardAgent=no" in args
    assert "ClearAllForwardings=yes" in args
    assert args[args.index("-p") + 1] == "2222"
    with pytest.raises(ValueError):
        ssh_arguments("ssh -o ProxyCommand=evil root@example", "pod-1")
    with pytest.raises(ValueError):
        ssh_arguments("ssh root@example; touch /tmp/unsafe", "pod-1")


def test_each_pod_trusts_its_own_host_key(tmp_path, set_config):
    # A new pod on a machine:port rented before has a new host key; trust follows the pod.
    set_config(SSH_KNOWN_HOSTS_DIR=tmp_path / "known_hosts")

    def known_hosts(pod_id):
        args = ssh_arguments("ssh root@93.120.231.186 -p 40500", pod_id)
        return next(a for a in args if a.startswith("UserKnownHostsFile="))

    assert known_hosts("pod-a") == f"UserKnownHostsFile={tmp_path / 'known_hosts' / 'pod-a'}"
    assert known_hosts("pod-a") != known_hosts("pod-b")
    with pytest.raises(ValueError):
        ssh_arguments("ssh root@example", "../shared")


def test_ssh_transport_cancellation_reaps_the_process(tmp_path, monkeypatch, set_config):
    import subprocess

    from evaluation.ssh import SshTransport

    processes = []
    original = subprocess.Popen

    def start(*args, **kwargs):
        process = original(["sleep", "60"], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr("evaluation.ssh.subprocess.Popen", start)

    def cancel():
        raise KeyboardInterrupt

    set_config(SSH_KNOWN_HOSTS_DIR=tmp_path / "hosts")
    transport = SshTransport("ssh root@example", "pod-1", cancel)
    with pytest.raises(KeyboardInterrupt):
        transport.run(["a command", "$(not-a-local-shell)"])
    assert processes[0].poll() is not None
