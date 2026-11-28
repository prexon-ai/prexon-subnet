from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from tests.factories import PrunedChainNode, target
from validator.chain import BittensorChain
from validator.providers.lium import LiumProvider


class ChainBoundary:
    def __init__(self):
        self.substrate = self
        self.queries = []

    def get_chain_finalised_head(self):
        return "finalized-hash"

    def get_block_number(self, block_hash):
        assert block_hash == "finalized-hash"
        return 100

    def metagraph(self, netuid, block):
        self.queries.append(block)
        return SimpleNamespace(neurons=[SimpleNamespace(uid=1, hotkey="miner")])

    def get_all_revealed_commitments(self, netuid, block):
        self.queries.append(block)
        return {"miner": ((99, "px1:" + "c" * 64), (101, "px1:" + "d" * 64))}

    def get_block_hash(self, block):
        return f"hash-{block}"

    def get_timestamp(self, block):
        return datetime.fromtimestamp(1_800_000_000 + block * 12, UTC)


def test_chain_reads_finalized_snapshot_and_excludes_future(set_config):
    boundary = ChainBoundary()
    set_config(NETUID=1)
    chain = BittensorChain(client=boundary)
    snapshot = chain.scan()
    assert snapshot.block == 100
    assert snapshot.owners == {1: "miner"}
    assert [c.artifact for c in snapshot.commitments] == ["c" * 64]
    assert snapshot.commitments[0].submitted_at == 1_800_000_000 + 99 * 12
    assert boundary.queries == [100, 100]


def test_chain_scan_does_not_reread_the_time_of_reveals_it_already_knows(set_config):
    # Reveals stay in chain storage indefinitely; their block state does not.
    node = PrunedChainNode(head=100, reveals={"miner": ((99, "px1:" + "c" * 64),)})
    set_config(NETUID=1)
    chain = BittensorChain(client=node)
    first = chain.scan()
    assert first.commitments[0].submitted_at == 1_800_000_000 + 99 * 12
    node.head, node.timestamp_queries = 190, []
    later = chain.scan(189, known={first.commitments[0].identity})
    assert [c.identity for c in later.commitments] == [first.commitments[0].identity]
    assert node.timestamp_queries == []


def test_chain_scan_reads_each_reveal_block_hash_once(set_config):
    # Every scanned block lists the whole reveal history, here one reveal at block 99.
    node = PrunedChainNode(head=100, reveals={"miner": ((99, "px1:" + "c" * 64),)})
    set_config(NETUID=1)
    chain = BittensorChain(client=node)
    first = chain.scan()
    node.head = 110
    later = chain.scan(100, known={first.commitments[0].identity})  # replays blocks 101-110
    assert [c.identity for c in later.commitments] == [first.commitments[0].identity]
    assert node.block_hash_queries == [99]


@pytest.mark.parametrize("failure", ["discarded", "zero"])
def test_chain_scan_keeps_a_new_reveal_whose_block_time_is_unavailable(failure, set_config):
    node = PrunedChainNode(head=190, reveals={"miner": ((99, "px1:" + "c" * 64),)})
    if failure == "zero":  # Some public nodes answer for discarded state with an empty value.
        node.get_timestamp = lambda block: datetime.fromtimestamp(0, UTC)
    set_config(NETUID=1)
    chain = BittensorChain(client=node)
    snapshot = chain.scan()
    # Admission falls back to the first-discovery time when chain time is absent.
    assert [c.artifact for c in snapshot.commitments] == ["c" * 64]
    assert snapshot.commitments[0].submitted_at is None


def test_lium_rent_uses_exact_idempotency_and_bounded_termination(tmp_path, set_config):
    key = tmp_path / "id.pub"
    key.write_text("ssh-ed25519 public")
    requests = []

    def handle(request):
        requests.append(request)
        if request.url.path == "/api/templates/template":
            return httpx.Response(
                200, json={"docker_image": "image", "docker_image_digest": "sha256:" + "a" * 64}
            )
        if request.method == "POST":
            assert request.headers["Idempotency-Key"] == "prexon-" + "a" * 32
            return httpx.Response(200, json={"success": True, "pod_id": "pod"})
        return httpx.Response(204)

    set_config(LIUM_TEMPLATE_ID="template", SSH_PUBLIC_KEY=key)
    client = httpx.Client(base_url="https://lium.io/api", transport=httpx.MockTransport(handle))
    provider = LiumProvider("private", 4, client=client)
    session = {"id": "a" * 32, "started_at": 100, "deadline": 3701}
    request = provider.build_request(session, target())
    assert request["termination_hours"] == 2
    assert request["max_price_per_gpu_hour"] == 2
    assert provider.rent(session["id"], request) == "pod"
    assert provider.rent(session["id"], request) == "pod"
    assert requests[-1].content == requests[-2].content
    provider.delete("pod")


@pytest.mark.parametrize(
    "status,body,refused",
    [
        (409, {"success": False, "code": "no_executor_matches_spec"}, True),
        (402, {"detail": "Insufficient balance"}, True),
        # The provider may still hold or be creating a pod for this identity.
        (409, {"detail": {"code": "idempotency_in_progress"}}, False),
        (409, {"detail": {"code": "idempotency_outcome_unknown"}}, False),
        (503, {"detail": "unavailable"}, False),
    ],
)
def test_lium_separates_a_definite_refusal_from_an_unknown_rental_outcome(status, body, refused):
    from evaluation.lium_api import RentalRefused

    client = httpx.Client(
        base_url="https://lium.io/api",
        transport=httpx.MockTransport(lambda _: httpx.Response(status, json=body)),
    )
    provider = LiumProvider("key", 100, client=client)
    with pytest.raises(RentalRefused if refused else httpx.HTTPStatusError):
        provider.rent("a" * 32, {"pod_name": "prexon-" + "a" * 32})


def test_lium_rejects_template_image_mismatch(tmp_path, set_config):
    client = httpx.Client(
        base_url="https://lium.io/api",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"docker_image": "untrusted", "docker_image_digest": "sha256:" + "b" * 64}
            )
        ),
    )
    set_config(LIUM_TEMPLATE_ID="template")
    provider = LiumProvider("key", 100, client=client)
    with pytest.raises(ValueError, match="image"):
        provider.build_request({"id": "a" * 32, "started_at": 1, "deadline": 3601}, target())


@pytest.mark.parametrize("tag,accepted", [("v0.5.23", True), ("v0.5.22", False), (None, False)])
def test_lium_template_names_image_and_tag_separately(tmp_path, set_config, tag, accepted):
    key = tmp_path / "id.pub"
    key.write_text("ssh-ed25519 public")
    digest = "sha256:" + "a" * 64
    template = {
        "docker_image": "ghcr.io/prexon-ai/prexon-worker",
        "docker_image_tag": tag,
        "docker_image_digest": digest,
    }
    client = httpx.Client(
        base_url="https://lium.io/api",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=template)),
    )
    set_config(LIUM_TEMPLATE_ID="template", SSH_PUBLIC_KEY=key)
    competition = target().model_copy(
        update={"worker_image": f"ghcr.io/prexon-ai/prexon-worker:v0.5.23@{digest}"}
    )
    provider = LiumProvider("key", 100, client=client)
    session = {"id": "a" * 32, "started_at": 1, "deadline": 3601}
    if accepted:
        assert provider.build_request(session, competition)["template_id"] == "template"
    else:
        with pytest.raises(ValueError, match="image"):
            provider.build_request(session, competition)


def test_lium_rental_mounts_only_its_own_competitions_volume(tmp_path, set_config):
    key = tmp_path / "id.pub"
    key.write_text("ssh-ed25519 public")
    template = {"docker_image": "image", "docker_image_digest": "sha256:" + "a" * 64}
    client = httpx.Client(
        base_url="https://lium.io/api",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=template)),
    )
    set_config(LIUM_TEMPLATE_ID="template", SSH_PUBLIC_KEY=key, LIUM_VOLUMES={"org/model1": "vol"})
    provider = LiumProvider("key", 100, client=client)
    session = {"id": "a" * 32, "started_at": 1, "deadline": 3601}
    assert provider.build_request(session, target(1))["volume_id"] == "vol"
    assert "volume_id" not in provider.build_request(session, target(2))


@pytest.mark.parametrize("model, volume", [(1, True), (2, False)])
def test_lium_worker_gets_settings_and_token_on_stdin_never_in_the_command(
    monkeypatch, set_config, model, volume
):
    import json

    from tests.factories import job

    set_config(HF_TOKEN="hf_secret", LIUM_VOLUMES={"org/model1": "vol"})
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(500)))
    provider = LiumProvider("key", 100, client=client)
    monkeypatch.setattr(provider, "status", lambda pod_id: {"ssh_connect_cmd": "ssh root@host"})
    runs = []
    monkeypatch.setattr(
        "validator.providers.lium.subprocess.run",
        lambda argv, **kwargs: runs.append((argv, kwargs)),
    )
    provider.launch("pod-1", job(target(model)), "https://job", "https://result")
    ((argv, kwargs),) = runs
    settings = json.loads(kwargs["input"])["env"]
    assert "hf_secret" not in " ".join(argv)
    assert settings["HF_TOKEN"] == "hf_secret" and settings["HF_XET_HIGH_PERFORMANCE"] == "1"
    assert ("PREXON_TARGET_VOLUME" in settings) is volume


def test_lium_launch_script_starts_the_worker_with_the_settings_it_receives(monkeypatch):
    import io
    import json
    import os
    import subprocess
    import sys

    from validator.providers.lium import _LAUNCH

    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: started.append(kwargs))
    monkeypatch.setattr(os, "makedirs", lambda *args, **kwargs: None)
    sent = {"job_url": "j", "result_url": "r", "env": {"HF_TOKEN": "hf_secret"}}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(sent)))
    exec(_LAUNCH, {"open": lambda *args: io.BytesIO()})
    assert started[0]["env"]["HF_TOKEN"] == "hf_secret" and started[0]["env"]["PATH"]
