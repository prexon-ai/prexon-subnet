from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import pytest

from shared.config import Settings
from tests.factories import target
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


def test_chain_reads_finalized_snapshot_and_excludes_future():
    boundary = ChainBoundary()
    chain = BittensorChain(Settings(role="validator", netuid=1), client=boundary)
    snapshot = chain.scan()
    assert snapshot.block == 100
    assert snapshot.owners == {1: "miner"}
    assert [c.artifact for c in snapshot.commitments] == ["c" * 64]
    assert snapshot.commitments[0].submitted_at == 1_800_000_000 + 99 * 12
    assert boundary.queries == [100, 100]


def test_lium_rent_uses_exact_idempotency_and_bounded_termination(tmp_path):
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

    settings = Settings(
        role="validator",
        lium_template_id="template",
        ssh_public_key=key,
        session_seconds=3601,
        session_max_cost=4,
    )
    client = httpx.Client(base_url="https://lium.io/api", transport=httpx.MockTransport(handle))
    provider = LiumProvider(settings, "private", client=client)
    session = {"id": "a" * 32, "started_at": 100, "deadline": 3701}
    request = provider.build_request(session, target())
    assert request["termination_hours"] == 2
    assert request["max_price_per_gpu_hour"] == 2
    assert provider.rent(session["id"], request) == "pod"
    assert provider.rent(session["id"], request) == "pod"
    assert requests[-1].content == requests[-2].content
    provider.delete("pod")


def test_lium_rejects_template_image_mismatch(tmp_path):
    client = httpx.Client(
        base_url="https://lium.io/api",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200, json={"docker_image": "untrusted", "docker_image_digest": "sha256:" + "b" * 64}
            )
        ),
    )
    provider = LiumProvider(
        Settings(role="validator", lium_template_id="template"), "key", client=client
    )
    with pytest.raises(ValueError, match="image"):
        provider.build_request({"id": "a" * 32, "started_at": 1, "deadline": 3601}, target())
