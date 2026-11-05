"""Finalized chain discovery and confirmed weight publication."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from shared.chain import connect, require_success, wallet
from shared.config import FrozenModel, Settings
from shared.contracts import Commitment


class ChainSnapshot(FrozenModel):
    block: int
    finalized_block: int | None = None
    commitments: tuple[Commitment, ...]
    owners: dict[int, str]
    block_hashes: dict[int, str] = {}


class Chain(Protocol):
    def scan(self, start_block: int | None = None) -> ChainSnapshot: ...
    def publish(self, weights: dict[int, float]) -> None: ...
    def current_owners(self) -> dict[int, str]: ...
    def block_hash(self, block: int) -> str: ...
    def finalized_height(self) -> int: ...


class BittensorChain:
    def __init__(self, settings: Settings, *, client: Any = None, signer: Any = None) -> None:
        self.settings = settings
        self.client = client if client is not None else connect(settings)
        self.signer = signer

    def scan(self, start_block: int | None = None) -> ChainSnapshot:
        finalized = self.finalized_height()
        # Replay snapshots after downtime to avoid skipping the bounded on-chain reveal history.
        end = finalized if start_block is None else min(finalized, start_block + 120)
        blocks = [end] if start_block is None else range(start_block + 1, end + 1)
        meta = self.client.metagraph(self.settings.netuid, block=end)
        owners = {int(n.uid): str(n.hotkey) for n in meta.neurons}
        by_hotkey = {hotkey: uid for uid, hotkey in owners.items()}
        found: dict[str, Commitment] = {}
        timestamps: dict[int, float] = {}
        for block in blocks:
            records = self.client.get_all_revealed_commitments(self.settings.netuid, block=block)
            for hotkey, entries in records.items():
                if hotkey not in by_hotkey:
                    continue
                for reveal_block, payload in entries:
                    match = re.fullmatch(r"px1:([0-9a-f]{64})", payload)
                    if not match or reveal_block > end:
                        continue
                    revealed_at = int(reveal_block)
                    if revealed_at not in timestamps:
                        timestamps[revealed_at] = self.client.get_timestamp(revealed_at).timestamp()
                    commitment = Commitment(
                        netuid=self.settings.netuid,
                        block=int(reveal_block),
                        block_hash=self.client.get_block_hash(int(reveal_block)),
                        # Storage does not expose extrinsic positions. Same-block ties are hotkey
                        # then artifact order; position must not depend on a sliding history window.
                        extrinsic_index=0,
                        uid=by_hotkey[hotkey],
                        hotkey=hotkey,
                        artifact=match.group(1),
                        submitted_at=timestamps[revealed_at],
                    )
                    found[commitment.identity] = commitment
        return ChainSnapshot(
            block=end,
            finalized_block=finalized,
            owners=owners,
            commitments=tuple(
                sorted(found.values(), key=lambda c: (c.block, c.hotkey, c.artifact))
            ),
        )

    def publish(self, weights: dict[int, float]) -> None:
        if self.signer is None:
            self.signer = wallet(self.settings)
        uids = sorted(weights)
        result = self.client.set_weights(
            wallet=self.signer,
            netuid=self.settings.netuid,
            uids=uids,
            weights=[weights[uid] for uid in uids],
            version_key=self.settings.weights_version,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        require_success(result)

    def current_owners(self) -> dict[int, str]:
        meta = self.client.metagraph(self.settings.netuid, block=self.finalized_height())
        return {int(n.uid): str(n.hotkey) for n in meta.neurons}

    def block_hash(self, block: int) -> str:
        if block > self.finalized_height():
            raise ValueError("Sampling block is not finalized")
        return str(self.client.get_block_hash(block))

    def finalized_height(self) -> int:
        head = self.client.substrate.get_chain_finalised_head()
        return int(self.client.substrate.get_block_number(head))


class FileChain:
    """Explicit local development input, never a network submission authority."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def scan(self, start_block: int | None = None) -> ChainSnapshot:
        del start_block
        return ChainSnapshot.model_validate_json(self.path.read_bytes())

    def publish(self, weights: dict[int, float]) -> None:
        output = self.path.with_name("weights.json")
        temp = output.with_suffix(".tmp")
        temp.write_text(json.dumps(weights, sort_keys=True) + "\n")
        temp.replace(output)

    def current_owners(self) -> dict[int, str]:
        return self.scan().owners

    def block_hash(self, block: int) -> str:
        snapshot = self.scan()
        if block > snapshot.block or block not in snapshot.block_hashes:
            raise ValueError("File chain needs an explicit finalized sampling block hash")
        return snapshot.block_hashes[block]

    def finalized_height(self) -> int:
        return self.scan().block
