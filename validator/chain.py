"""Finalized chain discovery and confirmed weight publication."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Container
from pathlib import Path
from typing import Any, Protocol

from shared.chain import connect, require_success, wallet
from shared.config import FrozenModel, Settings
from shared.contracts import Commitment

log = logging.getLogger(__name__)


class ChainSnapshot(FrozenModel):
    block: int
    finalized_block: int | None = None
    commitments: tuple[Commitment, ...]
    owners: dict[int, str]
    block_hashes: dict[int, str] = {}


class Chain(Protocol):
    def scan(self, start_block: int | None = None, known: Container[str] = ()) -> ChainSnapshot: ...
    def publish(self, weights: dict[int, float]) -> None: ...
    def current_owners(self) -> dict[int, str]: ...
    def block_hash(self, block: int) -> str: ...
    def finalized_height(self) -> int: ...


class BittensorChain:
    def __init__(self, settings: Settings, *, client: Any = None, signer: Any = None) -> None:
        self.settings = settings
        self.client = client if client is not None else connect(settings)
        self.signer = signer
        # Finalized block hashes never change; kept for the reveal blocks still listed.
        self._reveal_hashes: dict[int, str] = {}

    def scan(self, start_block: int | None = None, known: Container[str] = ()) -> ChainSnapshot:
        """Read finalized reveals; `known` holds commitment identities already discovered.

        Reveals stay in chain storage long after non-archive nodes discard the state of
        their blocks, so block times are read only for reveals that are new.
        """
        finalized = self.finalized_height()
        # Replay snapshots after downtime to avoid skipping the bounded on-chain reveal history.
        end = finalized if start_block is None else min(finalized, start_block + 120)
        blocks = [end] if start_block is None else range(start_block + 1, end + 1)
        meta = self.client.metagraph(self.settings.netuid, block=end)
        owners = {int(n.uid): str(n.hotkey) for n in meta.neurons}
        by_hotkey = {hotkey: uid for uid, hotkey in owners.items()}
        found: dict[str, Commitment] = {}
        timestamps: dict[int, float | None] = {}
        previous = self._reveal_hashes
        hashes: dict[int, str] = {}
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
                    if revealed_at not in hashes:
                        cached = previous.get(revealed_at)
                        hashes[revealed_at] = cached or self.client.get_block_hash(revealed_at)
                    commitment = Commitment(
                        netuid=self.settings.netuid,
                        block=revealed_at,
                        block_hash=hashes[revealed_at],
                        # Storage does not expose extrinsic positions. Same-block ties are hotkey
                        # then artifact order; position must not depend on a sliding history window.
                        extrinsic_index=0,
                        uid=by_hotkey[hotkey],
                        hotkey=hotkey,
                        artifact=match.group(1),
                    )
                    identity = commitment.identity
                    if identity in found:
                        continue
                    if identity not in known:
                        if revealed_at not in timestamps:
                            timestamps[revealed_at] = self._block_time(revealed_at)
                        commitment = commitment.model_copy(
                            update={"submitted_at": timestamps[revealed_at]}
                        )
                    found[identity] = commitment
        self._reveal_hashes = hashes
        return ChainSnapshot(
            block=end,
            finalized_block=finalized,
            owners=owners,
            commitments=tuple(
                sorted(found.values(), key=lambda c: (c.block, c.hotkey, c.artifact))
            ),
        )

    def _block_time(self, block: int) -> float | None:
        """Chain time of a block, or None once the node no longer holds that block's state."""
        try:
            seconds = float(self.client.get_timestamp(block).timestamp())
        except Exception as exc:
            log.warning("reveal_time_unavailable block=%s error=%s", block, type(exc).__name__)
            return None
        # Some nodes answer for discarded state with an empty value instead of an error.
        return seconds if seconds > 0 else None

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

    def scan(self, start_block: int | None = None, known: Container[str] = ()) -> ChainSnapshot:
        del start_block, known
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
