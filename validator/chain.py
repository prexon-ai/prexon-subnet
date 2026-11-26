"""Finalized chain discovery and confirmed weight publication."""

from __future__ import annotations

import logging
import re
from collections.abc import Container
from typing import Any, Protocol

from shared.chain import connect, require_success, wallet
from shared.contracts import Commitment
from shared.targets import FrozenModel
from validator.config import (
    NETUID,
    NETWORK,
    WALLET_HOTKEY,
    WALLET_NAME,
    WALLET_PATH,
    WEIGHTS_VERSION,
)

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
    def __init__(self, *, client: Any = None, signer: Any = None) -> None:
        self._client = client
        self.signer = signer
        # Finalized block hashes never change; kept for the reveal blocks still listed.
        self._reveal_hashes: dict[int, str] = {}

    @property
    def client(self) -> Any:
        # Connect on first use: a validator without targets never reads the chain.
        if self._client is None:
            self._client = connect(NETWORK)
        return self._client

    def scan(self, start_block: int | None = None, known: Container[str] = ()) -> ChainSnapshot:
        """Read finalized reveals; `known` holds commitment identities already discovered.

        Reveals stay in chain storage long after non-archive nodes discard the state of
        their blocks, so block times are read only for reveals that are new.
        """
        finalized = self.finalized_height()
        # Replay snapshots after downtime to avoid skipping the bounded on-chain reveal history.
        end = finalized if start_block is None else min(finalized, start_block + 120)
        blocks = [end] if start_block is None else range(start_block + 1, end + 1)
        meta = self.client.metagraph(NETUID, block=end)
        owners = {int(n.uid): str(n.hotkey) for n in meta.neurons}
        by_hotkey = {hotkey: uid for uid, hotkey in owners.items()}
        found: dict[str, Commitment] = {}
        timestamps: dict[int, float | None] = {}
        previous = self._reveal_hashes
        hashes: dict[int, str] = {}
        for block in blocks:
            records = self.client.get_all_revealed_commitments(NETUID, block=block)
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
                        netuid=NETUID,
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
            self.signer = wallet(WALLET_NAME, WALLET_HOTKEY, WALLET_PATH)
        uids = sorted(weights)
        result = self.client.set_weights(
            wallet=self.signer,
            netuid=NETUID,
            uids=uids,
            weights=[weights[uid] for uid in uids],
            version_key=WEIGHTS_VERSION,
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )
        require_success(result)

    def current_owners(self) -> dict[int, str]:
        meta = self.client.metagraph(NETUID, block=self.finalized_height())
        return {int(n.uid): str(n.hotkey) for n in meta.neurons}

    def block_hash(self, block: int) -> str:
        if block > self.finalized_height():
            raise ValueError("Sampling block is not finalized")
        return str(self.client.get_block_hash(block))

    def finalized_height(self) -> int:
        head = self.client.substrate.get_chain_finalised_head()
        return int(self.client.substrate.get_block_number(head))
