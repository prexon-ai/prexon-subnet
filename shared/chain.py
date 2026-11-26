"""Small signing/connection primitives used by miner and validator adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def connect(network: str) -> Any:
    import bittensor as bt

    return bt.Subtensor(network=network)


def wallet(name: str, hotkey: str, path: str) -> Any:
    import bittensor as bt

    return bt.Wallet(
        name=name,
        hotkey=hotkey,
        path=str(Path(path).expanduser()),
    )


def require_success(result: Any) -> None:
    # SDK 10.2.1 returns ExtrinsicResponse. Never interpret an arbitrary truthy object as success.
    if isinstance(result, tuple):
        success = result[0] is True
    else:
        success = getattr(result, "success", False) is True
    if not success:
        raise RuntimeError("Chain transaction was not confirmed successful")


def commit_artifact(
    digest: str, *, network: str, netuid: int, signer: Any, client: Any = None
) -> None:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Expected a SHA-256 artifact digest")
    chain = client if client is not None else connect(network)
    result = chain.set_reveal_commitment(
        wallet=signer,
        netuid=netuid,
        data=f"px1:{digest}",
        blocks_until_reveal=1,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    require_success(result)
