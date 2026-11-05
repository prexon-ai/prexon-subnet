"""Small signing/connection primitives used by miner and validator adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.config import Settings


def connect(settings: Settings) -> Any:
    import bittensor as bt

    return bt.Subtensor(network=settings.network)


def wallet(settings: Settings) -> Any:
    import bittensor as bt

    return bt.Wallet(
        name=settings.wallet_name,
        hotkey=settings.wallet_hotkey,
        path=str(Path(settings.wallet_path).expanduser()),
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
    settings: Settings, digest: str, *, client: Any = None, signer: Any = None
) -> None:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Expected a SHA-256 artifact digest")
    chain = client if client is not None else connect(settings)
    signer = signer if signer is not None else wallet(settings)
    result = chain.set_reveal_commitment(
        wallet=signer,
        netuid=settings.netuid,
        data=f"px1:{digest}",
        blocks_until_reveal=1,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    require_success(result)
