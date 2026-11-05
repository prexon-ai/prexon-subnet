"""Cancellation-safe cleanup for local inference and staging process groups."""

from __future__ import annotations

import asyncio
import os
import signal


async def terminate_process_group(process: asyncio.subprocess.Process) -> None:
    async def terminate() -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), 20)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    cleanup = asyncio.create_task(terminate())
    interrupted = False
    while True:
        try:
            await asyncio.shield(cleanup)
            break
        except asyncio.CancelledError:
            interrupted = True
            if cleanup.done():
                cleanup.result()
    if interrupted:
        raise asyncio.CancelledError
