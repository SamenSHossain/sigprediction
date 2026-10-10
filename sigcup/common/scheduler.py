"""Minimal asyncio periodic-task runner (replaces APScheduler)."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable

from sigcup.common.log import get_logger

log = get_logger(__name__)


async def every(
    seconds: float,
    fn: Callable[[], Awaitable[None]],
    *,
    name: str,
    jitter: float = 0.1,
    run_immediately: bool = True,
    stop: asyncio.Event | None = None,
) -> None:
    """Run `fn` every `seconds` (+/- jitter fraction); exceptions are logged, never fatal."""
    if not run_immediately:
        await asyncio.sleep(seconds)
    while not (stop and stop.is_set()):
        try:
            await fn()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("task %s failed", name)
        delay = seconds * (1 + random.uniform(-jitter, jitter))
        try:
            if stop:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            else:
                await asyncio.sleep(delay)
        except TimeoutError:
            pass
