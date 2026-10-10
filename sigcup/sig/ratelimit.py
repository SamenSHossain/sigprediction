"""Sliding-window rate limiters. Exactly ONE instance per budget (reads / writes) per process.

    limiter = SlidingWindowLimiter(max_calls=25, window_s=60)
    async with limiter.acquire(priority=1):
        ...
    limiter.freeze(60)        # after a 429: nothing acquires until the freeze ends
    limiter.available()       # free slots right now

Lower priority numbers are served first when contended. The clock is injectable for tests.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from collections import deque
from collections.abc import Callable
from contextlib import asynccontextmanager


class SlidingWindowLimiter:
    def __init__(self, max_calls: int, window_s: float = 60.0, clock: Callable[[], float] | None = None):
        if max_calls < 1:
            raise ValueError("max_calls must be >= 1")
        self.max_calls = max_calls
        self.window_s = window_s
        self._clock = clock or time.monotonic
        self._stamps: deque[float] = deque()
        self._frozen_until = 0.0
        self._waiters: list[tuple[int, int, asyncio.Future]] = []
        self._seq = itertools.count()
        self._lock = asyncio.Lock()

    # -- introspection -------------------------------------------------------------------------
    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self._stamps and self._stamps[0] <= cutoff:
            self._stamps.popleft()

    def available(self) -> int:
        now = self._clock()
        if now < self._frozen_until:
            return 0
        self._prune(now)
        return max(0, self.max_calls - len(self._stamps))

    def used(self) -> int:
        self._prune(self._clock())
        return len(self._stamps)

    def frozen_for(self) -> float:
        return max(0.0, self._frozen_until - self._clock())

    def freeze(self, seconds: float) -> None:
        self._frozen_until = max(self._frozen_until, self._clock() + seconds)

    def _next_free_in(self) -> float:
        now = self._clock()
        if now < self._frozen_until:
            return self._frozen_until - now
        self._prune(now)
        if len(self._stamps) < self.max_calls:
            return 0.0
        return max(0.0, self._stamps[0] + self.window_s - now)

    # -- acquisition ---------------------------------------------------------------------------
    def try_acquire(self) -> bool:
        """Non-blocking: take a slot if one is free (and nobody with a pending wait is ahead)."""
        if self._waiters:
            return False
        if self.available() <= 0:
            return False
        self._stamps.append(self._clock())
        return True

    async def wait(self, priority: int = 2) -> None:
        """Block until a slot is taken for the caller. Lower priority numbers go first."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        heapq.heappush(self._waiters, (priority, next(self._seq), fut))
        try:
            while True:
                head = self._waiters[0]
                if head[2] is fut:
                    delay = self._next_free_in()
                    if delay <= 0:
                        self._stamps.append(self._clock())
                        heapq.heappop(self._waiters)
                        fut.set_result(None)
                        self._wake_next()
                        return
                    await asyncio.sleep(min(delay, 0.5))
                else:
                    # not our turn: wait to be woken or poll
                    try:
                        await asyncio.wait_for(asyncio.shield(fut), timeout=0.5)
                        return
                    except TimeoutError:
                        continue
        except asyncio.CancelledError:
            self._waiters = [w for w in self._waiters if w[2] is not fut]
            heapq.heapify(self._waiters)
            self._wake_next()
            raise

    def _wake_next(self) -> None:
        # Nothing to do: waiters poll on a short interval; kept for symmetry/extension.
        return

    @asynccontextmanager
    async def acquire(self, priority: int = 2):
        await self.wait(priority)
        yield


class Budget:
    """The pair of limiters every exchange-facing process shares."""

    def __init__(self, reads_per_min: int, writes_per_min: int, clock: Callable[[], float] | None = None):
        self.reads = SlidingWindowLimiter(reads_per_min, 60.0, clock)
        self.writes = SlidingWindowLimiter(writes_per_min, 60.0, clock)

    def freeze_all(self, seconds: float) -> None:
        self.reads.freeze(seconds)
        self.writes.freeze(seconds)
