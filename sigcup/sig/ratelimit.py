"""Sliding-window rate limiter shared by every call in a process. Exactly ONE instance per key.

Contract (implemented by the `sig` module owner):
    limiter = SlidingWindowLimiter(max_calls=25, window_s=60)
    async with limiter:            # blocks until a slot is free in the trailing window
        ...
    limiter.freeze(seconds)        # after a 429: block all acquires until the freeze ends
    limiter.available()            # slots free right now (for OMS prioritisation)
Also provides `PriorityLimiter` wrapper: acquire(priority) where lower numbers go first when
contended (arb/election takes = 0, theta = 1, value = 2, mm = 3, housekeeping = 4).
"""

from __future__ import annotations

raise NotImplementedError("implemented in the sig workstream")
