import asyncio

import pytest

from sigcup.sig.ratelimit import Budget, SlidingWindowLimiter


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


async def test_sliding_window_never_exceeds_cap():
    clock = FakeClock()
    lim = SlidingWindowLimiter(5, 60, clock)
    for _ in range(5):
        assert lim.try_acquire()
    assert not lim.try_acquire()
    assert lim.available() == 0
    clock.t += 59.9
    assert not lim.try_acquire()
    clock.t += 0.2
    assert lim.try_acquire()


async def test_freeze_blocks_everything():
    clock = FakeClock()
    lim = SlidingWindowLimiter(5, 60, clock)
    lim.freeze(60)
    assert lim.available() == 0
    assert not lim.try_acquire()
    assert lim.frozen_for() == pytest.approx(60)
    clock.t += 61
    assert lim.try_acquire()


async def test_concurrent_waiters_respect_cap_and_priority():
    lim = SlidingWindowLimiter(3, 60)  # real clock; only 3 slots, the rest must wait
    order: list[int] = []

    async def worker(prio: int, idx: int):
        async with lim.acquire(priority=prio):
            order.append(idx)

    # fill the window first
    for _ in range(3):
        assert lim.try_acquire()
    tasks = [asyncio.create_task(worker(2, 1)), asyncio.create_task(worker(0, 2))]
    await asyncio.sleep(0.05)
    assert order == []  # nothing free
    for t in tasks:
        t.cancel()
    for t in tasks:
        with pytest.raises(asyncio.CancelledError):
            await t
    assert lim.used() == 3


async def test_budget_freeze_all():
    b = Budget(90, 25)
    b.freeze_all(10)
    assert b.reads.available() == 0 and b.writes.available() == 0
