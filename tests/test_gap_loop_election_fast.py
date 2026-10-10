"""B1: election-night takes run on the fast reference cadence, not the 60-s model cycle.

``Trader.election_tick`` (scheduled every ``election_night.ref_cadence_s``) reads the newest reference
prices from the DB with a freshness bound (3 collector cadences), refreshes SIG top-of-book for the
affected markets only, computes ``election_night.taker_orders`` and sends them through the shared OMS at
priority 0 without disturbing the resting book. Outside election night it idles without any read."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Market, Race, RefPrice, Strategy
from sigcup.settings import Secrets
from sigcup.trader import loop as trader_loop
from sigcup.trader.types import PRIORITY_ARB, DesiredOrder
from tests.fakes_exec import FakeAdapter, FakeClock

NIGHT = datetime(2026, 11, 4, 1, 0, tzinfo=UTC)  # inside election_night.start_utc .. deadline_utc
DAY = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _reset_module_cache():
    trader_loop._modules.clear()
    yield
    trader_loop._modules.clear()


def make_sec(tmp_path, **kw) -> Secrets:
    kw.setdefault("live", False)
    kw.setdefault("kill_file", str(tmp_path / "KILL"))
    kw.setdefault("sig_api_key", "")
    kw.setdefault("sig_read_key", "")
    return Secrets(_env_file=None, **kw)


def seed(ref_ts: datetime, ref_mid: float = 0.98) -> None:
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX Senate", exclusive=True, exhaustive=True))
        s.add(Race(race_id="SEN-OK", chamber="senate", state="OK", label="OK Senate", exclusive=True, exhaustive=True))
        s.add(Market(market_id="1", exchange_id="tx-d", race_id="SEN-TX", party="D", title="D", status="open", kalshi_ticker="SENATETX-26-D"))
        s.add(Market(market_id="2", exchange_id="tx-r", race_id="SEN-TX", party="R", title="R", status="open", kalshi_ticker="SENATETX-26-R"))
        s.add(Market(market_id="3", exchange_id="ok-r", race_id="SEN-OK", party="R", title="R", status="open", kalshi_ticker="SENATEOK-26-R"))
        s.add(RefPrice(ts=ref_ts, source="kalshi", key="SENATETX-26-D", yes_mid=ref_mid, liquidity=5000))
        s.add(RefPrice(ts=ref_ts, source="kalshi", key="SENATETX-26-R", yes_mid=1 - ref_mid, liquidity=5000))
        s.add(RefPrice(ts=ref_ts, source="kalshi", key="SENATEOK-26-R", yes_mid=0.60, liquidity=5000))  # not near a take


async def make_trader(cfg, tmp_path, clock: FakeClock) -> tuple[trader_loop.Trader, FakeAdapter]:
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("tx-d", 0.70, 0.80, bid_size=40, ask_size=50)  # SIG lags the 0.98 reference by 18c
    fake.set_top("tx-r", 0.20, 0.30, bid_size=40, ask_size=50)
    fake.set_top("ok-r", 0.55, 0.60)
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    await t.oms.startup_reconcile()
    await t.refresh_books()
    return t, fake


async def test_fresh_reference_takes_within_one_fast_tick(db, cfg, tmp_path):
    clock = FakeClock(start=NIGHT)
    seed(clock.now() - timedelta(seconds=5))
    t, fake = await make_trader(cfg, tmp_path, clock)
    fake.reads.clear()
    res = await t.election_tick()
    assert res is not None and res.active
    placed = [o for o in fake.orders.values() if o.req.strategy == Strategy.ELECTION.value]
    assert placed, "a fresh 0.98 reference with SIG lagging at 0.80 must take within one fast tick"
    by_market = {o.req.exchange_id: o.req for o in placed}
    assert by_market["tx-d"].side == "yes" and by_market["tx-d"].action == "buy" and by_market["tx-d"].price == pytest.approx(0.80)
    assert 0 < by_market["tx-d"].quantity <= 50
    # the NO mirror on the R leg (ref 0.02, SIG bid 0.20) is taken too: buy NO at 1 - bid
    assert by_market["tx-r"].side == "no" and by_market["tx-r"].price == pytest.approx(0.80)
    assert all(d.priority == PRIORITY_ARB and d.take for d in res.accepted)
    # only the affected markets were refreshed: one bulk read, the far-from-threshold market untouched
    assert fake.reads == ["prices"]
    assert set(res.refreshed) == {"tx-d", "tx-r"}


async def test_stale_reference_never_takes(db, cfg, tmp_path):
    clock = FakeClock(start=NIGHT)
    stale = clock.now() - timedelta(seconds=3 * cfg.election_night.ref_cadence_s + 1)
    seed(stale)
    t, fake = await make_trader(cfg, tmp_path, clock)
    fake.reads.clear()
    res = await t.election_tick()
    assert res is not None and res.active and res.takes == []
    assert fake.orders == {} and fake.writes == []
    assert fake.reads == [], "a stale reference is not an affected market: no SIG read either"


async def test_outside_election_night_the_task_idles(db, cfg, tmp_path):
    clock = FakeClock(start=DAY)
    seed(clock.now())
    t, fake = await make_trader(cfg, tmp_path, clock)
    fake.reads.clear()
    res = await t.election_tick()
    assert res is not None and not res.active
    assert fake.reads == [] and fake.writes == [] and fake.orders == {}


async def test_fast_take_keeps_the_resting_book(db, cfg, tmp_path):
    """The fast path adds takes; it must not cancel resting orders the model cycle wants (the OMS sync
    diffs against the whole desired set)."""
    clock = FakeClock(start=NIGHT)
    seed(clock.now() - timedelta(seconds=5))
    t, fake = await make_trader(cfg, tmp_path, clock)
    rest = DesiredOrder(exchange_id="ok-r", side="yes", action="buy", price=0.50, quantity=20, strategy=Strategy.VALUE)
    rep = await t.oms.sync([rest])
    assert len(rep.placed) == 1
    resting_id = next(iter(fake.orders))
    clock.advance(5)
    res = await t.election_tick()
    assert res.sync is not None and res.sync.cancelled == []
    assert fake.orders[resting_id].status == "open", "the resting value order survives the fast take"
    assert any(o.req.strategy == Strategy.ELECTION.value for o in fake.orders.values())


async def test_fast_tick_shares_the_single_budget_and_oms(db, cfg, tmp_path, monkeypatch):
    clock = FakeClock(start=NIGHT)
    seed(clock.now() - timedelta(seconds=5))
    t, fake = await make_trader(cfg, tmp_path, clock)

    def no_second_limiter(*a, **k):
        raise AssertionError("the fast election task must not build its own Budget")

    monkeypatch.setattr(trader_loop, "Budget", no_second_limiter)
    sent: list[list[DesiredOrder]] = []
    real_sync = t.oms.sync

    async def spy(desired):
        sent.append(list(desired))
        return await real_sync(desired)

    monkeypatch.setattr(t.oms, "sync", spy)
    await t.election_tick()
    assert t.oms.budget is t.budget and sent, "takes go through the trader's one OMS"
    takes = [d for d in sent[0] if d.strategy == Strategy.ELECTION]
    assert takes and all(d.priority == PRIORITY_ARB for d in takes)


async def test_killed_trader_sends_no_fast_takes(db, cfg, tmp_path):
    clock = FakeClock(start=NIGHT)
    seed(clock.now() - timedelta(seconds=5))
    t, fake = await make_trader(cfg, tmp_path, clock)
    await t.oms.kill("test kill")
    n_writes = len(fake.writes)
    res = await t.election_tick()
    assert res.takes == [] and len(fake.writes) == n_writes


async def test_run_forever_schedules_the_fast_task_on_the_reference_cadence(db, cfg, tmp_path, monkeypatch):
    clock = FakeClock(start=DAY)
    seed(clock.now())
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("tx-d", 0.70, 0.80)
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    scheduled: dict[str, float] = {}

    async def fake_every(seconds, fn, *, name, stop=None, **kw):
        scheduled[name] = seconds
        if stop is not None:
            await stop.wait()

    monkeypatch.setattr(trader_loop, "every", fake_every)
    task = asyncio.create_task(t.run_forever())
    for _ in range(200):
        if "election" in scheduled:
            break
        await asyncio.sleep(0.01)
    t.stop.set()
    await asyncio.wait_for(task, 5)
    assert scheduled.get("election") == cfg.election_night.ref_cadence_s
    with dbengine.session() as s:
        assert s.execute(select(Race)).scalars().all()
