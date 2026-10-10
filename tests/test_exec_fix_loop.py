"""Regression tests for the trader loop and paper/live isolation: paper and live keep separate kill flags
and daily-P&L baselines, a paper run on a live database leaves the live Position / AccountSnap rows alone
(F10), a transient exchange error at startup is retried instead of crashing the process (F12), and an
unconfirmed kill keeps the process alive until the kill-watch confirms nothing is open (F8)."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import AccountSnap, Position, Strategy
from sigcup.db.repo import kv_set
from sigcup.paper import PaperAdapter
from sigcup.settings import Secrets
from sigcup.sig.errors import TransientError
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import PositionInfo
from sigcup.trader import loop as trader_loop
from sigcup.trader.oms import KV_KILLED, OMS
from sigcup.trader.types import DesiredOrder
from tests.fakes_exec import FakeAdapter, FakeClock


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


def oms_for(cfg, tmp_path, adapter, clock: FakeClock, *, paper: bool) -> OMS:
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
    return OMS(adapter, cfg, budget, paper=paper, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, backoff_s=0.0)


# -- F10 --------------------------------------------------------------------------------------
async def test_paper_kill_flag_does_not_block_the_live_trader(db, cfg, tmp_path):
    clock = FakeClock()
    paper = oms_for(cfg, tmp_path, FakeAdapter(clock=clock.now), clock, paper=True)
    await paper.startup_reconcile()
    await paper.kill("paper canary stopped by operator")
    live = oms_for(cfg, tmp_path, FakeAdapter(clock=clock.now), clock, paper=False)
    await live.startup_reconcile()
    assert live.killed is None
    paper2 = oms_for(cfg, tmp_path, FakeAdapter(clock=clock.now), clock, paper=True)
    await paper2.startup_reconcile()
    assert paper2.killed == "paper canary stopped by operator", "the paper flag still persists for paper"
    # a flag a paper trader wrote under the shared key (older versions) never blocks live
    with dbengine.session() as s:
        kv_set(s, KV_KILLED, {"reason": "old paper kill", "paper": True})
    live2 = oms_for(cfg, tmp_path, FakeAdapter(clock=clock.now), clock, paper=False)
    await live2.startup_reconcile()
    assert live2.killed is None
    # and a live kill still blocks live
    await live2.kill("daily loss")
    live3 = oms_for(cfg, tmp_path, FakeAdapter(clock=clock.now), clock, paper=False)
    await live3.startup_reconcile()
    assert live3.killed == "daily loss"
    OMS.clear_killed()
    for paper_flag in (False, True):
        o = oms_for(cfg, tmp_path, FakeAdapter(clock=clock.now), clock, paper=paper_flag)
        await o.startup_reconcile()
        assert o.killed is None


async def test_cli_clear_of_the_shared_flag_also_clears_a_paper_kill(db, cfg, tmp_path):
    clock = FakeClock()
    paper = oms_for(cfg, tmp_path, FakeAdapter(clock=clock.now), clock, paper=True)
    await paper.kill("kill file present")
    await asyncio.sleep(0.01)
    with dbengine.session() as s:  # what `sigcup kill --clear` does
        kv_set(s, "killed", None)
    paper2 = oms_for(cfg, tmp_path, FakeAdapter(clock=clock.now), clock, paper=True)
    await paper2.startup_reconcile()
    assert paper2.killed is None


def test_daily_pnl_baselines_are_kept_per_mode(db, cfg, tmp_path):
    clock = FakeClock()
    live = trader_loop.build_trader(
        paper=False, cfg=cfg, sec=make_sec(tmp_path, live=True, sig_api_key="k"), adapter=FakeAdapter(clock=clock.now), clock=clock.now
    )
    paper = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=FakeAdapter(clock=clock.now), clock=clock.now)
    assert live.live and not paper.live
    with dbengine.session() as s:
        assert paper._daily_pnl(s, clock.now(), 100_000.0) == 0.0
    with dbengine.session() as s:
        assert live._daily_pnl(s, clock.now(), 94_000.0) == 0.0, "a paper baseline never sets the live one"
    with dbengine.session() as s:
        assert live._daily_pnl(s, clock.now(), 93_000.0) == pytest.approx(-1_000.0)
        assert paper._daily_pnl(s, clock.now(), 100_500.0) == pytest.approx(500.0)


async def test_paper_reconcile_leaves_live_positions_and_account_rows_alone(db, cfg, tmp_path):
    clock = FakeClock()
    fake = FakeAdapter(clock=clock.now)
    fake.positions["m1"] = PositionInfo(exchange_id="m1", quantity=500, avg_price=0.4, mark=0.5)
    live = oms_for(cfg, tmp_path, fake, clock, paper=False)
    await live.startup_reconcile()
    with dbengine.session() as s:
        snaps_live = len(s.execute(select(AccountSnap)).scalars().all())
        assert s.get(Position, "m1").quantity == 500
    paper = oms_for(cfg, tmp_path, PaperAdapter(cfg, clock=clock.now, use_db_quotes=False), clock, paper=True)
    await paper.startup_reconcile()
    await paper.reconcile()
    with dbengine.session() as s:
        assert s.get(Position, "m1").quantity == 500, "paper never zeroes live positions"
        assert len(s.execute(select(AccountSnap)).scalars().all()) == snaps_live


# -- F12 --------------------------------------------------------------------------------------
async def test_startup_errors_are_retried_not_fatal(db, cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(trader_loop, "_module", lambda name: None)

    class Flaky(FakeAdapter):
        calls = 0

        async def get_open_orders(self):
            Flaky.calls += 1
            if Flaky.calls <= 2:
                raise TransientError("service unavailable", 503, "UNAVAILABLE")
            return await super().get_open_orders()

    fake = Flaky()
    fake.set_top("m1", 0.40, 0.45)
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake)
    t.startup_retry_s = 0.01
    task = asyncio.create_task(t.run_forever())
    for _ in range(300):
        if t.last_cycle is not None or task.done():
            break
        await asyncio.sleep(0.01)
    assert not task.done(), "the process survived the 503s at startup"
    assert Flaky.calls >= 3 and t.last_cycle is not None
    t.stop.set()
    await asyncio.wait_for(task, 5.0)


# -- F8 (loop side) ---------------------------------------------------------------------------
async def test_unconfirmed_kill_keeps_running_until_watch_kill_confirms(db, cfg, tmp_path):
    clock = FakeClock()
    fake = FakeAdapter(clock=clock.now)
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    t.oms.sleep = clock.sleep
    t.oms.backoff_s = 0.0
    await t.oms.sync([DesiredOrder(exchange_id="m1", side="yes", action="buy", price=0.5, quantity=10, strategy=Strategy.VALUE)])
    assert len(fake.open_orders()) == 1

    async def broken_all() -> int:
        raise TransientError("service unavailable", 503, "UNAVAILABLE")

    async def broken_one(order_id: str) -> bool:
        raise TransientError("service unavailable", 503, "UNAVAILABLE")

    real_all, real_one = fake.cancel_all, fake.cancel_order
    fake.cancel_all, fake.cancel_order = broken_all, broken_one
    (tmp_path / "KILL").write_text("")
    await t.watch_kill()
    assert t.oms.killed and not t.stop.is_set(), "orders are still open: keep running and retry"
    fake.cancel_all, fake.cancel_order = real_all, real_one
    await t.watch_kill()
    assert fake.open_orders() == [] and t.stop.is_set()
