"""Trader loop: run(once=True) wiring, missing model modules skip the cycle, fake model modules drive
orders through risk + OMS, context building (ref prices, events, daily pnl), kill file, LIVE guard."""

from __future__ import annotations

import sys
from datetime import timedelta
from types import ModuleType

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import AccountSnap, Event, Heartbeat, LeaderboardSnap, Market, Race, RefPrice, Strategy
from sigcup.settings import Secrets
from sigcup.sig.ratelimit import Budget
from sigcup.trader import loop as trader_loop
from sigcup.trader.types import DesiredOrder, FairValue
from tests.fakes_exec import FakeAdapter, FakeClock


@pytest.fixture(autouse=True)
def _reset_module_cache():
    trader_loop._modules.clear()
    yield
    trader_loop._modules.clear()


def seed(clock: FakeClock) -> None:
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX Senate", exclusive=True, exhaustive=True))
        s.add(Market(market_id="1", exchange_id="tx-d", race_id="SEN-TX", party="D", title="D", status="open", kalshi_ticker="SENATETX-26-D", polymarket_condition_id="0xabc"))
        s.add(Market(market_id="2", exchange_id="tx-r", race_id="SEN-TX", party="R", title="R", status="open"))
        s.add(RefPrice(ts=clock.now(), source="kalshi", key="SENATETX-26-D", yes_mid=0.66, liquidity=500))
        s.add(RefPrice(ts=clock.now(), source="polymarket", key="0xabc", yes_bid=0.70, yes_ask=0.72, liquidity=5000))
        s.add(Event(ts=clock.now() - timedelta(hours=2), race_id="SEN-TX", kind="hard", description="dropout", source="rss", confidence=0.9))
        s.add(Event(ts=clock.now() - timedelta(hours=30), race_id="SEN-TX", kind="soft", description="old", source="rss", confidence=0.5))


def fake_models(monkeypatch, desired_fn, fair=0.7):
    """Install fake model modules under the contract names."""
    fv = ModuleType("sigcup.trader.fairvalue")
    fv.compute_all = lambda session, cfg, markets, now: {eid: FairValue(exchange_id=eid, fair=fair, sd=0.05) for eid in markets}
    fv.load_races = lambda session: {}
    cons = ModuleType("sigcup.trader.consistency")
    cons.scan = lambda ctx: []
    sig = ModuleType("sigcup.trader.signals")
    sig.generate = desired_fn
    eg = ModuleType("sigcup.trader.endgame")
    eg.sleeve_fraction = lambda cfg, now, leaderboard, equity: 0.15
    en = ModuleType("sigcup.trader.election_night")
    en.is_active = lambda cfg, now: False
    en.taker_orders = lambda ctx, ref_prices: []
    for m in (fv, cons, sig, eg, en):
        monkeypatch.setitem(sys.modules, m.__name__, m)


def make_sec(tmp_path, **kw) -> Secrets:
    kw.setdefault("live", False)
    kw.setdefault("kill_file", str(tmp_path / "KILL"))
    kw.setdefault("sig_api_key", "")
    kw.setdefault("sig_read_key", "")
    return Secrets(_env_file=None, **kw)


async def test_run_once_with_missing_models_skips_cycle_but_runs_housekeeping(db, cfg, tmp_path, monkeypatch):
    clock = FakeClock()
    seed(clock)
    monkeypatch.setattr(trader_loop, "_module", lambda name: None)
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("tx-d", 0.60, 0.64)
    res = await trader_loop.run(paper=True, once=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    assert res.skipped == "signals module unavailable" and res.sync is None
    assert fake.writes == []
    with dbengine.session() as s:
        hb = s.get(Heartbeat, "trader")
        assert hb is not None and hb.status == "paper"
        assert s.execute(select(LeaderboardSnap)).scalars().all()
        assert s.execute(select(AccountSnap)).scalars().all()
    assert res.reconcile is not None and res.reconcile.cash == fake.cash


async def test_run_once_with_fake_models_places_orders_through_risk_and_oms(db, cfg, tmp_path, monkeypatch):
    clock = FakeClock()
    seed(clock)
    captured = {}

    def generate(ctx, arbs):
        captured["ctx"] = ctx
        return [
            DesiredOrder(exchange_id="tx-d", side="yes", action="buy", price=0.6, quantity=100, strategy=Strategy.VALUE),
            DesiredOrder(exchange_id="tx-d", side="yes", action="buy", price=0.6, quantity=10**7, strategy=Strategy.THETA),  # scaled by risk
        ]

    fake_models(monkeypatch, generate)
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("tx-d", 0.58, 0.62)
    fake.set_top("tx-r", 0.36, 0.40)
    res = await trader_loop.run(paper=True, once=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    assert res.skipped is None and res.fair == 2 and res.desired == 2
    assert len(res.sync.placed) == res.accepted >= 1
    assert all(o.req.quantity <= cfg.risk.max_position_shares for o in fake.orders.values())
    ctx = captured["ctx"]
    assert ctx.ref_prices == {"tx-d": pytest.approx(0.71)}  # the deeper polymarket quote wins
    assert [e.kind for e in ctx.events] == ["hard"]
    assert ctx.markets["tx-d"].mid == 0.60 and ctx.markets["tx-d"].race_id == "SEN-TX"
    assert ctx.races["SEN-TX"].markets == {"D": "tx-d", "R": "tx-r"}
    assert ctx.hours_to_deadline > 0 and ctx.daily_pnl == 0.0 and ctx.sleeve_frac == 0.15
    assert ctx.cash == fake.cash and ctx.bankroll_start == cfg.bankroll_start


async def test_failing_model_step_skips_cycle(db, cfg, tmp_path, monkeypatch):
    clock = FakeClock()
    seed(clock)

    def boom(ctx, arbs):
        raise RuntimeError("model exploded")

    fake_models(monkeypatch, boom)
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("tx-d", 0.58, 0.62)
    res = await trader_loop.run(paper=True, once=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    assert res.skipped == "signals.generate failed" and fake.writes == []


async def test_kill_file_kills_in_cycle(db, cfg, tmp_path, monkeypatch):
    clock = FakeClock()
    seed(clock)
    fake_models(monkeypatch, lambda ctx, arbs: [DesiredOrder(exchange_id="tx-d", side="yes", action="buy", price=0.6, quantity=10, strategy=Strategy.VALUE)])
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("tx-d", 0.58, 0.62)
    (tmp_path / "KILL").write_text("")
    res = await trader_loop.run(paper=True, once=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    assert res.killed and "kill file" in res.killed
    assert [w[1] for w in fake.writes] == ["cancel_all"]
    with dbengine.session() as s:
        assert s.get(Heartbeat, "trader").status == "killed"


def test_live_without_key_is_refused(db, cfg, tmp_path):
    with pytest.raises(RuntimeError, match="SIG_API_KEY"):
        trader_loop.build_trader(paper=False, cfg=cfg, sec=make_sec(tmp_path, live=True))
    # paper mode never needs a key, and LIVE with --paper stays paper
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path, live=True))
    assert t.live is False and t.oms.paper is True and t.adapter.name == "paper"


async def test_realtime_consumer_and_budget_are_wired(db, cfg, tmp_path):
    clock = FakeClock()
    fake = FakeAdapter(clock=clock.now)
    budget = Budget(90, 25, clock=clock.mono)
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, budget=budget, clock=clock.now)
    assert t.oms.budget is budget and t.budget is budget

    async def events():
        yield type("Upd", (), {"exchange_id": "m1", "bids": [(0.4, 10)], "asks": [(0.46, 10)], "ts": None})()
        yield type("Resync", (), {"exchange_id": None})()

    await t.consume_realtime(events())
    assert t.book.top("m1").ask == 0.46 and t._resync_requested is True
    assert t.start_realtime() is False  # the realtime module is a stub: REST only


async def test_start_realtime_builds_feed_with_market_ids(db, cfg, tmp_path, monkeypatch):
    clock = FakeClock()
    seed(clock)
    built = {}

    class FakeFeed:
        def __init__(self, token_provider, tournament_id, *, markets=None, max_channels=None, **kw):
            built.update(token_provider=token_provider, tournament_id=tournament_id, markets=markets, max_channels=max_channels)
            self.stopped = False

        async def events(self):
            yield type("Upd", (), {"exchange_id": "tx-d", "bids": [(0.61, 10)], "asks": [(0.63, 10)], "ts": None})()

        def stop(self):
            self.stopped = True

    rt = ModuleType("sigcup.sig.realtime")
    rt.RealtimeFeed = FakeFeed
    monkeypatch.setitem(sys.modules, rt.__name__, rt)
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("tx-d", 0.58, 0.62)

    async def token(**kw):
        return "tok"

    fake.realtime_token = token
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    await t.refresh_books()
    assert t.start_realtime() is True
    await t._realtime_task
    assert built["tournament_id"] == "t-fake" and built["max_channels"] == cfg.limits.realtime_channels
    assert built["markets"]["tx-d"] == "1" and built["token_provider"] is token
    assert t.book.top("tx-d").ask == 0.63 and t.book.top("tx-d").source == "realtime"
