"""B3: every model cycle persists one Signal row per desired order that survived risk (the reporter's
calibration section reads them), in one transaction, at most once per market per strategy per 10 min."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Market, Race, Signal, Strategy
from sigcup.settings import Secrets
from sigcup.trader import loop as trader_loop
from sigcup.trader.types import DesiredOrder, FairValue
from tests.fakes_exec import FakeAdapter, FakeClock


@pytest.fixture(autouse=True)
def _reset_module_cache():
    trader_loop._modules.clear()
    yield
    trader_loop._modules.clear()


def make_sec(tmp_path) -> Secrets:
    return Secrets(_env_file=None, live=False, kill_file=str(tmp_path / "KILL"), sig_api_key="", sig_read_key="")


def install_models(monkeypatch, desired_fn) -> None:
    fv = ModuleType("sigcup.trader.fairvalue")
    fv.compute_all = lambda session, cfg, markets, now: {
        eid: FairValue(exchange_id=eid, fair=0.7, sd=0.05, components={"ref": 0.71, "polls": 0.66}, weights={"ref": 0.6, "polls": 0.4}, ts=now)
        for eid in markets
    }
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


def generate(ctx, arbs):
    return [
        DesiredOrder(exchange_id="tx-d", side="yes", action="buy", price=0.60, quantity=100, strategy=Strategy.VALUE, fair=0.70, edge=0.10, reason="value"),
        DesiredOrder(exchange_id="tx-r", side="no", action="buy", price=0.64, quantity=50, strategy=Strategy.THETA, fair=0.30, edge=0.06, reason="theta no"),
        DesiredOrder(exchange_id="tx-d", side="yes", action="buy", price=0.60, quantity=0, strategy=Strategy.MM, fair=0.70),  # dropped by risk
    ]


async def make_trader(cfg, tmp_path, clock: FakeClock):
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX", exclusive=True, exhaustive=True))
        s.add(Market(market_id="1", exchange_id="tx-d", race_id="SEN-TX", party="D", title="D", status="open"))
        s.add(Market(market_id="2", exchange_id="tx-r", race_id="SEN-TX", party="R", title="R", status="open"))
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("tx-d", 0.58, 0.62)
    fake.set_top("tx-r", 0.34, 0.38)
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    await t.oms.startup_reconcile()
    await t.refresh_books()
    return t


def signal_rows() -> list[Signal]:
    with dbengine.session() as s:
        return list(s.execute(select(Signal).order_by(Signal.id)).scalars().all())


async def test_cycle_writes_one_signal_per_accepted_order(db, cfg, tmp_path, monkeypatch):
    install_models(monkeypatch, generate)
    clock = FakeClock()
    t = await make_trader(cfg, tmp_path, clock)
    accepted: list[DesiredOrder] = []
    real_apply = trader_loop.risk.apply

    def spy(ctx, desired):
        out = real_apply(ctx, desired)
        accepted.extend(out)
        return out

    monkeypatch.setattr(trader_loop.risk, "apply", spy)
    res = await t.model_cycle()
    assert res.skipped is None and res.accepted == 2
    rows = signal_rows()
    assert len(rows) == 2 == res.signals
    by = {(r.exchange_id, r.strategy): r for r in rows}
    sizes = {(d.exchange_id, d.strategy.value): d.quantity for d in accepted}
    v = by[("tx-d", "value")]
    assert v.fair == pytest.approx(0.70) and v.price == pytest.approx(0.60) and v.edge == pytest.approx(0.10)
    assert v.size == sizes[("tx-d", "value")] > 0, "the size that survived risk"
    assert v.ts is not None
    assert v.meta["components"] == {"ref": pytest.approx(0.71), "polls": pytest.approx(0.66)}
    assert v.meta["side"] == "yes" and v.meta["action"] == "buy"
    th = by[("tx-r", "theta")]
    # fair is P(YES); price is stored in the same YES units (a NO bought at 0.64 is a YES at 0.36)
    assert th.fair == pytest.approx(0.30) and th.price == pytest.approx(0.36) and th.meta["order_price"] == pytest.approx(0.64)
    assert th.meta["side"] == "no" and th.size == sizes[("tx-r", "theta")]


async def test_second_cycle_within_ten_minutes_does_not_duplicate(db, cfg, tmp_path, monkeypatch):
    install_models(monkeypatch, generate)
    clock = FakeClock()
    t = await make_trader(cfg, tmp_path, clock)
    await t.model_cycle()
    assert len(signal_rows()) == 2
    clock.advance(60)
    await t.model_cycle()
    clock.advance(8 * 60)
    await t.model_cycle()
    assert len(signal_rows()) == 2, "at most one row per market per strategy per 10 minutes"
    clock.advance(2 * 60 + 1)
    res = await t.model_cycle()
    assert len(signal_rows()) == 4 and res.signals == 2


async def test_signal_write_failure_never_breaks_the_cycle(db, cfg, tmp_path, monkeypatch):
    install_models(monkeypatch, generate)
    clock = FakeClock()
    t = await make_trader(cfg, tmp_path, clock)

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(t, "record_signals", boom)
    res = await t.model_cycle()
    assert res.skipped is None and res.sync is not None and len(res.sync.placed) == 2
