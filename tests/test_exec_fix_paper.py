"""Regression tests for the paper adapter: displayed liquidity it already took is not re-armed by an
unchanged data refresh (F14), and fill / order ids are unique across processes so a restarted paper run's
fills reach the ledger (F15)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Fill, Quote, Strategy
from sigcup.paper import PaperAdapter
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import OrderRequest
from sigcup.trader.oms import OMS
from sigcup.trader.types import DesiredOrder
from tests.fakes_exec import FakeAdapter, FakeClock


def req(eid: str, price: float, qty: int, key: str) -> OrderRequest:
    return OrderRequest(exchange_id=eid, side="yes", action="buy", price=price, quantity=qty, idempotency_key=key)


def add_quote(clock: FakeClock, eid: str, bid: float, ask: float, bid_size: int, ask_size: int) -> None:
    with dbengine.session() as s:
        s.add(Quote(ts=clock.now(), exchange_id=eid, bid=bid, ask=ask, bid_size=bid_size, ask_size=ask_size))


# -- F14 --------------------------------------------------------------------------------------
async def test_unchanged_db_quote_refresh_does_not_refill_the_same_level(cfg, db):
    clock = FakeClock()
    add_quote(clock, "m9", 0.40, 0.44, 10, 30)
    p = PaperAdapter(cfg, cash=1000.0, clock=clock.now)
    await p.get_prices(["m9"])
    res = await p.place_order(req("m9", 0.45, 200, "k1"))
    assert res.quantity_traded == 30
    for _ in range(5):
        clock.advance(30)
        snaps = await p.get_prices(["m9"])
    await p.get_orderbook("m9")
    assert p.orders[res.order_id].filled == 30, "the same displayed 30 shares are not taken again"
    assert snaps["m9"].best_ask == 0.44
    # the quote changes size: that is new liquidity
    add_quote(clock, "m9", 0.40, 0.44, 10, 50)
    await p.get_prices(["m9"])
    assert p.orders[res.order_id].filled == 80
    # and a new price level is new liquidity too
    add_quote(clock, "m9", 0.40, 0.43, 10, 15)
    await p.get_prices(["m9"])
    assert p.orders[res.order_id].filled == 95


async def test_unchanged_source_refresh_does_not_refill_the_same_level(cfg, db):
    clock = FakeClock()
    src = FakeAdapter(clock=clock.now)
    src.set_top("m1", 0.30, 0.34, ask_size=5)
    p = PaperAdapter(cfg, source=src, cash=1000.0, clock=clock.now)
    await p.get_prices(["m1"])
    res = await p.place_order(req("m1", 0.35, 10, "k1"))
    assert res.quantity_traded == 5
    for _ in range(3):
        await p.get_prices(["m1"])
        await p.get_orderbook("m1")
    assert p.orders[res.order_id].filled == 5
    src.set_top("m1", 0.30, 0.34, ask_size=8)
    await p.get_prices(["m1"])
    assert p.orders[res.order_id].filled == 10


# -- F15 --------------------------------------------------------------------------------------
async def test_paper_ids_are_unique_across_runs_and_restarted_fills_reach_the_ledger(cfg, db, tmp_path):
    clock = FakeClock()

    def run() -> tuple[OMS, PaperAdapter]:
        p = PaperAdapter(cfg, cash=10_000.0, clock=clock.now, use_db_quotes=False)
        budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
        oms = OMS(p, cfg, budget, paper=True, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, backoff_s=0.0)
        return oms, p

    oms1, p1 = run()
    await oms1.startup_reconcile()
    p1.set_top("m1", 0.50, 0.55, 10, 10)
    await oms1.sync([DesiredOrder(exchange_id="m1", side="yes", action="buy", price=0.56, quantity=10, strategy=Strategy.VALUE)])
    rep = await oms1.reconcile()
    assert rep.new_fills == 1

    clock.advance(timedelta(hours=1).total_seconds())
    oms2, p2 = run()  # a restart: a new process, a new adapter
    await oms2.startup_reconcile()
    p2.set_top("m2", 0.55, 0.59, 20, 20)
    await oms2.sync([DesiredOrder(exchange_id="m2", side="yes", action="buy", price=0.60, quantity=20, strategy=Strategy.VALUE)])
    (intent2,) = [i for i in oms2.intents.values() if i.exchange_id == "m2"]
    rep = await oms2.reconcile()
    assert rep.new_fills == 1
    with dbengine.session() as s:
        fills = {f.exchange_id: f for f in s.execute(select(Fill)).scalars().all()}
    assert set(fills) == {"m1", "m2"} and fills["m2"].quantity == 20
    assert fills["m2"].intent_id == intent2.id
    assert fills["m1"].fill_id != fills["m2"].fill_id
    assert {o.order_id for o in p1.orders.values()}.isdisjoint(o.order_id for o in p2.orders.values())


async def test_restarted_paper_trader_continues_the_same_book(cfg, db, tmp_path):
    from sigcup.settings import Secrets
    from sigcup.trader import loop as trader_loop

    clock = FakeClock()
    p1 = PaperAdapter(cfg, clock=clock.now, use_db_quotes=False)
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
    oms1 = OMS(p1, cfg, budget, paper=True, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, backoff_s=0.0)
    await oms1.startup_reconcile()
    p1.set_top("m1", 0.50, 0.55, 10, 10)
    p1.set_top("m2", 0.50, 0.55, 10, 10)
    await oms1.sync([
        DesiredOrder(exchange_id="m1", side="yes", action="buy", price=0.56, quantity=10, strategy=Strategy.VALUE),
        DesiredOrder(exchange_id="m2", side="no", action="buy", price=0.50, quantity=6, strategy=Strategy.VALUE),
    ])
    await oms1.reconcile()
    assert {eid: p.quantity for eid, p in p1.positions.items()} == {"m1": 10, "m2": -6}

    sec = Secrets(_env_file=None, live=False, kill_file=str(tmp_path / "KILL"), sig_api_key="", sig_read_key="")
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=sec, clock=clock.now)
    assert isinstance(t.adapter, PaperAdapter) and t.adapter.run_id != p1.run_id
    assert t.adapter.cash == pytest.approx(p1.cash)
    assert {eid: p.quantity for eid, p in t.adapter.positions.items()} == {"m1": 10, "m2": -6}
    rep = await t.oms.startup_reconcile()
    assert not any("ledger" in d for d in rep.discrepancies)
