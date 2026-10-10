"""Regression tests for confirmed OMS findings: no local cancel of unacknowledged intents (F1), an
unconfirmed cancel never triggers a replacement (F2), idempotent fill accounting for fills at placement
(F5), cash and positions follow fills at placement (F6), a lock-free kill (F7), a verified and retried
kill (F8), withdrawal cancels ahead of new placements (F9) and the fills cursor after a restart (F11)."""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Fill, IntentState, OrderIntent, Strategy
from sigcup.paper import PaperAdapter
from sigcup.sig.errors import OrderStatusUnknown, RequestInFlight, TransientError
from sigcup.sig.ratelimit import Budget
from sigcup.trader.oms import OMS
from sigcup.trader.types import PRIORITY_THETA, PRIORITY_VALUE, DesiredOrder
from tests.fakes_exec import FakeAdapter, FakeClock


def desired(eid: str, price: float = 0.5, qty: int = 10, side: str = "yes", action: str = "buy", **kw) -> DesiredOrder:
    kw.setdefault("strategy", Strategy.VALUE)
    return DesiredOrder(exchange_id=eid, side=side, action=action, price=price, quantity=qty, **kw)


def make(cfg, tmp_path, *, fake=None, clock: FakeClock | None = None, **kw):
    clock = clock or FakeClock()
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
    fake = fake or FakeAdapter(budget=budget, clock=clock.now)
    kw.setdefault("backoff_s", 0.0)
    kw.setdefault("sleep", clock.sleep)
    oms = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, **kw)
    return oms, fake, budget, clock


def db_intents() -> list[OrderIntent]:
    with dbengine.session() as s:
        return list(s.execute(select(OrderIntent)).scalars().all())


def writes(fake: FakeAdapter) -> list[str]:
    return [w[1] for w in fake.writes]


# -- F1 ---------------------------------------------------------------------------------------
async def test_uncertain_intent_is_never_replaced_or_cancelled_locally(db, cfg, tmp_path):
    oms, fake, _, clock = make(cfg, tmp_path, max_retries=1)
    # the first POST lands but its response is lost; the retry times out too
    fake.default_script = ["uncertain", OrderStatusUnknown("timeout", None, "TIMEOUT")]
    await oms.sync([desired("m1", 0.50, 10)])
    (row,) = db_intents()
    intent = oms.intents[row.id]
    assert intent.state == IntentState.UNCERTAIN.value and intent.sig_order_id is None
    assert len(fake.open_orders()) == 1

    clock.advance(cfg.value.requote_cooldown_s + 1)
    rep = await oms.sync([desired("m1", 0.45, 10)])  # adverse move: would be a replace if ACKED
    assert rep.placed == [] and rep.cancelled == []
    assert len(fake.open_orders()) == 1, "no second order while the first one's fate is unknown"
    assert await oms._cancel(intent) is False
    assert intent.state == IntentState.UNCERTAIN.value, "an unacknowledged intent is not cancelled locally"

    # reconcile resolves it by key; only then may it be replaced
    await oms.reconcile()
    assert intent.state == IntentState.ACKED.value and intent.sig_order_id
    clock.advance(cfg.value.requote_cooldown_s + 1)
    rep = await oms.sync([desired("m1", 0.45, 10)])
    assert len(rep.cancelled) == 1 and len(rep.placed) == 1
    (o,) = fake.open_orders()
    assert o.req.price == 0.45
    assert oms.reserved == pytest.approx(0.45 * 10)


# -- F2 ---------------------------------------------------------------------------------------
async def test_unconfirmed_cancel_does_not_place_a_replacement(db, cfg, tmp_path):
    oms, fake, _, clock = make(cfg, tmp_path)
    await oms.sync([desired("m1", 0.50, 1000)])
    (intent,) = oms.working()
    clock.advance(cfg.value.requote_cooldown_s + 1)
    fake.cancel_script = ["false"]  # DELETE reports nothing cancelled, but the order is still live
    rep = await oms.sync([desired("m1", 0.45, 1000)])
    assert rep.placed == []
    assert [o.req.price for o in fake.open_orders()] == [0.50], "never two live orders for one key"
    assert intent.state == IntentState.ACKED.value
    assert oms.reserved == pytest.approx(500.0)
    # until a reconcile has classified the order, the next sync does not retry the cancel
    n = len(fake.writes)
    clock.advance(cfg.value.requote_cooldown_s + 1)
    await oms.sync([desired("m1", 0.45, 1000)])
    assert len(fake.writes) == n
    # reconcile sees it still open: the replace then goes through normally
    await oms.reconcile()
    assert intent.state == IntentState.ACKED.value
    rep = await oms.sync([desired("m1", 0.45, 1000)])
    assert len(rep.cancelled) == 1 and len(rep.placed) == 1
    assert [o.req.price for o in fake.open_orders()] == [0.45]


async def test_unconfirmed_cancel_of_a_filled_order_is_classified_by_reconcile(db, cfg, tmp_path):
    oms, fake, _, clock = make(cfg, tmp_path)
    await oms.sync([desired("m1", 0.50, 10)])
    (intent,) = oms.working()
    fake.fill(intent.sig_order_id)  # filled before our cancel arrives
    fake.cancel_script = ["false"]
    clock.advance(cfg.value.requote_cooldown_s + 1)
    rep = await oms.sync([])
    assert rep.cancelled == [] and intent.state == IntentState.ACKED.value
    await oms.reconcile()
    assert intent.state == IntentState.FILLED.value and intent.filled_qty == 10


# -- F5 / F6 ----------------------------------------------------------------------------------
def paper_oms(cfg, tmp_path, cash: float = 10_000.0):
    clock = FakeClock()
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
    p = PaperAdapter(cfg, cash=cash, clock=clock.now, use_db_quotes=False)
    oms = OMS(p, cfg, budget, paper=True, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, backoff_s=0.0)
    return oms, p, clock


async def test_fill_at_placement_is_not_double_counted_by_reconcile(db, cfg, tmp_path):
    oms, p, clock = paper_oms(cfg, tmp_path)
    p.set_top("m1", bid=0.50, ask=0.55, bid_size=20, ask_size=40)
    await oms.reconcile()
    await oms.sync([desired("m1", 0.56, 100)])
    (intent,) = oms.working()
    assert intent.filled_qty == 40 and intent.state == IntentState.ACKED.value
    clock.advance(5)
    rep = await oms.reconcile()
    assert rep.new_fills == 1
    assert intent.filled_qty == 40, "fills listed after the placement report are the same 40 shares"
    assert intent.avg_fill == pytest.approx(0.56)
    assert oms.reserved == pytest.approx(0.56 * 60)
    assert not any("ledger" in d for d in rep.discrepancies)

    p.set_top("m1", bid=0.50, ask=0.55, bid_size=20, ask_size=20)  # new liquidity: 20 more fill
    clock.advance(5)
    rep = await oms.reconcile()
    (row,) = db_intents()
    assert (row.state, row.filled_qty, row.strategy) == (IntentState.ACKED.value, 60, Strategy.VALUE.value)
    assert oms.reserved == pytest.approx(0.56 * 40)
    assert not any("no intent" in d for d in rep.discrepancies)


async def test_fill_at_placement_updates_cash_and_positions_before_reconcile(db, cfg, tmp_path):
    oms, p, clock = paper_oms(cfg, tmp_path, cash=30_000.0)
    p.set_top("m1", bid=0.85, ask=0.90, bid_size=100, ask_size=20_000)
    await oms.reconcile()
    equity = oms.equity
    await oms.sync([desired("m1", 0.90, 11_000, strategy=Strategy.THETA, priority=PRIORITY_THETA)])
    assert p.cash == pytest.approx(30_000 - 9_900)
    assert oms.cash == pytest.approx(p.cash), "spent cash is not free cash"
    assert oms.reserved == pytest.approx(0.0)
    assert oms.positions["m1"].quantity == 11_000
    assert oms.equity == pytest.approx(equity), "a fill at the trade price does not move equity"
    clock.advance(5)
    await oms.reconcile()
    assert oms.cash == pytest.approx(p.cash) and oms.positions["m1"].quantity == 11_000


async def test_sell_fill_at_placement_books_cash_both_ways(db, cfg, tmp_path):
    oms, p, clock = paper_oms(cfg, tmp_path, cash=1_000.0)
    p.set_top("m1", bid=0.40, ask=0.45, bid_size=100, ask_size=100)
    await oms.reconcile()
    await oms.sync([desired("m1", 0.45, 50)])  # buy 50 @0.45
    assert oms.cash == pytest.approx(p.cash)
    clock.advance(120)
    await oms.reconcile()
    # sell 80 @0.40: 50 owned return cash, 30 unowned fill as NO bought at 0.60
    await oms.sync([desired("m1", 0.40, 80, action="sell")])
    assert oms.cash == pytest.approx(p.cash)
    assert oms.positions["m1"].quantity == -30


# -- F7 ---------------------------------------------------------------------------------------
async def test_kill_does_not_wait_for_a_sync_sleeping_out_a_lease(db, cfg, tmp_path):
    oms, fake, _, _ = make(cfg, tmp_path)
    await oms.sync([desired("m0")])
    gate, sleeping = asyncio.Event(), asyncio.Event()

    async def blocking_sleep(_s: float) -> None:
        sleeping.set()
        await gate.wait()

    oms.sleep = blocking_sleep
    fake.default_script = [RequestInFlight("in flight", 409, "REQUEST_IN_FLIGHT")] * 10
    task = asyncio.create_task(oms.sync([desired("m0"), desired("m1")]))
    await asyncio.wait_for(sleeping.wait(), 1.0)
    await asyncio.wait_for(oms.kill("operator"), timeout=1.0)
    assert "cancel_all" in writes(fake)
    assert fake.open_orders() == []
    n_places = writes(fake).count("place")
    gate.set()
    rep = await asyncio.wait_for(task, 1.0)
    assert writes(fake).count("place") == n_places, "nothing is sent after the kill"
    assert fake.open_orders() == [] and oms.killed == "operator"
    assert rep.errors == []


async def test_order_landing_after_the_kill_is_swept_by_its_sync(db, cfg, tmp_path):
    oms, fake, _, _ = make(cfg, tmp_path)
    gate, posting = asyncio.Event(), asyncio.Event()

    class SlowPost(FakeAdapter):
        async def place_order(self, req):
            posting.set()
            await gate.wait()
            return await super().place_order(req)

    oms.adapter = slow = SlowPost(budget=oms.budget, clock=fake.clock)
    task = asyncio.create_task(oms.sync([desired("m1")]))
    await asyncio.wait_for(posting.wait(), 1.0)
    await asyncio.wait_for(oms.kill("operator"), 1.0)  # the POST is already on the wire
    gate.set()
    await asyncio.wait_for(task, 1.0)
    assert len(slow.orders) == 1 and slow.open_orders() == [], "the late order was cancelled"
    assert all(i.state == IntentState.CANCELLED.value for i in db_intents())


# -- F8 ---------------------------------------------------------------------------------------
async def test_kill_retries_a_failed_cancel_all_and_confirms_nothing_is_open(db, cfg, tmp_path):
    oms, fake, _, _ = make(cfg, tmp_path)
    await oms.sync([desired("m1"), desired("m2"), desired("m3")])
    calls = {"n": 0}
    real = fake.cancel_all

    async def flaky_cancel_all() -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            fake._write("cancel_all")
            raise TransientError("service unavailable", 503, "UNAVAILABLE")
        return await real()

    fake.cancel_all = flaky_cancel_all
    await oms.kill("daily loss")
    assert fake.open_orders() == []
    assert oms.kill_confirmed is True
    assert all(i.state == IntentState.CANCELLED.value for i in db_intents())


async def test_kill_cancels_leftovers_of_a_partial_cancel_all(db, cfg, tmp_path):
    oms, fake, _, _ = make(cfg, tmp_path)
    await oms.sync([desired("m1"), desired("m2"), desired("m3")])

    async def partial_cancel_all() -> int:  # a 207 that cancelled only the first order
        fake._write("cancel_all")
        first = fake.open_orders()[0]
        first.status = "cancelled"
        return 1

    fake.cancel_all = partial_cancel_all
    await oms.kill("operator")
    assert fake.open_orders() == []
    assert writes(fake).count("cancel") == 2
    assert all(i.state == IntentState.CANCELLED.value for i in db_intents())


async def test_unconfirmed_kill_keeps_intents_and_restart_with_the_flag_cancels_open_orders(db, cfg, tmp_path):
    oms, fake, budget, clock = make(cfg, tmp_path)
    await oms.sync([desired("m1"), desired("m2")])

    async def broken_cancel_all() -> int:
        raise TransientError("service unavailable", 503, "UNAVAILABLE")

    async def broken_cancel(order_id: str) -> bool:
        raise TransientError("service unavailable", 503, "UNAVAILABLE")

    real_all, real_one = fake.cancel_all, fake.cancel_order
    fake.cancel_all, fake.cancel_order = broken_cancel_all, broken_cancel
    await oms.kill("operator")
    assert oms.killed == "operator" and oms.kill_confirmed is False
    assert len(fake.open_orders()) == 2
    assert {i.state for i in db_intents()} == {IntentState.ACKED.value}, "orders still open are not marked cancelled"

    # restart with the persisted flag: open orders are cancelled before anything else
    fake.cancel_all, fake.cancel_order = real_all, real_one
    oms2 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, backoff_s=0.0)
    await oms2.startup_reconcile()
    assert oms2.killed == "operator" and oms2.kill_confirmed is True
    assert fake.open_orders() == []
    assert {i.state for i in db_intents()} == {IntentState.CANCELLED.value}


# -- F9 ---------------------------------------------------------------------------------------
async def test_withdrawal_cancel_goes_ahead_of_new_placements(db, cfg, tmp_path):
    oms, fake, _, clock = make(cfg, tmp_path)
    await oms.sync([desired("hot", 0.60, 2000)])
    (hot,) = oms.working()
    clock.advance(cfg.cadence.model_s)  # the next model cycle withdraws it (a hard event)
    theta = [desired(f"t{i}", 0.90, 10, strategy=Strategy.THETA, priority=PRIORITY_THETA) for i in range(40)]
    rep = await oms.sync(theta)
    assert hot.id in rep.cancelled, "the risk-reducing cancel is not starved by new placements"
    assert rep.deferred > 0 and "hot" not in {o.req.exchange_id for o in fake.open_orders()}


async def test_withdrawal_ranks_ahead_of_value_placements_too(db, cfg, tmp_path):
    oms, fake, budget, clock = make(cfg, tmp_path, kill_reserve=0)
    await oms.sync([desired("old", 0.60, 100)])
    clock.advance(cfg.cadence.model_s)
    while budget.writes.available() > 1:
        assert budget.writes.try_acquire()
    rep = await oms.sync([desired("new", 0.50, 10, priority=PRIORITY_VALUE)])
    assert len(rep.cancelled) == 1 and rep.placed == [] and rep.deferred == 1


# -- F11 --------------------------------------------------------------------------------------
async def test_restart_reads_fills_since_the_last_recorded_fill(db, cfg, tmp_path):
    sinces: list[datetime | None] = []

    class Spy(FakeAdapter):
        async def get_fills(self, since=None):
            sinces.append(since)
            return await super().get_fills(since)

    clock = FakeClock()
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
    fake = Spy(budget=budget, clock=clock.now)
    oms, _, _, _ = make(cfg, tmp_path, fake=fake, clock=clock)
    await oms.sync([desired("m1"), desired("m2")])
    for i in oms.working():
        fake.fill(i.sig_order_id)
    clock.advance(5)
    await oms.reconcile()
    with dbengine.session() as s:
        assert len(s.execute(select(Fill)).scalars().all()) == 2

    clock.advance(3600)
    sinces.clear()
    oms2 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep)
    await oms2.startup_reconcile()
    for _ in range(3):
        clock.advance(60)
        await oms2.reconcile()
    assert sinces and all(s is not None for s in sinces), "no full fill-history walk after a restart"
    assert oms2.last_fill_ts is not None


async def test_own_order_closed_locally_but_still_open_is_revived_not_adopted_as_manual(db, cfg, tmp_path):
    oms, fake, _, clock = make(cfg, tmp_path)
    await oms.sync([desired("m1", 0.90, 50, strategy=Strategy.THETA, priority=PRIORITY_THETA)])
    (intent,) = oms.working()
    oms._set_state(intent, IntentState.FILLED)  # closed locally by mistake; the order still rests
    clock.advance(5)
    await oms.reconcile()
    (row,) = db_intents()
    assert (row.id, row.state, row.strategy) == (intent.id, IntentState.ACKED.value, Strategy.THETA.value)
    n = len(fake.writes)
    clock.advance(cfg.cadence.model_s)
    rep = await oms.sync([desired("m1", 0.90, 50, strategy=Strategy.THETA, priority=PRIORITY_THETA)])
    assert rep.kept == 1 and len(fake.writes) == n, "no cancel + re-place churn"
