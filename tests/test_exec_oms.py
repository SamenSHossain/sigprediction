"""OMS: idempotency state machine, restart reconcile, write budget under burst, deadline, kill switch,
reconciliation invariants, re-quote policy, expiry, groups, cash reservation."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import AccountSnap, Fill, IntentState, OrderIntent, Position, Strategy
from sigcup.db.repo import kv_get
from sigcup.sig.errors import RequestInFlight
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import OrderRequest
from sigcup.trader.oms import KV_KILLED, OMS
from sigcup.trader.types import PRIORITY_ARB, PRIORITY_THETA, PRIORITY_VALUE, DesiredOrder
from tests.fakes_exec import FakeAdapter, FakeClock


def desired(eid: str, price: float = 0.5, qty: int = 10, side: str = "yes", action: str = "buy", **kw) -> DesiredOrder:
    kw.setdefault("strategy", Strategy.VALUE)
    return DesiredOrder(exchange_id=eid, side=side, action=action, price=price, quantity=qty, **kw)


def build(cfg, tmp_path, clock: FakeClock | None = None, fake: FakeAdapter | None = None, **kw):
    clock = clock or FakeClock()
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
    fake = fake or FakeAdapter(budget=budget, clock=clock.now)
    kw.setdefault("backoff_s", 0.0)
    oms = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, **kw)
    return oms, fake, budget, clock


def db_intents() -> list[OrderIntent]:
    with dbengine.session() as s:
        return list(s.execute(select(OrderIntent)).scalars().all())


def places(fake: FakeAdapter) -> list[tuple[datetime, str]]:
    return [w for w in fake.writes if w[1] == "place"]


# ---------------------------------------------------------------------------------------------
# (1) idempotent retry / restart
# ---------------------------------------------------------------------------------------------
async def test_uncertain_then_stored_order_single_order(db, cfg, tmp_path):
    oms, fake, _, _ = build(cfg, tmp_path)
    fake.default_script = ["uncertain"]  # first POST lands but the response is lost
    rep = await oms.sync([desired("m1")])
    assert len(rep.placed) == 1 and not rep.errors
    assert len(fake.orders) == 1, "retry with the same key must not create a duplicate"
    assert len(places(fake)) == 2
    (intent,) = db_intents()
    assert intent.state == IntentState.ACKED.value
    assert intent.sig_order_id == next(iter(fake.orders))
    keys = {o.req.idempotency_key for o in fake.orders.values()}
    assert keys == {intent.id}


async def test_intent_persisted_before_send(db, cfg, tmp_path):
    """The NEW row is on disk before the request goes out: crash between persist and response is safe."""
    seen: list[str] = []

    class Spy(FakeAdapter):
        async def place_order(self, req: OrderRequest):
            with dbengine.session() as s:
                row = s.get(OrderIntent, req.idempotency_key)
                seen.append(row.state if row else "missing")
            return await super().place_order(req)

    clock = FakeClock()
    budget = Budget(90, 25, clock=clock.mono)
    oms, fake, _, _ = build(cfg, tmp_path, clock=clock, fake=Spy(budget=budget, clock=clock.now))
    await oms.sync([desired("m1")])
    assert seen == [IntentState.SENT.value]


async def test_restart_resolves_uncertain_without_resend(db, cfg, tmp_path):
    clock = FakeClock()
    oms1, fake, budget, _ = build(cfg, tmp_path, clock=clock, max_retries=0)
    fake.default_script = ["uncertain"]
    await oms1.sync([desired("m1")])
    (intent,) = db_intents()
    assert intent.state == IntentState.UNCERTAIN.value
    assert len(fake.orders) == 1
    n_places = len(places(fake))

    # simulated restart: new OMS on the same exchange state
    oms2 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep)
    await oms2.startup_reconcile()
    (intent,) = db_intents()
    assert intent.state == IntentState.ACKED.value
    assert intent.sig_order_id in fake.orders
    assert len(fake.orders) == 1
    assert len(places(fake)) == n_places, "found via open orders: no re-POST needed"
    assert len(oms2.working()) == 1


async def test_restart_reposts_lost_request_exactly_once(db, cfg, tmp_path):
    clock = FakeClock()
    oms1, fake, budget, _ = build(cfg, tmp_path, clock=clock, max_retries=0)
    fake.default_script = ["lost"]  # never reached the exchange
    await oms1.sync([desired("m1")])
    assert fake.orders == {}
    (intent,) = db_intents()
    assert intent.state == IntentState.UNCERTAIN.value

    oms2 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep)
    await oms2.startup_reconcile()
    assert len(fake.orders) == 1
    (intent,) = db_intents()
    assert intent.state == IntentState.ACKED.value
    assert next(iter(fake.orders.values())).req.idempotency_key == intent.id

    # a second restart changes nothing
    oms3 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep)
    await oms3.startup_reconcile()
    assert len(fake.orders) == 1 and len(db_intents()) == 1


async def test_restart_cancels_never_sent_and_resolves_acked_gone(db, cfg, tmp_path):
    clock = FakeClock()
    oms, fake, budget, _ = build(cfg, tmp_path, clock=clock)
    await oms.sync([desired("m1"), desired("m2")])
    i1, i2 = sorted(db_intents(), key=lambda i: i.exchange_id)
    # m1 filled on the exchange while we were down; m2 vanished (cancelled by hand)
    fake.fill(i1.sig_order_id)
    fake.orders[i2.sig_order_id].status = "cancelled"
    now = clock.now()
    with dbengine.session() as s:
        s.add(
            OrderIntent(
                id="deadbeef" * 4, exchange_id="m3", side="yes", action="buy", price=0.5, quantity=5,
                strategy="value", state=IntentState.NEW.value, created_at=now, updated_at=now, expires_at=now + timedelta(minutes=15),
            )
        )
    oms2 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep)
    await oms2.startup_reconcile()
    states = {i.exchange_id: i.state for i in db_intents()}
    assert states == {"m1": IntentState.FILLED.value, "m2": IntentState.CANCELLED.value, "m3": IntentState.CANCELLED.value}
    assert len(fake.orders) == 2, "NEW intents are never sent on restart"
    with dbengine.session() as s:
        fills = s.execute(select(Fill)).scalars().all()
    assert len(fills) == 1 and fills[0].intent_id == i1.id


async def test_409_waits_out_lease_then_adopts_open_order(db, cfg, tmp_path):
    oms, fake, _, clock = build(cfg, tmp_path)
    intent = oms._new_intent(desired("m1"), clock.now())
    oms._persist(intent)
    # the first request is "in flight" on the exchange: it landed, our second attempt gets a 409
    await fake.place_order(oms._request(intent))
    fake.writes.clear()
    fake.script[intent.id] = [RequestInFlight("in flight", 409, "REQUEST_IN_FLIGHT")]
    t0 = clock.t
    await oms._submit(intent)
    assert intent.state == IntentState.ACKED.value
    assert clock.t - t0 >= oms.lease_s, "waited out the lease before polling"
    assert len(places(fake)) == 1 and len(fake.orders) == 1


# ---------------------------------------------------------------------------------------------
# (2) burst: the write budget is never exceeded in any sliding 60 s window
# ---------------------------------------------------------------------------------------------
def _max_in_window(stamps: list[datetime], window_s: float = 60.0) -> int:
    best = 0
    for t in stamps:
        best = max(best, sum(1 for u in stamps if t - timedelta(seconds=window_s) < u <= t))
    return best


async def test_burst_never_exceeds_writes_per_min(db, cfg, tmp_path):
    oms, fake, budget, clock = build(cfg, tmp_path)
    orders = [desired(f"m{i}", priority=PRIORITY_VALUE if i % 3 else PRIORITY_THETA) for i in range(60)]
    reps = await asyncio.gather(oms.sync(orders[:20]), oms.sync(orders[20:40]), oms.sync(orders[40:]))
    placed = sum(len(r.placed) for r in reps)
    assert placed <= cfg.limits.writes_per_min
    assert placed >= cfg.limits.writes_per_min - oms.kill_reserve
    # deferred = the unplaced placements plus housekeeping cancels the partial syncs planned
    assert sum(r.deferred for r in reps) >= 60 - placed
    assert all(not r.errors for r in reps)
    assert not any(i.state == IntentState.REJECTED.value for i in db_intents()), "nothing hit the limiter"

    # keep syncing the full desired set across the minute
    for _ in range(12):
        clock.advance(10)
        await oms.sync(orders)
    stamps = [t for t, _ in fake.writes]
    assert _max_in_window(stamps) <= cfg.limits.writes_per_min
    assert len(fake.orders) == 60, "everything eventually goes out, with no duplicates"
    assert len({o.req.idempotency_key for o in fake.orders.values()}) == 60
    assert not any(i.state == IntentState.REJECTED.value for i in db_intents())


async def test_priority_order_arb_and_takes_first(db, cfg, tmp_path):
    oms, fake, budget, clock = build(cfg, tmp_path, kill_reserve=0)
    for _ in range(cfg.limits.writes_per_min - 2):  # leave room for exactly two writes
        assert budget.writes.try_acquire()
    orders = [
        desired("v1", priority=PRIORITY_VALUE),
        desired("t1", priority=PRIORITY_THETA, strategy=Strategy.THETA),
        desired("a1", priority=PRIORITY_ARB, strategy=Strategy.ARB, take=True),
        desired("v2", priority=PRIORITY_VALUE),
    ]
    rep = await oms.sync(orders)
    sent = {o.req.exchange_id for o in fake.orders.values()}
    assert sent == {"a1", "t1"} and rep.deferred == 2


# ---------------------------------------------------------------------------------------------
# (3) deadline + kill switch
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("at", [datetime(2026, 11, 4, 17, 0, 1, tzinfo=UTC), datetime(2026, 11, 4, 16, 59, 45, tzinfo=UTC)])
async def test_no_post_at_or_after_deadline(db, cfg, tmp_path, at):
    clock = FakeClock(start=at)
    oms, fake, _, _ = build(cfg, tmp_path, clock=clock)
    rep = await oms.sync([desired("m1")])
    assert rep.blocked and "deadline" in rep.blocked
    assert fake.writes == [] and db_intents() == []
    assert not oms.may_send()


async def test_deadline_blocks_resend_on_startup_but_cancel_all_still_works(db, cfg, tmp_path):
    clock = FakeClock(start=datetime(2026, 11, 4, 16, 0, tzinfo=UTC))
    oms, fake, budget, _ = build(cfg, tmp_path, clock=clock, max_retries=0)
    fake.default_script = ["lost"]
    await oms.sync([desired("m1")])
    clock.advance(3600 + 5)  # past the deadline
    oms2 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep)
    await oms2.startup_reconcile()
    assert len(fake.orders) == 0, "no POST after the deadline, even for an unresolved intent"
    await oms2.cancel_all()
    assert fake.writes[-1][1] == "cancel_all"


async def test_kill_file_triggers_cancel_all_within_one_sync(db, cfg, tmp_path):
    oms, fake, _, _ = build(cfg, tmp_path)
    await oms.sync([desired("m1"), desired("m2")])
    assert len(fake.open_orders()) == 2
    (tmp_path / "KILL").write_text("stop")
    rep = await oms.sync([desired("m3")])
    assert rep.killed and "kill file" in rep.killed
    assert fake.writes[-1][1] == "cancel_all"
    assert fake.open_orders() == [] and "m3" not in {o.req.exchange_id for o in fake.orders.values()}
    assert all(i.state == IntentState.CANCELLED.value for i in db_intents())
    with dbengine.session() as s:
        assert kv_get(s, KV_KILLED)["reason"].startswith("kill file")
    assert oms.reserved == 0.0
    # stays dead
    rep2 = await oms.sync([desired("m4")])
    assert rep2.blocked and len(fake.orders) == 2


async def test_kill_flag_survives_restart(db, cfg, tmp_path):
    oms, fake, budget, clock = build(cfg, tmp_path)
    await oms.kill("daily loss")
    oms2 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep)
    await oms2.startup_reconcile()
    assert oms2.killed == "daily loss"
    rep = await oms2.sync([desired("m1")])
    assert rep.blocked and fake.orders == {}
    OMS.clear_killed()
    oms3 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep)
    await oms3.startup_reconcile()
    assert oms3.killed is None


# ---------------------------------------------------------------------------------------------
# (4) reconciliation invariants
# ---------------------------------------------------------------------------------------------
async def test_reconcile_fills_positions_intents_reservation(db, cfg, tmp_path):
    oms, fake, _, clock = build(cfg, tmp_path)
    await oms.sync([desired("m1", 0.5, 10), desired("m2", 0.4, 20)])
    i1, i2 = sorted(oms.working(), key=lambda i: i.exchange_id)
    assert oms.reserved == pytest.approx(0.5 * 10 + 0.4 * 20)
    fake.fill(i1.sig_order_id)  # full
    fake.fill(i2.sig_order_id, 5)  # partial
    clock.advance(5)
    rep = await oms.reconcile()
    assert rep.new_fills == 2 and rep.discrepancies == []
    with dbengine.session() as s:
        fills = s.execute(select(Fill)).scalars().all()
        positions = {p.exchange_id: p.quantity for p in s.execute(select(Position)).scalars().all()}
        snaps = s.execute(select(AccountSnap)).scalars().all()
    assert {f.intent_id for f in fills} == {i1.id, i2.id}
    assert positions == {"m1": 10, "m2": 5}
    states = {i.exchange_id: (i.state, i.filled_qty) for i in db_intents()}
    assert states == {"m1": (IntentState.FILLED.value, 10), "m2": (IntentState.ACKED.value, 5)}
    assert oms.reserved == pytest.approx(0.4 * 15)
    assert oms.reserved <= oms.cash
    assert rep.working_intents == rep.open_orders == 1
    assert snaps and snaps[-1].reserved == pytest.approx(oms.reserved)
    # idempotent: a second reconcile records nothing new and keeps filled_qty
    rep2 = await oms.reconcile()
    assert rep2.new_fills == 0 and rep2.discrepancies == []
    assert {i.exchange_id: i.filled_qty for i in db_intents()} == {"m1": 10, "m2": 5}


async def test_reconcile_adopts_orphans_and_repairs_vanished(db, cfg, tmp_path):
    oms, fake, _, clock = build(cfg, tmp_path)
    await oms.sync([desired("m1")])
    (i1,) = oms.working()
    # an order placed outside the OMS (e.g. by hand in the UI)
    await fake.place_order(OrderRequest(exchange_id="x1", side="yes", action="buy", price=0.3, quantity=7, idempotency_key="manual-1", tournament_id="t"))
    # our order vanished without fills
    fake.orders[i1.sig_order_id].status = "cancelled"
    rep = await oms.reconcile()
    assert any("no intent" in d for d in rep.discrepancies)
    assert any(i1.id[:8] in r and "CANCELLED" in r for r in rep.repaired)
    working = oms.working()
    assert len(working) == 1 and working[0].strategy == Strategy.MANUAL.value and working[0].sig_order_id
    assert {i.sig_order_id for i in working} == {o.order_id for o in fake.open_orders()}


async def test_reconcile_restores_reservation_leq_cash(db, cfg, tmp_path):
    oms, fake, _, _ = build(cfg, tmp_path)
    await oms.sync([desired(f"m{i}", 0.5, 1000, priority=PRIORITY_VALUE) for i in range(3)])
    assert oms.reserved == pytest.approx(1500)
    fake.cash = 600.0  # cash left the account some other way
    rep = await oms.reconcile()
    assert any("reservation" in d for d in rep.discrepancies)
    assert oms.reserved <= oms.cash
    assert len(fake.open_orders()) == 1


async def test_fills_ledger_mismatch_is_reported(db, cfg, tmp_path):
    oms, fake, _, _ = build(cfg, tmp_path)
    await oms.sync([desired("m1")])
    (i1,) = oms.working()
    fake.fill(i1.sig_order_id)
    await oms.reconcile()
    fake.positions["m1"].quantity = 99  # exchange disagrees with our ledger
    rep = await oms.reconcile()
    assert any("ledger" in d for d in rep.discrepancies)


# ---------------------------------------------------------------------------------------------
# re-quote policy, expiry, groups, reservation
# ---------------------------------------------------------------------------------------------
async def test_requote_policy_prefers_expiry_over_cancel_replace(db, cfg, tmp_path):
    oms, fake, _, clock = build(cfg, tmp_path)
    await oms.sync([desired("m1", 0.50)])
    n = len(fake.writes)
    rep = await oms.sync([desired("m1", 0.51)])  # below threshold
    assert rep.kept == 1 and len(fake.writes) == n
    clock.advance(cfg.value.requote_cooldown_s + 1)
    rep = await oms.sync([desired("m1", 0.53)])  # favourable, small: let it expire
    assert rep.kept == 1 and len(fake.writes) == n
    rep = await oms.sync([desired("m1", 0.47)])  # adverse: our bid is now too high -> cancel + replace
    assert len(rep.cancelled) == 1 and len(rep.placed) == 1
    assert [w[1] for w in fake.writes[n:]] == ["cancel", "place"]
    assert len(fake.open_orders()) == 1 and fake.open_orders()[0].req.price == 0.47
    n = len(fake.writes)
    rep = await oms.sync([desired("m1", 0.43)])  # inside the cooldown -> keep
    assert rep.kept == 1 and len(fake.writes) == n
    clock.advance(cfg.value.requote_cooldown_s + 1)
    rep = await oms.sync([desired("m1", 0.55)])  # large favourable move -> replace
    assert len(rep.placed) == 1


async def test_expired_orders_are_relaid(db, cfg, tmp_path):
    oms, fake, _, clock = build(cfg, tmp_path)
    await oms.sync([desired("m1")])
    (i1,) = oms.working()
    assert i1.expires_at == clock.now() + timedelta(seconds=cfg.risk.order_expiry_s)
    assert fake.orders[i1.sig_order_id].req.expires_at == i1.expires_at
    clock.advance(cfg.risk.order_expiry_s + 1)
    rep = await oms.sync([desired("m1")])
    assert len(rep.placed) == 1
    states = sorted(i.state for i in db_intents())
    assert states == [IntentState.ACKED.value, IntentState.EXPIRED.value]


async def test_undesired_orders_cancelled_only_with_spare_budget(db, cfg, tmp_path):
    oms, fake, _, clock = build(cfg, tmp_path)
    await oms.sync([desired("m1")])
    rep = await oms.sync([])
    assert len(rep.cancelled) == 1 and fake.open_orders() == []


async def test_multi_leg_group_is_one_write(db, cfg, tmp_path):
    oms, fake, _, _ = build(cfg, tmp_path)
    legs = [
        desired("d", 0.40, 10, strategy=Strategy.ARB, priority=PRIORITY_ARB, take=True, group_id="g1"),
        desired("r", 0.55, 10, strategy=Strategy.ARB, priority=PRIORITY_ARB, take=True, group_id="g1"),
    ]
    rep = await oms.sync(legs)
    assert len(rep.placed) == 2 and [w[1] for w in fake.writes] == ["multi"]
    assert {i.group_id for i in db_intents()} == {"g1"}
    assert all(i.state == IntentState.ACKED.value for i in db_intents())
    rep2 = await oms.sync(legs)  # still resting: do not re-place
    assert rep2.kept == 1 and len(fake.orders) == 2


async def test_cash_reservation_shrinks_and_blocks_buys(db, cfg, tmp_path):
    clock = FakeClock()
    budget = Budget(90, 25, clock=clock.mono)
    fake = FakeAdapter(budget=budget, clock=clock.now, cash=1000.0)
    oms, fake, _, _ = build(cfg, tmp_path, clock=clock, fake=fake)
    await oms.reconcile()
    rep = await oms.sync([desired("m1", 0.5, 10_000), desired("m2", 0.5, 10)])
    assert len(rep.placed) == 1 and len(rep.skipped) == 1
    o = fake.open_orders()[0]
    assert o.req.quantity == 2000 and oms.reserved == pytest.approx(1000.0)
    assert oms.reserved <= oms.cash


async def test_sell_of_owned_shares_needs_no_cash(db, cfg, tmp_path):
    clock = FakeClock()
    budget = Budget(90, 25, clock=clock.mono)
    fake = FakeAdapter(budget=budget, clock=clock.now, cash=0.0)
    from sigcup.sig.schemas import PositionInfo

    fake.positions["m1"] = PositionInfo(exchange_id="m1", quantity=50, avg_price=0.4, mark=0.5)
    oms, fake, _, _ = build(cfg, tmp_path, clock=clock, fake=fake)
    await oms.reconcile()
    rep = await oms.sync([desired("m1", 0.6, 50, action="sell")])
    assert len(rep.placed) == 1 and oms.reserved == 0.0
    rep = await oms.sync([desired("m1", 0.6, 80, action="sell", strategy=Strategy.THETA)])
    assert rep.skipped, "the uncovered part of the sell is a NO buy that needs cash"
