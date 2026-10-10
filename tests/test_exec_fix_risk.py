"""Regression tests for F4: risk.apply treats the desired set as the target book, so an order that is
re-emitted while it already rests is not charged twice (resting quantity + reservation), and the OMS does
not cancel and re-place half the book every other cycle."""

from __future__ import annotations

from datetime import UTC, datetime

from sigcup.db.models import IntentState, OrderIntent, Strategy
from sigcup.settings import Risk, StrategyConfig
from sigcup.sig.ratelimit import Budget
from sigcup.trader import risk
from sigcup.trader.oms import OMS
from sigcup.trader.types import PRIORITY_THETA, DesiredOrder, FairValue, MarketView, RaceInfo, TraderContext
from tests.fakes_exec import FakeAdapter, FakeClock

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def reserve_cfg(**risk_kw) -> StrategyConfig:
    base = dict(
        per_race_cap_frac=10.0, per_state_cap_frac=10.0, max_swing_loss_frac=10.0,
        liquid_reserve_frac=0.2, liquid_reserve_frac_final72h=0.2, max_position_shares=10**9,
    )
    base.update(risk_kw)
    return StrategyConfig(risk=Risk(**base))


def theta(eid: str, qty: int = 500, price: float = 0.90) -> DesiredOrder:
    return DesiredOrder(exchange_id=eid, side="yes", action="buy", price=price, quantity=qty, strategy=Strategy.THETA, priority=PRIORITY_THETA)


def resting(eid: str, qty: int = 500, price: float = 0.90, strategy: str = "theta") -> OrderIntent:
    return OrderIntent(
        id=f"k-{eid}-{strategy}", exchange_id=eid, side="yes", action="buy", price=price, quantity=qty,
        strategy=strategy, state=IntentState.ACKED.value, filled_qty=0,
    )


def context(cfg: StrategyConfig, eids: list[str], open_orders: dict[str, list], cash: float, reserved: float) -> TraderContext:
    markets = {
        eid: MarketView(exchange_id=eid, race_id=f"R-{eid}", party="D", chamber="senate", state=f"S{i}", bid=0.88, ask=0.91, open_orders=list(open_orders.get(eid, [])))
        for i, eid in enumerate(eids)
    }
    races = {mv.race_id: RaceInfo(race_id=mv.race_id, chamber="senate", state=mv.state, label=mv.race_id, exclusive=True, exhaustive=True, markets={"D": mv.exchange_id}) for mv in markets.values()}
    fair = {eid: FairValue(exchange_id=eid, fair=0.96, sd=0.02) for eid in eids}
    return TraderContext(cfg=cfg, now=NOW, markets=markets, races=races, fair=fair, cash=cash, reserved=reserved, equity=cash, bankroll_start=100_000.0, hours_to_deadline=600.0)


def test_reemitted_resting_orders_are_not_charged_twice_against_the_reserve():
    cfg = reserve_cfg()
    eids = [f"m{i}" for i in range(10)]
    rest = {eid: [resting(eid)] for eid in eids}
    ctx = context(cfg, eids, rest, cash=26_000.0, reserved=4_500.0)
    out = risk.apply(ctx, [theta(eid) for eid in eids])
    assert {d.exchange_id: d.quantity for d in out} == {eid: 500 for eid in eids}
    # a genuinely new order still has to fit the free cash (1 500 left after the resting book)
    value = DesiredOrder(exchange_id="m0", side="yes", action="buy", price=0.85, quantity=5_000, strategy=Strategy.VALUE)
    out = risk.apply(ctx, [theta(eid) for eid in eids] + [value])
    q = {(d.exchange_id, str(d.strategy)): d.quantity for d in out}
    assert all(q[(eid, "theta")] == 500 for eid in eids)
    assert 1_760 <= q[("m0", "value")] <= int(1_500 / 0.85)


def test_reemitted_resting_order_is_not_counted_against_the_position_cap():
    cfg = reserve_cfg(liquid_reserve_frac=0.0, liquid_reserve_frac_final72h=0.0, max_position_shares=1_000)
    ctx = context(cfg, ["m0"], {"m0": [resting("m0", 1_000)]}, cash=100_000.0, reserved=900.0)
    out = risk.apply(ctx, [theta("m0", 1_000)])
    assert [d.quantity for d in out] == [1_000]
    # a different strategy on the same market is additional exposure
    value = DesiredOrder(exchange_id="m0", side="yes", action="buy", price=0.90, quantity=500, strategy=Strategy.VALUE)
    assert risk.apply(ctx, [theta("m0", 1_000), value]) == [out[0]]


async def test_multi_cycle_target_book_does_not_thrash(db, tmp_path):
    cfg = reserve_cfg()
    clock = FakeClock()
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
    fake = FakeAdapter(budget=budget, clock=clock.now, cash=26_000.0)
    oms = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, backoff_s=0.0)
    await oms.reconcile()
    eids = [f"m{i}" for i in range(10)]
    accepted: list[int] = []
    cancelled = 0
    for _ in range(6):
        by_market: dict[str, list] = {}
        for i in oms.acked():
            by_market.setdefault(i.exchange_id, []).append(i)
        ctx = context(cfg, eids, by_market, cash=oms.cash, reserved=oms.reserved)
        out = risk.apply(ctx, [theta(eid) for eid in eids])
        accepted.append(len(out))
        rep = await oms.sync(out)
        cancelled += len(rep.cancelled)
        clock.advance(60)
        await oms.reconcile()
    assert accepted == [10] * 6
    assert cancelled == 0 and len(fake.orders) == 10
    assert len(fake.open_orders()) == 10
