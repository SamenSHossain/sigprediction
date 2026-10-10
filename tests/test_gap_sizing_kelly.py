"""A1: risk.apply sizes every non-arb order from fractional Kelly on the order's fair value.

target stake = kelly_fraction(fair, price, side) * multiplier(strategy) * equity; quantity = stake / cost per
share less what the market already holds in the same direction; every cap still applies on top; an order
whose Kelly size is below one share is dropped."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sigcup.db.models import IntentState, OrderIntent, Strategy
from sigcup.settings import Risk, StrategyConfig
from sigcup.trader import risk
from sigcup.trader.types import (
    PRIORITY_ARB,
    PRIORITY_MM,
    PRIORITY_THETA,
    PRIORITY_VALUE,
    DesiredOrder,
    FairValue,
    MarketView,
    RaceInfo,
    TraderContext,
)

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def loose(**risk_kw) -> StrategyConfig:
    base = dict(
        per_race_cap_frac=10.0, per_state_cap_frac=10.0, max_swing_loss_frac=10.0,
        liquid_reserve_frac=0.0, liquid_reserve_frac_final72h=0.0, max_position_shares=10**9,
    )
    base.update(risk_kw)
    return StrategyConfig(risk=Risk(**base))


def mv(eid: str, race: str = "SEN-TX", party: str = "D", state: str = "TX", position: int = 0, open_orders=None) -> MarketView:
    return MarketView(
        exchange_id=eid, race_id=race, party=party, chamber="senate", state=state, bid=0.45, ask=0.55,
        position=position, open_orders=list(open_orders or []),
    )


def ctx_for(cfg: StrategyConfig, markets: list[MarketView], fair: dict[str, float], **kw) -> TraderContext:
    races: dict[str, RaceInfo] = {}
    for m in markets:
        r = races.setdefault(m.race_id, RaceInfo(race_id=m.race_id, chamber="senate", state=m.state, label=m.race_id, exclusive=True, exhaustive=True))
        r.markets[m.party] = m.exchange_id
    base = dict(
        cfg=cfg, now=NOW, markets={m.exchange_id: m for m in markets}, races=races,
        fair={eid: FairValue(exchange_id=eid, fair=f, sd=0.02, components={"ref": f}) for eid, f in fair.items()},
        cash=100_000.0, reserved=0.0, equity=100_000.0, bankroll_start=100_000.0, hours_to_deadline=600.0,
    )
    base.update(kw)
    return TraderContext(**base)


def order(eid: str, price: float, fair: float | None, strategy: Strategy = Strategy.VALUE, qty: int = 200, side="yes", action="buy", **kw) -> DesiredOrder:
    kw.setdefault("priority", PRIORITY_THETA if strategy == Strategy.THETA else PRIORITY_VALUE)
    return DesiredOrder(exchange_id=eid, side=side, action=action, price=price, quantity=qty, strategy=strategy, fair=fair, **kw)


def expected(fair_yes: float, yes_price: float, long: bool, mult: float, equity: float) -> int:
    cost = yes_price if long else 1.0 - yes_price
    return int(risk.kelly_fraction(fair_yes, yes_price, long) * mult * equity / cost + 1e-6)


# -- the Kelly quantity -------------------------------------------------------------------------------
def test_value_order_is_sized_by_fractional_kelly_on_equity():
    cfg = loose()
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.60})
    out = risk.apply(ctx, [order("d", 0.50, 0.60, qty=200)])
    # f* = (0.60 - 0.50) / 0.50 = 0.2; x kelly_fraction_default 0.25 x 100k = 5 000 SUSQies / 0.50 = 10 000 shares
    assert [d.quantity for d in out] == [10_000]
    assert out[0].quantity == risk.kelly_size(0.60, 0.50, True, Strategy.VALUE, 100_000, cfg)
    # the bankroll is current equity, not the starting bankroll
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.60}, equity=80_000.0)
    assert [d.quantity for d in risk.apply(ctx, [order("d", 0.50, 0.60)])] == [8_000]
    # selling YES above fair: f* = (p - q) / p, cost per share 1 - p
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.40})
    out = risk.apply(ctx, [order("d", 0.50, 0.40, action="sell")])
    assert [d.quantity for d in out] == [expected(0.40, 0.50, False, 0.25, 100_000)] == [10_000]


def test_theta_multiplier_depends_on_the_favourite_fair_and_fair_cap():
    cfg = loose()
    # fair >= 0.90: kelly_fraction_theta (0.5)
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.95})
    out = risk.apply(ctx, [order("d", 0.90, 0.95, Strategy.THETA, qty=500)])
    assert [d.quantity for d in out] == [expected(0.95, 0.90, True, cfg.risk.kelly_fraction_theta, 100_000)]
    assert out[0].quantity > 20_000
    # fair below 0.90: the default multiplier (0.25)
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.88})
    out = risk.apply(ctx, [order("d", 0.83, 0.88, Strategy.THETA, qty=500)])
    assert [d.quantity for d in out] == [expected(0.88, 0.83, True, cfg.risk.kelly_fraction_default, 100_000)]
    # NO-side theta: the favourite is NO (fair_no = 0.95 >= 0.90), bought at NO price 0.88
    ctx = ctx_for(cfg, [mv("r", party="R")], {"r": 0.05})
    out = risk.apply(ctx, [order("r", 0.88, 0.05, Strategy.THETA, qty=500, side="no")])
    assert [d.quantity for d in out] == [expected(0.05, 0.12, False, cfg.risk.kelly_fraction_theta, 100_000)]
    # a fair above theta.fair_cap is sized at the cap (0.985), like the yield
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.999})
    out = risk.apply(ctx, [order("d", 0.93, 0.999, Strategy.THETA, qty=500)])
    assert [d.quantity for d in out] == [expected(cfg.theta.fair_cap, 0.93, True, cfg.risk.kelly_fraction_theta, 100_000)]


def test_convex_uses_the_sleeve_multiplier_and_never_exceeds_the_sleeve_sized_quantity():
    cfg = loose()
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.60})
    full = expected(0.60, 0.50, True, cfg.convex_sleeve.kelly_fraction, 100_000)  # 20 000
    assert full == 20_000
    out = risk.apply(ctx, [order("d", 0.50, 0.60, Strategy.CONVEX, qty=50_000)])
    assert [d.quantity for d in out] == [full]
    # signals sized it from the sleeve budget: that quantity is a ceiling
    out = risk.apply(ctx, [order("d", 0.50, 0.60, Strategy.CONVEX, qty=3_000)])
    assert [d.quantity for d in out] == [3_000]


def test_fade_and_follow_use_the_default_multiplier():
    cfg = loose()
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.50})
    for strategy in (Strategy.FADE, Strategy.FOLLOW):
        out = risk.apply(ctx, [order("d", 0.50, 0.60, strategy, qty=100)])
        assert [d.quantity for d in out] == [10_000], strategy


# -- caps still bind ----------------------------------------------------------------------------------
def test_caps_still_bind_on_the_kelly_quantity(cfg):
    # default config: per-race cap 3% of 100k = 3 000 SUSQies of exposure valued at fair
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.95})
    out = risk.apply(ctx, [order("d", 0.90, 0.95, Strategy.THETA, qty=500)])
    assert len(out) == 1
    q = out[0].quantity
    assert q > 500, "Kelly, not the 500-share placeholder, sets the size"
    assert q * 0.95 <= 3_000 + 1.0 and q * 0.95 > 2_950
    # max position shares
    ctx = ctx_for(loose(max_position_shares=1_000), [mv("d")], {"d": 0.60})
    assert [d.quantity for d in risk.apply(ctx, [order("d", 0.50, 0.60)])] == [1_000]
    # free cash above the reserve
    ctx = ctx_for(loose(liquid_reserve_frac=0.2), [mv("d")], {"d": 0.60}, cash=21_000.0)
    assert [d.quantity for d in risk.apply(ctx, [order("d", 0.50, 0.60)])] == [2_000]


# -- no edge -> dropped -------------------------------------------------------------------------------
def test_no_edge_or_sub_share_kelly_is_dropped():
    cfg = loose()
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.60})
    assert risk.apply(ctx, [order("d", 0.60, 0.60)]) == []  # fair == price
    assert risk.apply(ctx, [order("d", 0.60, 0.55)]) == []  # buying above fair
    assert risk.apply(ctx, [order("d", 0.50, 0.60, action="sell")]) == []  # selling below fair
    assert risk.apply(ctx, [order("d", 0.90, 0.85, Strategy.THETA, qty=500)]) == []
    # an edge too small for one share at this equity
    tiny = ctx_for(cfg, [mv("d")], {"d": 0.51}, equity=10.0)
    assert risk.apply(tiny, [order("d", 0.50, 0.51)]) == []
    # no equity, no stake
    broke = ctx_for(cfg, [mv("d")], {"d": 0.60}, equity=0.0)
    assert risk.apply(broke, [order("d", 0.50, 0.60)]) == []


# -- existing exposure reduces the increment ----------------------------------------------------------
def _resting(eid: str, qty: int, strategy: str, side: str = "yes", action: str = "buy", filled: int = 0) -> OrderIntent:
    return OrderIntent(
        id=f"{eid}-{strategy}", exchange_id=eid, side=side, action=action, price=0.50, quantity=qty,
        strategy=strategy, state=IntentState.ACKED.value, filled_qty=filled,
    )


def test_existing_position_and_other_resting_orders_reduce_the_increment():
    cfg = loose()
    target = 10_000  # value: fair 0.60 at 0.50 on 100k equity
    ctx = ctx_for(cfg, [mv("d", position=4_000)], {"d": 0.60})
    assert [d.quantity for d in risk.apply(ctx, [order("d", 0.50, 0.60)])] == [target - 4_000]
    # a position at or beyond the target: nothing more to buy
    ctx = ctx_for(cfg, [mv("d", position=12_000)], {"d": 0.60})
    assert risk.apply(ctx, [order("d", 0.50, 0.60)]) == []
    # a resting order of another strategy on the same side is exposure too
    ctx = ctx_for(cfg, [mv("d", position=4_000, open_orders=[_resting("d", 1_000, "convex")])], {"d": 0.60})
    assert [d.quantity for d in risk.apply(ctx, [order("d", 0.50, 0.60)])] == [target - 5_000]
    # ... but the order this desired order re-emits (same key) is replaced by it, not added to it
    ctx = ctx_for(cfg, [mv("d", position=4_000, open_orders=[_resting("d", 1_000, "value")])], {"d": 0.60})
    assert [d.quantity for d in risk.apply(ctx, [order("d", 0.50, 0.60)])] == [target - 4_000]


def test_two_strategies_on_the_same_bet_share_the_kelly_target():
    cfg = loose()
    ctx = ctx_for(cfg, [mv("d")], {"d": 0.60})
    out = risk.apply(ctx, [order("d", 0.50, 0.60, Strategy.VALUE), order("d", 0.50, 0.60, Strategy.CONVEX, qty=50_000)])
    total = sum(d.quantity for d in out)
    convex_target = expected(0.60, 0.50, True, cfg.convex_sleeve.kelly_fraction, 100_000)
    assert total == convex_target  # never the sum of both targets


# -- not Kelly-sized ----------------------------------------------------------------------------------
def test_arb_mm_election_and_orders_without_a_fair_keep_their_quantity():
    cfg = loose()
    markets = [mv("d"), mv("r", party="R"), mv("x", race="SEN-OH", state="OH")]
    ctx = ctx_for(cfg, markets, {"d": 0.60, "r": 0.40, "x": 0.50})
    legs = [
        order("d", 0.45, 0.60, Strategy.ARB, qty=1_234, priority=PRIORITY_ARB, take=True, group_id="g"),
        order("r", 0.50, 0.40, Strategy.ARB, qty=1_234, priority=PRIORITY_ARB, take=True, group_id="g"),
    ]
    out = risk.apply(ctx, legs)
    assert sorted(d.quantity for d in out) == [1_234, 1_234]
    assert [d.quantity for d in risk.apply(ctx, [order("x", 0.48, 0.50, Strategy.MM, qty=20, priority=PRIORITY_MM)])] == [20]
    assert [d.quantity for d in risk.apply(ctx, [order("x", 0.48, 0.99, Strategy.ELECTION, qty=700, priority=PRIORITY_ARB, take=True)])] == [700]
    # an order that does not state the fair it was priced on is not Kelly-sized (caps only)
    assert [d.quantity for d in risk.apply(ctx, [order("x", 0.50, None, qty=321)])] == [321]


@pytest.mark.parametrize("fair_, n", [(0.60, 10_000), (0.70, 20_000)])
def test_kelly_quantity_scales_with_edge(fair_, n):
    cfg = loose()
    ctx = ctx_for(cfg, [mv("d")], {"d": fair_})
    assert [d.quantity for d in risk.apply(ctx, [order("d", 0.50, fair_)])] == [n]
