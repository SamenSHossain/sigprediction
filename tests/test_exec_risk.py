"""risk.py: theta bid math, Kelly, per-race netting, per-state cap, swing stress, reserve, position cap,
groups, priority ordering, daily loss / kill switch."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from sigcup.db.models import Strategy
from sigcup.settings import Risk, StrategyConfig
from sigcup.sig.schemas import is_on_tick, round_tick
from sigcup.trader import risk
from sigcup.trader.probmath import shift_probability
from sigcup.trader.types import (
    PRIORITY_ARB,
    PRIORITY_THETA,
    PRIORITY_VALUE,
    DesiredOrder,
    FairValue,
    MarketView,
    RaceInfo,
    TraderContext,
)

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def mv(eid: str, race: str, party: str, chamber: str = "senate", state: str | None = "TX", bid=0.49, ask=0.51, position=0) -> MarketView:
    return MarketView(exchange_id=eid, race_id=race, party=party, chamber=chamber, state=state, bid=bid, ask=ask, position=position)


def ctx_for(cfg: StrategyConfig, markets: list[MarketView], fair: dict[str, float] | None = None, **kw) -> TraderContext:
    races: dict[str, RaceInfo] = {}
    for m in markets:
        r = races.setdefault(m.race_id, RaceInfo(race_id=m.race_id, chamber=m.chamber, state=m.state, label=m.race_id, exclusive=True, exhaustive=True))
        r.markets[m.party] = m.exchange_id
    fv = {eid: FairValue(exchange_id=eid, fair=f, sd=0.05) for eid, f in (fair or {}).items()}
    base = dict(cfg=cfg, now=NOW, markets={m.exchange_id: m for m in markets}, races=races, fair=fv, cash=100_000.0, reserved=0.0, equity=100_000.0, bankroll_start=100_000.0, hours_to_deadline=600.0)
    base.update(kw)
    return TraderContext(**base)


def order(eid: str, price=0.5, qty=100, side="yes", action="buy", **kw) -> DesiredOrder:
    kw.setdefault("strategy", Strategy.VALUE)
    return DesiredOrder(exchange_id=eid, side=side, action=action, price=price, quantity=qty, **kw)


def loose(**risk_kw) -> StrategyConfig:
    base = dict(per_race_cap_frac=10.0, per_state_cap_frac=10.0, max_swing_loss_frac=10.0, liquid_reserve_frac=0.0, liquid_reserve_frac_final72h=0.0, max_position_shares=10**9)
    base.update(risk_kw)
    return StrategyConfig(risk=Risk(**base))


# -- theta ------------------------------------------------------------------------------------
def test_theta_bid_math(cfg):
    # fair 0.95, ask 0.93: bound 0.95/1.055 = 0.9005 -> tick 0.900 (never above the bound)
    assert risk.theta_bid(0.95, 0.93, cfg) == pytest.approx(0.900)
    # ask below the bound: take at the ask
    assert risk.theta_bid(0.95, 0.90, cfg) == pytest.approx(0.900)
    assert risk.theta_bid(0.95, 0.885, cfg) == pytest.approx(0.885)
    # fair above the cap: capped at 0.985 -> 0.9336 -> rounds up to 0.935, so step down to 0.930
    b = risk.theta_bid(0.99, None, cfg)
    assert b == pytest.approx(0.930) and 0.985 / b - 1 >= cfg.theta.min_yield
    # below min_fair: no bid
    assert risk.theta_bid(0.84, 0.80, cfg) is None
    assert risk.theta_bid(None, 0.80, cfg) is None
    # every output is on tick and clears the yield
    for fair in (0.85, 0.9, 0.93, 0.97, 0.985, 0.999):
        for ask in (None, 0.99, 0.95, 0.9, 0.86):
            b = risk.theta_bid(fair, ask, cfg)
            if b is None:
                continue
            assert is_on_tick(b)
            assert b <= (ask if ask is not None else 1.0) + 1e-9
            assert min(fair, cfg.theta.fair_cap) / b - 1 >= cfg.theta.min_yield - 1e-9


def test_kelly_fraction_and_multipliers(cfg):
    assert risk.kelly_fraction(0.6, 0.5, long=True) == pytest.approx(0.2)
    assert risk.kelly_fraction(0.4, 0.5, long=False) == pytest.approx(0.2)
    assert risk.kelly_fraction(0.4, 0.5, long=True) == 0.0
    assert risk.kelly_fraction(0.6, 0.5, long=False) == 0.0
    assert risk.kelly_fraction(0.9, 1.0, long=True) == 0.0
    assert risk.strategy_multiplier(Strategy.THETA, cfg) == cfg.risk.kelly_fraction_theta
    assert risk.strategy_multiplier("arb", cfg) == cfg.risk.kelly_fraction_arb
    assert risk.strategy_multiplier(Strategy.VALUE, cfg) == cfg.risk.kelly_fraction_default
    # 0.2 * 0.25 * 100k / 0.5 = 10k shares
    assert risk.kelly_size(0.6, 0.5, True, Strategy.VALUE, 100_000, cfg) == 10_000


# -- per-race netting -------------------------------------------------------------------------
def test_per_race_netting_same_direction_adds_opposite_offsets():
    cfg = loose(per_race_cap_frac=0.03)  # 3 000 per race
    markets = [mv("d", "SEN-TX", "D"), mv("r", "SEN-TX", "R")]
    ctx = ctx_for(cfg, markets, fair={"d": 0.5, "r": 0.5})
    # YES on D + NO on R are the same bet: 2 000 + 2 000 at risk -> scaled to 3 000 together
    same = [order("d", 0.5, 4000), order("r", 0.5, 4000, side="no", action="buy")]
    out = risk.apply(ctx, same)
    qty = {d.exchange_id: d.quantity for d in out}
    assert qty["d"] == qty["r"]
    assert 2990 <= qty["d"] * 0.5 + qty["r"] * 0.5 <= 3000
    # YES on D + YES on R hedge each other: worst case loses one leg (2 000) -> untouched
    hedged = [order("d", 0.5, 4000), order("r", 0.5, 4000)]
    out = risk.apply(ctx, hedged)
    assert {d.exchange_id: d.quantity for d in out} == {"d": 4000, "r": 4000}
    legs = risk.combined_legs(ctx, [(o, 1.0) for o in hedged])
    assert risk.race_exposure(legs.values(), exhaustive=True) == pytest.approx(2000)
    # without API-verified exhaustiveness an independent could win: both YES legs lose
    assert risk.race_exposure(legs.values(), exhaustive=False) == pytest.approx(4000)
    legs = risk.combined_legs(ctx, [(o, 1.0) for o in same])
    assert risk.race_exposure(legs.values(), exhaustive=True) == pytest.approx(4000)
    # an unverified race scales the hedge too
    ctx.races["SEN-TX"].exhaustive = None
    out = risk.apply(ctx, hedged)
    assert {d.exchange_id: d.quantity for d in out} == {"d": 3000, "r": 3000}


def test_existing_position_counts_toward_race_cap():
    cfg = loose(per_race_cap_frac=0.03)
    markets = [mv("d", "SEN-TX", "D", position=5000), mv("r", "SEN-TX", "R")]
    ctx = ctx_for(cfg, markets, fair={"d": 0.5, "r": 0.5})
    out = risk.apply(ctx, [order("r", 0.5, 2000, side="no", action="buy")])
    assert out and out[0].quantity == 1000  # 2 500 held + 500 more = 3 000
    # a position-reducing sell is never scaled
    out = risk.apply(ctx, [order("d", 0.5, 3000, action="sell")])
    assert out[0].quantity == 3000


def test_per_state_cap_sums_races():
    cfg = loose(per_race_cap_frac=0.05, per_state_cap_frac=0.08)
    markets = [mv("s", "SEN-GA", "D", state="GA"), mv("g", "GOV-GA", "D", chamber="governor", state="GA"), mv("h", "HOUSE-GA-02", "D", chamber="house", state="GA")]
    ctx = ctx_for(cfg, markets, fair={"s": 0.5, "g": 0.5, "h": 0.5})
    out = risk.apply(ctx, [order("s", 0.5, 10000), order("g", 0.5, 10000), order("h", 0.5, 10000)])
    total = sum(d.quantity * 0.5 for d in out)
    assert 7900 <= total <= 8000


# -- swing stress -----------------------------------------------------------------------------
def test_swing_stress_scales_candidates_proportionally():
    cfg = loose(max_swing_loss_frac=0.25)
    markets = [mv(f"d{i}", f"SEN-{i}", "D", state=f"S{i}") for i in range(5)]
    ctx = ctx_for(cfg, markets, fair={m.exchange_id: 0.5 for m in markets})
    cands = [order(m.exchange_id, 0.5, 40_000) for m in markets]
    per_share = 0.5 - shift_probability(0.5, -cfg.risk.national_swing_pts, "senate", cfg)
    assert risk.portfolio_swing_loss(ctx, cands) == pytest.approx(200_000 * per_share, rel=1e-6)
    assert 200_000 * per_share > 25_000
    out = risk.apply(ctx, cands)
    assert len(out) == 5
    qtys = {d.quantity for d in out}
    assert max(qtys) - min(qtys) <= 1  # proportional
    loss = risk.portfolio_swing_loss(ctx, out)
    assert loss <= 25_000 + 1e-6 and loss > 24_900


def test_swing_stress_hedges_are_not_scaled_and_direction_is_adverse():
    cfg = loose(max_swing_loss_frac=0.25)
    markets = [mv("d", "SEN-TX", "D"), mv("r", "SEN-GA", "R", state="GA")]
    ctx = ctx_for(cfg, markets, fair={"d": 0.5, "r": 0.5})
    # long D and long R: a national swing hurts one side and helps the other
    both = [order("d", 0.5, 100_000), order("r", 0.5, 100_000)]
    assert risk.portfolio_swing_loss(ctx, both) == pytest.approx(0.0, abs=1e-6)
    assert risk.apply(ctx, both) and all(d.quantity == 100_000 for d in risk.apply(ctx, both))
    # a short-D position (NO on D) behaves like long R for the swing
    short_d = [order("d", 0.5, 100_000, side="no", action="buy")]
    assert risk.portfolio_swing_loss(ctx, short_d) == pytest.approx(risk.portfolio_swing_loss(ctx, [order("d", 0.5, 100_000)]))


# -- reserve, position cap, groups, priority, daily loss --------------------------------------
def test_reserve_blocks_and_shrinks_buys():
    cfg = loose(liquid_reserve_frac=0.2, liquid_reserve_frac_final72h=0.3)
    markets = [mv("d", "SEN-TX", "D")]
    ctx = ctx_for(cfg, markets, cash=30_000.0, reserved=0.0)
    out = risk.apply(ctx, [order("d", 0.5, 30_000)])  # 15k wanted, 10k free
    assert out[0].quantity == 20_000
    ctx2 = ctx_for(cfg, markets, cash=30_000.0, hours_to_deadline=48)  # final 72h: 30k reserve -> nothing free
    assert risk.apply(ctx2, [order("d", 0.5, 100)]) == []
    ctx3 = ctx_for(cfg, markets, cash=30_000.0, hours_to_deadline=48, election_night=True)  # may deploy 80% of it
    assert risk.apply(ctx3, [order("d", 0.5, 100_000)])[0].quantity == 48_000
    assert risk.reserve_fraction(cfg, 100) == 0.2 and risk.reserve_fraction(cfg, 72) == 0.3
    # existing reservation counts
    ctx4 = ctx_for(cfg, markets, cash=30_000.0, reserved=8_000.0)
    assert risk.apply(ctx4, [order("d", 0.5, 30_000)])[0].quantity == 4_000
    # sells of owned shares need no cash even with zero free cash
    ctx5 = ctx_for(cfg, [mv("d", "SEN-TX", "D", position=500)], cash=0.0)
    assert risk.apply(ctx5, [order("d", 0.6, 500, action="sell")])[0].quantity == 500


def test_max_position_shares_and_price_rounding():
    cfg = loose(max_position_shares=50_000)
    ctx = ctx_for(cfg, [mv("d", "SEN-TX", "D", position=49_990)])
    out = risk.apply(ctx, [order("d", 0.5012, 100)])
    assert out[0].quantity == 10 and out[0].price == round_tick(0.5012) == 0.5
    ctx = ctx_for(cfg, [mv("d", "SEN-TX", "D", position=50_000)])
    assert risk.apply(ctx, [order("d", 0.5, 1)]) == []


def test_groups_kept_or_dropped_whole_and_scaled_together():
    cfg = loose(per_race_cap_frac=0.03, liquid_reserve_frac=0.0)
    markets = [mv("d", "SEN-TX", "D"), mv("r", "SEN-TX", "R"), mv("x", "SEN-OH", "D", state="OH")]
    ctx = ctx_for(cfg, markets, fair={"d": 0.5, "r": 0.5, "x": 0.5}, cash=2_000.0)
    legs = [order("d", 0.45, 1000, strategy=Strategy.ARB, priority=PRIORITY_ARB, take=True, group_id="g"), order("r", 0.50, 1000, strategy=Strategy.ARB, priority=PRIORITY_ARB, take=True, group_id="g")]
    out = risk.apply(ctx, legs + [order("x", 0.5, 100)])
    assert {d.exchange_id for d in out} == {"d", "r", "x"}
    # group needs 950 cash, fits; now make cash too small for the group: both legs go, the single stays
    ctx2 = ctx_for(cfg, markets, fair={"d": 0.5, "r": 0.5, "x": 0.5}, cash=900.0)
    out = risk.apply(ctx2, legs + [order("x", 0.5, 100)])
    assert {d.exchange_id for d in out} == {"x"}
    # scaled by the race cap: both legs get the same factor
    legs_big = [order("d", 0.5, 8000, strategy=Strategy.ARB, priority=PRIORITY_ARB, take=True, group_id="g"), order("r", 0.5, 8000, side="no", strategy=Strategy.ARB, priority=PRIORITY_ARB, take=True, group_id="g")]
    out = risk.apply(ctx_for(cfg, markets, fair={"d": 0.5, "r": 0.5}), legs_big)
    assert len(out) == 2 and out[0].quantity == out[1].quantity < 8000


def test_priority_ordering_and_take_first():
    cfg = loose()
    ctx = ctx_for(cfg, [mv("a", "R1", "D"), mv("b", "R2", "D", state="OH"), mv("c", "R3", "D", state="GA")])
    out = risk.apply(ctx, [order("a", priority=PRIORITY_VALUE), order("b", priority=PRIORITY_ARB), order("c", priority=PRIORITY_ARB, take=True)])
    assert [d.exchange_id for d in out] == ["c", "b", "a"]
    # with scarce cash the highest priority is funded first
    ctx = ctx_for(cfg, [mv("a", "R1", "D"), mv("b", "R2", "D", state="OH")], cash=50.0)
    out = risk.apply(ctx, [order("a", 0.5, 100, priority=PRIORITY_VALUE), order("b", 0.5, 100, priority=PRIORITY_THETA)])
    assert [d.exchange_id for d in out] == ["b"]


def test_daily_loss_and_kill_switch(tmp_path):
    cfg = loose()
    ctx = ctx_for(cfg, [mv("d", "SEN-TX", "D", position=100)], daily_pnl=-5_000.0)
    assert risk.daily_loss_breached(ctx)
    out = risk.apply(ctx, [order("d", 0.5, 100), order("d", 0.5, 100, action="sell")])
    assert [d.action for d in out] == ["sell"], "only position-reducing orders survive a tripped daily limit"
    assert "daily loss" in risk.kill_switch_triggered(ctx, tmp_path / "KILL")
    ok = ctx_for(cfg, [], daily_pnl=-4_999.0)
    assert risk.kill_switch_triggered(ok, tmp_path / "KILL") is None
    (tmp_path / "KILL").write_text("")
    assert "kill file" in risk.kill_switch_triggered(ok, tmp_path / "KILL")


def test_closed_markets_and_past_deadline_are_dropped():
    cfg = loose()
    closed = mv("d", "SEN-TX", "D")
    closed.status = "settled"
    assert risk.apply(ctx_for(cfg, [closed]), [order("d")]) == []
    assert risk.apply(ctx_for(cfg, [mv("d", "SEN-TX", "D")], hours_to_deadline=0.0), [order("d")]) == []
