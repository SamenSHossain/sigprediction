"""Signal generation: theta by yield, value band/edge, hard-event gating, convex sleeve, mm, arb legs,
fade/follow, and the no-crossing invariant."""

from __future__ import annotations

from datetime import timedelta

import pytest

from sigcup.db import engine as dbengine
from sigcup.db.models import Quote, Strategy
from sigcup.sig.schemas import TICK, is_on_tick, round_tick
from sigcup.trader import signals as SG
from sigcup.trader.types import (
    PRIORITY_ARB,
    PRIORITY_MM,
    PRIORITY_THETA,
    PRIORITY_VALUE,
    ArbOpportunity,
    DesiredOrder,
)
from tests.test_model_support import NOW, ctx, event, fair, race, view


def _assert_never_crosses(orders, markets):
    for d in orders:
        assert is_on_tick(d.price), d
        if d.take:
            continue
        v = markets[d.exchange_id]
        yes_price = d.price if d.side == "yes" else 1 - d.price
        if d.is_long_yes:
            assert v.ask is None or yes_price <= v.ask + 1e-9, d
        else:
            assert v.bid is None or yes_price >= v.bid - 1e-9, d


# -- theta ------------------------------------------------------------------------------------------
def test_theta_price_formula(cfg):
    # fair above the cap: bound = 0.985 / 1.055 = 0.9336 -> 0.935 rounds up past the bound -> 0.93
    assert SG.theta_price(0.999, None, cfg) == pytest.approx(0.93)
    assert SG.theta_price(0.97, None, cfg) == pytest.approx(0.915)  # 0.97/1.055 = 0.9194 -> 0.92 up -> 0.915
    assert SG.theta_price(0.97, 0.90, cfg) == pytest.approx(0.90)  # ask below the bound binds
    assert SG.theta_price(0.80, None, cfg) is None  # below min_fair
    assert SG.theta_price(None, None, cfg) is None
    for f in (0.85, 0.9, 0.95, 0.985, 0.995):
        px = SG.theta_price(f, None, cfg)
        assert px is not None and SG.theta_yield(f, px, cfg) >= cfg.theta.min_yield - 1e-9


def test_theta_picks_the_better_yielding_leg(cfg):
    info = race("SEN-DE", "senate", "DE")
    # YES on D: min(ask 0.92, 0.95/1.055 = 0.9005) -> 0.90, yield 5.56%
    # NO on R: NO fair 0.95, NO ask = 1 - 0.12 = 0.88 binds -> 0.88, yield 7.95%  (better)
    views = [view(info, "D", bid=0.88, ask=0.92), view(info, "R", bid=0.12, ask=0.16)]
    fairs = [fair("SEN-DE-D", 0.95), fair("SEN-DE-R", 0.05)]
    c = ctx(cfg, [info], views, fairs)
    orders = [d for d in SG.generate(c, []) if d.strategy == Strategy.THETA]
    assert len(orders) == 1
    d = orders[0]
    assert d.exchange_id == "SEN-DE-R" and d.side == "no" and d.action == "buy"
    assert d.price == pytest.approx(0.88) and d.quantity == SG.THETA_QTY and d.priority == PRIORITY_THETA
    # lower the R bid: the NO ask rises to 0.94, the yield bound (0.90) binds on both -> tie on yield,
    # one order only
    views[1].bid = 0.06
    orders = [d for d in SG.generate(c, []) if d.strategy == Strategy.THETA]
    assert len(orders) == 1 and orders[0].price == pytest.approx(0.90)
    # raise the D ask bound: YES side now cheaper -> YES on D wins
    views[1].bid = 0.12
    views[0].ask = 0.86
    orders = [d for d in SG.generate(c, []) if d.strategy == Strategy.THETA]
    assert orders[0].exchange_id == "SEN-DE-D" and orders[0].side == "yes" and orders[0].price == pytest.approx(0.86)
    _assert_never_crosses(orders, c.markets)


def test_theta_respects_fair_cap_and_min_yield(cfg):
    info = race("SEN-WY", "senate", "WY")
    views = [view(info, "D", bid=0.02, ask=0.05), view(info, "R", bid=0.94, ask=0.97)]
    fairs = [fair("SEN-WY-D", 0.01), fair("SEN-WY-R", 0.99)]
    c = ctx(cfg, [info], views, fairs)
    orders = [d for d in SG.generate(c, []) if d.strategy == Strategy.THETA]
    assert len(orders) == 1
    d = orders[0]
    # both legs give the capped bound 0.93: never above min(fair, cap) / (1 + min_yield)
    assert d.price <= cfg.theta.fair_cap / (1 + cfg.theta.min_yield) + 1e-9
    assert SG.theta_yield(0.99, d.price, cfg) >= cfg.theta.min_yield
    # no theta when nothing reaches min_fair
    c2 = ctx(cfg, [info], views, [fair("SEN-WY-D", 0.3), fair("SEN-WY-R", 0.7)])
    assert [d for d in SG.generate(c2, []) if d.strategy == Strategy.THETA] == []


# -- value / convex ------------------------------------------------------------------------------
def test_value_band_and_edge(cfg):
    info = race("SEN-TX", "senate", "TX")
    views = [view(info, "D", bid=0.48, ask=0.52), view(info, "R", bid=0.48, ask=0.52)]
    # D: fair 0.60 vs mid 0.50 -> edge 0.10 > 0.05 -> buy at fair - 0.02 = 0.58, capped at the ask 0.52
    # R: fair 0.40 vs mid 0.50 -> sell at 0.42, floored at the bid 0.48
    c = ctx(cfg, [info], views, [fair("SEN-TX-D", 0.60), fair("SEN-TX-R", 0.40)])
    orders = [d for d in SG.generate(c, []) if d.strategy == Strategy.VALUE]
    assert {(d.exchange_id, d.action, d.price) for d in orders} == {("SEN-TX-D", "buy", 0.52), ("SEN-TX-R", "sell", 0.48)}
    assert all(d.quantity == SG.VALUE_QTY and d.priority == PRIORITY_VALUE and d.side == "yes" for d in orders)
    _assert_never_crosses(orders, c.markets)
    # wide book: the resting price is fair -/+ offset
    views = [view(info, "D", bid=0.40, ask=0.60), view(info, "R", bid=0.40, ask=0.60)]
    c = ctx(cfg, [info], views, [fair("SEN-TX-D", 0.60), fair("SEN-TX-R", 0.40)])
    orders = {d.exchange_id: d for d in SG.generate(c, []) if d.strategy == Strategy.VALUE}
    assert orders["SEN-TX-D"].price == pytest.approx(0.58) and orders["SEN-TX-R"].price == pytest.approx(0.42)
    # edge at / below min_edge -> nothing
    c = ctx(cfg, [info], views, [fair("SEN-TX-D", 0.55), fair("SEN-TX-R", 0.45)])
    assert [d for d in SG.generate(c, []) if d.strategy == Strategy.VALUE] == []
    # outside the band -> theta territory, not value
    c = ctx(cfg, [info], views, [fair("SEN-TX-D", 0.90), fair("SEN-TX-R", 0.10)])
    out = SG.generate(c, [])
    assert [d for d in out if d.strategy == Strategy.VALUE] == []
    assert [d for d in out if d.strategy == Strategy.THETA]


def test_value_skips_stale_fair_after_hard_event(cfg):
    info = race("SEN-TX", "senate", "TX")
    views = [view(info, "D", bid=0.40, ask=0.60)]
    stale = fair("SEN-TX-D", 0.60, ts=NOW - timedelta(hours=3))
    hard = event("SEN-TX", "hard", NOW - timedelta(hours=1), "candidate withdrew")
    c = ctx(cfg, [info], views, [stale], events=[hard])
    assert [d for d in SG.generate(c, []) if d.strategy in (Strategy.VALUE, Strategy.CONVEX)] == []
    # fair refreshed after the event -> trade
    fresh = fair("SEN-TX-D", 0.60, ts=NOW - timedelta(minutes=10))
    c = ctx(cfg, [info], views, [fresh], events=[hard])
    assert [d for d in SG.generate(c, []) if d.strategy == Strategy.VALUE]
    # a soft event, a low-confidence or an old hard event does not block
    for ev in (
        event("SEN-TX", "soft", NOW - timedelta(hours=1)),
        event("SEN-TX", "hard", NOW - timedelta(hours=1), confidence=0.2),
        event("SEN-TX", "hard", NOW - timedelta(hours=30)),
        event("SEN-OTHER", "hard", NOW - timedelta(hours=1)),
    ):
        c = ctx(cfg, [info], views, [stale], events=[ev])
        assert [d for d in SG.generate(c, []) if d.strategy == Strategy.VALUE], ev


def test_convex_sleeve(cfg):
    tossup = race("SEN-ME", "senate", "ME")
    ctl = race("CONTROL-HOUSE", "control", None, label="House control")
    views = [view(tossup, "D", bid=0.40, ask=0.50), view(ctl, "D", bid=0.70, ask=0.80)]
    fairs = [fair("SEN-ME-D", 0.55), fair("CONTROL-HOUSE-D", 0.88)]
    c = ctx(cfg, [tossup, ctl], views, fairs)
    out = SG.generate(c, [])
    convex = {d.exchange_id: d for d in out if d.strategy == Strategy.CONVEX}
    value = {d.exchange_id: d for d in out if d.strategy == Strategy.VALUE}
    assert set(convex) == {"SEN-ME-D", "CONTROL-HOUSE-D"}  # toss-up and control
    assert set(value) == {"SEN-ME-D"}  # control fair 0.88 is outside the value band
    assert convex["SEN-ME-D"].quantity == SG.CONVEX_QTY and convex["SEN-ME-D"].action == "buy"
    assert convex["CONTROL-HOUSE-D"].price == pytest.approx(0.80)  # capped at the ask
    _assert_never_crosses(out, c.markets)
    # sleeve closed -> no convex orders
    c = ctx(cfg, [tossup, ctl], views, fairs, sleeve_frac=0.0)
    assert [d for d in SG.generate(c, []) if d.strategy == Strategy.CONVEX] == []
    # small edge -> nothing
    c = ctx(cfg, [tossup], views[:1], [fair("SEN-ME-D", 0.47)])
    assert [d for d in SG.generate(c, []) if d.strategy == Strategy.CONVEX] == []


# -- mm ---------------------------------------------------------------------------------------------
def test_mm_only_when_enabled_and_wide(cfg):
    info = race("GOV-NV", "governor", "NV")
    views = [view(info, "D", bid=0.45, ask=0.55), view(info, "R", bid=0.48, ask=0.52)]
    c = ctx(cfg, [info], views, [fair("GOV-NV-D", 0.50)])
    assert [d for d in SG.generate(c, []) if d.strategy == Strategy.MM] == []
    cfg.mm.enabled = True
    out = [d for d in SG.generate(c, []) if d.strategy == Strategy.MM]
    assert {(d.exchange_id, d.action, d.price) for d in out} == {("GOV-NV-D", "buy", 0.48), ("GOV-NV-D", "sell", 0.52)}
    assert all(d.quantity == cfg.mm.size and d.priority == PRIORITY_MM for d in out)
    # a side through our fair value is dropped
    c = ctx(cfg, [info], views, [fair("GOV-NV-D", 0.47)])
    out = [d for d in SG.generate(c, []) if d.strategy == Strategy.MM]
    assert [(d.action, d.price) for d in out] == [("sell", 0.52)]


# -- arbs ---------------------------------------------------------------------------------------
def test_arb_legs_first_and_informational_excluded(cfg):
    info = race("SEN-DE", "senate", "DE")
    views = [view(info, "D", bid=0.95, ask=0.97), view(info, "R", bid=0.08, ask=0.12)]
    legs = [
        DesiredOrder("SEN-DE-D", "yes", "sell", 0.95, 40, Strategy.ARB, PRIORITY_ARB, group_id="g1", take=True),
        DesiredOrder("SEN-DE-R", "yes", "sell", 0.08, 40, Strategy.ARB, PRIORITY_ARB, group_id="g1", take=True),
    ]
    arb = ArbOpportunity(kind="race_no_sum", legs=legs, profit_per_set=0.03, max_sets=40, race_id="SEN-DE")
    info_leg = DesiredOrder("SEN-DE-D", "yes", "buy", 0.97, 0, Strategy.ARB, PRIORITY_ARB, take=False)
    info_opp = ArbOpportunity(kind="control", legs=[info_leg], profit_per_set=0.1, max_sets=0)
    c = ctx(cfg, [info], views, [fair("SEN-DE-D", 0.97), fair("SEN-DE-R", 0.03)])
    out = SG.generate(c, [arb, info_opp])
    assert out[:2] == legs
    assert all(d.quantity > 0 for d in out)
    assert [d for d in out if d.strategy == Strategy.THETA]  # theta still generated after the arb
    assert all(a.priority <= b.priority for a, b in zip(out, out[1:], strict=False))


def test_past_deadline_yields_nothing(cfg):
    info = race("SEN-DE", "senate", "DE")
    views = [view(info, "D", bid=0.95, ask=0.97)]
    c = ctx(cfg, [info], views, [fair("SEN-DE-D", 0.97)], hours_to_deadline=-1.0)
    assert SG.generate(c, []) == []


# -- fade / follow ---------------------------------------------------------------------------------
def test_fade_follow_gated_and_priced_from_pre_event_quote(db, cfg):
    info = race("SEN-TX", "senate", "TX")
    pre_ts = NOW - timedelta(hours=2)
    with dbengine.session() as s:
        s.add(Quote(ts=pre_ts - timedelta(minutes=1), exchange_id="SEN-TX-D", bid=0.49, ask=0.51, last=0.50, source="rest"))
        s.add(Quote(ts=NOW - timedelta(minutes=1), exchange_id="SEN-TX-D", bid=0.59, ask=0.61, last=0.60, source="rest"))
    views = [view(info, "D", bid=0.59, ask=0.61), view(info, "R", bid=0.39, ask=0.41)]
    soft = event("SEN-TX", "soft", pre_ts, "single poll")
    c = ctx(cfg, [info], views, [], events=[soft])
    assert SG.generate(c, []) == []  # disabled by default
    cfg.sentiment.enable_fade_follow = True
    out = SG.generate(c, [])
    assert len(out) == 1
    d = out[0]
    assert d.strategy == Strategy.FADE and d.exchange_id == "SEN-TX-D" and d.action == "sell"
    # pre-event mid 0.50 + 0.045 = 0.545 would cross: rests at the bid 0.59
    assert d.price == pytest.approx(0.59) and d.quantity == SG.EVENT_QTY
    _assert_never_crosses(out, c.markets)
    # hard event: follow toward pre + move / pass-through = 0.50 + 0.10 / 0.64 = 0.656 -> buy at 0.636 -> 0.61 (ask)
    hard = event("SEN-TX", "hard", pre_ts, "indictment")
    c = ctx(cfg, [info], views, [], events=[hard])
    out = SG.generate(c, [])
    assert len(out) == 1 and out[0].strategy == Strategy.FOLLOW and out[0].action == "buy"
    assert out[0].price == pytest.approx(0.61)
    assert out[0].fair == pytest.approx(min(0.995, 0.50 + 0.10 / SG.HARD_PASS_THROUGH))
    # a move below the threshold is ignored
    with dbengine.session() as s:
        s.add(Quote(ts=pre_ts - timedelta(seconds=30), exchange_id="SEN-TX-D", bid=0.58, ask=0.60, last=0.59, source="rest"))
    assert SG.generate(c, []) == []


def test_prices_on_tick_and_rounding_helper(cfg):
    info = race("SEN-TX", "senate", "TX")
    views = [view(info, "D", bid=0.401, ask=0.603)]
    c = ctx(cfg, [info], views, [fair("SEN-TX-D", 0.5123)])
    for d in SG.generate(c, []):
        assert is_on_tick(d.price)
        assert d.price == round_tick(d.price)
    assert round_tick(0.9336) == pytest.approx(0.935)
    assert TICK == 0.005
