"""Dutch-book scanner: exclusivity gate, race_no_sum / race_yes_sum, depth bound, control bounds."""

from __future__ import annotations

from sigcup.db.models import Strategy
from sigcup.trader import consistency as C
from sigcup.trader import simulate as S
from sigcup.trader.types import PRIORITY_ARB
from tests.test_model_support import book, ctx, fair, race, view


def _two_leg(cfg, info, d_bid, d_ask, r_bid, r_ask, size=100, **kw):
    views = [
        view(info, "D", bid=d_bid, ask=d_ask, bid_size=size, ask_size=size),
        view(info, "R", bid=r_bid, ask=r_ask, bid_size=size, ask_size=size),
    ]
    return ctx(cfg, [info], views, **kw)


def test_exclusivity_gate_rejects_unverified_races(cfg):
    """RI fusion, NE independent and GA runoff style races have no API relationship evidence."""
    assert cfg.arb.require_api_exclusivity is True
    ri = race("SEN-RI", "senate", "RI", exclusive=None, exhaustive=None, settlement_note="fusion ticket counts for every party")
    ne = race("SEN-NE", "senate", "NE", parties=("D", "R", "I"), exclusive=True, exhaustive=None, settlement_note="independent candidate")
    ga = race("SEN-GA", "senate", "GA", exclusive=None, exhaustive=True, settlement_note="runoff settles in December")
    bad = race("GOV-RI", "governor", "RI", exclusive=False, exhaustive=True)
    for info in (ri, ne, ga, bad):
        assert not C.exclusivity_ok(info, cfg)
        views = [view(info, p, bid=0.6, ask=0.7, bid_size=50, ask_size=50) for p in info.markets]  # bids sum > 1
        assert C.scan(ctx(cfg, [info], views)) == []
    good = race("SEN-DE", "senate", "DE", exclusive=True, exhaustive=True)
    assert C.exclusivity_ok(good, cfg)
    # without the API requirement, unknown flags pass but explicit False still fails
    cfg.arb.require_api_exclusivity = False
    assert C.exclusivity_ok(ri, cfg) and not C.exclusivity_ok(bad, cfg)


def test_profitable_race_no_sum(cfg):
    info = race("SEN-DE", "senate", "DE")
    c = _two_leg(cfg, info, d_bid=0.95, d_ask=0.97, r_bid=0.08, r_ask=0.12, size=40)
    opps = C.scan(c)
    assert len(opps) == 1
    opp = opps[0]
    assert opp.kind == "race_no_sum" and opp.race_id == "SEN-DE"
    assert abs(opp.profit_per_set - 0.03) < 1e-9
    assert opp.max_sets == 40
    assert len(opp.legs) == 2
    gids = {leg.group_id for leg in opp.legs}
    assert len(gids) == 1 and next(iter(gids))
    for leg in opp.legs:
        assert leg.side == "yes" and leg.action == "sell" and leg.take is True
        assert leg.priority == PRIORITY_ARB and leg.strategy == Strategy.ARB
        assert leg.quantity == 40
    prices = {leg.exchange_id: leg.price for leg in opp.legs}
    assert prices == {"SEN-DE-D": 0.95, "SEN-DE-R": 0.08}


def test_profitable_race_yes_sum(cfg):
    info = race("GOV-GA", "governor", "GA")
    c = _two_leg(cfg, info, d_bid=0.40, d_ask=0.44, r_bid=0.46, r_ask=0.50, size=25)
    opps = C.scan(c)
    assert [o.kind for o in opps] == ["race_yes_sum"]
    opp = opps[0]
    assert abs(opp.profit_per_set - 0.06) < 1e-9 and opp.max_sets == 25
    assert all(leg.action == "buy" and leg.side == "yes" and leg.take for leg in opp.legs)
    assert {leg.price for leg in opp.legs} == {0.44, 0.50}


def test_min_profit_threshold(cfg):
    info = race("SEN-DE", "senate", "DE")
    c = _two_leg(cfg, info, d_bid=0.92, d_ask=0.97, r_bid=0.085, r_ask=0.12)  # sum 1.005 == min_profit: not enough
    assert C.scan(c) == []
    c = _two_leg(cfg, info, d_bid=0.92, d_ask=0.97, r_bid=0.09, r_ask=0.12)  # 1.01
    assert [o.kind for o in C.scan(c)] == ["race_no_sum"]


def test_depth_bound(cfg):
    info = race("SEN-DE", "senate", "DE")
    # the smaller displayed size bounds the set
    views = [
        view(info, "D", bid=0.95, ask=0.97, bid_size=500, ask_size=500),
        view(info, "R", bid=0.08, ask=0.12, bid_size=7, ask_size=7),
    ]
    opps = C.scan(ctx(cfg, [info], views))
    assert opps[0].max_sets == 7 and all(leg.quantity == 7 for leg in opps[0].legs)
    # unknown size on one leg -> no opportunity (an unfilled leg is not an arbitrage)
    views[1].bid_size = None
    assert C.scan(ctx(cfg, [info], views)) == []
    # ... unless the cached depth gives the best-level size
    views[1].book = book("SEN-DE-R", bids=[(0.08, 12), (0.07, 100)], asks=[(0.12, 3)])
    opps = C.scan(ctx(cfg, [info], views))
    assert opps[0].max_sets == 12
    views[1].ask_size = None
    assert C.displayed_size(views[1], "ask") == 3


def test_closed_leg_and_max_legs(cfg):
    info = race("SEN-DE", "senate", "DE")
    views = [
        view(info, "D", bid=0.95, ask=0.97, bid_size=50, ask_size=50, status="settled"),
        view(info, "R", bid=0.08, ask=0.12, bid_size=50, ask_size=50),
    ]
    assert C.scan(ctx(cfg, [info], views)) == []
    big = race("X", "senate", "XX", parties=tuple(f"P{i}" for i in range(12)))
    views = [view(big, p, bid=0.1, ask=0.2, bid_size=10, ask_size=10) for p in big.markets]
    assert C.scan(ctx(cfg, [big], views)) == []


def test_control_vs_seats_informational(cfg):
    # all 35 seats present; D needs 17 (51 - 34)
    states = sorted(S.SENATE_D_HELD_UP | {"AK", "IA", "KS", "ME", "MI", "OH", "TX", "NC", "AL", "AR", "ID", "KY", "LA", "MS", "MT", "NE", "OK", "SC", "SD", "TN", "WV", "WY", "FL"})
    seats = [race(f"SEN-{st}", "senate", st) for st in states]
    ctl = race("CONTROL-SENATE", "control", None, label="Senate control")
    views = []
    fairs = []
    for info in seats:
        views += [view(info, "D", bid=0.5, ask=0.6), view(info, "R", bid=0.4, ask=0.5)]
        fairs += [fair(info.markets["D"], 0.995), fair(info.markets["R"], 0.005)]  # every seat near-certain D
    views += [view(ctl, "D", bid=0.30, ask=0.35, bid_size=100, ask_size=100), view(ctl, "R", bid=0.65, ask=0.70, bid_size=100, ask_size=100)]
    c = ctx(cfg, [*seats, ctl], views, fairs)
    k, ids = S.control_requirements(c.races)["senate"]
    assert k == 17 and len(ids) == 35
    lo, hi = S.frechet_bounds([0.995] * 35, 17)
    assert lo > 0.35 + cfg.arb.min_profit
    opps = [o for o in C.scan(c) if o.kind == "control"]
    assert opps, "control D ask 0.35 is far below the Frechet lower bound of 35 near-certain seats"
    for o in opps:
        assert o.max_sets == 0 and o.race_id == "CONTROL-SENATE"
        assert all(not leg.take and leg.quantity == 0 for leg in o.legs)
    d_leg = next(o for o in opps if o.legs[0].exchange_id == "CONTROL-SENATE-D").legs[0]
    assert d_leg.action == "buy" and d_leg.price == 0.35
    r_leg = next(o for o in opps if o.legs[0].exchange_id == "CONTROL-SENATE-R").legs[0]
    assert r_leg.action == "sell" and r_leg.price == 0.65
    # a missing seat fair value makes the check incomplete -> no report
    c.fair.pop(seats[0].markets["D"])
    c.fair.pop(seats[0].markets["R"])
    c.markets[seats[0].markets["D"]].bid = None
    c.markets[seats[0].markets["D"]].ask = None
    c.markets[seats[0].markets["D"]].last = None
    c.markets[seats[0].markets["R"]].bid = None
    c.markets[seats[0].markets["R"]].ask = None
    assert [o for o in C.scan(c) if o.kind == "control"] == []


def test_control_check_requires_full_coverage(cfg):
    seats = [race(f"SEN-{st}", "senate", st) for st in ("AK", "IA")]
    ctl = race("CONTROL-SENATE", "control", None, label="Senate control")
    views = [view(ctl, "D", bid=0.60, ask=0.65), view(ctl, "R", bid=0.35, ask=0.40)]
    fairs = []
    for info in seats:
        views += [view(info, "D", bid=0.5, ask=0.6), view(info, "R", bid=0.4, ask=0.5)]
        fairs += [fair(info.markets["D"], 0.5), fair(info.markets["R"], 0.5)]
    assert [o for o in C.scan(ctx(cfg, [*seats, ctl], views, fairs)) if o.kind == "control"] == []
