"""Regression tests for review findings on consistency.py: F6 (our own resting orders priced as the
market, self-trading arb legs) and F11 (race_yes_sum on an incomplete set of legs)."""

from __future__ import annotations

from sigcup.db.models import Strategy
from sigcup.trader import consistency as C
from tests.test_model_fix_signals import intent
from tests.test_model_support import book, ctx, race, view


def test_f6_own_theta_bid_is_not_a_dutch_book(cfg):
    info = race("SEN-DE", "senate", "DE")
    d = view(info, "D", bid=0.92, ask=0.97, bid_size=500, ask_size=100)
    d.open_orders = [intent("SEN-DE-D", "yes", "buy", 0.92, 500, Strategy.THETA)]  # the whole best D bid is ours
    r = view(info, "R", bid=0.10, ask=0.14, bid_size=20, ask_size=20)
    assert [o.kind for o in C.scan(ctx(cfg, [info], [d, r]))] == []
    # with the next outside level known the sum is computed without us: 0.85 + 0.10 < 1
    d.book = book("SEN-DE-D", bids=[(0.92, 500), (0.85, 40)], asks=[(0.97, 100)])
    assert C.scan(ctx(cfg, [info], [d, r])) == []
    # others share our level: the bids still sum above one, but selling into that level could match
    # our own order first (self-trade), so no group is sent
    d.bid_size = 800
    assert C.scan(ctx(cfg, [info], [d, r])) == []
    # our order deeper in the book does not matter: the outside best level is a genuine opportunity
    d.open_orders = [intent("SEN-DE-D", "yes", "buy", 0.80, 500, Strategy.THETA)]
    opps = C.scan(ctx(cfg, [info], [d, r]))
    assert [o.kind for o in opps] == ["race_no_sum"]
    assert opps[0].max_sets == 20


def test_f6_outside_view_removes_own_orders(cfg):
    info = race("SEN-DE", "senate", "DE")
    v = view(info, "D", bid=0.58, ask=0.62, bid_size=None, ask_size=50, book=book("SEN-DE-D", bids=[(0.58, 200), (0.40, 50)], asks=[(0.62, 50), (0.70, 10)]))
    v.open_orders = [
        intent("SEN-DE-D", "yes", "buy", 0.58, 200, Strategy.VALUE),
        intent("SEN-DE-D", "no", "buy", 0.38, 30, Strategy.THETA),  # buy NO @0.38 == sell YES @0.62: ask side
    ]
    o = C.outside_view(v)
    assert (o.bid, o.bid_size) == (0.40, 50)
    assert (o.ask, o.ask_size) == (0.62, 20)
    assert o.mid == (0.40 + 0.62) / 2
    assert C.displayed_size(o, "bid") == 50
    assert v.bid == 0.58 and v.bid_size is None  # the original view is untouched
    plain = view(info, "R", bid=0.1, ask=0.2)
    assert C.outside_view(plain) is plain


def test_f11_race_yes_sum_needs_every_leg(cfg):
    info = race("SEN-NE", "senate", "NE", parties=("D", "R", "I"))
    views = [
        view(info, "D", bid=0.40, ask=0.45, bid_size=50, ask_size=50),
        view(info, "R", bid=0.40, ask=0.45, bid_size=50, ask_size=50),
        view(info, "I", bid=0.05, ask=0.08, bid_size=50, ask_size=50, status="paused"),
    ]
    # D + R asks sum to 0.90 but the paused I leg can still win: buying D and R YES is not an arbitrage
    assert [o for o in C.scan(ctx(cfg, [info], views)) if o.kind == "race_yes_sum"] == []
    # a missing leg (no view at all) is the same
    assert [o for o in C.scan(ctx(cfg, [info], views[:2])) if o.kind == "race_yes_sum"] == []
    # with every leg open and quoted the sum 0.98 is a genuine yes-sum
    views[2].status = "open"
    assert [o.kind for o in C.scan(ctx(cfg, [info], views))] == ["race_yes_sum"]
