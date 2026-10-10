"""Regression tests for review findings on election_night.py: F9 (a stale reference price triggers a
take) and the self-trade case of F6 (a take that would hit our own resting order)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sigcup.db import engine as dbengine
from sigcup.db.models import Market, Race, RefPrice, Strategy
from sigcup.trader import election_night as EN
from sigcup.trader import fairvalue as FV
from tests.test_model_fix_signals import intent
from tests.test_model_support import ctx, fair, race, view

STALL = datetime(2026, 11, 3, 23, 10, tzinfo=UTC)
LATER = datetime(2026, 11, 4, 1, 30, tzinfo=UTC)


def _seed(s):
    s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", district=None, label="Texas Senate", incumbent_party=None, pvi=None, exclusive=True, exhaustive=True))
    for p in ("D", "R"):
        s.add(Market(market_id=f"m-{p}", exchange_id=f"SEN-TX-{p}", race_id="SEN-TX", party=p, title=p, status="open", kalshi_ticker=f"K-SEN-TX-{p}"))


def _take_ctx(cfg, now):
    info = race("SEN-TX", "senate", "TX")
    # the race flipped: SIG has repriced D down to 0.38/0.40
    v = view(info, "D", bid=0.38, ask=0.40, bid_size=100, ask_size=100)
    with dbengine.session() as s:
        fairs = FV.compute_all(s, cfg, {v.exchange_id: v}, now)
    return ctx(cfg, [info], [v], list(fairs.values()), now=now, election_night=True)


def test_f9_stale_reference_never_triggers_a_take(db, cfg):
    with dbengine.session() as s:
        _seed(s)
        s.add(RefPrice(ts=STALL, source="kalshi", key="K-SEN-TX-D", yes_bid=0.965, yes_ask=0.975, yes_mid=0.97, liquidity=50_000))
    # the collector stalled at 23:10; the loop still hands over the last recorded price
    c = _take_ctx(cfg, LATER)
    assert EN.taker_orders(c, {"SEN-TX-D": 0.97}) == []
    # a fresh confirming reference re-enables the rule
    with dbengine.session() as s:
        s.add(RefPrice(ts=LATER - timedelta(seconds=10), source="kalshi", key="K-SEN-TX-D", yes_bid=0.965, yes_ask=0.975, yes_mid=0.97, liquidity=50_000))
    c = _take_ctx(cfg, LATER)
    out = EN.taker_orders(c, {"SEN-TX-D": 0.97})
    assert [(d.exchange_id, d.side, d.price) for d in out] == [("SEN-TX-D", "yes", 0.40)]


def test_f9_reference_contradicted_by_fresh_data_is_ignored(cfg):
    info = race("SEN-TX", "senate", "TX")
    now = datetime(2026, 11, 4, 1, 30, tzinfo=UTC)
    v = view(info, "D", bid=0.38, ask=0.40, bid_size=100, ask_size=100)
    # one source stalled at 0.97, the fresh blend says 0.40
    c = ctx(cfg, [info], [v], [fair("SEN-TX-D", 0.40, ts=now - timedelta(seconds=5))], now=now, election_night=True)
    assert EN.taker_orders(c, {"SEN-TX-D": 0.97}) == []
    # no fair value at all: nothing confirms the reference
    c = ctx(cfg, [info], [v], [], now=now, election_night=True)
    assert EN.taker_orders(c, {"SEN-TX-D": 0.97}) == []


def test_f6_take_never_hits_our_own_resting_order(cfg):
    info = race("SEN-TX", "senate", "TX")
    now = datetime(2026, 11, 4, 1, 30, tzinfo=UTC)
    # D is dead on the reference; the best bid 0.92 is our own stale theta order
    v = view(info, "D", bid=0.92, ask=0.95, bid_size=500, ask_size=50)
    v.open_orders = [intent("SEN-TX-D", "yes", "buy", 0.92, 500, Strategy.THETA)]
    c = ctx(cfg, [info], [v], [fair("SEN-TX-D", 0.02, ts=now - timedelta(seconds=5))], now=now, election_night=True)
    assert EN.taker_orders(c, {"SEN-TX-D": 0.02}) == []
