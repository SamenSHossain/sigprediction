"""Regression tests for CLV pricing: the 24 h price is the book as of fill + 24 h (Quote rows are
written only when the top of book changes, so a quiet book has no row near the target) and is never
finalised before the target (F5); the closing line is fixed at the trading deadline even when a
market settles later (F13)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sigcup.db import engine as dbengine
from sigcup.db.models import CLV, Fill, Market, Quote, Race
from sigcup.reporter import clv

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
DEADLINE = datetime(2026, 11, 4, 17, 0, tzinfo=UTC)


def fill(fid: str, price: float, ts: datetime = T0, eid: str = "m1") -> Fill:
    return Fill(fill_id=fid, exchange_id=eid, ts=ts, side="yes", action="buy", price=price, quantity=100, fee=0.0, strategy="theta", paper=True)


def quote(ts: datetime, bid: float, ask: float, eid: str = "m1") -> Quote:
    return Quote(ts=ts, exchange_id=eid, bid=bid, ask=ask, source="rest")


def test_quiet_book_gets_the_as_of_price(db):
    with dbengine.session() as s:
        s.add(fill("f1", 0.94))
        s.add(quote(T0 - timedelta(hours=1), 0.94, 0.96))  # unchanged top until T0 + 30 h
        s.add(quote(T0 + timedelta(hours=30), 0.97, 0.99))
    for h in range(20, 40):  # hourly passes
        with dbengine.session() as s:
            clv.compute(s, T0 + timedelta(hours=h, minutes=5), deadline=DEADLINE)
    with dbengine.session() as s:
        row = s.get(CLV, "f1")
        assert row.price_24h is not None and abs(row.price_24h - 0.95) < 1e-9
        assert abs(row.clv_24h - 0.01) < 1e-9
        rows = clv.aggregate_by_strategy(s)
    assert {r["strategy"]: r for r in rows}["theta"]["n_24h"] == 1


def test_pass_before_the_target_does_not_finalise_early(db):
    with dbengine.session() as s:
        s.add(fill("f1", 0.60))
        s.add(quote(T0 + timedelta(hours=23), 0.59, 0.61))
    with dbengine.session() as s:
        st = clv.compute(s, T0 + timedelta(hours=23, minutes=5), deadline=DEADLINE)
    assert st.updated_24h == 0 and st.pending == 1
    with dbengine.session() as s:
        assert s.get(CLV, "f1").price_24h is None
        s.add(quote(T0 + timedelta(hours=24), 0.69, 0.71))  # the book moves exactly at the target
    with dbengine.session() as s:
        st = clv.compute(s, T0 + timedelta(hours=24, minutes=5), deadline=DEADLINE)
    assert st.updated_24h == 1
    with dbengine.session() as s:
        row = s.get(CLV, "f1")
        assert abs(row.price_24h - 0.70) < 1e-9 and abs(row.clv_24h - 0.10) < 1e-9


def test_first_quote_shortly_after_the_target_when_nothing_before(db):
    with dbengine.session() as s:
        s.add(fill("f1", 0.60))
    with dbengine.session() as s:
        st = clv.compute(s, T0 + timedelta(hours=24), deadline=DEADLINE)
    assert st.updated_24h == 0 and st.pending == 1  # no quote yet, the +1 h grace is still open
    with dbengine.session() as s:
        s.add(quote(T0 + timedelta(hours=24, minutes=10), 0.64, 0.66))
    with dbengine.session() as s:
        assert clv.compute(s, T0 + timedelta(hours=24, minutes=30), deadline=DEADLINE).updated_24h == 1
        assert abs(s.get(CLV, "f1").clv_24h - 0.05) < 1e-9


def test_close_is_capped_at_the_trading_deadline(db):
    settle = datetime(2026, 12, 8, 17, 0, tzinfo=UTC)  # GA runoff: settles after the competition ends
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-GA", chamber="senate", state="GA", label="GA Senate"))
        s.add(Market(market_id="1", exchange_id="ga-d", race_id="SEN-GA", party="D", title="D", status="open", settlement_date=settle))
        s.add(fill("f1", 0.60, eid="ga-d"))
        s.add(quote(DEADLINE - timedelta(minutes=5), 0.79, 0.81, eid="ga-d"))
        s.add(quote(DEADLINE + timedelta(days=3), 0.97, 0.99, eid="ga-d"))  # after the deadline: not our line
    with dbengine.session() as s:
        assert clv.market_close(s, "ga-d", DEADLINE) == DEADLINE
        st = clv.compute(s, DEADLINE + timedelta(minutes=1), deadline=DEADLINE)
    assert st.updated_close == 1
    with dbengine.session() as s:
        row = s.get(CLV, "f1")
        assert abs(row.price_close - 0.80) < 1e-9 and abs(row.clv_close - 0.20) < 1e-9
