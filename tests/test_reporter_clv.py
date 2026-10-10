"""CLV: sign convention for all four side/action combinations, the 24 h window, the closing price,
pending vs permanently unavailable quotes, idempotency and the per-strategy aggregate."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import CLV, Fill, Market, Quote, Race
from sigcup.reporter import clv

T0 = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
DEADLINE = datetime(2026, 11, 4, 17, 0, tzinfo=UTC)


def fill(fid: str, side: str, action: str, price: float, qty: int = 100, ts: datetime = T0, eid: str = "m1", strategy: str | None = "theta", paper: bool = True) -> Fill:
    return Fill(fill_id=fid, exchange_id=eid, ts=ts, side=side, action=action, price=price, quantity=qty, fee=0.0, strategy=strategy, paper=paper)


def quote(eid: str, ts: datetime, bid: float | None, ask: float | None, last: float | None = None) -> Quote:
    return Quote(ts=ts, exchange_id=eid, bid=bid, ask=ask, last=last, source="rest")


def test_clv_sign_convention():
    assert abs(clv.clv_value("yes", "buy", 0.60, 0.70) - 0.10) < 1e-12
    assert abs(clv.clv_value("yes", "sell", 0.60, 0.70) + 0.10) < 1e-12
    assert abs(clv.clv_value("no", "buy", 0.40, 0.70) - (-0.10)) < 1e-12  # buy NO @0.40 == sell YES @0.60
    assert abs(clv.clv_value("no", "sell", 0.40, 0.70) - 0.10) < 1e-12  # sell NO @0.40 == buy YES @0.60
    assert clv.clv_value("yes", "buy", 0.60, None) is None
    assert clv.fill_direction("YES", "Buy") == 1 and clv.fill_direction("no", "buy") == -1


def test_quote_mid_fallbacks():
    assert clv.quote_mid(quote("m", T0, 0.4, 0.5)) == 0.45
    assert clv.quote_mid(quote("m", T0, None, None, last=0.3)) == 0.3
    assert clv.quote_mid(quote("m", T0, 0.4, None)) == 0.4
    assert clv.quote_mid(quote("m", T0, None, None)) is None
    assert clv.quote_mid(None) is None


def test_compute_24h_uses_nearest_quote_in_window(db):
    with dbengine.session() as s:
        s.add_all([
            fill("f-buy-yes", "yes", "buy", 0.60),
            fill("f-sell-yes", "yes", "sell", 0.60, strategy="value"),
            fill("f-buy-no", "no", "buy", 0.40, strategy="value"),
            fill("f-sell-no", "no", "sell", 0.40),
            quote("m1", T0 + timedelta(hours=23, minutes=40), 0.68, 0.72),  # 20 min from target: chosen
            quote("m1", T0 + timedelta(hours=24, minutes=50), 0.78, 0.82),  # 50 min: not the nearest
            quote("m1", T0 + timedelta(hours=26), 0.90, 0.92),  # outside the window
        ])
    with dbengine.session() as s:
        stats = clv.compute(s, T0 + timedelta(hours=24, minutes=30), deadline=DEADLINE)
    assert stats.examined == 4 and stats.created == 4 and stats.updated_24h == 4 and stats.updated_close == 0
    with dbengine.session() as s:
        rows = {r.fill_id: r for r in s.execute(select(CLV)).scalars().all()}
    assert all(abs(rows[k].price_24h - 0.70) < 1e-9 for k in rows)
    assert abs(rows["f-buy-yes"].clv_24h - 0.10) < 1e-9
    assert abs(rows["f-sell-yes"].clv_24h + 0.10) < 1e-9
    assert abs(rows["f-buy-no"].clv_24h + 0.10) < 1e-9
    assert abs(rows["f-sell-no"].clv_24h - 0.10) < 1e-9
    assert all(r.price_close is None and r.clv_close is None for r in rows.values())


def test_compute_is_incremental_and_idempotent(db):
    with dbengine.session() as s:
        s.add(fill("f1", "yes", "buy", 0.60))
    # too early: the 24 h window has not opened -> row created, price null, pending
    with dbengine.session() as s:
        st = clv.compute(s, T0 + timedelta(hours=2), deadline=DEADLINE)
    assert st.created == 1 and st.updated_24h == 0 and st.pending == 1
    # window open, no quote yet -> still pending
    with dbengine.session() as s:
        st = clv.compute(s, T0 + timedelta(hours=24), deadline=DEADLINE)
    assert st.created == 0 and st.updated_24h == 0 and st.pending == 1
    # quote arrives inside the window -> filled in on the next pass
    with dbengine.session() as s:
        s.add(quote("m1", T0 + timedelta(hours=24, minutes=10), 0.64, 0.66))
    with dbengine.session() as s:
        st = clv.compute(s, T0 + timedelta(hours=24, minutes=30), deadline=DEADLINE)
    assert st.updated_24h == 1
    with dbengine.session() as s:
        st = clv.compute(s, T0 + timedelta(hours=30), deadline=DEADLINE)
        assert abs(s.execute(select(CLV)).scalars().one().clv_24h - 0.05) < 1e-9
    assert st.created == 0 and st.updated_24h == 0  # nothing recomputed


def test_missing_quote_after_window_is_left_null_and_not_pending(db):
    with dbengine.session() as s:
        s.add(fill("f1", "yes", "buy", 0.60))
    with dbengine.session() as s:
        st = clv.compute(s, T0 + timedelta(hours=27), deadline=DEADLINE)
    assert st.updated_24h == 0 and st.pending == 1  # still pending only because the close has not passed
    with dbengine.session() as s:
        row = s.execute(select(CLV)).scalars().one()
        assert row.price_24h is None and row.clv_24h is None


def test_closing_price_uses_settlement_date_then_deadline(db):
    settle = datetime(2026, 10, 20, 10, 0, tzinfo=UTC)
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX"))
        s.add(Market(market_id="1", exchange_id="m1", race_id="SEN-TX", party="D", title="D", status="open", settlement_date=settle))
        s.add(Market(market_id="2", exchange_id="m2", race_id="SEN-TX", party="R", title="R", status="open"))
        s.add(fill("f1", "yes", "buy", 0.60))
        s.add(fill("f2", "yes", "buy", 0.30, eid="m2", strategy="value"))
        s.add(quote("m1", settle - timedelta(hours=1), 0.94, 0.96))  # last before settlement
        s.add(quote("m1", settle + timedelta(hours=1), 0.99, 1.0))  # after: ignored
        s.add(quote("m2", DEADLINE - timedelta(minutes=5), 0.04, 0.06))
    with dbengine.session() as s:
        st = clv.compute(s, settle + timedelta(hours=2), deadline=DEADLINE)
    assert st.updated_close == 1
    with dbengine.session() as s:
        rows = {r.fill_id: r for r in s.execute(select(CLV)).scalars().all()}
    assert abs(rows["f1"].price_close - 0.95) < 1e-9 and abs(rows["f1"].clv_close - 0.35) < 1e-9
    assert rows["f2"].price_close is None  # m2 closes at the deadline, which has not passed
    with dbengine.session() as s:
        st = clv.compute(s, DEADLINE + timedelta(minutes=1), deadline=DEADLINE)
    assert st.updated_close == 1
    with dbengine.session() as s:
        r2 = s.get(CLV, "f2")
        assert abs(r2.price_close - 0.05) < 1e-9 and abs(r2.clv_close + 0.25) < 1e-9


def test_aggregate_by_strategy(db):
    with dbengine.session() as s:
        s.add_all([
            fill("a", "yes", "buy", 0.60, qty=100, strategy="theta"),
            fill("b", "yes", "buy", 0.80, qty=50, strategy="theta"),
            fill("c", "no", "buy", 0.40, qty=10, strategy="value"),
            fill("d", "yes", "buy", 0.50, qty=10, strategy=None, paper=False),
            fill("e", "yes", "buy", 0.50, qty=10, strategy="theta", ts=T0 + timedelta(days=3)),  # no CLV yet
            quote("m1", T0 + timedelta(hours=24), 0.69, 0.71),
        ])
    with dbengine.session() as s:
        clv.compute(s, T0 + timedelta(hours=25), deadline=DEADLINE)
        rows = clv.aggregate_by_strategy(s)
        paper_only = clv.aggregate_by_strategy(s, paper=True)
    by = {r["strategy"]: r for r in rows}
    assert set(by) == {"theta", "value", "unknown", "all"}
    theta = by["theta"]
    assert theta["n_fills"] == 3 and theta["n_24h"] == 2 and theta["shares"] == 160
    assert abs(theta["clv_24h_mean"] - 0.0) < 1e-9  # (+0.10 - 0.10) / 2
    assert abs(theta["clv_24h_weighted"] - (100 * 0.10 + 50 * -0.10) / 150) < 1e-9
    assert abs(theta["clv_24h_total"] - 5.0) < 1e-9
    assert theta["hit_24h"] == 0.5
    assert abs(by["value"]["clv_24h_mean"] + 0.10) < 1e-9  # buy NO @0.40 vs YES 0.70
    assert abs(by["unknown"]["clv_24h_mean"] - 0.20) < 1e-9
    assert by["all"]["n_fills"] == 5 and by["all"]["n_24h"] == 4
    assert {r["strategy"] for r in paper_only} == {"theta", "value", "all"}
    table = clv.render_table(rows)
    assert table.row_count == 4
    assert clv.render_table([]).row_count == 1


def test_aggregate_empty(db):
    with dbengine.session() as s:
        assert clv.aggregate_by_strategy(s) == []
        assert clv.compute(s, T0, deadline=DEADLINE).examined == 0
