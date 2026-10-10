"""Regression tests for the status / daily report: a LIVE=1 environment with a paper trader is an alert
(F1); the daily CLV table never pools paper and live fills (F6); exposure reads Position.avg_price in
the producer's convention, the NO price paid for a NO position (F7); calibration is populated without
Signal rows and never reads FairValue weights as a reference price (F8)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sigcup.db import engine as dbengine
from sigcup.db.models import Fill, Heartbeat, Market, Position, Quote, Race, RefPrice, Signal
from sigcup.paper import PaperAdapter
from sigcup.reporter import clv, report
from sigcup.reporter.loop import render_text
from sigcup.settings import Secrets
from sigcup.sig.schemas import OrderRequest

NOW = datetime(2026, 10, 10, 15, 0, tzinfo=UTC)


def sec_for(tmp_path, **kw) -> Secrets:
    return Secrets(_env_file=None, kill_file=str(tmp_path / "KILL"), data_dir=str(tmp_path / "data"), **kw)


# -- F1: LIVE=1 but the trader heartbeat says paper ---------------------------------------------
def test_live_env_with_paper_trader_is_an_alert(db, tmp_path):
    with dbengine.session() as s:
        s.add(Heartbeat(process="trader", ts=NOW - timedelta(seconds=20), status="paper", detail="x"))
    with dbengine.session() as s:
        live = report.status(s, now=NOW, sec=sec_for(tmp_path, live=True, sig_api_key="k"))
        paper = report.status(s, now=NOW, sec=sec_for(tmp_path, live=False))
    assert any("LIVE=1" in a and "paper" in a for a in live["alerts"]), live["alerts"]
    assert not any("LIVE=1" in a for a in paper["alerts"])


# -- F6: paper and live CLV are reported separately ---------------------------------------------
def test_daily_clv_never_pools_paper_and_live(db):
    t = NOW - timedelta(hours=30)
    with dbengine.session() as s:
        s.add(Fill(fill_id="fp", exchange_id="m1", ts=t, side="yes", action="buy", price=0.60, quantity=100, fee=0.0, strategy="theta", paper=True))
        s.add(Fill(fill_id="fl", exchange_id="m1", ts=t, side="yes", action="buy", price=0.66, quantity=10, fee=0.0, strategy="theta", paper=False))
        s.add(Quote(ts=t + timedelta(hours=24), exchange_id="m1", bid=0.69, ask=0.71))
    with dbengine.session() as s:
        clv.compute(s, NOW, deadline=datetime(2026, 11, 4, 17, 0, tzinfo=UTC))
        d = report.daily(s, now=NOW)
    rows = {(r["mode"], r["strategy"]): r for r in d["clv"]}
    assert set(rows) == {("live", "theta"), ("live", "all"), ("paper", "theta"), ("paper", "all")}
    assert rows[("live", "theta")]["n_fills"] == 1 and abs(rows[("live", "theta")]["clv_24h_mean"] - 0.04) < 1e-9
    assert rows[("paper", "theta")]["n_fills"] == 1 and abs(rows[("paper", "theta")]["clv_24h_mean"] - 0.10) < 1e-9
    assert rows[("live", "all")]["n_fills"] == 1
    text = render_text(report.render_daily(d))
    assert "Closing-line value" in text and "mode" in text


# -- F7: NO positions carry the NO price paid ---------------------------------------------------
def test_exposure_uses_the_no_price_convention(db):
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX"))
        s.add(Market(market_id="1", exchange_id="tx-r", race_id="SEN-TX", party="R", title="R", status="open"))
        s.add(Market(market_id="2", exchange_id="tx-d", race_id="SEN-TX", party="D", title="D", status="open"))
        s.add(Position(exchange_id="tx-r", quantity=-50, avg_price=0.26, mark=0.70, updated_at=NOW))  # bought 50 NO @0.26
        s.add(Position(exchange_id="tx-d", quantity=-20, avg_price=0.40, mark=None, updated_at=NOW))  # no mark, no quote
    with dbengine.session() as s:
        ex = report.exposure(s)
    rows = {r["exchange_id"]: r for r in ex["positions"]}
    # YES fell from 0.74 to 0.70: the NO holder gains 50 * 0.04
    assert abs(rows["tx-r"]["unrealised"] - 2.0) < 1e-9
    assert abs(rows["tx-r"]["at_risk"] - 15.0) < 1e-9  # 50 NO shares worth 0.30 each
    # the fallback mark is the YES-terms cost (1 - 0.40), not the NO price
    assert abs(rows["tx-d"]["mark"] - 0.60) < 1e-9 and abs(rows["tx-d"]["unrealised"]) < 1e-9
    assert abs(rows["tx-d"]["value"] + 12.0) < 1e-9
    assert abs(ex["at_risk"] - (15.0 + 20 * 0.40)) < 1e-9
    assert abs(ex["by_state"]["TX"]["at_risk"] - 23.0) < 1e-9


async def test_paper_position_round_trip_matches_the_paper_book(db, cfg):
    paper = PaperAdapter(cfg, use_db_quotes=False, clock=lambda: NOW)
    paper.set_top("tx-r", 0.74, 0.76)
    await paper.place_order(OrderRequest(exchange_id="tx-r", side="no", action="buy", price=0.26, quantity=50, idempotency_key="k1", tournament_id="paper"))
    paper.set_top("tx-r", 0.69, 0.71)
    (p,) = await paper.get_positions()
    acct = await paper.get_account()
    assert p.quantity == -50 and abs(p.avg_price - 0.26) < 1e-9
    with dbengine.session() as s:  # what OMS._write_positions stores
        s.add(Position(exchange_id=p.exchange_id, quantity=p.quantity, avg_price=p.avg_price, mark=p.mark, updated_at=NOW))
    with dbengine.session() as s:
        ex = report.exposure(s)
    (row,) = ex["positions"]
    assert abs(row["unrealised"] - (acct.positions_value - 50 * 0.26)) < 1e-9
    assert abs(ex["at_risk"] - acct.positions_value) < 1e-9


# -- F8: calibration without Signal rows --------------------------------------------------------
def test_calibration_from_current_fair_values_when_no_signals(db, cfg):
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX Senate"))
        s.add(Market(market_id="1", exchange_id="tx-d", race_id="SEN-TX", party="D", title="TX D", status="open", kalshi_ticker="KX-TX-D"))
        s.add(Quote(ts=NOW - timedelta(minutes=3), exchange_id="tx-d", bid=0.60, ask=0.62))
        s.add(RefPrice(ts=NOW - timedelta(minutes=2), source="kalshi", key="KX-TX-D", yes_mid=0.70, liquidity=5000))
    with dbengine.session() as s:
        d = report.daily(s, now=NOW, cfg=cfg)
    cal = d["calibration"]
    assert cal["source"] == "fair_values"
    assert cal["n_signals"] == 1 and cal["fair_vs_ref"]["n"] == 1
    assert cal["fair_vs_ref"]["mean_abs_gap"] is not None and cal["fair_vs_ref"]["mean_abs_gap"] < 0.2
    assert sum(b["n"] for b in cal["buckets"].values()) == 1
    assert "(no signals yet)" not in render_text(report.render_daily(d))


def test_calibration_ignores_fairvalue_weights(db):
    with dbengine.session() as s:
        s.add(Signal(ts=NOW - timedelta(hours=1), exchange_id="tx-d", strategy="value", fair=0.70, price=0.60, edge=0.10, size=10,
                     meta={"weights": {"ref": 0.6}, "components": {"ref": 0.72}}))
    with dbengine.session() as s:
        cal = report.calibration(s, NOW)
    assert cal["source"] == "signals"
    assert abs(cal["fair_vs_ref"]["mean_gap"] - (0.70 - 0.72)) < 1e-9
