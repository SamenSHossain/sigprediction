"""Status and daily report on an in-memory DB: heartbeats/staleness, account, open intents by
strategy, exposure by party/state, leaderboard, alerts; P&L ledger (realised + mark-to-market),
calibration from Signal rows, SIG-vs-reference gaps, rank history; empty-DB robustness; rendering."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sigcup.db import engine as dbengine
from sigcup.db.models import (
    AccountSnap,
    Fill,
    Heartbeat,
    IntentState,
    LeaderboardSnap,
    Market,
    OrderIntent,
    Position,
    Quote,
    Race,
    RefPrice,
    Signal,
)
from sigcup.db.repo import kv_set
from sigcup.reporter import report
from sigcup.reporter.loop import render_text
from sigcup.settings import Secrets

NOW = datetime(2026, 10, 10, 15, 0, tzinfo=UTC)


def sec_for(tmp_path) -> Secrets:
    return Secrets(_env_file=None, kill_file=str(tmp_path / "KILL"), data_dir=str(tmp_path / "data"))


def intent(iid: str, eid: str, strategy: str, state: str, price: float = 0.90, qty: int = 100, action: str = "buy", filled: int = 0, paper: bool = True) -> OrderIntent:
    return OrderIntent(
        id=iid, exchange_id=eid, side="yes", action=action, price=price, quantity=qty, strategy=strategy, state=state,
        created_at=NOW - timedelta(minutes=5), updated_at=NOW - timedelta(minutes=5), filled_qty=filled, paper=paper,
    )


def seed(s) -> None:
    s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX Senate"))
    s.add(Race(race_id="GOV-GA", chamber="governor", state="GA", label="GA Governor"))
    s.add(Market(market_id="1", exchange_id="tx-d", race_id="SEN-TX", party="D", title="TX D", status="open", kalshi_ticker="KX-TX-D"))
    s.add(Market(market_id="2", exchange_id="tx-r", race_id="SEN-TX", party="R", title="TX R", status="open"))
    s.add(Market(market_id="3", exchange_id="ga-d", race_id="GOV-GA", party="D", title="GA D", status="open", polymarket_condition_id="0xga"))
    s.add(Heartbeat(process="trader", ts=NOW - timedelta(seconds=20), status="paper", detail="cash=90000"))
    s.add(Heartbeat(process="feeds", ts=NOW - timedelta(minutes=30), status="ok", detail="x"))
    s.add(AccountSnap(ts=NOW - timedelta(hours=10), cash=95_000, positions_value=5_500, total=100_500, reserved=1_000))
    s.add(AccountSnap(ts=NOW - timedelta(minutes=1), cash=90_000, positions_value=12_000, total=102_000, reserved=2_500))
    s.add(intent("i-theta", "tx-d", "theta", IntentState.ACKED.value, price=0.90, qty=100))
    s.add(intent("i-value", "ga-d", "value", IntentState.ACKED.value, price=0.40, qty=50, action="sell", filled=10))
    s.add(intent("i-unc", "tx-r", "theta", IntentState.UNCERTAIN.value))
    s.add(intent("i-done", "tx-r", "theta", IntentState.FILLED.value))
    s.add(Position(exchange_id="tx-d", quantity=100, avg_price=0.60, mark=0.70, updated_at=NOW))
    s.add(Position(exchange_id="tx-r", quantity=-50, avg_price=0.75, mark=0.30, updated_at=NOW))
    s.add(Position(exchange_id="ga-d", quantity=0, avg_price=None, mark=None, updated_at=NOW))
    s.add(LeaderboardSnap(ts=NOW - timedelta(days=2), rank=12, our_total=100_200, top_totals=[120_000, 115_000, 110_000], member_count=100))
    s.add(LeaderboardSnap(ts=NOW - timedelta(hours=25), rank=10, our_total=100_900, top_totals=[121_000, 116_000, 111_000], member_count=100))
    s.add(LeaderboardSnap(ts=NOW - timedelta(minutes=10), rank=7, our_total=102_000, top_totals=[125_000, 118_000, 112_000], member_count=100))
    s.add(Quote(ts=NOW - timedelta(minutes=3), exchange_id="tx-d", bid=0.64, ask=0.66))
    s.add(Quote(ts=NOW - timedelta(minutes=3), exchange_id="ga-d", bid=0.40, ask=0.44))
    s.add(RefPrice(ts=NOW - timedelta(minutes=2), source="kalshi", key="KX-TX-D", yes_mid=0.70, liquidity=5000))
    s.add(RefPrice(ts=NOW - timedelta(minutes=2), source="polymarket", key="0xga", yes_bid=0.46, yes_ask=0.48, liquidity=100))
    s.add(Signal(ts=NOW - timedelta(hours=1), exchange_id="tx-d", strategy="value", fair=0.70, price=0.60, edge=0.10, size=100, meta={"ref": 0.72}))
    s.add(Signal(ts=NOW - timedelta(hours=1), exchange_id="ga-d", strategy="theta", fair=0.92, price=0.88, edge=None, size=50, meta={"components": {"ref": {"mid": 0.95}}}))
    s.add(Signal(ts=NOW - timedelta(days=3), exchange_id="ga-d", strategy="theta", fair=0.90, price=0.80, edge=0.1, size=50, meta=None))  # outside the window
    # fills: theta long tx-d (partially closed), value short tx-r via buy NO
    s.add(Fill(fill_id="f1", exchange_id="tx-d", ts=NOW - timedelta(hours=5), side="yes", action="buy", price=0.60, quantity=100, fee=0.0, strategy="theta", paper=True))
    s.add(Fill(fill_id="f2", exchange_id="tx-d", ts=NOW - timedelta(hours=3), side="yes", action="sell", price=0.70, quantity=40, fee=0.5, strategy="theta", paper=True))
    s.add(Fill(fill_id="f3", exchange_id="tx-r", ts=NOW - timedelta(hours=4), side="no", action="buy", price=0.25, quantity=50, fee=0.0, strategy="value", paper=True))
    s.add(Fill(fill_id="f4", exchange_id="ga-d", ts=NOW - timedelta(hours=2), side="yes", action="buy", price=0.40, quantity=10, fee=0.0, strategy=None, paper=False))


def test_status_sections(db, tmp_path):
    with dbengine.session() as s:
        seed(s)
    with dbengine.session() as s:
        d = report.status(s, now=NOW, sec=sec_for(tmp_path))
    hb = {h["process"]: h for h in d["heartbeats"]}
    assert hb["trader"]["stale"] is False and hb["trader"]["age_s"] == 20
    assert hb["feeds"]["stale"] is True
    assert hb["reporter"]["status"] == "missing" and hb["reporter"]["stale"] is True
    a = d["account"]
    assert a["total"] == 102_000 and a["reserved"] == 2_500 and a["free_cash"] == 87_500 and a["pnl"] == 2_000
    oi = d["open_intents"]
    assert oi["n_acked"] == 2 and oi["n_uncertain"] == 1 and oi["n_sent"] == 0
    assert oi["by_strategy"]["theta"]["n"] == 1 and abs(oi["by_strategy"]["theta"]["notional"] - 90.0) < 1e-9
    assert oi["by_strategy"]["value"]["shares"] == 40 and abs(oi["by_strategy"]["value"]["notional"] - 0.60 * 40) < 1e-9
    ex = d["exposure"]
    assert ex["n_positions"] == 2
    assert abs(ex["by_party"]["D"]["value"] - 70.0) < 1e-9 and abs(ex["by_party"]["R"]["value"] + 15.0) < 1e-9
    assert abs(ex["by_state"]["TX"]["value"] - 55.0) < 1e-9 and abs(ex["by_state"]["TX"]["gross"] - 85.0) < 1e-9
    assert abs(ex["net_dem_equivalent"] - 85.0) < 1e-9
    assert ex["positions"][0]["exchange_id"] == "tx-d" and abs(ex["positions"][0]["unrealised"] - 10.0) < 1e-9
    assert d["leaderboard"]["rank"] == 7 and d["leaderboard"]["member_count"] == 100
    assert d["kill"]["kill_file_present"] is False and d["kill"]["kv_flag"] is None
    assert any("feeds" in a_ for a_ in d["alerts"]) and any("UNCERTAIN" in a_ for a_ in d["alerts"])
    line = report.status_line(d)
    assert "rank=7/100" in line and "feeds=STALE" in line and "uncertain=1" in line
    text = render_text(report.render_status(d))
    assert "Heartbeats" in text and "theta" in text and "Alerts" in text


def test_status_kill_and_reserve_alerts(db, tmp_path):
    sec = sec_for(tmp_path)
    sec.kill_path.parent.mkdir(parents=True, exist_ok=True)
    sec.kill_path.touch()
    with dbengine.session() as s:
        s.add(AccountSnap(ts=NOW, cash=10_000, positions_value=85_000, total=95_000, reserved=5_000))
        kv_set(s, "killed", {"reason": "daily loss limit"})
    with dbengine.session() as s:
        d = report.status(s, now=NOW, sec=sec)
    assert d["kill"]["kill_file_present"] is True and d["kill"]["kv_flag"]["reason"] == "daily loss limit"
    assert any("KILL file" in a_ for a_ in d["alerts"]) and any("liquid reserve" in a_ for a_ in d["alerts"])
    assert any("equity -5.0%" in a_ for a_ in d["alerts"])
    assert "KILL" in render_text(report.render_status(d))


def test_pnl_ledger_realised_and_mark_to_market(db):
    with dbengine.session() as s:
        seed(s)
    with dbengine.session() as s:
        rows = report.pnl_by_strategy(s)
    by = {(r["mode"], r["strategy"]): r for r in rows}
    theta = by[("paper", "theta")]
    # bought 100 @0.60, sold 40 @0.70 -> realised 4.0; 60 left marked at 0.70 -> unrealised 6.0; fee 0.5
    assert abs(theta["realised"] - 4.0) < 1e-9 and abs(theta["unrealised"] - 6.0) < 1e-9
    assert abs(theta["fees"] - 0.5) < 1e-9 and abs(theta["total"] - 9.5) < 1e-9
    assert theta["n_fills"] == 2 and theta["shares"] == 140 and theta["open_shares"] == 60
    value = by[("paper", "value")]
    # buy NO 50 @0.25 == short 50 YES @0.75, marked 0.30 -> +22.5 unrealised
    assert abs(value["unrealised"] - 22.5) < 1e-9 and value["realised"] == 0
    unknown = by[("live", "unknown")]
    assert unknown["n_fills"] == 1 and abs(unknown["unrealised"] - 10 * (0.42 - 0.40)) < 1e-9  # no mark: quote mid
    assert abs(by[("paper", "all")]["total"] - (9.5 + 22.5)) < 1e-9
    assert [r["mode"] for r in rows] == ["live", "live", "paper", "paper", "paper"]


def test_ledger_flip_through_zero():
    led = report._Ledger()
    led.apply(100, 0.50, 0.0)  # long 100 @0.50
    led.apply(-150, 0.60, 0.0)  # sell 150: close 100 (+10), open short 50 @0.60
    assert abs(led.realised - 10.0) < 1e-9 and led.pos == -50 and led.avg == 0.60
    led.apply(50, 0.55, 0.0)  # cover the short @0.55 -> +2.5
    assert abs(led.realised - 12.5) < 1e-9 and led.pos == 0
    assert led.unrealised(0.9) == 0.0


def test_daily_sections(db, tmp_path):
    with dbengine.session() as s:
        seed(s)
    with dbengine.session() as s:
        d = report.daily(s, now=NOW)
    assert d["day"] == "2026-10-10"
    eq = d["equity"]
    assert eq["start_of_day"] == 100_500 and eq["now"] == 102_000 and eq["change_today"] == 1_500 and eq["change_total"] == 2_000
    cal = d["calibration"]
    assert cal["n_signals"] == 2
    assert abs(cal["by_strategy"]["value"]["mean_edge"] - 0.10) < 1e-9
    assert abs(cal["by_strategy"]["theta"]["mean_edge"] - 0.04) < 1e-9  # edge null -> fair - price
    assert cal["fair_vs_ref"]["n"] == 2 and abs(cal["fair_vs_ref"]["mean_gap"] - ((0.70 - 0.72) + (0.92 - 0.95)) / 2) < 1e-9
    assert "0.6-0.7" in cal["buckets"] and cal["buckets"]["0.6-0.7"]["n"] == 0 or "0.7-0.8" in cal["buckets"]
    sr = d["sig_vs_ref"]
    assert sr["n"] == 2
    gaps = {g["exchange_id"]: g for g in sr["largest"]}
    assert abs(gaps["tx-d"]["gap"] + 0.05) < 1e-9 and gaps["tx-d"]["source"] == "kalshi"
    assert abs(gaps["ga-d"]["gap"] + 0.05) < 1e-9 and gaps["ga-d"]["source"] == "polymarket"
    rh = d["rank_history"]
    assert rh["rank"] == 7 and rh["rank_change_24h"] == -3  # improved from 10 to 7
    assert [p["rank"] for p in rh["points"]] == [12, 10, 7]
    assert isinstance(d["clv"], list) and d["exposure"]["n_positions"] == 2
    text = render_text(report.render_daily(d))
    assert "Daily report 2026-10-10" in text and "P&L by strategy" in text and "Rank history" in text


def test_empty_db_status_and_daily(db, tmp_path):
    with dbengine.session() as s:
        d = report.status(s, now=NOW, sec=sec_for(tmp_path))
        day = report.daily(s, now=NOW)
    assert d["account"]["total"] is None and d["exposure"]["n_positions"] == 0 and d["leaderboard"]["rank"] is None
    assert all(h["stale"] for h in d["heartbeats"]) and len(d["alerts"]) == 3
    assert "cash=-" in report.status_line(d)
    assert day["pnl"] == [] and day["clv"] == [] and day["calibration"]["n_signals"] == 0 and day["rank_history"]["points"] == []
    assert "(flat)" in render_text(report.render_status(d))
    assert "no fills yet" in render_text(report.render_daily(day))
