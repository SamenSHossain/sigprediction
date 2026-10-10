"""C2: the daily report runs the correlated simulation over the current positions and reports the
settlement P&L distribution (mean, 5th/50th/95th percentile, P(loss > 10% of bankroll)); it degrades
gracefully with no positions or when fair values cannot be computed, and computes the fair values at
most once per report (shared with the calibration fallback)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from sigcup.db import engine as dbengine
from sigcup.db.models import Market, Position, Quote, Race, RefPrice
from sigcup.reporter import report
from sigcup.reporter.loop import render_text
from sigcup.settings import StrategyConfig
from sigcup.trader import fairvalue
from sigcup.trader import simulate as S
from sigcup.trader.types import FairValue

NOW = datetime(2026, 10, 10, 15, 0, tzinfo=UTC)


def seed_races(s) -> None:
    s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX Senate"))
    s.add(Race(race_id="SEN-GA", chamber="senate", state="GA", label="GA Senate"))
    for eid, rid, party in (("tx-d", "SEN-TX", "D"), ("tx-r", "SEN-TX", "R"), ("ga-d", "SEN-GA", "D"), ("ga-r", "SEN-GA", "R")):
        s.add(Market(market_id=eid, exchange_id=eid, race_id=rid, party=party, title=eid, status="open"))


def fixed_fairs(values: dict[str, float]):
    def compute_all(session, cfg, markets, now):
        return {eid: FairValue(exchange_id=eid, fair=p, sd=0.02) for eid, p in values.items()}

    return compute_all


def test_no_positions_degrades_gracefully_without_computing_fair_values(db):
    def boom():
        raise AssertionError("fair values must not be computed for a flat book")

    with dbengine.session() as s:
        seed_races(s)
    with dbengine.session() as s:
        pr = report.portfolio_risk(s, NOW, StrategyConfig(), fair_values=boom)
    assert pr["n_positions"] == 0 and pr["n_simulated"] == 0
    assert pr["mean"] is None and pr["p05"] is None and pr["p50"] is None and pr["p95"] is None and pr["p_loss"] is None
    assert pr["error"] is None
    with dbengine.session() as s:
        d = report.daily(s, now=NOW, cfg=StrategyConfig())
    assert d["portfolio_pnl"]["n_positions"] == 0
    text = render_text(report.render_daily(d))
    assert "Portfolio settlement P&L" in text and "(no positions)" in text
    json.dumps(d["portfolio_pnl"])


def test_distribution_over_correlated_positions(db, monkeypatch):
    monkeypatch.setattr(fairvalue, "compute_all", fixed_fairs({"tx-d": 0.70, "tx-r": 0.30, "ga-d": 0.50, "ga-r": 0.50}))
    cfg = StrategyConfig(bankroll_start=3000)  # loss threshold 300
    with dbengine.session() as s:
        seed_races(s)
        s.add(Position(exchange_id="tx-d", quantity=1000, avg_price=0.40, mark=0.68, updated_at=NOW))  # +600 / -400
        s.add(Position(exchange_id="ga-r", quantity=-500, avg_price=0.45, mark=0.52, updated_at=NOW))  # NO @0.45: +275 / -225
    with dbengine.session() as s:
        d = report.daily(s, now=NOW, cfg=cfg)
    pr = d["portfolio_pnl"]
    assert pr["n_positions"] == 2 and pr["n_simulated"] == 2 and pr["unsimulated"] == []
    assert pr["n_sims"] == cfg.simulation.n_sims
    assert pr["loss_threshold"] == pytest.approx(300.0)
    # mean = 1000 * (0.70 - 0.40) - 500 * (0.50 - 0.55) = 325 (Monte-Carlo error ~5)
    assert pr["mean"] == pytest.approx(325.0, abs=25.0)
    assert pr["p05"] == pytest.approx(-625.0) and pr["p50"] == pytest.approx(375.0) and pr["p95"] == pytest.approx(875.0)
    # a loss > 300 needs TX-D to lose AND GA-R to win: correlated (same national swing), so above 0.3 * 0.5
    assert 0.15 < pr["p_loss"] < 0.30
    # cost basis in YES terms; today's marks are 1000 * (0.68 - 0.40) - 500 * (0.52 - 0.55)
    assert pr["unrealised_at_mark"] == pytest.approx(295.0)
    json.dumps(pr)
    text = render_text(report.render_daily(d))
    assert "Portfolio settlement P&L" in text and "P(loss > 300)" in text


def test_matches_a_direct_simulation(db, monkeypatch):
    fairs = {"tx-d": 0.70, "tx-r": 0.30, "ga-d": 0.50, "ga-r": 0.50}
    monkeypatch.setattr(fairvalue, "compute_all", fixed_fairs(fairs))
    cfg = StrategyConfig()
    with dbengine.session() as s:
        seed_races(s)
        s.add(Position(exchange_id="tx-d", quantity=200, avg_price=0.55, mark=None, updated_at=NOW))
    with dbengine.session() as s:
        pr = report.portfolio_risk(s, NOW, cfg)
        meta = fairvalue.load_races(s)
    sim = S.simulate({"SEN-TX": {"D": 0.70, "R": 0.30}, "SEN-GA": {"D": 0.50, "R": 0.50}}, meta, cfg, n=cfg.simulation.n_sims, seed=cfg.simulation.seed)
    pnl = S.portfolio_pnl({"tx-d": (200, 0.55)}, sim)
    assert pr["mean"] == pytest.approx(float(pnl.mean()))
    assert pr["p05"] == pytest.approx(float(np.percentile(pnl, 5)))


def test_control_empty_leg_mark_fallback_and_unsimulated(db, monkeypatch):
    # NE Senate: an R and an independent; the D leg fields no candidate and cannot win
    monkeypatch.setattr(fairvalue, "compute_all", fixed_fairs({"tx-d": 0.70, "tx-r": 0.30, "ne-r": 0.60, "ne-i": 0.40, "ne-d": 0.02, "ctl-d": 0.4, "ctl-r": 0.6}))
    with dbengine.session() as s:
        seed_races(s)
        s.add(Race(race_id="SEN-NE", chamber="senate", state="NE", label="NE Senate", candidates={"R": "Ricketts", "I": "Osborn"}))
        for eid, party in (("ne-r", "R"), ("ne-i", "I"), ("ne-d", "D")):
            s.add(Market(market_id=eid, exchange_id=eid, race_id="SEN-NE", party=party, title=eid, status="open"))
        s.add(Race(race_id="CONTROL-SENATE", chamber="control", state=None, label="Senate control"))
        s.add(Market(market_id="ctl-d", exchange_id="ctl-d", race_id="CONTROL-SENATE", party="D", title="ctl-d", status="open"))
        s.add(Market(market_id="ctl-r", exchange_id="ctl-r", race_id="CONTROL-SENATE", party="R", title="ctl-r", status="open"))
        s.add(Position(exchange_id="ne-d", quantity=-100, avg_price=0.97, mark=0.03, updated_at=NOW))  # NO @0.97 on a leg that cannot win
        s.add(Position(exchange_id="ctl-r", quantity=10, avg_price=0.50, mark=0.6, updated_at=NOW))
        s.add(Position(exchange_id="ga-d", quantity=10, avg_price=0.50, mark=0.45, updated_at=NOW))  # no fair: priced at its mark
        s.add(Position(exchange_id="zz-unknown", quantity=7, avg_price=0.20, mark=0.25, updated_at=NOW))  # no race
    cfg = StrategyConfig()
    with dbengine.session() as s:
        pr = report.portfolio_risk(s, NOW, cfg)
    assert pr["n_positions"] == 4 and pr["n_simulated"] == 3
    assert [u["exchange_id"] for u in pr["unsimulated"]] == ["zz-unknown"]
    assert pr["mark_fallback"] == ["ga-d"]
    with dbengine.session() as s:
        meta = fairvalue.load_races(s)
    races = {"SEN-TX": {"D": 0.70, "R": 0.30}, "SEN-NE": {"R": 0.60, "I": 0.40}, "SEN-GA": {"D": 0.45}}
    sim = S.simulate(races, meta, cfg, n=cfg.simulation.n_sims, seed=cfg.simulation.seed)
    r_control = (sim.seats["senate"] < S.SENATE_MAJORITY).astype(float)
    expected = 100 * 0.03 + 10 * (r_control - 0.50) + 10 * (sim.samples["ga-d"].astype(float) - 0.50)
    assert pr["mean"] == pytest.approx(float(expected.mean()))
    assert pr["p95"] == pytest.approx(float(np.percentile(expected, 95)))


def test_fair_value_failure_is_reported_not_raised(db, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("model down")

    monkeypatch.setattr(fairvalue, "compute_all", broken)
    with dbengine.session() as s:
        seed_races(s)
        s.add(Position(exchange_id="tx-d", quantity=10, avg_price=0.40, mark=0.5, updated_at=NOW))
    with dbengine.session() as s:
        d = report.daily(s, now=NOW, cfg=StrategyConfig())
    pr = d["portfolio_pnl"]
    assert pr["n_positions"] == 1 and pr["mean"] is None and "model down" in pr["error"]
    assert "simulation unavailable" in render_text(report.render_daily(d))


def test_fair_values_are_computed_once_per_daily_report(db, monkeypatch):
    calls: list[int] = []
    inner = fixed_fairs({"tx-d": 0.70, "tx-r": 0.30})

    def counting(*a, **k):
        calls.append(1)
        return inner(*a, **k)

    monkeypatch.setattr(fairvalue, "compute_all", counting)
    with dbengine.session() as s:
        seed_races(s)
        s.add(Position(exchange_id="tx-d", quantity=10, avg_price=0.40, mark=0.5, updated_at=NOW))
    with dbengine.session() as s:
        d = report.daily(s, now=NOW, cfg=StrategyConfig())  # no Signal rows: calibration also needs fair values
    assert d["calibration"]["source"] == "fair_values" and d["portfolio_pnl"]["n_simulated"] == 1
    assert len(calls) == 1


def test_end_to_end_with_the_real_fair_value_model(db, cfg):
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX Senate"))
        s.add(Market(market_id="1", exchange_id="tx-d", race_id="SEN-TX", party="D", title="TX D", status="open", kalshi_ticker="KX-TX-D"))
        s.add(Quote(ts=NOW - timedelta(minutes=3), exchange_id="tx-d", bid=0.60, ask=0.62))
        s.add(RefPrice(ts=NOW - timedelta(minutes=2), source="kalshi", key="KX-TX-D", yes_mid=0.70, liquidity=5000))
        s.add(Position(exchange_id="tx-d", quantity=100, avg_price=0.60, mark=0.61, updated_at=NOW))
    with dbengine.session() as s:
        d = report.daily(s, now=NOW, cfg=cfg)
    pr = d["portfolio_pnl"]
    assert pr["error"] is None and pr["n_simulated"] == 1
    assert pr["p05"] == pytest.approx(-60.0) and pr["p95"] == pytest.approx(40.0)
    json.dumps(d, default=str)
