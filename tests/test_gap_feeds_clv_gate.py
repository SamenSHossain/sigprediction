"""C3: the daily report carries a per-strategy CLV gate line: a strategy tag whose mean 24-h CLV is
negative over at least ``CLV_GATE_MIN_FILLS`` (50) scored fills is flagged "consider disabling". The gate
only reports; it never changes the configuration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sigcup.db import engine as dbengine
from sigcup.db.models import CLV, Fill
from sigcup.reporter import clv, report
from sigcup.reporter.loop import render_text
from sigcup.settings import StrategyConfig

NOW = datetime(2026, 10, 10, 15, 0, tzinfo=UTC)


def add_fills(s, strategy: str, n: int, clv_24h: float | None, *, paper: bool = False, prefix: str | None = None) -> None:
    prefix = prefix or f"{strategy}-{'p' if paper else 'l'}"
    t = NOW - timedelta(hours=30)
    for i in range(n):
        fid = f"{prefix}{i}"
        s.add(Fill(fill_id=fid, exchange_id="m1", ts=t, side="yes", action="buy", price=0.5, quantity=10, fee=0.0, strategy=strategy, paper=paper))
        s.add(CLV(fill_id=fid, price_24h=None if clv_24h is None else 0.5 + clv_24h, clv_24h=clv_24h, computed_at=NOW))


def test_gate_flags_negative_clv_only_with_enough_fills(db):
    with dbengine.session() as s:
        add_fills(s, "value", 50, -0.01)  # exactly 50 scored fills, negative -> flagged
        add_fills(s, "theta", 120, 0.02)  # positive -> ok
        add_fills(s, "convex", 49, -0.05)  # negative but too few -> not flagged
        add_fills(s, "mm", 30, -0.02, prefix="mm-scored")
        add_fills(s, "mm", 40, None, prefix="mm-pending")  # 70 fills, only 30 scored at 24 h -> not flagged
    with dbengine.session() as s:
        rows = clv.aggregate_by_strategy(s, paper=False)
    gate = {g["strategy"]: g for g in clv.strategy_gate(rows)}
    assert "all" not in gate
    assert gate["value"]["flag"] is True and gate["value"]["status"] == "consider disabling"
    assert gate["value"]["n_24h"] == 50 and abs(gate["value"]["clv_24h_mean"] + 0.01) < 1e-12
    assert gate["theta"]["flag"] is False and gate["theta"]["status"] == "ok"
    assert gate["convex"]["flag"] is False and gate["convex"]["status"].startswith("too few")
    assert gate["mm"]["flag"] is False and gate["mm"]["n_24h"] == 30
    # the threshold is a parameter
    small = {g["strategy"]: g for g in clv.strategy_gate(rows, min_fills=30)}
    assert small["convex"]["flag"] is True and small["mm"]["flag"] is True


def test_daily_report_carries_the_gate_per_mode_and_renders_it(db):
    cfg = StrategyConfig()
    with dbengine.session() as s:
        add_fills(s, "value", 60, -0.02)  # live, flagged
        add_fills(s, "value", 60, 0.03, paper=True)  # paper, ok
    with dbengine.session() as s:
        d = report.daily(s, now=NOW, cfg=cfg)
    gate = {(g["mode"], g["strategy"]): g for g in d["clv_gate"]}
    assert gate[("live", "value")]["flag"] is True
    assert gate[("paper", "value")]["flag"] is False
    assert d["clv_gate_min_fills"] == clv.CLV_GATE_MIN_FILLS == 50
    text = render_text(report.render_daily(d))
    assert "consider disabling" in text
    flagged = [line for line in text.splitlines() if "consider disabling" in line and "value" in line]
    assert flagged and "live" in flagged[0]
    # report only: the configuration object is untouched
    assert cfg == StrategyConfig()


def test_gate_is_empty_without_fills(db):
    with dbengine.session() as s:
        d = report.daily(s, now=NOW, cfg=StrategyConfig())
    assert d["clv_gate"] == []
    assert "consider disabling" not in render_text(report.render_daily(d))
