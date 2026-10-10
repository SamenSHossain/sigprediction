"""Regression tests for lead-lag (F12): a reaction inside the first hour after the daily z-score is
stamped is found at lag 0, a numerically zero correlation is not a "best lag", and a race counts as
priced only when it has price data."""

from __future__ import annotations

from sigcup.db import engine as dbengine
from sigcup.db.models import Feature
from sigcup.reporter import leadlag
from sigcup.reporter.loop import render_text
from tests.test_reporter_leadlag import NOW, seed_lagged_series, seed_universe


def test_reaction_within_the_first_hour_is_lag_zero(db):
    with dbengine.session() as s:
        seed_universe(s)
        seed_lagged_series(s, lag_h=0, ref_lag_h=0)  # the price jumps at 00:30, 30 min after z is stamped
    with dbengine.session() as s:
        res = leadlag.compute(s, NOW, days=14)
    p = res["platforms"]["bluesky"]
    assert 0 in p["xcorr_sig"] and p["xcorr_sig"][0]["r"] > 0.5
    assert p["best_lag_sig"] == 0, {k: v["r"] for k, v in p["xcorr_sig"].items()}
    assert p["best_lag_ref"] == 0
    assert "0 h" in render_text(leadlag.render(res))


def test_tiny_positive_correlation_is_not_a_best_lag():
    assert leadlag.best_lag({1: {"r": 0.001, "n": 300}, 2: {"r": -0.2, "n": 300}}) is None
    assert leadlag.best_lag({1: {"r": 0.001, "n": 300}, 5: {"r": 0.3, "n": 300}}) == 5


def test_race_without_quotes_or_refs_is_not_priced(db):
    with dbengine.session() as s:
        seed_universe(s, with_ref=False)  # tx-d has neither quotes nor reference prices
        s.add(Feature(platform="youtube", race_id="SEN-TX", day="2026-10-09", n_posts=3, net_stance_dem=0.1, z_score=0.9, computed_at=NOW))
    with dbengine.session() as s:
        res = leadlag.compute(s, NOW, days=7)
    assert res["platforms"]["youtube"]["n_races_with_prices"] == 0
