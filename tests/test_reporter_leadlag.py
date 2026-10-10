"""Lead-lag: the pure numerics (a lagged copy peaks at the right lag, IC sign), hourly resampling,
and the DB-level compute on synthetic Feature / Quote / RefPrice rows plus sparse-data robustness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np

from sigcup.db import engine as dbengine
from sigcup.db.models import Feature, Market, Quote, Race, RefPrice
from sigcup.reporter import leadlag
from sigcup.reporter.loop import render_text

NOW = datetime(2026, 10, 10, 15, 30, tzinfo=UTC)


# -- numerics ---------------------------------------------------------------------------------
def test_cross_correlation_peaks_at_the_true_lag():
    rng = np.random.default_rng(7)
    x = rng.normal(size=400)
    y = np.zeros_like(x)
    y[3:] = x[:-3]  # y lags x by exactly 3 steps
    y += rng.normal(scale=0.1, size=400)
    xc = leadlag.cross_correlation(x, y, max_lag=10)
    assert set(xc) == set(range(1, 11))
    assert leadlag.best_lag(xc) == 3
    assert xc[3]["r"] > 0.9 and xc[3]["n"] == 397
    assert all(abs(xc[k]["r"]) < 0.3 for k in xc if k != 3)


def test_cross_correlation_handles_short_or_constant_input():
    assert leadlag.cross_correlation([1.0, 2.0], [1.0, 2.0], max_lag=3)[1]["r"] is None
    xc = leadlag.cross_correlation([1.0] * 20, list(range(20)), max_lag=2)
    assert xc[1]["r"] is None and xc[1]["n"] == 19
    assert leadlag.best_lag(xc) is None
    assert leadlag.best_lag({}) is None
    # a negative-only correlation is not a "best" positive lag
    assert leadlag.best_lag({1: {"r": -0.5, "n": 10}}) is None


def test_spearman_sign():
    z = [0.1, 0.5, -0.2, 1.0, 0.8, -0.7, 0.3, 0.0]
    dp = [0.5 * v + 0.01 for v in z]
    rho, p = leadlag.spearman(z, dp)
    assert rho > 0.99 and 0 <= p <= 1
    rho_neg, _ = leadlag.spearman(z, [-v for v in dp])
    assert rho_neg < -0.99
    assert leadlag.spearman([1.0, 2.0], [1.0, 2.0]) is None
    assert leadlag.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None


def test_resample_forward_fills_without_backfill():
    start = datetime(2026, 10, 1, tzinfo=UTC)
    pts = [(start + timedelta(hours=1, minutes=30), 0.5), (start + timedelta(hours=4), 0.7)]
    s = leadlag.resample(pts, start, start + timedelta(hours=6), timedelta(hours=1))
    assert s == [None, 0.5, 0.5, 0.5, 0.5, 0.7]
    assert leadlag.diff(s) == [0.0, 0.0, 0.0, 0.0, 0.2]
    assert leadlag.resample([], start, start + timedelta(hours=2), timedelta(hours=1)) == [None, None]


# -- database -----------------------------------------------------------------------------------
def seed_universe(s, with_ref: bool = True) -> None:
    s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX Senate"))
    s.add(Market(market_id="1", exchange_id="tx-d", race_id="SEN-TX", party="D", title="D", status="open", kalshi_ticker="KX-TX-D" if with_ref else None))
    s.add(Market(market_id="2", exchange_id="tx-r", race_id="SEN-TX", party="R", title="R", status="open"))
    s.add(Race(race_id="SEN-OH", chamber="senate", state="OH", label="OH Senate"))
    s.add(Market(market_id="3", exchange_id="oh-r", race_id="SEN-OH", party="R", title="R", status="open"))


def seed_lagged_series(s, *, lag_h: int, ref_lag_h: int, days: int = 10) -> list[float]:
    """z-score per day; the D leg's SIG mid jumps by 0.02*z(D) at D+1 00:00 + lag_h (ref at ref_lag_h)."""
    first = (NOW - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
    zs = [0.5, -1.0, 2.0, 0.2, -0.6, 1.5, -2.0, 0.8, 1.1, -0.3][:days]
    sig = 0.50
    ref = 0.52
    for i, z in enumerate(zs):
        day = first + timedelta(days=i)
        s.add(Feature(platform="bluesky", race_id="SEN-TX", day=day.strftime("%Y-%m-%d"), n_posts=20, net_stance_dem=z / 5, z_score=z, computed_at=day))
        # a quote every hour (constant) so the hourly grid is dense, plus the lagged jump
        nxt = day + timedelta(days=1)
        for h in range(24):
            ts = nxt + timedelta(hours=h, minutes=30)
            if h == lag_h:
                sig += 0.02 * z
            if h == ref_lag_h:
                ref += 0.02 * z
            if ts < NOW:
                s.add(Quote(ts=ts, exchange_id="tx-d", bid=sig - 0.01, ask=sig + 0.01, source="rest"))
                s.add(RefPrice(ts=ts, source="kalshi", key="KX-TX-D", market_id="1", yes_mid=ref, liquidity=1000))
    return zs


def test_compute_recovers_the_lag_and_a_positive_ic(db):
    with dbengine.session() as s:
        seed_universe(s)
        seed_lagged_series(s, lag_h=3, ref_lag_h=5)
    with dbengine.session() as s:
        res = leadlag.compute(s, NOW, days=14)
    assert res["n_platforms"] == 1
    p = res["platforms"]["bluesky"]
    assert p["n_features"] == 10 and p["n_races"] == 1 and p["n_races_with_prices"] == 1
    assert p["best_lag_sig"] == 3, {k: v["r"] for k, v in p["xcorr_sig"].items()}
    assert p["best_lag_ref"] == 5, {k: v["r"] for k, v in p["xcorr_ref"].items()}
    assert p["xcorr_sig"][3]["r"] > 0.5 and p["xcorr_sig"][3]["n"] > 100
    assert p["ic_sig"]["rho"] is not None and p["ic_sig"]["rho"] > 0.8 and p["ic_sig"]["n"] >= 8
    assert p["ic_ref"]["rho"] is not None and p["ic_ref"]["rho"] > 0.8
    text = render_text(leadlag.render(res))
    assert "bluesky" in text and "3 h" in text


def test_compute_uses_republican_leg_with_flipped_sign(db):
    with dbengine.session() as s:
        seed_universe(s)
        first = (NOW - timedelta(days=8)).replace(hour=0, minute=0, second=0, microsecond=0)
        price = 0.50
        zs = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5]
        for i, z in enumerate(zs):
            day = first + timedelta(days=i)
            s.add(Feature(platform="x", race_id="SEN-OH", day=day.strftime("%Y-%m-%d"), n_posts=5, net_stance_dem=z / 5, z_score=z, computed_at=day))
            nxt = day + timedelta(days=1)
            price -= 0.03 * z  # pro-Dem stance -> the R leg falls
            for h in (2, 14):
                ts = nxt + timedelta(hours=h)
                if ts < NOW:
                    s.add(Quote(ts=ts, exchange_id="oh-r", bid=price - 0.01, ask=price + 0.01))
    with dbengine.session() as s:
        res = leadlag.compute(s, NOW, days=14)
    p = res["platforms"]["x"]
    assert p["n_races_with_prices"] == 1
    assert p["ic_sig"]["rho"] is not None and p["ic_sig"]["rho"] > 0.8  # sign flipped: positive IC
    assert p["ic_ref"]["n"] == 0 and p["ic_ref"]["rho"] is None  # no reference prices: no crash
    assert p["xcorr_ref"] == {}


def test_sparse_data_never_crashes(db):
    with dbengine.session() as s:
        res = leadlag.compute(s, NOW)
    assert res["platforms"] == {} and res["n_platforms"] == 0
    assert "no feature rows" in render_text(leadlag.render(res))
    with dbengine.session() as s:  # features for a race without markets, and a market without quotes
        s.add(Feature(platform="bluesky", race_id="GOV-ZZ", day="2026-10-08", n_posts=3, net_stance_dem=0.1, z_score=0.4, computed_at=NOW))
        s.add(Feature(platform="bluesky", race_id="GOV-ZZ", day="2026-10-09", n_posts=3, net_stance_dem=0.1, z_score=None, computed_at=NOW))
        seed_universe(s, with_ref=False)
        s.add(Feature(platform="youtube", race_id="SEN-TX", day="2026-10-09", n_posts=3, net_stance_dem=0.1, z_score=0.9, computed_at=NOW))
    with dbengine.session() as s:
        res = leadlag.compute(s, NOW, days=7)
    b = res["platforms"]["bluesky"]
    assert b["n_features"] == 1 and b["n_races_with_prices"] == 0 and b["xcorr_sig"] == {} and b["ic_sig"]["n"] == 0
    y = res["platforms"]["youtube"]
    assert y["n_races_with_prices"] == 1 and y["xcorr_sig"] == {} and y["best_lag_sig"] is None
    render_text(leadlag.render(res))
    with dbengine.session() as s:
        assert leadlag.compute(s, NOW, days=0)["platforms"] == {}
