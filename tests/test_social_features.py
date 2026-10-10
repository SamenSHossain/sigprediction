"""Daily stance features: aggregation, z-score math on synthetic series, and the DB pipeline."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Feature, Stance
from sigcup.feeds.nlp import features
from sigcup.settings import StrategyConfig
from tests.test_social_helpers import NOW, make_post


def test_stance_value_sign_convention():
    assert features.stance_value("D", "support") == 1
    assert features.stance_value("D", "oppose") == -1
    assert features.stance_value("R", "support") == -1
    assert features.stance_value("R", "oppose") == 1
    assert features.stance_value("R", "neutral") == 0
    assert features.stance_value("I", "support") == 0
    assert features.stance_value("D", "unclear") == 0


def test_aggregate_counts_posts_once_and_clips():
    recs = [
        ("x", "SEN-GA", "2026-10-10", "p1", "D", "support"),
        ("x", "SEN-GA", "2026-10-10", "p2", "R", "oppose"),
        ("x", "SEN-GA", "2026-10-10", "p3", "D", "oppose"),
        ("x", "SEN-GA", "2026-10-10", "p4", "R", "neutral"),
        ("x", "SEN-GA", "2026-10-10", "p5", "D", "support"),  # p5 is about both candidates: clipped to +1
        ("x", "SEN-GA", "2026-10-10", "p5", "R", "oppose"),
        ("bluesky", "SEN-GA", "2026-10-10", "p6", "D", "oppose"),
    ]
    out = features.aggregate(recs)
    assert out[("x", "SEN-GA", "2026-10-10")] == (5, pytest.approx((1 + 1 - 1 + 0 + 1) / 5))
    assert out[("bluesky", "SEN-GA", "2026-10-10")] == (1, -1.0)


def test_zscores_math_and_minimum_days():
    series = {"2026-10-01": 0.1, "2026-10-02": 0.3, "2026-10-03": 0.2, "2026-10-04": 0.4, "2026-10-05": 0.55}
    z = features.zscores(series, baseline_days=14)
    assert z["2026-10-01"] is None and z["2026-10-02"] is None and z["2026-10-03"] is None
    # day 4 baseline = [0.1, 0.3, 0.2]: mean 0.2, std 0.0816 -> z = 0.2 / 0.0816
    assert z["2026-10-04"] == pytest.approx(0.2 / 0.08165, abs=1e-3)
    # day 5 baseline = [0.1, 0.3, 0.2, 0.4]: mean 0.25, std 0.1118 -> z = 0.3 / 0.1118
    assert z["2026-10-05"] == pytest.approx(0.3 / 0.11180, abs=1e-3)


def test_zscores_std_floor_and_window_cutoff():
    flat = {f"2026-10-{d:02d}": 0.0 for d in range(1, 15)}
    flat["2026-10-15"] = 0.5
    z = features.zscores(flat, baseline_days=14)
    assert z["2026-10-15"] == pytest.approx(0.5 / features.STD_FLOOR)
    # a 3-day baseline window only sees the three preceding days
    series = {"2026-10-01": 1.0, "2026-10-02": 1.0, "2026-10-03": 1.0, "2026-10-04": 0.0, "2026-10-05": 0.0, "2026-10-06": 0.0, "2026-10-07": 0.0}
    z3 = features.zscores(series, baseline_days=3)
    assert z3["2026-10-07"] == 0.0  # baseline 4,5,6 are all zero


def test_compute_pipeline():
    recs = [("x", "SEN-GA", f"2026-10-{d:02d}", f"p{d}", "D", "support" if d == 5 else "neutral") for d in range(1, 6)]
    out = features.compute(recs, baseline_days=14)
    assert out[("x", "SEN-GA", "2026-10-05")] == (1, 1.0, pytest.approx(1.0 / features.STD_FLOOR))
    assert out[("x", "SEN-GA", "2026-10-01")] == (1, 0.0, None)


async def test_run_once_excludes_bots_and_news_and_upserts(db):
    cfg = StrategyConfig(sentiment={"baseline_days": 3})
    with dbengine.session() as s:
        for back in range(5, 0, -1):
            day = NOW - timedelta(days=back)
            s.add(make_post(f"x:s{back}", f"support {back}", ["SEN-GA"], ts=day, classified=True))
            s.add(Stance(post_id=f"x:s{back}", race_id="SEN-GA", party="D", stance="support", confidence=0.9, is_news_or_poll=False, model="m", ts=day))
            s.add(make_post(f"x:o{back}", f"oppose {back}", ["SEN-GA"], ts=day, classified=True))
            s.add(Stance(post_id=f"x:o{back}", race_id="SEN-GA", party="D", stance="oppose", confidence=0.9, is_news_or_poll=False, model="m", ts=day))
        for i in range(3):
            s.add(make_post(f"x:t{i}", f"today {i}", ["SEN-GA"], ts=NOW, classified=True))
            s.add(Stance(post_id=f"x:t{i}", race_id="SEN-GA", party="D", stance="support", confidence=0.9, is_news_or_poll=False, model="m", ts=NOW))
        s.add(make_post("x:bot", "bot oppose", ["SEN-GA"], ts=NOW, bot_score=0.95, classified=True))
        s.add(Stance(post_id="x:bot", race_id="SEN-GA", party="D", stance="oppose", confidence=0.9, is_news_or_poll=False, model="m", ts=NOW))
        s.add(make_post("rss:news", "headline", ["SEN-GA"], platform="rss", ts=NOW, bot_score=None, classified=True))
        s.add(Stance(post_id="rss:news", race_id="SEN-GA", party="D", stance="oppose", confidence=0.9, is_news_or_poll=True, model="m", ts=NOW))

    n = await features.run_once(dbengine.session, cfg, None, now=NOW)
    assert n == 6
    with dbengine.session() as s:
        rows = {(f.platform, f.day): f for f in s.execute(select(Feature)).scalars()}
    today = rows[("x", NOW.strftime("%Y-%m-%d"))]
    assert (today.n_posts, today.net_stance_dem) == (3, 1.0)
    assert today.z_score == pytest.approx(1.0 / features.STD_FLOOR)
    assert today.race_id == "SEN-GA"
    assert ("rss", NOW.strftime("%Y-%m-%d")) not in rows
    first = rows[("x", (NOW - timedelta(days=5)).strftime("%Y-%m-%d"))]
    assert (first.n_posts, first.net_stance_dem, first.z_score) == (2, 0.0, None)
    third = rows[("x", (NOW - timedelta(days=2)).strftime("%Y-%m-%d"))]
    assert third.z_score == 0.0

    # re-running upserts rather than duplicating
    assert await features.run_once(dbengine.session, cfg, None, now=NOW) == 6
    with dbengine.session() as s:
        assert len(s.execute(select(Feature)).scalars().all()) == 6
