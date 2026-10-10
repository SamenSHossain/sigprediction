"""Regression tests for the event detector and daily features: small partial-day samples do not spike
(F2), top posts are filtered by race before the limit (F6), the GDELT 3x rule ignores days the
collector never covered (F7), and copy-paste campaigns cannot move the daily net (F13)."""

from __future__ import annotations

from datetime import timedelta

import httpx
import respx
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Event, Feature, Stance
from sigcup.db.repo import kv_get, kv_set
from sigcup.feeds import events, news_gdelt
from sigcup.feeds.nlp import features
from sigcup.settings import StrategyConfig
from tests.test_social_helpers import NOW, load_json, make_gazetteer, make_post

DAY = NOW.strftime("%Y-%m-%d")


def day(back: int) -> str:
    return (NOW - timedelta(days=back)).strftime("%Y-%m-%d")


def feat(platform: str, race_id: str, z: float | None, n: int, net: float = 1.0) -> Feature:
    return Feature(platform=platform, race_id=race_id, day=DAY, n_posts=n, net_stance_dem=net, z_score=z, computed_at=NOW)


def upsert(platform: str, race_id: str, z: float, n: int) -> None:
    with dbengine.session() as s:
        row = s.execute(select(Feature).where(Feature.platform == platform, Feature.race_id == race_id, Feature.day == DAY)).scalars().first()
        if row is None:
            s.add(feat(platform, race_id, z, n))
        else:
            row.z_score, row.n_posts = z, n


# -- F2 ---------------------------------------------------------------------------------------------


def test_spike_rule_ignores_tiny_samples():
    feats = [feat("x", "SEN-GA", 18.0, n=1), feat("bluesky", "SEN-GA", 18.0, n=2)]
    assert events.detect_social_spikes(feats, spike_z=2.5, min_platforms=2) == {}


async def test_an_early_utc_blip_neither_fires_nor_blocks_the_real_spike(db, cfg):
    calls: list[str] = []

    async def classifier(race_context, top_posts, trigger):
        calls.append(trigger)
        return ("hard" if len(calls) > 1 else "soft"), f"event {len(calls)}", 0.8

    early = NOW.replace(hour=0, minute=5)
    upsert("x", "SEN-GA", 18.0, 1)
    upsert("bluesky", "SEN-GA", 18.0, 1)
    assert await events.run_once(dbengine.session, cfg, None, now=early, classifier=classifier, gazetteer=make_gazetteer()) == 0
    assert calls == []

    upsert("x", "SEN-GA", 3.0, 60)
    upsert("bluesky", "SEN-GA", 2.8, 45)
    assert await events.run_once(dbengine.session, cfg, None, now=NOW, classifier=classifier, gazetteer=make_gazetteer()) == 1

    # a materially stronger spike later the same day is recorded too; an unchanged one is not
    assert await events.run_once(dbengine.session, cfg, None, now=NOW + timedelta(minutes=15), classifier=classifier, gazetteer=make_gazetteer()) == 0
    upsert("x", "SEN-GA", 7.5, 300)
    upsert("bluesky", "SEN-GA", 6.0, 200)
    upsert("youtube", "SEN-GA", 5.0, 80)
    assert await events.run_once(dbengine.session, cfg, None, now=NOW + timedelta(hours=3), classifier=classifier, gazetteer=make_gazetteer()) == 1
    with dbengine.session() as s:
        evs = s.execute(select(Event).order_by(Event.id)).scalars().all()
    assert [(e.race_id, e.source, e.kind) for e in evs] == [("SEN-GA", "social_spike", "soft"), ("SEN-GA", "social_spike", "hard")]
    assert len(calls) == 2


# -- F6 ---------------------------------------------------------------------------------------------


def test_top_posts_are_filtered_by_race_before_the_limit(db):
    day_start = NOW.replace(hour=0, minute=0)
    with dbengine.session() as s:
        for i in range(450):
            s.add(make_post(f"gdelt:tx{i}", f"Texas article {i}", ["SEN-TX"], platform="gdelt", ts=NOW - timedelta(minutes=i % 600), bot_score=0.0))
        for i in range(5):
            s.add(make_post(f"bluesky:ga{i}", f"Ossoff post {i}", ["SEN-GA"], platform="bluesky", ts=NOW - timedelta(hours=1), bot_score=0.05))
        s.add(make_post("x:gagov", "Burt Jones post", ["GOV-GA"], ts=NOW, bot_score=0.0))
    with dbengine.session() as s:
        top = events._top_posts(s, "SEN-GA", day_start)
    assert sorted(top) == sorted(f"[bluesky] Ossoff post {i}" for i in range(5))


def test_top_posts_mix_platforms(db):
    day_start = NOW.replace(hour=0, minute=0)
    with dbengine.session() as s:
        for i in range(20):
            s.add(make_post(f"gdelt:ga{i}", f"Georgia article {i}", ["SEN-GA"], platform="gdelt", ts=NOW - timedelta(minutes=i), bot_score=0.0))
        for i in range(3):
            s.add(make_post(f"x:ga{i}", f"Ossoff tweet {i}", ["SEN-GA"], ts=NOW - timedelta(hours=2), bot_score=0.1))
    with dbengine.session() as s:
        top = events._top_posts(s, "SEN-GA", day_start)
    assert len(top) == events.TOP_POSTS
    assert sum(t.startswith("[x]") for t in top) == 3
    assert top[0] == "[gdelt] Georgia article 0"


# -- F7 ---------------------------------------------------------------------------------------------


def test_cold_start_partial_first_day_is_not_a_baseline():
    assert events.detect_gdelt_spikes({("SEN-GA", day(1)): 15, ("SEN-GA", DAY): 15}, DAY, 14) == {}


def test_steady_coverage_after_a_cold_start_never_spikes():
    # collector deployed 5 days ago with a 1d timespan: 15 articles on the partial first day, then 30/day
    for today_back in range(4, -1, -1):
        counts = {("SEN-GA", day(5)): 15}
        for b in range(4, today_back - 1, -1):
            counts[("SEN-GA", day(b))] = 30
        assert events.detect_gdelt_spikes(counts, day(today_back), 14) == {}, today_back


def test_covered_since_counts_quiet_covered_days_as_zero():
    counts = {("SEN-GA", DAY): 9}
    # the collector has covered SEN-GA for 6 full days with no articles: 9 today is a spike
    assert events.detect_gdelt_spikes(counts, DAY, 14, covered_since={"SEN-GA": day(6)}) == {"SEN-GA": (9, 0.0)}
    # covered for only 2 full days: not enough baseline yet
    assert events.detect_gdelt_spikes(counts, DAY, 14, covered_since={"SEN-GA": day(2)}) == {}
    # articles from before coverage started (a partial backfill day) do not enter the mean
    counts2 = {("SEN-GA", day(5)): 2, ("SEN-GA", day(3)): 3, ("SEN-GA", day(2)): 3, ("SEN-GA", day(1)): 3, ("SEN-GA", DAY): 10}
    assert events.detect_gdelt_spikes(counts2, DAY, 14, covered_since={"SEN-GA": day(3)}) == {"SEN-GA": (10, 3.0)}


@respx.mock
async def test_gdelt_collector_records_coverage_on_success_only(db, cfg):
    async def no_sleep(_: float) -> None:
        return None

    def side(request: httpx.Request) -> httpx.Response:
        if "Ossoff" in request.url.params["query"]:
            return httpx.Response(429)
        return httpx.Response(200, json=load_json("gdelt_artlist.json"))

    respx.get(host="api.gdeltproject.org").mock(side_effect=side)
    await news_gdelt.run_once(dbengine.session, cfg, None, gazetteer=make_gazetteer(), sleep=no_sleep, now=NOW, max_requests=10)
    with dbengine.session() as s:
        covered = kv_get(s, news_gdelt.KV_COVERED_SINCE)
    assert "SEN-GA" not in covered  # its query failed
    assert covered["SEN-TX"] == DAY  # a 1d timespan at 12:00 reaches back past midnight: today is fully covered
    # coverage never moves later
    await news_gdelt.run_once(dbengine.session, cfg, None, gazetteer=make_gazetteer(), sleep=no_sleep, now=NOW + timedelta(days=2), max_requests=10)
    with dbengine.session() as s:
        assert kv_get(s, news_gdelt.KV_COVERED_SINCE)["SEN-TX"] == DAY


async def test_run_once_gdelt_uses_recorded_coverage(db):
    cfg = StrategyConfig(sentiment={"baseline_days": 14})
    with dbengine.session() as s:
        kv_set(s, news_gdelt.KV_COVERED_SINCE, {"SEN-TX": day(1)})
        for i in range(15):
            s.add(make_post(f"gdelt:y{i}", f"Texas Senate y {i}", ["SEN-TX"], platform="gdelt", ts=NOW - timedelta(days=1), bot_score=0.0))
            s.add(make_post(f"gdelt:t{i}", f"Texas Senate t {i}", ["SEN-TX"], platform="gdelt", ts=NOW, bot_score=0.0))
    assert await events.run_once(dbengine.session, cfg, None, now=NOW, gazetteer=make_gazetteer()) == 0


# -- F13 --------------------------------------------------------------------------------------------


async def test_copy_paste_campaign_counts_once(db):
    cfg = StrategyConfig(sentiment={"baseline_days": 3})
    with dbengine.session() as s:
        for i in range(50):  # aged accounts, ordinary handles: each copy scores well under 0.7
            s.add(make_post(f"x:camp{i}", "Jon Ossoff is the only choice for Georgia", ["SEN-GA"], ts=NOW - timedelta(minutes=i), bot_score=0.3, classified=True))
            s.add(Stance(post_id=f"x:camp{i}", race_id="SEN-GA", party="D", stance="support", confidence=0.9, is_news_or_poll=False, model="m", ts=NOW))
        for i in range(10):
            s.add(make_post(f"x:org{i}", f"organic take {i}", ["SEN-GA"], ts=NOW, bot_score=0.0, classified=True))
            s.add(Stance(post_id=f"x:org{i}", race_id="SEN-GA", party="D", stance="oppose", confidence=0.9, is_news_or_poll=False, model="m", ts=NOW))
    await features.run_once(dbengine.session, cfg, None, now=NOW)
    with dbengine.session() as s:
        row = s.execute(select(Feature).where(Feature.platform == "x", Feature.day == DAY)).scalars().one()
    assert (row.n_posts, row.net_stance_dem) == (11, round((1 - 10) / 11, 4))
