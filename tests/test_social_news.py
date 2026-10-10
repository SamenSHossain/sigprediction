"""News collectors: GDELT artlist/timelinevol normalisers and RSS parsing from fixtures; run_once via respx."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Post
from sigcup.db.repo import kv_get
from sigcup.feeds import news_gdelt, news_rss
from tests.test_social_helpers import NOW, load_json, load_text, make_gazetteer

GDELT_HOST, GDELT_PATH = "api.gdeltproject.org", "/api/v2/doc/doc"


async def no_sleep(_: float) -> None:
    return None


# -- GDELT ---------------------------------------------------------------------------------------


def test_build_query():
    assert news_gdelt.build_query(["Jon Ossoff"]) == '"Jon Ossoff" sourcelang:english'
    assert news_gdelt.build_query(["Jon Ossoff", "Buddy Carter", ""]) == '("Jon Ossoff" OR "Buddy Carter") sourcelang:english'
    assert news_gdelt.build_query([]) is None
    assert news_gdelt.build_query(["a"] * 20).count('"a"') == news_gdelt.MAX_QUERY_TERMS


def test_parse_artlist_normalises_and_dedupes():
    drafts = news_gdelt.parse_artlist(load_json("gdelt_artlist.json"), "SEN-GA", now=NOW)
    assert [d.post_id for d in drafts] == [news_gdelt.article_id("https://www.ajc.com/politics/ossoff-poll-2026/"),
                                           news_gdelt.article_id("https://www.texastribune.org/2026/10/10/texas-senate-debate/")]
    first, second = drafts
    assert first.platform == "gdelt" and first.race_ids == ["SEN-GA"] and first.author_handle == "ajc.com"
    assert first.ts == datetime(2026, 10, 10, 12, tzinfo=UTC)
    assert first.text == "New poll shows Jon Ossoff ahead in Georgia Senate race"
    assert second.text == "Texas Senate debate: candidates clash over the border The two candidates traded barbs for ninety minutes."
    assert news_gdelt.parse_artlist({"weird": 1}) == []
    assert news_gdelt.parse_artlist(None) == []


def test_parse_timelinevol():
    points = news_gdelt.parse_timelinevol(load_json("gdelt_timelinevol.json"))
    assert points == [(datetime(2026, 10, 8, tzinfo=UTC), 0.12), (datetime(2026, 10, 9, tzinfo=UTC), 0.15), (datetime(2026, 10, 10, tzinfo=UTC), 0.9)]
    assert news_gdelt.parse_timelinevol({}) == []


@respx.mock
async def test_gdelt_run_once_round_robin_and_linking(db, cfg):
    art = load_json("gdelt_artlist.json")
    queries: list[str] = []

    def side(request: httpx.Request) -> httpx.Response:
        q = request.url.params.get("query", "")
        queries.append(q)
        assert request.url.params.get("mode") == "artlist" and request.url.params.get("format") == "json"
        return httpx.Response(200, json=art if "Ossoff" in q else {"articles": []})

    respx.get(host=GDELT_HOST, path=GDELT_PATH).mock(side_effect=side)
    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    g = make_gazetteer()
    # six races carry candidate names (CONTROL-SENATE has none); sorted order puts SEN-GA fifth
    n = await news_gdelt.run_once(dbengine.session, cfg, None, gazetteer=g, sleep=sleep, now=NOW, max_requests=5)
    assert n == 2
    assert len(queries) == 5 and len(sleeps) == 4 and all(s >= 5 for s in sleeps)
    assert queries[0].startswith('("Burt Jones" OR "Keisha Lance Bottoms")')  # sorted race order: GOV-GA first
    assert queries[4].startswith('("Jon Ossoff" OR "Buddy Carter" OR "Earl Carter" OR "Chuck Jones")')
    with dbengine.session() as s:
        rows = {p.post_id: p for p in s.execute(select(Post)).scalars()}
        assert kv_get(s, news_gdelt.KV_CURSOR) == 5
    texas = rows[news_gdelt.article_id("https://www.texastribune.org/2026/10/10/texas-senate-debate/")]
    assert texas.race_ids == ["SEN-GA", "SEN-TX"]  # queried race + gazetteer race code
    assert texas.bot_score == 0.0 and texas.classified is False
    assert rows[news_gdelt.article_id("https://www.ajc.com/politics/ossoff-poll-2026/")].race_ids == ["SEN-GA"]

    # the next cycle continues the rotation (SEN-TX, then wraps to GOV-GA ...) and finds nothing new
    n = await news_gdelt.run_once(dbengine.session, cfg, None, gazetteer=g, sleep=sleep, now=NOW, max_requests=5)
    assert n == 0
    with dbengine.session() as s:
        assert kv_get(s, news_gdelt.KV_CURSOR) == 4
    assert len(queries) == 10 and queries[5].startswith('("Colin Allred" OR "Ken Paxton")')


@respx.mock
@pytest.mark.parametrize("response", [httpx.Response(429, text="rate limited"), httpx.Response(200, text="The query was too short."), httpx.Response(500)])
async def test_gdelt_run_once_tolerates_failures(db, cfg, response):
    respx.get(host=GDELT_HOST, path=GDELT_PATH).mock(return_value=response)
    assert await news_gdelt.run_once(dbengine.session, cfg, None, gazetteer=make_gazetteer(), sleep=no_sleep, now=NOW) == 0


async def test_gdelt_run_once_without_candidates_returns_zero(db, cfg):
    from sigcup.feeds.nlp.gazetteer import Gazetteer

    assert await news_gdelt.run_once(dbengine.session, cfg, None, gazetteer=Gazetteer([]), sleep=no_sleep) == 0


@respx.mock
async def test_gdelt_volume_once_stores_kv(db, cfg):
    respx.get(host=GDELT_HOST, path=GDELT_PATH, params__contains={"mode": "timelinevol"}).mock(
        return_value=httpx.Response(200, json=load_json("gdelt_timelinevol.json"))
    )
    n = await news_gdelt.volume_once(dbengine.session, cfg, ["SEN-GA", "CONTROL-SENATE"], gazetteer=make_gazetteer(), sleep=no_sleep)
    assert n == 1
    with dbengine.session() as s:
        assert len(kv_get(s, news_gdelt.KV_VOLUME_PREFIX + "SEN-GA")["points"]) == 3


# -- RSS -----------------------------------------------------------------------------------------


def test_feed_urls_from_sources():
    sources = {"global": {"rss": ["https://a/feed", "https://b/feed", "https://a/feed"]}, "races": {"SEN-GA": {"rss": ["https://c/feed"]}, "SEN-TX": {}}}
    assert news_rss.feed_urls(sources) == [("https://a/feed", None), ("https://b/feed", None), ("https://c/feed", "SEN-GA")]
    assert news_rss.feed_urls({}) == []


def test_parse_feed_from_local_string():
    drafts = news_rss.parse_feed(load_text("rss_feed.xml"), source_url="https://www.ajc.com/politics/feed", now=NOW)
    assert len(drafts) == 3
    d = drafts[0]
    assert d.post_id.startswith("rss:") and d.platform == "rss"
    assert d.text == "Jon Ossoff rallies in Savannah ahead of debate The senator drew a crowd of 2,000 on the riverfront."
    assert d.ts == datetime(2026, 10, 10, 9, 30, tzinfo=UTC)
    assert d.author_handle == "Atlanta Journal-Constitution Politics"
    assert d.url == "https://www.ajc.com/politics/ossoff-savannah-rally/"
    assert drafts[2].post_id != drafts[1].post_id
    tagged = news_rss.parse_feed(load_text("rss_feed.xml").encode(), race_id="SEN-GA")
    assert all(x.race_ids == ["SEN-GA"] for x in tagged)
    assert news_rss.parse_feed("<not really xml") == []


def test_parse_feed_then_write_posts_keeps_linked_entries_only(db):
    from sigcup.feeds.nlp.gazetteer import write_posts

    g = make_gazetteer()
    drafts = news_rss.parse_feed(load_text("rss_feed.xml"), now=NOW)
    assert write_posts(dbengine.session, drafts, g, now=NOW) == 2  # Ossoff (full name) + "Carter ... Georgia Senate" (surname + token)
    with dbengine.session() as s:
        texts = sorted(p.text[:20] for p in s.execute(select(Post)).scalars())
    assert texts == ["Carter is up in the ", "Jon Ossoff rallies i"]


@respx.mock
async def test_rss_run_once_conditional_get(db, cfg):
    feed = load_text("rss_feed.xml")
    hits: list[dict[str, str]] = []

    def global_side(request: httpx.Request) -> httpx.Response:
        hits.append(dict(request.headers))
        if request.headers.get("if-none-match") == '"v1"':
            return httpx.Response(304)
        return httpx.Response(200, text=feed, headers={"ETag": '"v1"', "Last-Modified": "Sat, 10 Oct 2026 09:30:00 GMT"})

    respx.get("https://news.example/global.xml").mock(side_effect=global_side)
    respx.get("https://news.example/ga.xml").mock(return_value=httpx.Response(200, text=feed))
    sources = {"global": {"rss": ["https://news.example/global.xml"]}, "races": {"SEN-GA": {"rss": ["https://news.example/ga.xml"]}}}

    n = await news_rss.run_once(dbengine.session, cfg, None, sources=sources, gazetteer=make_gazetteer(), now=NOW)
    assert n == 3  # 2 linked from the global feed, the Braves item only via the race feed
    with dbengine.session() as s:
        braves = [p for p in s.execute(select(Post)).scalars() if p.text.startswith("Braves")]
        assert braves[0].race_ids == ["SEN-GA"]
        assert kv_get(s, news_rss._kv_key("https://news.example/global.xml"))["etag"] == '"v1"'

    n = await news_rss.run_once(dbengine.session, cfg, None, sources=sources, gazetteer=make_gazetteer(), now=NOW)
    assert n == 0
    assert hits[1].get("if-none-match") == '"v1"' and hits[1].get("if-modified-since")


@respx.mock
async def test_rss_run_once_unconfigured_and_failures(db, cfg):
    assert await news_rss.run_once(dbengine.session, cfg, None, sources={}) == 0
    respx.get("https://down.example/feed").mock(side_effect=httpx.ConnectError("blocked"))
    respx.get("https://gone.example/feed").mock(return_value=httpx.Response(404))
    sources = {"global": {"rss": ["https://down.example/feed", "https://gone.example/feed"]}}
    assert await news_rss.run_once(dbengine.session, cfg, None, sources=sources, gazetteer=make_gazetteer()) == 0
