"""Platform collectors: X vendor normalisers, YouTube, Bluesky Jetstream/search and the disabled Reddit path."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Post
from sigcup.db.repo import kv_get
from sigcup.feeds import social_bluesky, social_reddit, social_x, social_youtube
from sigcup.feeds.nlp.gazetteer import Gazetteer
from sigcup.settings import Secrets
from tests.test_social_helpers import NOW, load_json, load_text, make_gazetteer

# Every credential blank unless a test passes it: pydantic-settings would otherwise read exported
# BLUESKY_* / X_* / ... variables and a test could log in to a live service.
NO_CREDENTIALS: dict[str, str] = {
    "anthropic_api_key": "", "x_vendor": "none", "x_vendor_key": "", "bluesky_handle": "", "bluesky_app_password": "",
    "youtube_api_key": "", "reddit_client_id": "", "reddit_client_secret": "", "reddit_user_agent": "",
}


def secrets(**kw: Any) -> Secrets:
    return Secrets(_env_file=None, **{**NO_CREDENTIALS, **kw})


def all_posts() -> dict[str, Post]:
    with dbengine.session() as s:
        return {p.post_id: p for p in s.execute(select(Post)).scalars()}


# =============================================================================================
# X
# =============================================================================================


def test_x_normalise_official():
    rows = social_x.normalise_official(load_json("x_official.json"))
    assert [r["id"] for r in rows] == ["1841000000000000001", "1841000000000000002"]  # empty text dropped
    assert rows[0] == {"id": "1841000000000000001", "text": "Jon Ossoff just crushed the debate #GASen", "created_at": "2026-10-10T01:02:03.000Z",
                       "author_id": "12345", "author": {"username": "gapolitico", "created_at": "2015-04-01T00:00:00.000Z"}}
    assert rows[1]["author"] == {"username": None, "created_at": None}


def test_x_normalise_socialdata_and_twitterapi():
    sd = social_x.normalise_socialdata(load_json("x_socialdata.json"))
    assert sd == [{"id": "1841000000000000003", "text": "Buddy Carter is a fraud and Georgia knows it", "created_at": "2026-10-10T02:00:00.000000Z",
                   "author_id": "777", "author": {"username": "peachstate_dem", "created_at": "Tue Mar 03 10:00:00 +0000 2020"}}]
    ta = social_x.normalise_twitterapi(load_json("x_twitterapi.json"))
    assert ta == [{"id": "1841000000000000004", "text": "Vote Ossoff! @ossoff", "created_at": "Sat Oct 10 03:00:00 +0000 2026",
                   "author_id": "888", "author": {"username": "ga_voter_2026", "created_at": "Fri Jan 03 00:00:00 +0000 2025"}}]
    assert social_x.normalise("bogus", {}) == []
    assert social_x.normalise("official", None) == []


def test_x_to_drafts_parses_vendor_timestamps():
    d = social_x.to_drafts(social_x.normalise_twitterapi(load_json("x_twitterapi.json")), "SEN-GA", now=NOW)[0]
    assert d.post_id == "x:1841000000000000004" and d.platform == "x" and d.race_ids == ["SEN-GA"]
    assert d.ts == datetime(2026, 10, 10, 3, tzinfo=UTC) and d.author_created_at == datetime(2025, 1, 3, tzinfo=UTC)
    assert d.url == "https://x.com/ga_voter_2026/status/1841000000000000004" and d.author_handle == "ga_voter_2026"
    sd = social_x.to_drafts(social_x.normalise_socialdata(load_json("x_socialdata.json")), now=NOW)[0]
    assert sd.author_created_at == datetime(2020, 3, 3, 10, tzinfo=UTC) and sd.race_ids == []


def test_x_build_query_per_vendor():
    g = make_gazetteer()
    q = social_x.build_query(g, "SEN-GA")
    # hashtags and handles are normalised to lower case by the gazetteer (X search is case-insensitive)
    assert q == '("Jon Ossoff" OR @ossoff OR #teamossoff OR "Buddy Carter" OR @buddycarter OR "Earl Carter" OR "Chuck Jones") -is:retweet lang:en'
    assert social_x.build_query(g, "SEN-GA", "socialdata").endswith("-filter:retweets lang:en")
    assert social_x.build_query(g, "CONTROL-SENATE") is None
    assert len(social_x.build_query(Gazetteer.from_races({"SEN-ZZ": {"candidates": {str(i): {"name": f"Name Number{i}"} for i in range(60)}}}), "SEN-ZZ")) <= social_x.MAX_QUERY_CHARS


def test_x_request_spec_headers_never_leak_into_params():
    url, params, headers = social_x.request_spec("official", "sek", "q", since_id="5")
    assert url == social_x.OFFICIAL_URL and params["since_id"] == "5" and headers == {"Authorization": "Bearer sek"}
    assert "sek" not in json.dumps(params)
    assert social_x.request_spec("twitterapi", "k", "q")[2] == {"X-API-Key": "k"}
    with pytest.raises(ValueError):
        social_x.request_spec("nope", "k", "q")


@respx.mock
async def test_x_run_once_disabled_or_unconfigured(db, cfg):
    route = respx.get(host="api.x.com").mock(return_value=httpx.Response(200, json={}))
    assert await social_x.run_once(dbengine.session, cfg, secrets(x_vendor="none", x_vendor_key="k")) == 0
    assert await social_x.run_once(dbengine.session, cfg, secrets(x_vendor="official", x_vendor_key="")) == 0
    assert await social_x.run_once(dbengine.session, cfg, secrets(x_vendor="mystery", x_vendor_key="k")) == 0
    assert await social_x.run_once(dbengine.session, cfg, None) == 0
    assert not route.called


@respx.mock
async def test_x_run_once_official(db, cfg):
    official = load_json("x_official.json")
    seen: list[httpx.Request] = []

    def side(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        q = request.url.params["query"]
        return httpx.Response(200, json=official if "Ossoff" in q else {"data": [], "meta": {"result_count": 0}})

    respx.get(host="api.x.com", path="/2/tweets/search/recent").mock(side_effect=side)
    n = await social_x.run_once(dbengine.session, cfg, secrets(x_vendor="official", x_vendor_key="tok"), gazetteer=make_gazetteer(), now=NOW)
    assert n == 2
    assert len(seen) == 6 and all(r.headers["authorization"] == "Bearer tok" for r in seen)
    assert seen[0].url.params["expansions"] == "author_id" and seen[0].url.params["max_results"] == "100"
    posts = all_posts()
    p = posts["x:1841000000000000001"]
    assert p.race_ids == ["SEN-GA"] and p.author_handle == "gapolitico" and p.author_created_at.year == 2015
    assert posts["x:1841000000000000002"].race_ids == ["SEN-GA"]
    with dbengine.session() as s:
        assert kv_get(s, social_x.KV_SINCE_PREFIX + "SEN-GA") == "1841000000000000002"
        assert kv_get(s, social_x.KV_CURSOR) == 0  # six races, six queries, wrapped
    # second cycle sends since_id
    await social_x.run_once(dbengine.session, cfg, secrets(x_vendor="official", x_vendor_key="tok"), gazetteer=make_gazetteer(), now=NOW)
    ga = [r for r in seen[6:] if "Ossoff" in r.url.params["query"]][0]
    assert ga.url.params["since_id"] == "1841000000000000002"


@respx.mock
async def test_x_run_once_socialdata_and_twitterapi(db, cfg):
    sd = respx.get(host="api.socialdata.tools", path="/twitter/search").mock(return_value=httpx.Response(200, json=load_json("x_socialdata.json")))
    n = await social_x.run_once(dbengine.session, cfg, secrets(x_vendor="socialdata", x_vendor_key="sdk"), gazetteer=make_gazetteer(), now=NOW, max_queries=1)
    assert n == 1 and sd.calls[0].request.headers["authorization"] == "Bearer sdk"
    assert sd.calls[0].request.url.params["type"] == "Latest"
    ta = respx.get(host="api.twitterapi.io", path="/twitter/tweet/advanced_search").mock(return_value=httpx.Response(200, json=load_json("x_twitterapi.json")))
    n = await social_x.run_once(dbengine.session, cfg, secrets(x_vendor="twitterapi", x_vendor_key="tak"), gazetteer=make_gazetteer(), now=NOW, max_queries=1)
    assert n == 1 and ta.calls[0].request.headers["x-api-key"] == "tak"
    posts = all_posts()
    assert posts["x:1841000000000000003"].platform == "x" and posts["x:1841000000000000004"].author_handle == "ga_voter_2026"


@respx.mock
async def test_x_run_once_stops_on_rate_limit(db, cfg):
    route = respx.get(host="api.x.com").mock(return_value=httpx.Response(429))
    assert await social_x.run_once(dbengine.session, cfg, secrets(x_vendor="official", x_vendor_key="tok"), gazetteer=make_gazetteer()) == 0
    assert route.call_count == 1
    with dbengine.session() as s:
        assert kv_get(s, social_x.KV_CURSOR) == 0


# =============================================================================================
# YouTube
# =============================================================================================


def test_youtube_parsers():
    assert social_youtube.uploads_playlist_id("UCabc123") == "UUabc123"
    assert social_youtube.uploads_playlist_id("PLxyz") == "PLxyz"
    videos = social_youtube.parse_playlist_items(load_json("youtube_playlist.json"))
    assert [v["video_id"] for v in videos] == ["vid001", "vidold"]
    assert videos[0]["title"] == "Georgia Senate debate: Ossoff vs Carter" and videos[0]["published_at"] == datetime(2026, 10, 9, 18, tzinfo=UTC)
    drafts = social_youtube.parse_comment_threads(load_json("youtube_comments.json"), videos[0], race_id=None, now=NOW)
    assert [d.post_id for d in drafts] == ["youtube:Ugx001", "youtube:Ugx002"]
    d = drafts[0]
    assert d.platform == "youtube" and d.text == "Ossoff won this debate easily" and d.author_handle == "Peach State Voter"
    assert d.author_id == "UCauthor1" and d.url == "https://www.youtube.com/watch?v=vid001&lc=Ugx001"
    assert d.link_text == "Georgia Senate debate: Ossoff vs Carter" and d.ts == datetime(2026, 10, 9, 20, tzinfo=UTC)
    assert social_youtube.channel_specs({"global": {"youtube_channels": ["UCa", {"id": "UCb", "race_id": "SEN-TX"}, "UCa"]}, "races": {"SEN-GA": {"youtube_channels": ["UCc"]}}}) == [("UCa", None), ("UCb", "SEN-TX"), ("UCc", "SEN-GA")]


@respx.mock
async def test_youtube_run_once_unconfigured(db, cfg):
    route = respx.get(host="www.googleapis.com").mock(return_value=httpx.Response(200, json={}))
    assert await social_youtube.run_once(dbengine.session, cfg, secrets(youtube_api_key=""), sources={"global": {"youtube_channels": ["UCx"]}}) == 0
    assert await social_youtube.run_once(dbengine.session, cfg, secrets(youtube_api_key="k"), sources={}) == 0
    assert not route.called


@respx.mock
async def test_youtube_run_once_uses_playlist_and_comments_never_search(db, cfg):
    search = respx.get(host="www.googleapis.com", path="/youtube/v3/search").mock(return_value=httpx.Response(200, json={}))
    playlist = respx.get(host="www.googleapis.com", path="/youtube/v3/playlistItems").mock(return_value=httpx.Response(200, json=load_json("youtube_playlist.json")))
    comments = respx.get(host="www.googleapis.com", path="/youtube/v3/commentThreads").mock(return_value=httpx.Response(200, json=load_json("youtube_comments.json")))
    sources = {"global": {"youtube_channels": ["UCchan1"]}}
    n = await social_youtube.run_once(dbengine.session, cfg, secrets(youtube_api_key="key"), sources=sources, gazetteer=make_gazetteer(), now=NOW)
    assert n == 2  # both comments sit under "Georgia Senate debate: Ossoff vs Carter", so the title links them
    assert not search.called
    assert playlist.calls[0].request.url.params["playlistId"] == "UUchan1" and playlist.calls[0].request.url.params["key"] == "key"
    assert comments.call_count == 1  # the 2025 upload is older than the cutoff
    assert comments.calls[0].request.url.params["videoId"] == "vid001" and comments.calls[0].request.url.params["textFormat"] == "plainText"
    posts = all_posts()
    assert posts["youtube:Ugx001"].race_ids == ["SEN-GA"] and posts["youtube:Ugx001"].text == "Ossoff won this debate easily"
    assert posts["youtube:Ugx002"].race_ids == ["SEN-GA"]

    # a race-specific channel keeps comments the gazetteer cannot link, tagged with the race
    playlist.mock(return_value=httpx.Response(200, json={"items": [{"contentDetails": {"videoId": "vid777", "videoPublishedAt": "2026-10-09T18:00:00Z"}, "snippet": {"title": "Weekly roundup"}}]}))
    comments.mock(return_value=httpx.Response(200, json={"items": [{"id": "Ugx777", "snippet": {"topLevelComment": {"id": "Ugx777", "snippet": {"textOriginal": "lol", "videoId": "vid777", "publishedAt": "2026-10-09T20:00:00Z"}}}}]}))
    sources = {"races": {"SEN-GA": {"youtube_channels": ["UCchan2"]}}}
    n = await social_youtube.run_once(dbengine.session, cfg, secrets(youtube_api_key="key"), sources=sources, gazetteer=make_gazetteer(), now=NOW)
    assert n == 1 and all_posts()["youtube:Ugx777"].race_ids == ["SEN-GA"]
    assert await social_youtube.run_once(dbengine.session, cfg, secrets(youtube_api_key="key"), sources={"global": {"youtube_channels": ["UCchan3"]}}, gazetteer=make_gazetteer(), now=NOW) == 0  # unlinked on a global channel

    # the quota cap stops after the playlist call
    before = comments.call_count
    assert await social_youtube.run_once(dbengine.session, cfg, secrets(youtube_api_key="key"), sources=sources, gazetteer=make_gazetteer(), now=NOW, max_units=1) == 0
    assert comments.call_count == before


# =============================================================================================
# Bluesky
# =============================================================================================


JETSTREAM_LINES = load_text("jetstream_messages.jsonl").splitlines()


class FakeWS:
    def __init__(self, msgs: list[str]) -> None:
        self._msgs = list(msgs)

    def __aiter__(self) -> FakeWS:
        return self

    async def __anext__(self) -> str:
        if not self._msgs:
            raise StopAsyncIteration
        return self._msgs.pop(0)


class FakeConnect:
    def __init__(self, msgs: list[str], fail: bool = False) -> None:
        self.msgs, self.fail, self.urls = msgs, fail, []

    def __call__(self, url: str) -> FakeConnect:
        self.urls.append(url)
        return self

    async def __aenter__(self) -> FakeWS:
        if self.fail:
            raise OSError("proxy blocked")
        return FakeWS(self.msgs)

    async def __aexit__(self, *exc: Any) -> bool:
        return False


def test_bluesky_parse_jetstream_message():
    d = social_bluesky.parse_jetstream_message(JETSTREAM_LINES[0])
    assert d is not None
    assert d.post_id == "bluesky:did:plc:abc123xyz/3kpost1" and d.platform == "bluesky" and d.author_id == "did:plc:abc123xyz"
    assert d.ts == datetime(2026, 10, 10, 4, tzinfo=UTC) and d.url == "https://bsky.app/profile/did:plc:abc123xyz/post/3kpost1"
    assert d.text == "Jon Ossoff up 4 in the new AJC poll" and d.author_handle is None
    assert social_bluesky.parse_jetstream_message(JETSTREAM_LINES[1]) is None  # identity event
    assert social_bluesky.parse_jetstream_message(JETSTREAM_LINES[2]).text.startswith("lovely")
    assert social_bluesky.parse_jetstream_message(JETSTREAM_LINES[3]) is None  # delete
    assert social_bluesky.parse_jetstream_message(JETSTREAM_LINES[4]) is None  # not English
    assert social_bluesky.parse_jetstream_message(JETSTREAM_LINES[5]) is None  # not JSON
    assert social_bluesky.parse_jetstream_message(json.loads(JETSTREAM_LINES[0])) is not None
    no_created = json.loads(JETSTREAM_LINES[0])
    del no_created["commit"]["record"]["createdAt"]
    assert social_bluesky.parse_jetstream_message(no_created).ts == datetime(2026, 10, 13, 14, tzinfo=UTC)  # falls back to time_us


async def test_bluesky_consume_jetstream_filters_by_gazetteer():
    connect = FakeConnect(JETSTREAM_LINES)
    kept = await social_bluesky.consume_jetstream(make_gazetteer(), 5.0, connect=connect)
    assert [d.post_id for d in kept] == ["bluesky:did:plc:abc123xyz/3kpost1"] and kept[0].race_ids == ["SEN-GA"]
    assert connect.urls == [social_bluesky.JETSTREAM_URL]
    assert await social_bluesky.consume_jetstream(None, 5.0, connect=FakeConnect(JETSTREAM_LINES)) and len(await social_bluesky.consume_jetstream(None, 5.0, connect=FakeConnect(JETSTREAM_LINES))) == 2
    assert await social_bluesky.consume_jetstream(make_gazetteer(), 5.0, connect=FakeConnect(JETSTREAM_LINES, fail=True)) == []
    assert await social_bluesky.consume_jetstream(make_gazetteer(), 0.0, connect=FakeConnect(JETSTREAM_LINES)) == []


def test_bluesky_parse_search_post():
    posts = load_json("bluesky_search.json")["posts"]
    d = social_bluesky.parse_search_post(posts[0])
    assert d.post_id == "bluesky:did:plc:abc123xyz/3ksearch1" and d.author_handle == "reporter.bsky.social"
    assert d.author_created_at == datetime(2023, 5, 1, tzinfo=UTC) and d.ts == datetime(2026, 10, 10, 4, 30, tzinfo=UTC)
    assert social_bluesky.parse_search_post(posts[1]) is None
    obj = SimpleNamespace(uri=posts[0]["uri"], record=SimpleNamespace(text="hi", created_at=None), author=SimpleNamespace(did="did:plc:abc123xyz", handle="h", created_at=None), indexed_at="2026-10-10T05:00:00Z")
    assert social_bluesky.parse_search_post(obj).ts == datetime(2026, 10, 10, 5, tzinfo=UTC)


class FakeSearchClient:
    def __init__(self) -> None:
        self.queries: list[dict[str, Any]] = []
        self.app = SimpleNamespace(bsky=SimpleNamespace(feed=SimpleNamespace(search_posts=self.search_posts)))

    def search_posts(self, params: dict[str, Any]) -> SimpleNamespace:
        self.queries.append(params)
        return SimpleNamespace(posts=load_json("bluesky_search.json")["posts"] if "Ossoff" in params["q"] else [])


async def test_bluesky_run_once_jetstream_and_backfill(db, cfg):
    n = await social_bluesky.run_once(dbengine.session, cfg, None, duration_s=5, connect=FakeConnect(JETSTREAM_LINES), gazetteer=make_gazetteer(), now=NOW)
    assert n == 1 and all_posts()["bluesky:did:plc:abc123xyz/3kpost1"].race_ids == ["SEN-GA"]
    # no credentials -> no backfill; a blocked firehose -> 0, not an exception
    assert await social_bluesky.run_once(dbengine.session, cfg, secrets(), duration_s=5, connect=FakeConnect(JETSTREAM_LINES, fail=True), gazetteer=make_gazetteer()) == 0

    client = FakeSearchClient()
    n = await social_bluesky.run_once(dbengine.session, cfg, None, duration_s=0, search_client=client, search_queries=20, search_pause_s=0, gazetteer=make_gazetteer(), now=NOW)
    assert n == 1
    assert {q["q"] for q in client.queries} >= {'"Jon Ossoff"', '"Buddy Carter"'} and all(q["limit"] == 25 for q in client.queries)
    p = all_posts()["bluesky:did:plc:abc123xyz/3ksearch1"]
    assert p.race_ids == ["SEN-GA"] and p.author_handle == "reporter.bsky.social"
    with dbengine.session() as s:
        assert kv_get(s, social_bluesky.KV_SEARCH_CURSOR) is not None


async def test_bluesky_run_once_empty_gazetteer(db, cfg):
    assert await social_bluesky.run_once(dbengine.session, cfg, None, duration_s=5, connect=FakeConnect(JETSTREAM_LINES), gazetteer=Gazetteer([])) == 0


# =============================================================================================
# Reddit
# =============================================================================================


async def test_reddit_disabled_without_credentials_and_never_imports_praw(db, cfg, monkeypatch):
    monkeypatch.setitem(sys.modules, "praw", None)  # any `import praw` would now raise ImportError
    assert await social_reddit.run_once(dbengine.session, cfg, None) == 0
    assert await social_reddit.run_once(dbengine.session, cfg, secrets(reddit_client_id="id", reddit_client_secret="")) == 0
    assert social_reddit.credentials_present(secrets(reddit_client_id="id", reddit_client_secret="s", reddit_user_agent="ua"))


def test_reddit_subreddits_and_normalisers():
    subs = social_reddit.subreddits_for(make_gazetteer())
    assert subs[:5] == ["Arkansas", "Georgia", "NewYork", "Pennsylvania", "Texas"] and subs[-3:] == list(social_reddit.BETTOR_SUBS)
    sub = SimpleNamespace(id="abc", title="Jon Ossoff town hall recap", selftext="He spoke for an hour.", created_utc=1791900000, author=SimpleNamespace(name="u_one"), permalink="/r/Georgia/comments/abc/x/")
    d = social_reddit.normalise_submission(sub, "Georgia")
    assert d.post_id == "reddit:t3_abc" and d.text == "Jon Ossoff town hall recap He spoke for an hour." and d.author_handle == "u_one"
    assert d.url == "https://www.reddit.com/r/Georgia/comments/abc/x/" and d.ts == datetime(2026, 10, 13, 14, tzinfo=UTC) and d.link_text == "r/Georgia"
    c = social_reddit.normalise_comment({"id": "c1", "body": "Carter is toast", "created_utc": 1791900000, "author": "u_two", "permalink": "/r/Georgia/comments/abc/x/c1/"}, "Georgia")
    assert c.post_id == "reddit:t1_c1" and c.author_handle == "u_two"
    assert social_reddit.normalise_comment({"id": "c2", "body": "[deleted]"}) is None
    assert social_reddit.normalise_submission({"id": "", "title": "x"}) is None


class FakeSub:
    def __init__(self, name: str) -> None:
        self.name = name

    def new(self, limit: int) -> list[Any]:
        return [SimpleNamespace(id=f"{self.name}1", title=f"Jon Ossoff visits r/{self.name}", selftext="", created_utc=1791900000, author=SimpleNamespace(name="u"), permalink=f"/r/{self.name}/1/"),
                SimpleNamespace(id=f"{self.name}2", title="unrelated", selftext="", created_utc=1791900000, author=None, permalink=f"/r/{self.name}/2/")]

    def comments(self, limit: int) -> list[Any]:
        return [SimpleNamespace(id=f"{self.name}c", body="Carter is up in Georgia", created_utc=1791900000, author=SimpleNamespace(name="v"), permalink=f"/r/{self.name}/1/c/")]


class FakeReddit:
    def __init__(self) -> None:
        self.names: list[str] = []

    def subreddit(self, name: str) -> FakeSub:
        self.names.append(name)
        if name == "Texas":
            raise RuntimeError("forbidden")
        return FakeSub(name)


async def test_reddit_run_once_with_injected_client(db, cfg):
    reddit = FakeReddit()
    n = await social_reddit.run_once(dbengine.session, cfg, None, reddit=reddit, gazetteer=make_gazetteer(), max_subs=5, now=NOW)
    assert reddit.names == ["Arkansas", "Georgia", "NewYork", "Pennsylvania", "Texas"]
    assert n == 8  # 4 working subs x (1 linked submission + 1 linked comment)
    posts = all_posts()
    assert posts["reddit:t3_Georgia1"].race_ids == ["SEN-GA"] and posts["reddit:t1_Georgiac"].platform == "reddit"
    with dbengine.session() as s:
        assert kv_get(s, social_reddit.KV_CURSOR) == 5
