"""X/Twitter via a vendor-agnostic interface (official v2 recent-search shape is canonical):
vendors = socialdata | twitterapi | official | none. Writes Post rows (platform='x').

``secrets.x_vendor`` picks the backend and ``secrets.x_vendor_key`` authenticates it:

* ``official``   GET https://api.x.com/2/tweets/search/recent (pay-per-use, 7-day window, Bearer token)
* ``socialdata`` GET https://api.socialdata.tools/twitter/search (Bearer token; breaches X ToS - user's call)
* ``twitterapi`` GET https://api.twitterapi.io/twitter/tweet/advanced_search (``X-API-Key`` header)
* ``none``       collector disabled; ``run_once`` returns 0 without a request

Every vendor payload is normalised into the v2 shape ``{id, text, created_at, author_id,
author: {username, created_at}}`` before it becomes a Post, so downstream code never sees vendor
quirks. Queries come from the gazetteer per race: ``("Full Name" OR @handle OR #tag) -is:retweet lang:en``
(``-filter:retweets`` for the web-search-syntax vendors). Each run queries ``max_queries`` races
round-robin (KV cursor) and remembers the newest id per race (``x:since:<race_id>``).

LinkedIn, Instagram and Facebook are NOT collected anywhere in this package: no keyword-search API
exists for them and scraping is a ToS breach with litigation history (docs/research/practitioners_and_data.md).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import httpx

from sigcup.common.time import utc_now
from sigcup.db.repo import kv_get, kv_set
from sigcup.feeds.nlp.gazetteer import Gazetteer, PostDraft, load_gazetteer, parse_ts, pick, write_posts
from sigcup.settings import Secrets, StrategyConfig

log = logging.getLogger(__name__)

OFFICIAL_URL = "https://api.x.com/2/tweets/search/recent"
SOCIALDATA_URL = "https://api.socialdata.tools/twitter/search"
TWITTERAPI_URL = "https://api.twitterapi.io/twitter/tweet/advanced_search"
VENDORS = ("official", "socialdata", "twitterapi")
HTTP_TIMEOUT_S = 30.0
MAX_RESULTS = 100
MAX_QUERIES_PER_RUN = 40
MAX_QUERY_CHARS = 480  # the official basic tier caps queries at 512 characters
KV_CURSOR = "x:cursor"
KV_SINCE_PREFIX = "x:since:"


# ---------------------------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------------------------


def build_query(gazetteer: Gazetteer, race_id: str, vendor: str = "official") -> str | None:
    """Per-race search query in X syntax; None when the race has nothing searchable."""
    terms: list[str] = []
    for c in gazetteer.candidates_for(race_id):
        if len(c.name.split()) >= 2:
            terms.append(f'"{c.name}"')
        terms.extend(f"@{h}" for h in c.handles)
        terms.extend(f"#{t}" for t in c.hashtags)
        terms.extend(f'"{a}"' for a in c.aliases if len(a.split()) >= 2)
    terms = list(dict.fromkeys(terms))
    if not terms:
        return None
    suffix = " -is:retweet lang:en" if vendor == "official" else " -filter:retweets lang:en"
    while terms:
        core = terms[0] if len(terms) == 1 else "(" + " OR ".join(terms) + ")"
        q = core + suffix
        if len(q) <= MAX_QUERY_CHARS:
            return q
        terms.pop()
    return None


# ---------------------------------------------------------------------------------------------
# Normalisers: every vendor -> v2 shape (pure)
# ---------------------------------------------------------------------------------------------


def _v2(tweet_id: Any, text: Any, created: Any, author_id: Any, username: Any, author_created: Any) -> dict[str, Any] | None:
    tid = str(tweet_id or "").strip()
    txt = str(text or "").strip()
    if not tid or not txt:
        return None
    return {
        "id": tid,
        "text": txt,
        "created_at": created,
        "author_id": str(author_id or "") or None,
        "author": {"username": str(username or "").lstrip("@") or None, "created_at": author_created},
    }


def normalise_official(payload: Any) -> list[dict[str, Any]]:
    """v2 recent-search: ``data[]`` + ``includes.users[]`` joined on ``author_id``."""
    users = {str(pick(u, "id", default="")): u for u in (pick(payload, "includes.users", default=[]) or [])}
    out: list[dict[str, Any]] = []
    for t in pick(payload, "data", default=[]) or []:
        aid = str(pick(t, "author_id", default="") or "")
        u = users.get(aid, {})
        row = _v2(pick(t, "id"), pick(t, "text"), pick(t, "created_at"), aid, pick(u, "username"), pick(u, "created_at"))
        if row:
            out.append(row)
    return out


def normalise_socialdata(payload: Any) -> list[dict[str, Any]]:
    """SocialData ``/twitter/search``: ``tweets[]`` with ``id_str``, ``full_text``, ``tweet_created_at``, ``user{}``."""
    out: list[dict[str, Any]] = []
    for t in pick(payload, "tweets", "data", default=[]) or []:
        u = pick(t, "user", "author", default={}) or {}
        row = _v2(
            pick(t, "id_str", "id"),
            pick(t, "full_text", "text"),
            pick(t, "tweet_created_at", "created_at"),
            pick(u, "id_str", "id"),
            pick(u, "screen_name", "username", "userName"),
            pick(u, "created_at", "createdAt"),
        )
        if row:
            out.append(row)
    return out


def normalise_twitterapi(payload: Any) -> list[dict[str, Any]]:
    """twitterapi.io ``advanced_search``: ``tweets[]`` with ``id``, ``text``, ``createdAt``, ``author{userName,createdAt}``."""
    out: list[dict[str, Any]] = []
    for t in pick(payload, "tweets", "data", default=[]) or []:
        u = pick(t, "author", "user", default={}) or {}
        row = _v2(
            pick(t, "id", "id_str"),
            pick(t, "text", "full_text"),
            pick(t, "createdAt", "created_at"),
            pick(u, "id", "id_str"),
            pick(u, "userName", "username", "screen_name"),
            pick(u, "createdAt", "created_at"),
        )
        if row:
            out.append(row)
    return out


def normalise(vendor: str, payload: Any) -> list[dict[str, Any]]:
    fn = {"official": normalise_official, "socialdata": normalise_socialdata, "twitterapi": normalise_twitterapi}.get(vendor)
    return fn(payload) if fn else []


def to_drafts(tweets: Sequence[dict[str, Any]], race_id: str | None = None, *, now: datetime | None = None) -> list[PostDraft]:
    """v2-shaped tweets -> drafts (``post_id`` = ``x:<id>``)."""
    now = now or utc_now()
    out: list[PostDraft] = []
    for t in tweets:
        username = pick(t, "author.username")
        out.append(
            PostDraft(
                post_id=f"x:{t['id']}",
                platform="x",
                ts=parse_ts(t.get("created_at")) or now,
                text=t["text"],
                author_id=t.get("author_id"),
                author_handle=username,
                author_created_at=parse_ts(pick(t, "author.created_at")),
                url=f"https://x.com/{username or 'i'}/status/{t['id']}",
                race_ids=[race_id] if race_id else [],
            )
        )
    return out


# ---------------------------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------------------------


def request_spec(vendor: str, key: str, query: str, *, since_id: str | None = None, max_results: int = MAX_RESULTS) -> tuple[str, dict[str, Any], dict[str, str]]:
    """``(url, params, headers)`` for one search call on the given vendor."""
    if vendor == "official":
        params: dict[str, Any] = {
            "query": query,
            "max_results": max(10, min(100, int(max_results))),
            "tweet.fields": "created_at,author_id,lang",
            "expansions": "author_id",
            "user.fields": "created_at,username",
        }
        if since_id:
            params["since_id"] = since_id
        return OFFICIAL_URL, params, {"Authorization": f"Bearer {key}"}
    if vendor == "socialdata":
        return SOCIALDATA_URL, {"query": query, "type": "Latest"}, {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if vendor == "twitterapi":
        return TWITTERAPI_URL, {"query": query, "queryType": "Latest"}, {"X-API-Key": key}
    raise ValueError(f"unknown X vendor {vendor!r}")


async def fetch(http: httpx.AsyncClient, vendor: str, key: str, query: str, *, since_id: str | None = None) -> Any | None:
    """One search request; None on any failure (status and vendor logged, never the key or headers)."""
    url, params, headers = request_spec(vendor, key, query, since_id=since_id)
    try:
        r = await http.get(url, params=params, headers=headers, timeout=HTTP_TIMEOUT_S)
    except httpx.HTTPError as exc:
        log.warning("x[%s]: request failed: %s", vendor, type(exc).__name__)
        return None
    if r.status_code == 429:
        log.warning("x[%s]: rate limited; stopping this cycle", vendor)
        return None
    if r.status_code != 200:
        log.warning("x[%s]: HTTP %s", vendor, r.status_code)
        return None
    try:
        return r.json()
    except ValueError:
        log.warning("x[%s]: non-JSON response", vendor)
        return None


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


def _max_id(tweets: Sequence[dict[str, Any]], current: str | None) -> str | None:
    best = int(current) if current and str(current).isdigit() else 0
    for t in tweets:
        tid = str(t.get("id", ""))
        if tid.isdigit():
            best = max(best, int(tid))
    return str(best) if best else current


async def run_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    http: httpx.AsyncClient | None = None,
    vendor: str | None = None,
    api_key: str | None = None,
    max_queries: int = MAX_QUERIES_PER_RUN,
    gazetteer: Gazetteer | None = None,
    now: datetime | None = None,
) -> int:
    """Query the next ``max_queries`` races on the configured vendor; returns new Post rows (0 when vendor is none)."""
    now = now or utc_now()
    vendor = (vendor or (secrets.x_vendor if secrets else "none") or "none").strip().lower()
    key = (api_key if api_key is not None else (secrets.x_vendor_key if secrets else "")) or ""
    if vendor == "none" or vendor not in VENDORS:
        if vendor != "none":
            log.warning("x: unknown vendor %r; collector disabled", vendor)
        return 0
    if not key.strip():
        log.info("x: vendor %s configured without a key; skipping", vendor)
        return 0
    gazetteer = gazetteer or load_gazetteer(session_factory)
    queue = [(rid, q) for rid in gazetteer.race_ids() if (q := build_query(gazetteer, rid, vendor))]
    if not queue:
        return 0
    with session_factory() as s:
        cursor = int(kv_get(s, KV_CURSOR, 0) or 0) % len(queue)
    jobs = [queue[(cursor + i) % len(queue)] for i in range(min(max_queries, len(queue)))]
    owns = http is None
    client = http or httpx.AsyncClient()
    total = 0
    done = 0
    try:
        for rid, query in jobs:
            with session_factory() as s:
                since_id = kv_get(s, KV_SINCE_PREFIX + rid)
            payload = await fetch(client, vendor, key, query, since_id=since_id)
            if payload is None:
                break  # a 429 or outage: do not burn the budget on the remaining races
            done += 1
            tweets = normalise(vendor, payload)
            total += write_posts(session_factory, to_drafts(tweets, rid, now=now), gazetteer, require_link=True, now=now)
            with session_factory() as s:
                kv_set(s, KV_SINCE_PREFIX + rid, _max_id(tweets, since_id))
    finally:
        if owns:
            await client.aclose()
    with session_factory() as s:
        kv_set(s, KV_CURSOR, (cursor + done) % len(queue))
    log.info("x[%s]: %d/%d queries, %d new posts", vendor, done, len(jobs), total)
    return total
