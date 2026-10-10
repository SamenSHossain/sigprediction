"""GDELT 2.0 DOC API: per-candidate article volume and tone timelines; writes Post rows (platform='gdelt',
one per article) and feeds the event detector with volume spikes.

Endpoint ``cfg.feeds.gdelt_doc`` (``.../api/v2/doc/doc``), free, no key, ~1 request per 5 s or a 429.
Per race the query is the candidates' full names OR-ed in quotes (``("Jon Ossoff" OR "Buddy Carter")
sourcelang:english``), ``mode=artlist&format=json``. Each article becomes a Post: ``post_id`` =
``gdelt:<sha256(url)[:24]>``, ``text`` = title + ' ' + snippet (artlist has no snippet; tolerated),
``ts`` from ``seendate``, ``author_handle`` = source domain, ``race_ids`` = the queried race plus
whatever the gazetteer links. ``mode=timelinevol`` is exposed as :func:`fetch_timelinevol` /
:func:`parse_timelinevol` and stored in KV by :func:`volume_once`; the event detector's primary volume
measure is simply the count of gdelt Post rows per (race, day).

Request budget: requests are serialised with ``asyncio.sleep`` (``GDELT_MIN_INTERVAL_S``) and each
``run_once`` handles at most ``max_requests`` races, round-robin via a KV cursor, so a 117-race universe
is swept in two or three 15-minute cycles rather than blocking one cycle for ten minutes.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Any

import httpx

from sigcup.common.time import utc_now
from sigcup.db.repo import kv_get, kv_set
from sigcup.feeds.nlp.gazetteer import (
    Gazetteer,
    PostDraft,
    load_gazetteer,
    parse_ts,
    pick,
    strip_html,
    write_posts,
)
from sigcup.settings import Secrets, StrategyConfig

log = logging.getLogger(__name__)

GDELT_MIN_INTERVAL_S = 5.2
MAX_REQUESTS_PER_RUN = 60
MAX_RECORDS = 75
DEFAULT_TIMESPAN = "1d"
HTTP_TIMEOUT_S = 30.0
KV_CURSOR = "gdelt:cursor"
KV_VOLUME_PREFIX = "gdelt:timelinevol:"
MAX_QUERY_TERMS = 6

Sleep = Callable[[float], Awaitable[None]]


def build_query(terms: Sequence[str], *, extra: str = "sourcelang:english") -> str | None:
    """GDELT query: quoted phrases OR-ed inside parentheses (GDELT requires the parentheses for OR)."""
    phrases = [f'"{t.strip()}"' for t in terms if t and t.strip()][:MAX_QUERY_TERMS]
    if not phrases:
        return None
    core = phrases[0] if len(phrases) == 1 else "(" + " OR ".join(phrases) + ")"
    return f"{core} {extra}".strip()


def article_id(url: str) -> str:
    return "gdelt:" + hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:24]


def parse_artlist(payload: Any, race_id: str | None = None, *, now: datetime | None = None) -> list[PostDraft]:
    """Normalise an ``artlist`` payload into drafts (lenient: tolerates missing or renamed fields)."""
    now = now or utc_now()
    arts = pick(payload, "articles", "results", "items", default=[]) or []
    out: list[PostDraft] = []
    seen: set[str] = set()
    for a in arts:
        url = str(pick(a, "url", "link", "url_mobile", default="") or "").strip()
        title = strip_html(pick(a, "title", "headline", default=""))
        if not url or not title:
            continue
        pid = article_id(url)
        if pid in seen:
            continue
        seen.add(pid)
        snippet = strip_html(pick(a, "snippet", "excerpt", "description", "summary", default=""))
        text = f"{title} {snippet}".strip()
        ts = parse_ts(pick(a, "seendate", "date", "published", "publishdate")) or now
        out.append(
            PostDraft(
                post_id=pid,
                platform="gdelt",
                ts=ts,
                text=text,
                author_handle=str(pick(a, "domain", "source", default="") or "")[:128] or None,
                url=url,
                race_ids=[race_id] if race_id else [],
            )
        )
    return out


def parse_timelinevol(payload: Any) -> list[tuple[datetime, float]]:
    """``mode=timelinevol`` -> [(ts, value)]; value is GDELT's percentage of monitored coverage."""
    series = pick(payload, "timeline", default=[]) or []
    data = pick(series[0], "data", default=[]) if series else []
    out: list[tuple[datetime, float]] = []
    for point in data or []:
        ts = parse_ts(pick(point, "date", "ts"))
        try:
            val = float(pick(point, "value", "count", default=0.0))
        except (TypeError, ValueError):
            continue
        if ts is not None:
            out.append((ts, val))
    return out


async def fetch_json(http: httpx.AsyncClient, url: str, params: dict[str, Any]) -> Any | None:
    """GET and decode; None on any HTTP, network or decode error (GDELT returns plain text on bad queries)."""
    try:
        r = await http.get(url, params=params, timeout=HTTP_TIMEOUT_S)
    except httpx.HTTPError as exc:
        log.warning("gdelt: request failed: %s", type(exc).__name__)
        return None
    if r.status_code == 429:
        log.warning("gdelt: rate limited (429); backing off this cycle")
        return None
    if r.status_code != 200:
        log.warning("gdelt: HTTP %s", r.status_code)
        return None
    try:
        return r.json()
    except ValueError:
        log.warning("gdelt: non-JSON response (%s)", (r.text or "")[:80].replace("\n", " "))
        return None


async def fetch_artlist(http: httpx.AsyncClient, base: str, query: str, *, timespan: str = DEFAULT_TIMESPAN, max_records: int = MAX_RECORDS) -> Any | None:
    params = {"query": query, "mode": "artlist", "format": "json", "maxrecords": int(max_records), "timespan": timespan, "sort": "datedesc"}
    return await fetch_json(http, base, params)


async def fetch_timelinevol(http: httpx.AsyncClient, base: str, query: str, *, timespan: str = "7d") -> Any | None:
    params = {"query": query, "mode": "timelinevol", "format": "json", "timespan": timespan}
    return await fetch_json(http, base, params)


def _race_queue(gazetteer: Gazetteer) -> list[tuple[str, str]]:
    """(race_id, query) for every race with searchable candidate names, in a stable order."""
    out: list[tuple[str, str]] = []
    for rid in gazetteer.race_ids():
        q = build_query(gazetteer.search_terms(rid))
        if q:
            out.append((rid, q))
    return out


def _slice(queue: Sequence[Any], cursor: int, n: int) -> tuple[list[Any], int]:
    if not queue:
        return [], 0
    cursor = cursor % len(queue)
    picked = [queue[(cursor + i) % len(queue)] for i in range(min(n, len(queue)))]
    return picked, (cursor + len(picked)) % len(queue)


async def collect(
    http: httpx.AsyncClient,
    cfg: StrategyConfig,
    jobs: Sequence[tuple[str, str]],
    *,
    timespan: str = DEFAULT_TIMESPAN,
    min_interval_s: float = GDELT_MIN_INTERVAL_S,
    sleep: Sleep = asyncio.sleep,
    now: datetime | None = None,
) -> list[PostDraft]:
    """Run the artlist queries serially, ``min_interval_s`` apart, and normalise the articles."""
    drafts: list[PostDraft] = []
    for i, (rid, query) in enumerate(jobs):
        if i:
            await sleep(min_interval_s)
        payload = await fetch_artlist(http, cfg.feeds.gdelt_doc, query, timespan=timespan)
        if payload is None:
            continue
        drafts.extend(parse_artlist(payload, rid, now=now))
    return drafts


async def run_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    http: httpx.AsyncClient | None = None,
    gazetteer: Gazetteer | None = None,
    max_requests: int = MAX_REQUESTS_PER_RUN,
    timespan: str = DEFAULT_TIMESPAN,
    sleep: Sleep = asyncio.sleep,
    now: datetime | None = None,
) -> int:
    """One pass over the next ``max_requests`` races (round-robin); returns new Post rows written."""
    now = now or utc_now()
    gazetteer = gazetteer or load_gazetteer(session_factory)
    queue = _race_queue(gazetteer)
    if not queue:
        log.info("gdelt: no races with candidate names; nothing to query")
        return 0
    with session_factory() as s:
        cursor = int(kv_get(s, KV_CURSOR, 0) or 0)
    jobs, next_cursor = _slice(queue, cursor, max_requests)
    owns = http is None
    client = http or httpx.AsyncClient(headers={"User-Agent": "sigcup/0.1 (+research)"})
    try:
        drafts = await collect(client, cfg, jobs, timespan=timespan, sleep=sleep, now=now)
    finally:
        if owns:
            await client.aclose()
    n = write_posts(session_factory, drafts, gazetteer, require_link=True, now=now)
    with session_factory() as s:
        kv_set(s, KV_CURSOR, next_cursor)
    log.info("gdelt: %d races queried, %d articles, %d new posts", len(jobs), len(drafts), n)
    return n


async def volume_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    race_ids: Sequence[str],
    *,
    http: httpx.AsyncClient | None = None,
    gazetteer: Gazetteer | None = None,
    sleep: Sleep = asyncio.sleep,
    timespan: str = "7d",
) -> int:
    """Store ``timelinevol`` series in KV (``gdelt:timelinevol:<race_id>``) for the given races; returns races stored."""
    gazetteer = gazetteer or load_gazetteer(session_factory)
    owns = http is None
    client = http or httpx.AsyncClient()
    stored = 0
    try:
        for i, rid in enumerate(race_ids):
            q = build_query(gazetteer.search_terms(rid))
            if not q:
                continue
            if i:
                await sleep(GDELT_MIN_INTERVAL_S)
            payload = await fetch_timelinevol(client, cfg.feeds.gdelt_doc, q, timespan=timespan)
            series = parse_timelinevol(payload) if payload is not None else []
            if not series:
                continue
            with session_factory() as s:
                kv_set(s, KV_VOLUME_PREFIX + rid, {"points": [[ts.isoformat(), v] for ts, v in series]})
            stored += 1
    finally:
        if owns:
            await client.aclose()
    return stored
