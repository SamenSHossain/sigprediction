"""RSS collector over config/sources.yaml (local papers, Substacks); writes Post rows (platform='rss').

``sources.yaml`` shape: ``global.rss`` (a list of feed URLs that apply to every race) and
``races[race_id].rss`` (feeds tied to one race). Entries from a race feed are always kept and tagged
with that race; entries from a global feed are kept only when the gazetteer links them to a race.
Feeds are fetched with httpx (conditional GET via ETag / Last-Modified remembered in KV) and parsed
with ``feedparser`` from the response body, so the parser is unit-testable on a local string.

Post rows: ``post_id`` = ``rss:<sha256(entry id or link)[:24]>``, ``text`` = title + ' ' + summary
(HTML stripped), ``ts`` from published/updated (else fetch time), ``author_handle`` = feed title.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import feedparser
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

HTTP_TIMEOUT_S = 20.0
MAX_ENTRIES_PER_FEED = 100
KV_PREFIX = "rss:feed:"


def load_sources() -> dict[str, Any]:
    """config/sources.yaml as a dict (empty when absent or malformed)."""
    from sigcup.settings import load_yaml

    raw = load_yaml("sources")
    return dict(raw) if isinstance(raw, Mapping) else {}


def feed_urls(sources: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    """``[(url, race_id | None)]`` from ``global.rss`` and ``races[race_id].rss``; de-duplicated, order kept."""
    out: list[tuple[str, str | None]] = []
    seen: set[str] = set()

    def add(url: Any, race_id: str | None) -> None:
        u = str(url or "").strip()
        if u and u not in seen:
            seen.add(u)
            out.append((u, race_id))

    for u in pick(sources, "global.rss", default=[]) or []:
        add(u, None)
    for rid, spec in (pick(sources, "races", default={}) or {}).items():
        for u in pick(spec, "rss", default=[]) or []:
            add(u, str(rid))
    return out


def entry_id(entry: Any) -> str | None:
    key = str(pick(entry, "id", "guid", "link", default="") or "").strip()
    if not key:
        return None
    return "rss:" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def _entry_summary(entry: Any) -> str:
    content = pick(entry, "content")
    if isinstance(content, list) and content:
        v = pick(content[0], "value", default="")
        if v:
            return strip_html(v)
    return strip_html(pick(entry, "summary", "description", "subtitle", default=""))


def parse_feed(
    content: str | bytes,
    *,
    race_id: str | None = None,
    source_url: str | None = None,
    now: datetime | None = None,
    max_entries: int = MAX_ENTRIES_PER_FEED,
) -> list[PostDraft]:
    """Parse a feed document (string or bytes) into drafts. Never raises on malformed feeds."""
    now = now or utc_now()
    parsed = feedparser.parse(content)
    feed_title = strip_html(pick(parsed, "feed.title", default="")) or (urlparse(source_url).netloc if source_url else "")
    out: list[PostDraft] = []
    for entry in (pick(parsed, "entries", default=[]) or [])[:max_entries]:
        pid = entry_id(entry)
        title = strip_html(pick(entry, "title", default=""))
        if pid is None or not title:
            continue
        summary = _entry_summary(entry)
        text = f"{title} {summary}".strip()[:4000]
        ts = (
            parse_ts(pick(entry, "published_parsed", "updated_parsed", "created_parsed"))
            or parse_ts(pick(entry, "published", "updated", "created"))
            or now
        )
        out.append(
            PostDraft(
                post_id=pid,
                platform="rss",
                ts=ts,
                text=text,
                author_handle=(feed_title or None) and feed_title[:128],
                url=str(pick(entry, "link", default="") or "")[:512] or None,
                race_ids=[race_id] if race_id else [],
                link_text=feed_title or None,
            )
        )
    return out


async def fetch_feed(
    http: httpx.AsyncClient, url: str, *, etag: str | None = None, modified: str | None = None
) -> tuple[bytes | None, str | None, str | None]:
    """Conditional GET: ``(body or None when unchanged/failed, etag, last_modified)``."""
    headers: dict[str, str] = {}
    if etag:
        headers["If-None-Match"] = etag
    if modified:
        headers["If-Modified-Since"] = modified
    try:
        r = await http.get(url, headers=headers, timeout=HTTP_TIMEOUT_S, follow_redirects=True)
    except httpx.HTTPError as exc:
        log.warning("rss: %s failed: %s", urlparse(url).netloc, type(exc).__name__)
        return None, etag, modified
    if r.status_code == 304:
        return None, etag, modified
    if r.status_code != 200:
        log.warning("rss: %s HTTP %s", urlparse(url).netloc, r.status_code)
        return None, etag, modified
    return r.content, r.headers.get("etag") or etag, r.headers.get("last-modified") or modified


def _kv_key(url: str) -> str:
    return KV_PREFIX + hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]


async def run_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    http: httpx.AsyncClient | None = None,
    sources: Mapping[str, Any] | None = None,
    gazetteer: Gazetteer | None = None,
    now: datetime | None = None,
) -> int:
    """Fetch every configured feed once; returns new Post rows written (0 when no feeds are configured)."""
    now = now or utc_now()
    urls = feed_urls(sources if sources is not None else load_sources())
    if not urls:
        log.info("rss: no feeds configured in config/sources.yaml")
        return 0
    gazetteer = gazetteer or load_gazetteer(session_factory)
    owns = http is None
    client = http or httpx.AsyncClient(headers={"User-Agent": "sigcup/0.1 (+research)"})
    total = 0
    try:
        for url, race_id in urls:
            with session_factory() as s:
                state = kv_get(s, _kv_key(url), {}) or {}
            body, etag, modified = await fetch_feed(client, url, etag=state.get("etag"), modified=state.get("modified"))
            if body is None:
                continue
            drafts = parse_feed(body, race_id=race_id, source_url=url, now=now)
            n = write_posts(session_factory, drafts, gazetteer, require_link=True, now=now)
            total += n
            with session_factory() as s:
                kv_set(s, _kv_key(url), {"etag": etag, "modified": modified, "fetched_at": now.isoformat()})
            log.debug("rss: %s -> %d entries, %d new", urlparse(url).netloc, len(drafts), n)
    finally:
        if owns:
            await client.aclose()
    log.info("rss: %d feeds, %d new posts", len(urls), total)
    return total
