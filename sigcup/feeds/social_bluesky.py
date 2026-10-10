"""Bluesky: Jetstream websocket firehose filtered client-side by the gazetteer, plus searchPosts backfill
with an app password. Writes Post rows (platform='bluesky').

* **Jetstream** (``wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post``,
  free, no auth) is consumed for a bounded slice per ``run_once`` (``duration_s``, 60 s by default) and
  every ``app.bsky.feed.post`` create is kept only when the gazetteer links it to a race. The websocket
  factory is injectable (``connect=``) so tests feed canned messages.
* **searchPosts backfill** runs only when ``bluesky_handle`` + ``bluesky_app_password`` are configured:
  unauthenticated ``public.api.bsky.app`` calls 403 (mid-2026), so the ``atproto`` client logs in and
  queries ``app.bsky.feed.searchPosts`` with each race's candidate names, a few races per cycle (KV
  cursor) because the route hits a wall after ~10 requests.

Post rows: ``post_id`` = ``bluesky:<did>/<rkey>``, ``url`` = ``https://bsky.app/profile/<did>/post/<rkey>``,
``author_id`` = did, ``author_handle`` from search results (Jetstream carries no handle).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from sigcup.common.time import utc_now
from sigcup.db.repo import kv_get, kv_set
from sigcup.feeds.nlp.gazetteer import Gazetteer, PostDraft, load_gazetteer, parse_ts, pick, write_posts_async
from sigcup.settings import Secrets, StrategyConfig

log = logging.getLogger(__name__)

JETSTREAM_URL = "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post"
POST_COLLECTION = "app.bsky.feed.post"
DEFAULT_DURATION_S = 60.0
MAX_POSTS_PER_SLICE = 5000
SEARCH_LIMIT = 25
SEARCH_QUERIES_PER_RUN = 8
SEARCH_PAUSE_S = 1.0
KV_SEARCH_CURSOR = "bluesky:search_cursor"


# ---------------------------------------------------------------------------------------------
# Normalisers (pure)
# ---------------------------------------------------------------------------------------------


def _split_uri(uri: str) -> tuple[str, str] | None:
    """``at://did/app.bsky.feed.post/rkey`` -> (did, rkey)."""
    if not uri or not uri.startswith("at://"):
        return None
    parts = uri[len("at://") :].split("/")
    if len(parts) != 3:
        return None
    return parts[0], parts[2]


def post_url(did: str, rkey: str) -> str:
    return f"https://bsky.app/profile/{did}/post/{rkey}"


def parse_jetstream_message(raw: str | bytes | dict[str, Any], *, now: datetime | None = None) -> PostDraft | None:
    """A Jetstream ``commit`` creating an ``app.bsky.feed.post`` -> draft; anything else -> None."""
    msg: Any = raw
    if isinstance(raw, bytes | str):
        try:
            msg = json.loads(raw)
        except (ValueError, TypeError):
            return None
    if not isinstance(msg, dict):
        return None
    if pick(msg, "kind") != "commit":
        return None
    commit = pick(msg, "commit", default={}) or {}
    if pick(commit, "operation", default="create") != "create":
        return None
    if pick(commit, "collection", default=POST_COLLECTION) != POST_COLLECTION:
        return None
    record = pick(commit, "record", default={}) or {}
    text = str(pick(record, "text", default="") or "").strip()
    did = str(pick(msg, "did", "repo", default="") or "")
    rkey = str(pick(commit, "rkey", default="") or "")
    if not text or not did or not rkey:
        return None
    langs = pick(record, "langs")
    if isinstance(langs, list) and langs and not any(str(lang).lower().startswith("en") for lang in langs):
        return None
    ts = parse_ts(pick(record, "createdAt", "created_at")) or parse_ts(pick(msg, "time_us")) or now or utc_now()
    return PostDraft(
        post_id=f"bluesky:{did}/{rkey}",
        platform="bluesky",
        ts=ts,
        text=text,
        author_id=did,
        author_handle=None,
        url=post_url(did, rkey),
    )


def parse_search_post(obj: Any, *, now: datetime | None = None) -> PostDraft | None:
    """An ``app.bsky.feed.defs#postView`` (atproto model or dict) -> draft."""
    uri = str(pick(obj, "uri", default="") or "")
    ids = _split_uri(uri)
    record = pick(obj, "record", "value", default={}) or {}
    text = str(pick(record, "text", default="") or "").strip()
    if ids is None or not text:
        return None
    did, rkey = ids
    author = pick(obj, "author", default={}) or {}
    return PostDraft(
        post_id=f"bluesky:{did}/{rkey}",
        platform="bluesky",
        ts=parse_ts(pick(record, "created_at", "createdAt")) or parse_ts(pick(obj, "indexed_at", "indexedAt")) or now or utc_now(),
        text=text,
        author_id=str(pick(author, "did", default=did) or did),
        author_handle=(str(pick(author, "handle", default="") or "") or None),
        author_created_at=parse_ts(pick(author, "created_at", "createdAt")),
        url=post_url(did, rkey),
    )


# ---------------------------------------------------------------------------------------------
# Jetstream
# ---------------------------------------------------------------------------------------------


def _default_connect(url: str) -> Any:
    import websockets

    return websockets.connect(url, max_size=2**20, open_timeout=20)


async def consume_jetstream(
    gazetteer: Gazetteer | None,
    duration_s: float = DEFAULT_DURATION_S,
    *,
    url: str = JETSTREAM_URL,
    connect: Callable[[str], Any] | None = None,
    max_posts: int = MAX_POSTS_PER_SLICE,
    now: datetime | None = None,
) -> list[PostDraft]:
    """Read the firehose for ``duration_s`` seconds; keep posts the gazetteer links (all posts if no gazetteer).

    ``connect(url)`` must return an async context manager yielding an async iterable of messages.
    Connection errors are logged and yield an empty list.
    """
    connect = connect or _default_connect
    deadline = time.monotonic() + max(0.0, duration_s)
    kept: list[PostDraft] = []
    seen = 0
    try:
        async with connect(url) as ws:
            it = ws.__aiter__()
            while len(kept) < max_posts:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = await asyncio.wait_for(it.__anext__(), timeout=remaining)
                except StopAsyncIteration:
                    break
                except TimeoutError:
                    break
                seen += 1
                d = parse_jetstream_message(raw, now=now)
                if d is None:
                    continue
                if gazetteer is not None:
                    d.race_ids = gazetteer.link_race_ids(d.text, d.author_handle)
                    if not d.race_ids:
                        continue
                kept.append(d)
    except Exception as exc:  # noqa: BLE001 - websockets raises many types; none is fatal for the loop
        log.warning("bluesky: jetstream unavailable: %s", type(exc).__name__)
    log.info("bluesky: jetstream slice saw %d events, kept %d", seen, len(kept))
    return kept


# ---------------------------------------------------------------------------------------------
# searchPosts backfill
# ---------------------------------------------------------------------------------------------


def _make_search_client(secrets: Secrets | None) -> Any | None:
    """A logged-in ``atproto.Client`` or None when credentials are missing or login fails."""
    if secrets is None or not secrets.bluesky_handle or not secrets.bluesky_app_password:
        return None
    try:
        from atproto import Client

        client = Client()
        client.login(secrets.bluesky_handle, secrets.bluesky_app_password)
        return client
    except Exception as exc:  # noqa: BLE001
        log.warning("bluesky: login failed: %s", type(exc).__name__)
        return None


def search_sync(client: Any, query: str, *, limit: int = SEARCH_LIMIT, since: str | None = None) -> list[Any]:
    """``app.bsky.feed.searchPosts`` for one query; returns raw post views (empty on error)."""
    params: dict[str, Any] = {"q": query, "limit": int(limit), "sort": "latest"}
    if since:
        params["since"] = since
    try:
        resp = client.app.bsky.feed.search_posts(params)
    except Exception as exc:  # noqa: BLE001
        log.warning("bluesky: searchPosts failed: %s", type(exc).__name__)
        return []
    return list(pick(resp, "posts", default=[]) or [])


def backfill_sync(
    client: Any,
    gazetteer: Gazetteer,
    race_ids: Sequence[str],
    *,
    limit: int = SEARCH_LIMIT,
    pause_s: float = SEARCH_PAUSE_S,
    now: datetime | None = None,
) -> list[PostDraft]:
    """One search per candidate name of the given races; drafts carry the race id."""
    out: list[PostDraft] = []
    first = True
    for rid in race_ids:
        for term in gazetteer.search_terms(rid):
            if not first and pause_s:
                time.sleep(pause_s)
            first = False
            for view in search_sync(client, f'"{term}"', limit=limit):
                d = parse_search_post(view, now=now)
                if d is not None:
                    d.race_ids = [rid]
                    out.append(d)
    return out


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


async def run_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    duration_s: float = DEFAULT_DURATION_S,
    jetstream_url: str = JETSTREAM_URL,
    connect: Callable[[str], Any] | None = None,
    search_client: Any | None = None,
    search_queries: int = SEARCH_QUERIES_PER_RUN,
    search_pause_s: float = SEARCH_PAUSE_S,
    gazetteer: Gazetteer | None = None,
    now: datetime | None = None,
) -> int:
    """Jetstream slice (if ``duration_s`` > 0) plus a searchPosts backfill slice (if credentials); returns new rows."""
    now = now or utc_now()
    gazetteer = gazetteer or load_gazetteer(session_factory)
    if not gazetteer.candidates:
        log.info("bluesky: empty gazetteer; nothing to filter on")
        return 0
    written = 0
    if duration_s > 0:
        drafts = await consume_jetstream(gazetteer, duration_s, url=jetstream_url, connect=connect, now=now)
        written += await write_posts_async(session_factory, drafts, gazetteer, require_link=True, now=now)

    client = search_client if search_client is not None else _make_search_client(secrets)
    if client is not None and search_queries > 0:
        queue = [rid for rid in gazetteer.race_ids() if gazetteer.search_terms(rid)]
        if queue:
            with session_factory() as s:
                cursor = int(kv_get(s, KV_SEARCH_CURSOR, 0) or 0) % len(queue)
            picked: list[str] = []
            budget = search_queries
            i = cursor
            while budget > 0 and len(picked) < len(queue):
                rid = queue[i % len(queue)]
                picked.append(rid)
                budget -= max(1, len(gazetteer.search_terms(rid)))
                i += 1
            drafts = await asyncio.to_thread(backfill_sync, client, gazetteer, picked, pause_s=search_pause_s, now=now)
            written += await write_posts_async(session_factory, drafts, gazetteer, require_link=True, now=now)
            with session_factory() as s:
                kv_set(s, KV_SEARCH_CURSOR, i % len(queue))
    log.info("bluesky: %d new posts", written)
    return written
