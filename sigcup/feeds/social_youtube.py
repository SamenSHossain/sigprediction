"""YouTube Data API v3: comments on configured channels' recent uploads (playlistItems + commentThreads,
never search.list); writes Post rows (platform='youtube').

Channels come from config/sources.yaml (``global.youtube_channels`` and ``races[race_id].youtube_channels``,
each entry a channel id ``UC...`` or ``{id, race_id}``). A channel's uploads playlist is derived locally
(``UU`` + channel id tail), listed with ``playlistItems.list`` (1 quota unit) and each recent video's
newest comments are read with ``commentThreads.list`` (1 unit per page of up to 100). ``search.list``
costs 100 units and is never used; a run is capped at ``max_units`` so the 10,000 units/day default
quota is never exhausted by one process. Channels are read round-robin from a KV cursor
(``youtube:cursor``), so a list longer than one run's budget is still covered over successive runs.

Post rows: ``post_id`` = ``youtube:<comment id>``, ``text`` = the comment, ``url`` deep-links the comment,
``author_handle`` = display name, ``author_id`` = the commenter's channel id. Comments on a race-specific
channel are tagged with that race; comments on global channels are kept only when the gazetteer links
them (the video title is passed as extra linking context, not stored).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
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
    write_posts_async,
)
from sigcup.settings import Secrets, StrategyConfig

log = logging.getLogger(__name__)

API_BASE = "https://www.googleapis.com/youtube/v3"
HTTP_TIMEOUT_S = 20.0
MAX_UNITS_PER_RUN = 200
MAX_VIDEOS_PER_CHANNEL = 5
MAX_VIDEO_AGE_DAYS = 14
COMMENTS_PER_PAGE = 100
KV_CURSOR = "youtube:cursor"


def load_sources() -> dict[str, Any]:
    from sigcup.settings import load_yaml

    raw = load_yaml("sources")
    return dict(raw) if isinstance(raw, Mapping) else {}


def channel_specs(sources: Mapping[str, Any]) -> list[tuple[str, str | None]]:
    """``[(channel_id, race_id | None)]`` from sources.yaml; de-duplicated."""
    out: list[tuple[str, str | None]] = []
    seen: set[str] = set()

    def add(entry: Any, race_id: str | None) -> None:
        cid = str(pick(entry, "id", "channel_id", "channel", default=entry if isinstance(entry, str) else "") or "").strip()
        rid = str(pick(entry, "race_id", default=race_id or "") or "") or None
        if cid and cid not in seen:
            seen.add(cid)
            out.append((cid, rid))

    for e in pick(sources, "global.youtube_channels", default=[]) or []:
        add(e, None)
    for rid, spec in (pick(sources, "races", default={}) or {}).items():
        for e in pick(spec, "youtube_channels", default=[]) or []:
            add(e, str(rid))
    return out


def uploads_playlist_id(channel_id: str) -> str:
    """``UCxxxx`` -> ``UUxxxx`` (the channel's uploads playlist); other ids are assumed to be playlists."""
    cid = channel_id.strip()
    return "UU" + cid[2:] if cid.startswith("UC") and len(cid) > 2 else cid


def parse_playlist_items(payload: Any) -> list[dict[str, Any]]:
    """``[{video_id, title, published_at}]`` from a playlistItems.list response."""
    out: list[dict[str, Any]] = []
    for item in pick(payload, "items", default=[]) or []:
        vid = pick(item, "contentDetails.videoId", "snippet.resourceId.videoId", "id")
        if not vid:
            continue
        out.append(
            {
                "video_id": str(vid),
                "title": strip_html(pick(item, "snippet.title", default="")),
                "published_at": parse_ts(pick(item, "contentDetails.videoPublishedAt", "snippet.publishedAt")),
            }
        )
    return out


def parse_comment_threads(payload: Any, video: Mapping[str, Any] | None = None, *, race_id: str | None = None, now: datetime | None = None) -> list[PostDraft]:
    """Top-level comments from a commentThreads.list response -> drafts."""
    now = now or utc_now()
    out: list[PostDraft] = []
    video_id = str(pick(video, "video_id", default="") or "") if video else ""
    title = str(pick(video, "title", default="") or "") if video else ""
    for item in pick(payload, "items", default=[]) or []:
        sn = pick(item, "snippet.topLevelComment.snippet", "snippet", default={}) or {}
        cid = str(pick(item, "snippet.topLevelComment.id", "id", default="") or "")
        text = strip_html(pick(sn, "textOriginal", "textDisplay", default=""))
        if not cid or not text:
            continue
        vid = str(pick(sn, "videoId", default=video_id) or video_id)
        out.append(
            PostDraft(
                post_id=f"youtube:{cid}",
                platform="youtube",
                ts=parse_ts(pick(sn, "publishedAt", "updatedAt")) or now,
                text=text[:4000],
                author_id=(str(pick(sn, "authorChannelId.value", "authorChannelId", default="") or "") or None),
                author_handle=(strip_html(pick(sn, "authorDisplayName", default="")) or None),
                url=f"https://www.youtube.com/watch?v={vid}&lc={cid}" if vid else None,
                race_ids=[race_id] if race_id else [],
                link_text=title or None,
            )
        )
    return out


async def _get(http: httpx.AsyncClient, path: str, params: dict[str, Any]) -> Any | None:
    try:
        r = await http.get(f"{API_BASE}/{path}", params=params, timeout=HTTP_TIMEOUT_S)
    except httpx.HTTPError as exc:
        log.warning("youtube: %s failed: %s", path, type(exc).__name__)
        return None
    if r.status_code == 403:
        log.warning("youtube: %s forbidden (quota exhausted or comments disabled)", path)
        return None
    if r.status_code != 200:
        log.warning("youtube: %s HTTP %s", path, r.status_code)
        return None
    try:
        return r.json()
    except ValueError:
        return None


async def run_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    http: httpx.AsyncClient | None = None,
    api_key: str | None = None,
    sources: Mapping[str, Any] | None = None,
    max_units: int = MAX_UNITS_PER_RUN,
    max_videos_per_channel: int = MAX_VIDEOS_PER_CHANNEL,
    max_video_age_days: int = MAX_VIDEO_AGE_DAYS,
    gazetteer: Gazetteer | None = None,
    now: datetime | None = None,
) -> int:
    """List recent uploads per channel, read their newest comments; returns new Post rows (0 when unconfigured)."""
    now = now or utc_now()
    key = (api_key if api_key is not None else (secrets.youtube_api_key if secrets else "")) or ""
    if not key.strip():
        log.info("youtube: no youtube_api_key; skipping")
        return 0
    channels = channel_specs(sources if sources is not None else load_sources())
    if not channels:
        log.info("youtube: no channels configured in config/sources.yaml")
        return 0
    gazetteer = gazetteer or load_gazetteer(session_factory)
    cutoff = now - timedelta(days=max_video_age_days)
    with session_factory() as s:
        cursor = int(kv_get(s, KV_CURSOR, 0) or 0) % len(channels)
    owns = http is None
    client = http or httpx.AsyncClient()
    units = 0
    total = 0
    read = 0
    try:
        for channel_id, race_id in channels[cursor:] + channels[:cursor]:
            if units >= max_units:
                break
            units += 1
            read += 1
            playlist = await _get(
                client,
                "playlistItems",
                {"part": "contentDetails,snippet", "playlistId": uploads_playlist_id(channel_id), "maxResults": 10, "key": key},
            )
            if playlist is None:
                continue
            videos = [v for v in parse_playlist_items(playlist) if v["published_at"] is None or v["published_at"] >= cutoff]
            for video in videos[:max_videos_per_channel]:
                if units >= max_units:
                    break
                units += 1
                threads = await _get(
                    client,
                    "commentThreads",
                    {
                        "part": "snippet",
                        "videoId": video["video_id"],
                        "maxResults": COMMENTS_PER_PAGE,
                        "order": "time",
                        "textFormat": "plainText",
                        "key": key,
                    },
                )
                if threads is None:
                    continue
                drafts = parse_comment_threads(threads, video, race_id=race_id, now=now)
                total += await write_posts_async(session_factory, drafts, gazetteer, require_link=True, now=now)
    finally:
        if owns:
            await client.aclose()
        with session_factory() as s:
            kv_set(s, KV_CURSOR, (cursor + read) % len(channels))
    log.info("youtube: %d/%d channels, %d quota units, %d new posts", read, len(channels), units, total)
    return total
