"""Reddit via PRAW; DISABLED unless credentials exist (approval takes weeks). Writes Post rows (platform='reddit').

Reddit's Responsible Builder Policy (Nov 2025) closed self-service OAuth; new credentials need a manual
review that takes 2-4 weeks and unauthenticated ``.json`` endpoints 403. So this collector is scaffolded
but inert: ``run_once`` returns 0 *without importing praw* unless ``reddit_client_id``,
``reddit_client_secret`` and ``reddit_user_agent`` are all set (or a client is injected).

When enabled it reads, read-only, the newest submissions and comments of the state subreddits of the
races in the gazetteer (``r/Georgia``, ``r/NorthCarolina`` ...) plus the bettor subs ``r/PredictIt``,
``r/Polymarket`` and ``r/Kalshi`` (a flow indicator), ``max_subs`` per cycle round-robin, and keeps what
the gazetteer links. Post rows: ``post_id`` = ``reddit:t3_<id>`` / ``reddit:t1_<id>``, ``author_handle`` =
username, ``url`` = permalink. Author account age is not fetched (one extra request per author).

LinkedIn, Instagram and Facebook are NOT collected (no keyword API; scraping is a ToS breach).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from sigcup.common.time import utc_now
from sigcup.db.repo import kv_get, kv_set
from sigcup.feeds.nlp.gazetteer import (
    STATE_NAMES,
    Gazetteer,
    PostDraft,
    load_gazetteer,
    parse_ts,
    pick,
    write_posts_async,
)
from sigcup.settings import Secrets, StrategyConfig

log = logging.getLogger(__name__)

BETTOR_SUBS = ("PredictIt", "Polymarket", "Kalshi")
MAX_SUBS_PER_RUN = 12
SUBMISSIONS_PER_SUB = 50
COMMENTS_PER_SUB = 100
KV_CURSOR = "reddit:cursor"


def credentials_present(secrets: Secrets | None) -> bool:
    return bool(
        secrets is not None
        and (secrets.reddit_client_id or "").strip()
        and (secrets.reddit_client_secret or "").strip()
        and (secrets.reddit_user_agent or "").strip()
    )


def subreddits_for(gazetteer: Gazetteer) -> list[str]:
    """State subs for every state in the gazetteer (spaces removed: ``NorthCarolina``) plus the bettor subs."""
    states = sorted({m.state for m in gazetteer.races.values() if m.state})
    subs = [STATE_NAMES[st].replace(" ", "") for st in states if st in STATE_NAMES]
    return list(dict.fromkeys(subs + list(BETTOR_SUBS)))


def _author_name(obj: Any) -> str | None:
    author = pick(obj, "author")
    if author is None:
        return None
    name = pick(author, "name", default=None)
    return str(name) if name else (str(author) or None)


def normalise_submission(sub: Any, subreddit: str | None = None, *, now: datetime | None = None) -> PostDraft | None:
    """A PRAW ``Submission`` (or dict) -> draft; title + selftext is the text."""
    sid = str(pick(sub, "id", default="") or "")
    title = str(pick(sub, "title", default="") or "").strip()
    body = str(pick(sub, "selftext", "text", default="") or "").strip()
    if not sid or not title:
        return None
    permalink = str(pick(sub, "permalink", default="") or "")
    return PostDraft(
        post_id=f"reddit:t3_{sid}",
        platform="reddit",
        ts=parse_ts(pick(sub, "created_utc", "created")) or now or utc_now(),
        text=f"{title} {body}".strip()[:4000],
        author_handle=_author_name(sub),
        url=("https://www.reddit.com" + permalink) if permalink.startswith("/") else (permalink or None),
        link_text=f"r/{subreddit}" if subreddit else None,
    )


def normalise_comment(c: Any, subreddit: str | None = None, *, now: datetime | None = None) -> PostDraft | None:
    """A PRAW ``Comment`` (or dict) -> draft."""
    cid = str(pick(c, "id", default="") or "")
    body = str(pick(c, "body", "text", default="") or "").strip()
    if not cid or not body or body in {"[deleted]", "[removed]"}:
        return None
    permalink = str(pick(c, "permalink", default="") or "")
    return PostDraft(
        post_id=f"reddit:t1_{cid}",
        platform="reddit",
        ts=parse_ts(pick(c, "created_utc", "created")) or now or utc_now(),
        text=body[:4000],
        author_handle=_author_name(c),
        url=("https://www.reddit.com" + permalink) if permalink.startswith("/") else (permalink or None),
        link_text=f"r/{subreddit}" if subreddit else None,
    )


def collect_sync(reddit: Any, subs: Sequence[str], *, now: datetime | None = None) -> list[PostDraft]:
    """Newest submissions and comments per subreddit via a PRAW-like client (errors per sub are logged)."""
    out: list[PostDraft] = []
    for name in subs:
        try:
            sr = reddit.subreddit(name)
            for sub in sr.new(limit=SUBMISSIONS_PER_SUB):
                d = normalise_submission(sub, name, now=now)
                if d:
                    out.append(d)
            for c in sr.comments(limit=COMMENTS_PER_SUB):
                d = normalise_comment(c, name, now=now)
                if d:
                    out.append(d)
        except Exception as exc:  # noqa: BLE001 - prawcore raises many types; skip the sub
            log.warning("reddit: r/%s failed: %s", name, type(exc).__name__)
    return out


def _make_client(secrets: Secrets) -> Any | None:
    try:
        import praw

        return praw.Reddit(
            client_id=secrets.reddit_client_id,
            client_secret=secrets.reddit_client_secret,
            user_agent=secrets.reddit_user_agent,
            check_for_async=False,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("reddit: client init failed: %s", type(exc).__name__)
        return None


async def run_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    reddit: Any | None = None,
    max_subs: int = MAX_SUBS_PER_RUN,
    gazetteer: Gazetteer | None = None,
    now: datetime | None = None,
) -> int:
    """Returns 0 (without importing praw) unless credentials exist or a client is injected."""
    now = now or utc_now()
    if reddit is None:
        if not credentials_present(secrets):
            log.debug("reddit: disabled (no credentials)")
            return 0
        reddit = _make_client(secrets)  # type: ignore[arg-type]
        if reddit is None:
            return 0
    gazetteer = gazetteer or load_gazetteer(session_factory)
    subs = subreddits_for(gazetteer)
    if not subs:
        return 0
    with session_factory() as s:
        cursor = int(kv_get(s, KV_CURSOR, 0) or 0) % len(subs)
    picked = [subs[(cursor + i) % len(subs)] for i in range(min(max_subs, len(subs)))]
    drafts = await asyncio.to_thread(collect_sync, reddit, picked, now=now)
    n = await write_posts_async(session_factory, drafts, gazetteer, require_link=True, now=now)
    with session_factory() as s:
        kv_set(s, KV_CURSOR, (cursor + len(picked)) % len(subs))
    log.info("reddit: %d subs, %d items, %d new posts", len(picked), len(drafts), n)
    return n
