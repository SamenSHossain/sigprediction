"""Heuristic bot score in [0,1]: account age, posting rate, duplicate text hash frequency, handle pattern.

``score(post)`` is monotone in every risk factor: a younger account, a higher posting rate, more
duplicates of the same text, or a more bot-like handle never *lowers* the score. The components are
additive and clamped; it takes several signals together (a young account posting duplicates at a high
rate from a numbered handle) to cross the 0.7 threshold that features.py uses to drop a post.

The score is computed once, at ingest, from the counts known then: the first copies of a copy-paste
campaign score as originals and duplication alone tops out at +0.40. Coordinated duplicates are
therefore handled at aggregation time instead (features.py counts a heavily duplicated text once per
platform, race and day). No network, no DB: the caller supplies the two counts that need a query
(``dup_count``, ``author_posts_24h``).
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

from sigcup.common.time import ensure_utc, utc_now

BOT_THRESHOLD = 0.7  # posts above this are excluded from stance shares (features.py)

_TRAILING_DIGITS = re.compile(r"\d{5,}$")
_MOSTLY_DIGITS = re.compile(r"^[A-Za-z]{0,3}\d{6,}[A-Za-z]{0,3}$")
_RANDOM_SUFFIX = re.compile(r"^[A-Za-z]+[_.]?[A-Za-z]*\d{8,}$")
_URL = re.compile(r"https?://\S+")
_HASHTAG = re.compile(r"(?<!\w)#\w+")
_MENTION = re.compile(r"(?<!\w)@\w+")


def _get(post: Any, key: str, default: Any = None) -> Any:
    if post is None:
        return default
    if isinstance(post, dict):
        return post.get(key, default)
    return getattr(post, key, default)


def account_age_component(created_at: datetime | None, now: datetime) -> float:
    """0 for accounts older than ~180 days, up to 0.35 for accounts under a week old; 0.05 if unknown."""
    if created_at is None:
        return 0.05
    age_days = max(0.0, (now - ensure_utc(created_at)).total_seconds() / 86400.0)
    if age_days < 7:
        return 0.35
    if age_days < 30:
        return 0.25
    if age_days < 90:
        return 0.12
    if age_days < 180:
        return 0.05
    return 0.0


def posting_rate_component(posts_24h: int | None) -> float:
    """Saturating log curve: 20 posts/day ~0.08, 50 ~0.15, 100 ~0.22, 300+ ~0.30."""
    if not posts_24h or posts_24h <= 5:
        return 0.0
    return min(0.30, 0.08 * math.log2(posts_24h / 10.0 + 1.0))


def duplicate_component(dup_count: int | None) -> float:
    """Posts sharing this text hash: 1 extra copy ~0.08, 3 ~0.16, 10 ~0.28, 30+ = 0.40."""
    if not dup_count or dup_count <= 0:
        return 0.0
    return min(0.40, 0.08 * math.log2(dup_count + 1.0))


def handle_component(handle: str | None) -> float:
    """Handles like ``maga_patriot48291734`` or ``jd93847561`` score up to 0.25; names score 0."""
    if not handle:
        return 0.0
    h = str(handle).strip().lstrip("@")
    if not h:
        return 0.0
    comp = 0.0
    if _MOSTLY_DIGITS.match(h) or _RANDOM_SUFFIX.match(h):
        comp = 0.25
    elif _TRAILING_DIGITS.search(h):
        comp = 0.15
    digits = sum(ch.isdigit() for ch in h)
    if digits >= 4:
        comp = max(comp, 0.10 + 0.02 * min(digits - 4, 5))
    return min(0.25, comp)


def text_component(text: str | None) -> float:
    """Spammy shape: many hashtags/mentions, link-only posts, very short shouting."""
    if not text:
        return 0.0
    t = str(text)
    comp = 0.0
    n_tags = len(_HASHTAG.findall(t))
    n_mentions = len(_MENTION.findall(t))
    if n_tags >= 5:
        comp += 0.10
    elif n_tags >= 3:
        comp += 0.05
    if n_mentions >= 5:
        comp += 0.08
    stripped = _URL.sub("", t).strip()
    if _URL.search(t) and len(stripped) < 15:
        comp += 0.08
    letters = [ch for ch in stripped if ch.isalpha()]
    if len(letters) >= 20 and sum(ch.isupper() for ch in letters) / len(letters) > 0.8:
        comp += 0.05
    return min(0.20, comp)


def score(
    post: Any,
    *,
    dup_count: int | None = None,
    author_posts_24h: int | None = None,
    now: datetime | None = None,
) -> float:
    """Bot likelihood in [0, 1] for a post-like object (dict, Post row or PostDraft).

    Fields read (all optional): ``author_handle``, ``author_created_at``, ``text``. ``dup_count`` is the
    number of *other* posts with the same text hash; ``author_posts_24h`` the author's posts in the
    trailing day. Platforms without author metadata (rss, gdelt) land near 0.05.
    """
    now = now or utc_now()
    platform = str(_get(post, "platform", "") or "")
    if platform in {"rss", "gdelt"}:
        return 0.0
    total = (
        account_age_component(_get(post, "author_created_at"), now)
        + posting_rate_component(author_posts_24h)
        + duplicate_component(dup_count)
        + handle_component(_get(post, "author_handle"))
        + text_component(_get(post, "text"))
    )
    return round(max(0.0, min(1.0, total)), 4)
