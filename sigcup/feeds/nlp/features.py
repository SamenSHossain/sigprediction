"""Daily per-(platform, race) net stance and z-score against the trailing sentiment.baseline_days.

For every UTC day in the window and every (platform, race_id):

* ``n_posts``        distinct posts with a Stance row, excluding posts with ``bot_score > 0.7`` and
                     stance rows flagged ``is_news_or_poll`` (headlines carry no opinion); a text
                     posted more than ``DUP_COPIES_MAX`` times in the window (a copy-paste campaign)
                     counts as ONE post per (platform, race, day) - the ingest-time bot score cannot
                     catch it, since it is fixed when the first copies arrive;
* ``net_stance_dem`` (support_D + oppose_R - oppose_D - support_R) / n_posts, in [-1, 1];
* ``z_score``        (net - mean) / max(std, STD_FLOOR) over the days in the trailing
                     ``baseline_days`` window that have data (at least MIN_BASELINE_DAYS of them),
                     else None.

Z-scoring each (platform, race) series against its *own* baseline is what makes platform skew
(Bluesky left, X right) drop out: only changes count. Rows are upserted on (platform, race_id, day).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select

from sigcup.common.time import ensure_utc, utc_now
from sigcup.db.models import Feature, Post, Stance
from sigcup.feeds.nlp.botfilter import BOT_THRESHOLD
from sigcup.settings import Secrets, StrategyConfig

log = logging.getLogger(__name__)

MIN_BASELINE_DAYS = 3
STD_FLOOR = 0.05
DUP_COPIES_MAX = 3  # a text_hash seen more often than this in the window counts once per (platform, race, day)

Record = tuple[str, str, str, str, str, str]
"""(platform, race_id, day 'YYYY-MM-DD', post_id, party, stance)."""


def stance_value(party: str, stance: str) -> int:
    """+1 for a pro-Democratic reading, -1 for pro-Republican, 0 otherwise (neutral/unclear/independents)."""
    p, st = (party or "").upper(), (stance or "").lower()
    if p == "D":
        return {"support": 1, "oppose": -1}.get(st, 0)
    if p == "R":
        return {"support": -1, "oppose": 1}.get(st, 0)
    return 0


def aggregate(records: Iterable[Record]) -> dict[tuple[str, str, str], tuple[int, float]]:
    """``{(platform, race_id, day): (n_posts, net_stance_dem)}`` from stance records.

    A post contributes once to ``n_posts``; its stance rows are summed and clipped to [-1, 1] so a
    post about both candidates cannot count twice.
    """
    posts: dict[tuple[str, str, str], dict[str, int]] = {}
    for platform, race_id, day, post_id, party, stance in records:
        bucket = posts.setdefault((platform, race_id, day), {})
        bucket[post_id] = bucket.get(post_id, 0) + stance_value(party, stance)
    out: dict[tuple[str, str, str], tuple[int, float]] = {}
    for key, per_post in posts.items():
        n = len(per_post)
        total = sum(max(-1, min(1, v)) for v in per_post.values())
        out[key] = (n, total / n if n else 0.0)
    return out


def zscores(
    series: Mapping[str, float],
    baseline_days: int,
    *,
    min_days: int = MIN_BASELINE_DAYS,
    std_floor: float = STD_FLOOR,
) -> dict[str, float | None]:
    """Z-score of each day's value against the trailing ``baseline_days`` days (strictly before it).

    ``series`` maps 'YYYY-MM-DD' -> value; days without data are simply absent from the baseline.
    """
    out: dict[str, float | None] = {}
    days = sorted(series)
    parsed = {d: datetime.strptime(d, "%Y-%m-%d") for d in days}
    for d in days:
        start = parsed[d] - timedelta(days=baseline_days)
        base = [series[b] for b in days if start <= parsed[b] < parsed[d]]
        if len(base) < min_days:
            out[d] = None
            continue
        mean = sum(base) / len(base)
        var = sum((x - mean) ** 2 for x in base) / len(base)
        std = max(math.sqrt(var), std_floor)
        out[d] = round((series[d] - mean) / std, 4)
    return out


def compute(records: Iterable[Record], baseline_days: int) -> dict[tuple[str, str, str], tuple[int, float, float | None]]:
    """Pure pipeline: records -> ``{(platform, race_id, day): (n_posts, net, z)}``."""
    agg = aggregate(records)
    series: dict[tuple[str, str], dict[str, float]] = {}
    for (platform, race_id, day), (_n, net) in agg.items():
        series.setdefault((platform, race_id), {})[day] = net
    z_by_key: dict[tuple[str, str, str], float | None] = {}
    for (platform, race_id), s in series.items():
        for day, z in zscores(s, baseline_days).items():
            z_by_key[(platform, race_id, day)] = z
    return {k: (n, net, z_by_key.get(k)) for k, (n, net) in agg.items()}


def load_records(s: Any, since: datetime) -> list[Record]:
    """Stance rows joined to posts since ``since``; bot-like posts and news/poll items are excluded, and
    a heavily duplicated text (more than ``DUP_COPIES_MAX`` posts in the window) keeps only its earliest
    post per (platform, race, day)."""
    rows = s.execute(
        select(Stance.race_id, Stance.party, Stance.stance, Post.post_id, Post.platform, Post.ts, Post.text_hash)
        .join(Post, Stance.post_id == Post.post_id)
        .where(Post.ts >= since, Stance.is_news_or_poll.is_(False))
        .order_by(Post.ts, Post.post_id)
    ).all()
    bot_scores = dict(s.execute(select(Post.post_id, Post.bot_score).where(Post.ts >= since)).all())
    heavy = set(
        s.execute(
            select(Post.text_hash).where(Post.ts >= since).group_by(Post.text_hash).having(func.count() > DUP_COPIES_MAX)
        ).scalars()
    )
    representative: dict[tuple[str, str, str, str], str] = {}
    out: list[Record] = []
    for race_id, party, stance, post_id, platform, ts, th in rows:
        score = bot_scores.get(post_id)
        if score is not None and score > BOT_THRESHOLD:
            continue
        day = ensure_utc(ts).strftime("%Y-%m-%d")
        if th in heavy and representative.setdefault((platform, race_id, day, th), post_id) != post_id:
            continue
        out.append((platform, race_id, day, post_id, party, stance))
    return out


def upsert_features(
    s: Any, computed: Mapping[tuple[str, str, str], tuple[int, float, float | None]], now: datetime
) -> int:
    """Insert or update Feature rows; returns the number touched."""
    existing = {
        (f.platform, f.race_id, f.day): f
        for f in s.execute(select(Feature).where(Feature.day.in_({k[2] for k in computed}))).scalars().all()
    }
    n = 0
    for (platform, race_id, day), (n_posts, net, z) in computed.items():
        row = existing.get((platform, race_id, day))
        if row is None:
            s.add(
                Feature(
                    platform=platform, race_id=race_id, day=day, n_posts=n_posts,
                    net_stance_dem=round(net, 4), z_score=z, computed_at=now,
                )
            )
        else:
            row.n_posts, row.net_stance_dem, row.z_score, row.computed_at = n_posts, round(net, 4), z, now
        n += 1
    return n


async def run_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    now: datetime | None = None,
    window_days: int | None = None,
) -> int:
    """Recompute daily features over the last ``window_days`` (default 2 x baseline_days + 1). Returns rows touched."""
    now = now or utc_now()
    baseline = max(1, int(cfg.sentiment.baseline_days))
    window = window_days or (2 * baseline + 1)
    since = (now - timedelta(days=window)).replace(hour=0, minute=0, second=0, microsecond=0)
    with session_factory() as s:
        records = load_records(s, since)
        computed = compute(records, baseline)
        n = upsert_features(s, computed, now)
    log.info("features: %d stance records -> %d (platform, race, day) rows", len(records), n)
    return n
