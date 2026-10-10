"""Event detector: turns feature z-score spikes (>= sentiment.spike_z on >= min_platforms) and GDELT volume
spikes into Event rows, classified soft/hard/unknown by an LLM call with the race context.

Two triggers, evaluated for the current UTC day on every ``run_once``:

* **social spike** - ``|z_score| >= spike_z`` on at least ``min_platforms`` distinct platforms the same
  day (negative spikes matter as much as positive ones: a leader's crowd turning is the signal);
* **GDELT volume** - today's gdelt Post count for the race exceeds ``GDELT_RATIO`` x the trailing
  ``baseline_days`` daily mean (and at least ``GDELT_MIN_ARTICLES`` articles).

Each trigger yields at most one Event per (race, source, day). ``kind`` comes from an LLM given the
race context and the day's top posts (hard = dropout, indictment, major endorsement, ballot ruling,
health; soft = a poll, a debate take, an ad, fundraising, chatter); without an API key or on any
failure the kind is ``unknown``. Events are written with ``acted=False``; the trader decides.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from sigcup.common.time import ensure_utc, utc_now
from sigcup.db.models import Event, Feature, Post
from sigcup.feeds.nlp.gazetteer import Gazetteer, load_gazetteer, pick
from sigcup.feeds.nlp.stance import _make_client, response_text
from sigcup.settings import Secrets, StrategyConfig

log = logging.getLogger(__name__)

SOURCE_SOCIAL = "social_spike"
SOURCE_GDELT = "gdelt_volume"
KINDS = ("soft", "hard", "unknown")
GDELT_RATIO = 3.0
GDELT_MIN_ARTICLES = 5
TOP_POSTS = 8

Classifier = Callable[[Mapping[str, Any], list[str], str], Awaitable[tuple[str, str, float]]]
"""``async (race_context, top_posts, trigger_summary) -> (kind, description, confidence)``."""

SYSTEM_PROMPT = (
    "You triage a burst of attention on a 2026 US midterm race for a trading desk. Given the race, what "
    "triggered the alert and the day's most representative posts, answer with ONLY a JSON object: "
    '{"kind": "hard"|"soft"|"unknown", "description": "<= 200 chars, what happened", "confidence": 0-1}. '
    "hard = an event that durably changes win probability (candidate withdrawal, death or serious health "
    "news, indictment or credible scandal with evidence, a major endorsement such as the President's, "
    "a ballot or court ruling, a party switch). soft = a single poll, a debate reaction, an ad, a "
    "fundraising number, a gaffe, pundit chatter, or market talk. unknown = cannot tell from the posts."
)


# ---------------------------------------------------------------------------------------------
# Detection (pure)
# ---------------------------------------------------------------------------------------------


def detect_social_spikes(
    features: Iterable[Feature], spike_z: float, min_platforms: int
) -> dict[str, list[Feature]]:
    """``{race_id: [Feature...]}`` for races whose |z| >= spike_z on >= min_platforms platforms."""
    by_race: dict[str, dict[str, Feature]] = {}
    for f in features:
        if f.z_score is None or abs(f.z_score) < spike_z:
            continue
        by_race.setdefault(f.race_id, {})[f.platform] = f
    return {rid: list(v.values()) for rid, v in by_race.items() if len(v) >= max(1, min_platforms)}


def detect_gdelt_spikes(
    counts: Mapping[tuple[str, str], int],
    today: str,
    baseline_days: int,
    *,
    ratio: float = GDELT_RATIO,
    min_articles: int = GDELT_MIN_ARTICLES,
) -> dict[str, tuple[int, float]]:
    """``{race_id: (today_count, trailing_mean)}`` where today > ratio x the trailing daily mean.

    ``counts`` maps (race_id, 'YYYY-MM-DD') -> article count. Days with no articles count as 0 in the
    mean (a quiet race that suddenly gets 5 articles is a spike); at least two baseline days are needed.
    """
    t = datetime.strptime(today, "%Y-%m-%d")
    base_days = [(t - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, baseline_days + 1)]
    races = {rid for rid, _day in counts}
    out: dict[str, tuple[int, float]] = {}
    for rid in races:
        today_n = counts.get((rid, today), 0)
        if today_n < min_articles:
            continue
        hist = [counts.get((rid, d), 0) for d in base_days]
        seen_days = sum(1 for d in base_days if (rid, d) in counts)
        if len(hist) < 2 or seen_days < 1:
            continue
        mean = sum(hist) / len(hist)
        if today_n > ratio * max(mean, 1e-9):
            out[rid] = (today_n, round(mean, 3))
    return out


def gdelt_counts(posts: Iterable[Post]) -> dict[tuple[str, str], int]:
    """(race_id, day) -> number of gdelt articles, from Post rows (race_ids is a JSON list)."""
    counts: dict[tuple[str, str], int] = {}
    for p in posts:
        day = ensure_utc(p.ts).strftime("%Y-%m-%d")
        for rid in p.race_ids or []:
            counts[(rid, day)] = counts.get((rid, day), 0) + 1
    return counts


# ---------------------------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------------------------


def parse_classification(text: str) -> tuple[str, str, float] | None:
    """Lenient parse of the LLM's JSON object; None when nothing usable came back."""
    if not text:
        return None
    raw = text.strip()
    if "{" in raw and "}" in raw:
        raw = raw[raw.find("{") : raw.rfind("}") + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, Mapping):
        return None
    kind = str(pick(data, "kind", "type", "class", default="unknown")).strip().lower()
    if kind not in KINDS:
        kind = "unknown"
    desc = str(pick(data, "description", "summary", "what", default="")).strip()[:500]
    try:
        conf = float(pick(data, "confidence", "conf", default=0.5))
    except (TypeError, ValueError):
        conf = 0.5
    if conf > 1.0:
        conf /= 100.0
    return kind, desc, max(0.0, min(1.0, conf))


def llm_classifier(client: Any, model: str) -> Classifier:
    """Classifier backed by the Messages API (synchronous SDK run in a thread)."""

    async def classify(race_context: Mapping[str, Any], top_posts: list[str], trigger: str) -> tuple[str, str, float]:
        payload = {"race": dict(race_context), "trigger": trigger, "posts": top_posts[:TOP_POSTS]}
        params = {
            "model": model,
            "max_tokens": 512,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            "output_config": {"effort": "low"},
        }

        def call() -> str:
            try:
                return response_text(client.messages.create(**params))
            except Exception as exc:  # noqa: BLE001
                log.error("events: classifier call failed: %s", type(exc).__name__)
                return ""

        parsed = parse_classification(await asyncio.to_thread(call))
        if parsed is None:
            return "unknown", trigger, 0.3
        kind, desc, conf = parsed
        return kind, (desc or trigger), conf

    return classify


async def fallback_classifier(race_context: Mapping[str, Any], top_posts: list[str], trigger: str) -> tuple[str, str, float]:
    """No LLM available: record the spike as ``unknown`` so the trader can at least widen quotes."""
    return "unknown", trigger, 0.3


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


def _top_posts(s: Any, race_id: str, day_start: datetime, limit: int = TOP_POSTS) -> list[str]:
    rows = (
        s.execute(select(Post).where(Post.ts >= day_start).order_by(Post.bot_score.asc().nulls_last(), Post.ts.desc()).limit(400))
        .scalars()
        .all()
    )
    out: list[str] = []
    for p in rows:
        if race_id in (p.race_ids or []):
            out.append(f"[{p.platform}] {(p.text or '')[:280]}")
            if len(out) >= limit:
                break
    return out


def _already_flagged(s: Any, race_id: str, source: str, day_start: datetime) -> bool:
    row = s.execute(
        select(Event.id).where(Event.race_id == race_id, Event.source == source, Event.ts >= day_start).limit(1)
    ).first()
    return row is not None


async def run_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    now: datetime | None = None,
    classifier: Classifier | None = None,
    client: Any | None = None,
    gazetteer: Gazetteer | None = None,
) -> int:
    """Detect today's spikes and write Event rows (one per race/source/day). Returns rows written."""
    now = now or utc_now()
    day = now.strftime("%Y-%m-%d")
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    sent = cfg.sentiment
    if classifier is None:
        client = client if client is not None else _make_client(secrets)
        classifier = llm_classifier(client, sent.escalation_model) if client is not None else fallback_classifier
    gazetteer = gazetteer or load_gazetteer(session_factory)

    with session_factory() as s:
        feats = s.execute(select(Feature).where(Feature.day == day)).scalars().all()
        social = detect_social_spikes(feats, sent.spike_z, sent.min_platforms)
        since = day_start - timedelta(days=sent.baseline_days)
        gposts = s.execute(select(Post).where(Post.platform == "gdelt", Post.ts >= since)).scalars().all()
        gdelt = detect_gdelt_spikes(gdelt_counts(gposts), day, sent.baseline_days)

    triggers: list[tuple[str, str, str, float]] = []  # race_id, source, summary, prior confidence
    for rid, fs in social.items():
        parts = ", ".join(f"{f.platform} z={f.z_score:+.1f} (n={f.n_posts})" for f in sorted(fs, key=lambda f: f.platform))
        zmax = max(abs(f.z_score or 0.0) for f in fs)
        triggers.append((rid, SOURCE_SOCIAL, f"stance spike on {len(fs)} platforms: {parts}", min(0.9, 0.4 + 0.1 * zmax)))
    for rid, (today_n, mean) in gdelt.items():
        triggers.append((rid, SOURCE_GDELT, f"news volume {today_n} articles vs trailing mean {mean:.1f}/day", 0.5))

    written = 0
    for rid, source, summary, prior in triggers:
        with session_factory() as s:
            if _already_flagged(s, rid, source, day_start):
                continue
            top = _top_posts(s, rid, day_start)
        try:
            kind, desc, conf = await classifier(gazetteer.race_context(rid), top, summary)
        except Exception:  # noqa: BLE001
            log.exception("events: classifier failed for %s", rid)
            kind, desc, conf = "unknown", summary, 0.3
        if kind not in KINDS:
            kind = "unknown"
        with session_factory() as s:
            if _already_flagged(s, rid, source, day_start):
                continue
            s.add(
                Event(
                    ts=now,
                    race_id=rid,
                    kind=kind,
                    description=f"{desc} | {summary}"[:2000] if desc != summary else summary,
                    source=source,
                    confidence=round(max(0.0, min(1.0, conf if kind != "unknown" else min(conf, prior))), 3),
                    acted=False,
                )
            )
            written += 1
        log.info("events: %s %s kind=%s conf=%.2f", rid, source, kind, conf)
    return written
