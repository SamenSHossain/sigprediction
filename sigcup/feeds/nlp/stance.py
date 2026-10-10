"""Stance classification with the Claude API (Batch API, sentiment.max_posts_per_batch posts per request,
JSON output {post_id, race_id, party, stance, confidence, is_news_or_poll}); cached by text_hash;
escalates 'unclear' to the escalation model. Writes Stance rows and marks posts classified.

Flow of ``run_once`` (non-blocking; safe to call every cycle):

1. harvest a pending Message Batch whose id is persisted in KV (``stance:batch``) - a restart resumes;
2. cache: a post whose ``text_hash`` already has Stance rows gets copies (model ``cache``), no API call;
3. the remaining unclassified, race-linked posts are packed ``max_posts_per_batch`` per request:
   fewer than ``BATCH_THRESHOLD`` (50) posts go through the regular Messages API now; more are submitted
   as one Message Batch (50 % cheaper) and harvested on a later cycle;
4. results with stance ``unclear`` are re-asked on ``sentiment.escalation_model`` (bounded per cycle).

Stance is toward the *candidate* (anger at the opponent == support), never the sentiment of the text.
The classifier model is ``cfg.sentiment.classifier_model`` (Haiku 5.5 by default, effort ``low``);
no API key -> returns 0 without importing the SDK. Tests inject a fake client.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select

from sigcup.common.time import utc_now
from sigcup.db.models import Post, Stance
from sigcup.db.repo import kv_get, kv_set
from sigcup.feeds.nlp.gazetteer import Gazetteer, load_gazetteer, pick
from sigcup.settings import Secrets, StrategyConfig

log = logging.getLogger(__name__)

BATCH_THRESHOLD = 50  # posts; at or above this a Message Batch is used instead of synchronous calls
KV_BATCH_KEY = "stance:batch"
MAX_POSTS_PER_RUN = 500
MAX_ESCALATIONS_PER_RUN = 50
MAX_TOKENS_PER_REQUEST = 4096
STANCES = ("support", "oppose", "neutral", "unclear")
CACHE_MODEL = "cache"

SYSTEM_PROMPT = (
    "You classify posts about 2026 US midterm elections. Input: a JSON object with `races` "
    "(race_id -> label, state, chamber, candidates by party) and `posts` (post_id, race_ids, author, text).\n"
    "Output: ONLY a JSON array, one object per (post, race) pair the post is actually about: "
    '{"post_id": str, "race_id": str, "party": "D"|"R"|"I", "stance": "support"|"oppose"|"neutral"|"unclear", '
    '"confidence": number 0-1, "is_news_or_poll": bool}.\n'
    "Rules: `party` is the candidate the post takes a position on; `stance` is the author's stance toward that "
    "candidate (attacking the opponent counts as support; sarcasm inverts). `is_news_or_poll` is true for news "
    "reports, poll releases, headlines, and campaign-finance or schedule items that carry no opinion. "
    "Use `unclear` only when the target or the position genuinely cannot be told. If a post is about several "
    "candidates of a race, emit one object per candidate. No prose, no code fences."
)


@dataclass(frozen=True)
class StanceResult:
    post_id: str
    race_id: str
    party: str
    stance: str
    confidence: float
    is_news_or_poll: bool


# ---------------------------------------------------------------------------------------------
# Prompt building and lenient parsing (pure; unit-tested)
# ---------------------------------------------------------------------------------------------


def build_user_content(posts: Sequence[Post], gazetteer: Gazetteer | None) -> str:
    """JSON payload for one request: race contexts for every race the posts touch, then the posts."""
    race_ids: list[str] = []
    for p in posts:
        for rid in p.race_ids or []:
            if rid not in race_ids:
                race_ids.append(rid)
    races = {rid: (gazetteer.race_context(rid) if gazetteer else {"race_id": rid}) for rid in race_ids}
    items = [
        {
            "post_id": p.post_id,
            "race_ids": list(p.race_ids or []),
            "author": p.author_handle or "",
            "platform": p.platform,
            "text": (p.text or "")[:1500],
        }
        for p in posts
    ]
    return json.dumps({"races": races, "posts": items}, ensure_ascii=False, separators=(",", ":"))


def build_params(posts: Sequence[Post], gazetteer: Gazetteer | None, model: str) -> dict[str, Any]:
    """Messages API params shared by the synchronous and batch paths."""
    return {
        "model": model,
        "max_tokens": MAX_TOKENS_PER_REQUEST,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_user_content(posts, gazetteer)}],
        "output_config": {"effort": "low"},
    }


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_results(text: str, allowed: Mapping[str, Sequence[str]] | None = None) -> list[StanceResult] | None:
    """Parse the model's JSON array leniently; ``None`` when no array can be recovered at all.

    ``allowed`` maps post_id -> permitted race_ids; items for unknown posts are dropped, and an item
    whose race_id is not among the post's race_ids is kept only when the post has no race_ids.
    """
    if not text:
        return None
    raw = text.strip()
    m = _FENCE_RE.search(raw)
    if m:
        raw = m.group(1).strip()
    data: Any = None
    for candidate in (raw, raw[raw.find("[") : raw.rfind("]") + 1] if "[" in raw and "]" in raw else ""):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        return None
    if isinstance(data, Mapping):  # {"results": [...]} or a single object
        inner = pick(data, "results", "stances", "items", "data")
        data = inner if isinstance(inner, list) else [data]
    if not isinstance(data, list):
        return None
    out: list[StanceResult] = []
    seen: set[tuple[str, str, str]] = set()
    for item in data:
        r = _normalise_item(item, allowed)
        if r is None:
            continue
        key = (r.post_id, r.race_id, r.party)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _normalise_item(item: Any, allowed: Mapping[str, Sequence[str]] | None) -> StanceResult | None:
    if not isinstance(item, Mapping):
        return None
    post_id = str(pick(item, "post_id", "id", "post", default="")).strip()
    race_id = str(pick(item, "race_id", "race", default="")).strip().upper()
    if not post_id or not race_id:
        return None
    if allowed is not None:
        if post_id not in allowed:
            return None
        permitted = list(allowed[post_id] or [])
        if permitted and race_id not in permitted:
            return None
    party = str(pick(item, "party", "target_party", "candidate_party", default="")).strip().upper()
    party = {"DEM": "D", "DEMOCRAT": "D", "REP": "R", "REPUBLICAN": "R", "GOP": "R", "IND": "I", "INDEPENDENT": "I"}.get(
        party, party
    )
    if party not in {"D", "R", "I"}:
        return None
    stance = str(pick(item, "stance", "label", default="unclear")).strip().lower()
    stance = {"supports": "support", "pro": "support", "for": "support", "opposes": "oppose", "against": "oppose",
              "anti": "oppose", "none": "neutral", "mixed": "unclear", "unknown": "unclear"}.get(stance, stance)
    if stance not in STANCES:
        stance = "unclear"
    conf_raw = pick(item, "confidence", "conf", "score", default=0.5)
    try:
        conf = float(conf_raw)
    except (TypeError, ValueError):
        conf = 0.5
    if conf > 1.0:  # a percentage
        conf = conf / 100.0
    conf = max(0.0, min(1.0, conf))
    news_raw = pick(item, "is_news_or_poll", "is_poll_or_news", "news", default=False)
    is_news = news_raw if isinstance(news_raw, bool) else str(news_raw).strip().lower() in {"true", "1", "yes"}
    return StanceResult(post_id, race_id, party, stance, round(conf, 3), bool(is_news))


def response_text(message: Any) -> str:
    """Concatenate text blocks from a Message (or a dict shaped like one); empty on refusal."""
    if message is None:
        return ""
    if pick(message, "stop_reason") == "refusal":
        return ""
    content = pick(message, "content", default=[])
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        if pick(block, "type") == "text":
            parts.append(str(pick(block, "text", default="")))
    return "".join(parts)


# ---------------------------------------------------------------------------------------------
# Client plumbing
# ---------------------------------------------------------------------------------------------


def _make_client(secrets: Secrets | None) -> Any | None:
    """A synchronous Anthropic client, or None when no API key is configured (the SDK is imported lazily)."""
    key = (secrets.anthropic_api_key or "").strip() if secrets is not None else ""
    if not key:
        return None
    import anthropic

    return anthropic.Anthropic(api_key=key, max_retries=2)


def _call_sync(client: Any, params: Mapping[str, Any]) -> str:
    """One Messages API call; returns the text (empty on any API error, which is logged without the payload)."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - the SDK is a declared dependency
        anthropic = None  # type: ignore[assignment]
    try:
        msg = client.messages.create(**params)
    except Exception as exc:  # noqa: BLE001 - every SDK error type ends up here; classify for the log
        status = getattr(exc, "status_code", None)
        if anthropic is not None and isinstance(exc, anthropic.RateLimitError):
            log.warning("stance: rate limited by the API; will retry next cycle")
        elif status is not None and status >= 500:
            log.warning("stance: API server error %s; will retry next cycle", status)
        else:
            log.error("stance: API call failed: %s", type(exc).__name__)
        return ""
    return response_text(msg)


def chunked(items: Sequence[Any], size: int) -> list[list[Any]]:
    size = max(1, int(size))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


# ---------------------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------------------


def _write_results(
    s: Any,
    results: Iterable[StanceResult],
    chunk_post_ids: Iterable[str],
    model: str,
    now: datetime,
    *,
    replace_unclear: bool = False,
) -> int:
    """Insert Stance rows and mark every post of the chunk classified. Returns rows inserted."""
    by_post: dict[str, list[StanceResult]] = {}
    for r in results:
        by_post.setdefault(r.post_id, []).append(r)
    n = 0
    for pid in chunk_post_ids:
        post = s.get(Post, pid)
        if post is None:
            continue
        rows = by_post.get(pid, [])
        if replace_unclear and rows:
            for old in s.execute(select(Stance).where(Stance.post_id == pid, Stance.stance == "unclear")).scalars():
                s.delete(old)
        for r in rows:
            s.add(
                Stance(
                    post_id=pid,
                    race_id=r.race_id[:32],
                    party=r.party,
                    stance=r.stance,
                    confidence=r.confidence,
                    is_news_or_poll=r.is_news_or_poll,
                    model=model[:48],
                    ts=now,
                )
            )
            n += 1
        post.classified = True
    return n


def apply_cache(s: Any, posts: Sequence[Post], now: datetime) -> tuple[int, list[Post]]:
    """Copy Stance rows from an already-classified post with the same text_hash. Returns (rows, remaining)."""
    hashes = {p.text_hash for p in posts if p.text_hash}
    if not hashes:
        return 0, list(posts)
    cached = s.execute(
        select(Stance, Post.text_hash)
        .join(Post, Stance.post_id == Post.post_id)
        .where(Post.text_hash.in_(hashes), Post.classified.is_(True))
    ).all()
    by_hash: dict[str, list[Stance]] = {}
    for st, th in cached:
        by_hash.setdefault(th, []).append(st)
    written = 0
    remaining: list[Post] = []
    for p in posts:
        rows = by_hash.get(p.text_hash or "")
        if not rows:
            remaining.append(p)
            continue
        permitted = set(p.race_ids or [])
        usable = [r for r in rows if not permitted or r.race_id in permitted]
        if not usable:
            remaining.append(p)
            continue
        seen: set[tuple[str, str]] = set()
        for r in usable:
            if (r.race_id, r.party) in seen:
                continue
            seen.add((r.race_id, r.party))
            s.add(
                Stance(
                    post_id=p.post_id,
                    race_id=r.race_id,
                    party=r.party,
                    stance=r.stance,
                    confidence=r.confidence,
                    is_news_or_poll=r.is_news_or_poll,
                    model=CACHE_MODEL,
                    ts=now,
                )
            )
            written += 1
        p.classified = True
    return written, remaining


def _load_unclassified(s: Any, limit: int) -> list[Post]:
    rows = s.execute(select(Post).where(Post.classified.is_(False)).order_by(Post.ts.asc()).limit(limit * 2)).scalars().all()
    out: list[Post] = []
    for p in rows:
        if not p.race_ids:
            p.classified = True  # nothing to classify; keep the queue clean
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


# ---------------------------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------------------------


def submit_batch(client: Any, chunks: Sequence[Sequence[Post]], gazetteer: Gazetteer | None, model: str) -> dict[str, Any] | None:
    """Create a Message Batch with one request per chunk; returns the KV state to persist (None on failure)."""
    requests = [
        {"custom_id": f"chunk-{i}", "params": build_params(chunk, gazetteer, model)} for i, chunk in enumerate(chunks)
    ]
    try:
        batch = client.messages.batches.create(requests=requests)
    except Exception as exc:  # noqa: BLE001
        log.error("stance: batch create failed: %s", type(exc).__name__)
        return None
    batch_id = str(pick(batch, "id", default=""))
    if not batch_id:
        return None
    return {
        "batch_id": batch_id,
        "model": model,
        "submitted_at": utc_now().isoformat(),
        "chunks": {f"chunk-{i}": [p.post_id for p in chunk] for i, chunk in enumerate(chunks)},
    }


def harvest_batch(client: Any, state: Mapping[str, Any]) -> tuple[bool, dict[str, str]]:
    """Return ``(ended, {custom_id: text})``. ``ended`` False means the batch is still processing."""
    batch_id = str(state.get("batch_id", ""))
    try:
        batch = client.messages.batches.retrieve(batch_id)
    except Exception as exc:  # noqa: BLE001
        log.error("stance: batch retrieve failed for %s: %s", batch_id, type(exc).__name__)
        status = getattr(exc, "status_code", None)
        return (status == 404), {}  # a vanished batch is "ended" with nothing to harvest
    if str(pick(batch, "processing_status", default="")) != "ended":
        return False, {}
    texts: dict[str, str] = {}
    try:
        for item in client.messages.batches.results(batch_id):
            cid = str(pick(item, "custom_id", default=""))
            result = pick(item, "result")
            if str(pick(result, "type", default="")) == "succeeded":
                texts[cid] = response_text(pick(result, "message"))
            else:
                log.warning("stance: batch item %s %s", cid, pick(result, "type", default="?"))
    except Exception as exc:  # noqa: BLE001
        log.error("stance: batch results failed for %s: %s", batch_id, type(exc).__name__)
        return False, {}
    return True, texts


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


async def run_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    client: Any | None = None,
    gazetteer: Gazetteer | None = None,
    now: datetime | None = None,
    use_batches: bool = True,
    batch_threshold: int = BATCH_THRESHOLD,
    max_posts: int = MAX_POSTS_PER_RUN,
) -> int:
    """Classify unclassified posts; returns Stance rows written this cycle (cache copies included)."""
    now = now or utc_now()
    client = client if client is not None else _make_client(secrets)
    if client is None:
        log.info("stance: no anthropic_api_key configured; skipping")
        return 0
    sent = cfg.sentiment
    gazetteer = gazetteer or load_gazetteer(session_factory)
    written = 0

    # 1. a pending batch from an earlier cycle (or an earlier process life)
    with session_factory() as s:
        pending = kv_get(s, KV_BATCH_KEY)
    if pending:
        ended, texts = await asyncio.to_thread(harvest_batch, client, pending)
        if not ended:
            log.info("stance: batch %s still processing", pending.get("batch_id"))
        else:
            with session_factory() as s:
                for cid, post_ids in (pending.get("chunks") or {}).items():
                    text = texts.get(cid, "")
                    allowed = {pid: ((s.get(Post, pid).race_ids or []) if s.get(Post, pid) else []) for pid in post_ids}
                    results = parse_results(text, allowed)
                    if results is None:
                        continue  # left unclassified -> retried in a later batch
                    written += _write_results(s, results, post_ids, str(pending.get("model") or sent.classifier_model), now)
                kv_set(s, KV_BATCH_KEY, None)
            log.info("stance: harvested batch %s -> %d rows", pending.get("batch_id"), written)
            pending = None

    # 2. cache hits, then the queue
    with session_factory() as s:
        posts = _load_unclassified(s, max_posts)
        cache_rows, posts = apply_cache(s, posts, now)
        written += cache_rows
        post_ids = [p.post_id for p in posts]
    if not posts:
        return written

    chunks = chunked(posts, sent.max_posts_per_batch)
    if use_batches and len(posts) >= batch_threshold:
        if pending:
            return written  # one batch in flight at a time
        state = await asyncio.to_thread(submit_batch, client, chunks, gazetteer, sent.classifier_model)
        if state:
            with session_factory() as s:
                kv_set(s, KV_BATCH_KEY, state)
            log.info("stance: submitted batch %s with %d posts in %d requests", state["batch_id"], len(post_ids), len(chunks))
        return written

    # 3. synchronous classification for small queues
    unclear: list[str] = []
    for chunk in chunks:
        params = build_params(chunk, gazetteer, sent.classifier_model)
        text = await asyncio.to_thread(_call_sync, client, params)
        allowed = {p.post_id: list(p.race_ids or []) for p in chunk}
        results = parse_results(text, allowed)
        if results is None:
            log.warning("stance: unparseable response for a chunk of %d posts; retrying next cycle", len(chunk))
            continue
        with session_factory() as s:
            written += _write_results(s, results, [p.post_id for p in chunk], sent.classifier_model, now)
        unclear.extend({r.post_id for r in results if r.stance == "unclear"})

    # 4. escalation of 'unclear' results
    written += await _escalate(session_factory, client, gazetteer, sent.escalation_model, unclear, sent.max_posts_per_batch, now)
    return written


async def _escalate(
    session_factory: Callable[[], Any],
    client: Any,
    gazetteer: Gazetteer | None,
    model: str,
    post_ids: Sequence[str],
    per_request: int,
    now: datetime,
) -> int:
    ids = list(dict.fromkeys(post_ids))[:MAX_ESCALATIONS_PER_RUN]
    if not ids:
        return 0
    with session_factory() as s:
        posts = [p for p in (s.get(Post, pid) for pid in ids) if p is not None]
    written = 0
    for chunk in chunked(posts, per_request):
        text = await asyncio.to_thread(_call_sync, client, build_params(chunk, gazetteer, model))
        results = parse_results(text, {p.post_id: list(p.race_ids or []) for p in chunk})
        if not results:
            continue
        improved = [r for r in results if r.stance != "unclear"]
        if not improved:
            continue
        with session_factory() as s:
            written += _write_results(s, improved, [r.post_id for r in improved], model, now, replace_unclear=True)
    return written
