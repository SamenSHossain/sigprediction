"""Stance classification with the Claude API (Batch API, sentiment.max_posts_per_batch posts per request,
JSON output {post_id, race_id, party, stance, confidence, is_news_or_poll}); cached by text_hash;
escalates 'unclear' to the escalation model. Writes Stance rows and marks posts classified.

Flow of ``run_once`` (non-blocking; safe to call every cycle):

1. harvest a pending Message Batch whose id is persisted in KV (``stance:batch``) - a restart resumes;
2. the queue is the unclassified, race-linked posts, NEWEST first; posts older than the features
   window (``2 x baseline_days + 1`` days) and posts the bot filter flags (``bot_score >
   BOT_THRESHOLD``, the cut features.py applies at aggregation) are marked classified without a call
   or a Stance row (they can never move a feature) and posts already in the pending batch are left alone;
3. cache: a post whose ``text_hash`` already has Stance rows gets copies (model ``cache``), no API call;
4. the remaining posts are packed ``max_posts_per_batch`` per request: fewer than ``BATCH_THRESHOLD``
   (50) go through the regular Messages API now; more are submitted as one Message Batch (50 %
   cheaper, up to ``MAX_POSTS_PER_BATCH`` posts, so one batch absorbs a cycle's whole inflow) that is
   harvested on a later cycle, after which the next batch is submitted in the same run;
5. results with stance ``unclear`` (synchronous or harvested) are re-asked on
   ``sentiment.escalation_model`` (bounded per cycle); the escalated rows replace the post's earlier
   rows for the same (race, party) and its ``unclear`` rows.

A request whose answer cannot be parsed (a refusal, JSON cut off at ``max_tokens``, prose) costs each
of its posts one attempt (KV ``stance:attempts``); a post with a failed attempt is retried alone, and
after ``MAX_ATTEMPTS`` it is marked classified with no Stance rows. API errors are not attempts.

Stance is toward the candidate the post is about: an attack on candidate X is (X, oppose), never
"support" for X's opponent, and never the sentiment of the text.
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
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from sigcup.common.time import utc_now
from sigcup.db.models import Post, Stance
from sigcup.db.repo import kv_get, kv_set
from sigcup.feeds.nlp.botfilter import BOT_THRESHOLD
from sigcup.feeds.nlp.gazetteer import Gazetteer, load_gazetteer, pick
from sigcup.settings import Secrets, StrategyConfig

log = logging.getLogger(__name__)

BATCH_THRESHOLD = 50  # posts; at or above this a Message Batch is used instead of synchronous calls
KV_BATCH_KEY = "stance:batch"
KV_ATTEMPTS_KEY = "stance:attempts"
MAX_POSTS_PER_RUN = 500  # synchronous path
MAX_POSTS_PER_BATCH = 10_000  # one Message Batch (400 requests of 25): a cycle's whole X inflow fits
MAX_ATTEMPTS = 3  # unparseable answers before a post is given up
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
    "Rules: `party` is the party of the candidate the post is about; `stance` is the author's attitude toward THAT "
    "candidate. A post attacking candidate X is party=X's party, stance=oppose; do not relabel it as support for "
    "X's opponent. A post praising X is party=X's party, stance=support. Sarcasm inverts the literal reading. "
    'Examples (Ken Paxton is the R candidate): "Paxton is a crook" -> {"party": "R", "stance": "oppose"}; '
    '"Proud to vote Paxton" -> {"party": "R", "stance": "support"}; "Ken Paxton, a real genius /s" -> '
    '{"party": "R", "stance": "oppose"}. `is_news_or_poll` is true for news reports, poll releases, headlines, and '
    "campaign-finance or schedule items that carry no opinion. Use `unclear` only when the target or the position "
    "genuinely cannot be told. If a post is about several candidates of a race, emit one object per candidate. "
    "No prose, no code fences."
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


def _call(client: Any, params: Mapping[str, Any]) -> tuple[bool, str]:
    """One Messages API call -> ``(answered, text)``. ``answered`` is False on an API error (logged without
    the payload); a refusal or a ``max_tokens`` cut-off is an answer whose text may not parse."""
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
        return False, ""
    if pick(msg, "stop_reason") == "max_tokens":
        log.warning("stance: answer cut off at max_tokens")
    return True, response_text(msg)


def _call_sync(client: Any, params: Mapping[str, Any]) -> str:
    """One Messages API call; returns the text (empty on any API error, which is logged without the payload)."""
    return _call(client, params)[1]


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
    """Insert Stance rows and mark every post of the chunk classified. Returns rows inserted.

    ``replace_unclear`` (escalation): before inserting a post's new rows, delete its ``unclear`` rows and
    every earlier row for a (race, party) the new results cover, so a pair never carries two labels.
    """
    by_post: dict[str, list[StanceResult]] = {}
    for r in results:
        by_post.setdefault(r.post_id, []).append(r)
    n = 0
    for pid in dict.fromkeys(chunk_post_ids):
        post = s.get(Post, pid)
        if post is None:
            continue
        rows = by_post.get(pid, [])
        if replace_unclear and rows:
            covered = {(r.race_id[:32], r.party) for r in rows}
            for old in s.execute(select(Stance).where(Stance.post_id == pid)).scalars().all():
                if old.stance == "unclear" or (old.race_id, old.party) in covered:
                    s.delete(old)
            s.flush()
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


def _load_unclassified(
    s: Any, limit: int, *, since: datetime | None = None, exclude: Iterable[str] = ()
) -> list[Post]:
    """Up to ``limit`` unclassified race-linked posts, newest first.

    Posts older than ``since`` and bot-flagged posts (``bot_score > BOT_THRESHOLD``: features.py drops
    them, so a label would be wasted spend) are marked classified (no Stance) without a call, in SQL
    before the limit so they never crowd real posts out of a cycle; ``exclude`` (posts in the pending
    batch) are skipped; posts without race_ids are marked classified.
    """
    if since is not None:
        stale = s.execute(update(Post).where(Post.classified.is_(False), Post.ts < since).values(classified=True))
        if stale.rowcount:
            log.info("stance: skipped %d unclassified posts older than %s", stale.rowcount, since.date())
    bots = s.execute(update(Post).where(Post.classified.is_(False), Post.bot_score > BOT_THRESHOLD).values(classified=True))
    if bots.rowcount:
        log.info("stance: skipped %d bot-flagged posts (bot_score > %.2f)", bots.rowcount, BOT_THRESHOLD)
    skip = set(exclude)
    rows = (
        s.execute(select(Post).where(Post.classified.is_(False)).order_by(Post.ts.desc()).limit(limit * 2 + len(skip)))
        .scalars()
        .all()
    )
    out: list[Post] = []
    for p in rows:
        if p.post_id in skip:
            continue
        if not p.race_ids:
            p.classified = True  # nothing to classify; keep the queue clean
            continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


def _note_attempts(s: Any, failed: Iterable[str], succeeded: Iterable[str]) -> list[str]:
    """Count unparseable answers per post (KV); give a post up after MAX_ATTEMPTS. Returns given-up ids."""
    failed, succeeded = list(dict.fromkeys(failed)), set(succeeded)
    attempts = {str(k): int(v) for k, v in (kv_get(s, KV_ATTEMPTS_KEY) or {}).items()}
    if not failed and not (succeeded & attempts.keys()):
        return []
    for pid in succeeded:
        attempts.pop(pid, None)
    given_up: list[str] = []
    for pid in failed:
        n = attempts.get(pid, 0) + 1
        if n < MAX_ATTEMPTS:
            attempts[pid] = n
            continue
        attempts.pop(pid, None)
        post = s.get(Post, pid)
        if post is not None:
            post.classified = True
        given_up.append(pid)
    if attempts:  # drop counters of posts that left the queue some other way (cache, age cutoff)
        keys = list(attempts)
        live: set[str] = set()
        for i in range(0, len(keys), 500):
            live.update(
                s.execute(select(Post.post_id).where(Post.post_id.in_(keys[i : i + 500]), Post.classified.is_(False))).scalars()
            )
        attempts = {k: v for k, v in attempts.items() if k in live}
    kv_set(s, KV_ATTEMPTS_KEY, attempts)
    if given_up:
        log.warning("stance: gave up on %d posts after %d unparseable answers", len(given_up), MAX_ATTEMPTS)
    return given_up


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
    max_batch_posts: int = MAX_POSTS_PER_BATCH,
) -> int:
    """Classify unclassified posts; returns Stance rows written this cycle (cache copies included)."""
    now = now or utc_now()
    client = client if client is not None else _make_client(secrets)
    if client is None:
        log.info("stance: no anthropic_api_key configured; skipping")
        return 0
    sent = cfg.sentiment
    gazetteer = gazetteer or load_gazetteer(session_factory)
    cutoff = now - timedelta(days=2 * max(1, int(sent.baseline_days)) + 1)  # features' recompute window
    written = 0
    unclear: list[str] = []

    # 1. a pending batch from an earlier cycle (or an earlier process life)
    with session_factory() as s:
        pending = kv_get(s, KV_BATCH_KEY)
    if pending:
        ended, texts = await asyncio.to_thread(harvest_batch, client, pending)
        if not ended:
            log.info("stance: batch %s still processing", pending.get("batch_id"))
        else:
            failed: list[str] = []
            succeeded: list[str] = []
            harvested = 0
            with session_factory() as s:
                for cid, post_ids in (pending.get("chunks") or {}).items():
                    if cid not in texts:
                        continue  # errored or expired item: left unclassified, retried, not an attempt
                    allowed = {pid: ((s.get(Post, pid).race_ids or []) if s.get(Post, pid) else []) for pid in post_ids}
                    results = parse_results(texts[cid], allowed)
                    if results is None:
                        failed.extend(post_ids)
                        continue
                    harvested += _write_results(s, results, post_ids, str(pending.get("model") or sent.classifier_model), now)
                    succeeded.extend(post_ids)
                    unclear.extend(r.post_id for r in results if r.stance == "unclear")
                kv_set(s, KV_BATCH_KEY, None)
                _note_attempts(s, failed, succeeded)
            written += harvested
            log.info("stance: harvested batch %s -> %d rows", pending.get("batch_id"), harvested)
            pending = None

    # 2. the queue (newest first, stale skipped, the pending batch's posts left alone), then cache hits
    in_flight = [pid for ids in ((pending or {}).get("chunks") or {}).values() for pid in ids]
    limit = max(max_posts, max_batch_posts) if use_batches else max_posts
    with session_factory() as s:
        posts = _load_unclassified(s, limit, since=cutoff, exclude=in_flight)
        cache_rows, posts = apply_cache(s, posts, now)
        written += cache_rows
        attempts = kv_get(s, KV_ATTEMPTS_KEY) or {}

    # 3. a Message Batch for a backlog, synchronous calls for a trickle
    if posts and use_batches and len(posts) >= batch_threshold:
        if pending:
            log.info("stance: %d posts wait for batch %s", len(posts), pending.get("batch_id"))
        else:
            chunks = _pack(posts[:max_batch_posts], attempts, sent.max_posts_per_batch)
            state = await asyncio.to_thread(submit_batch, client, chunks, gazetteer, sent.classifier_model)
            if state:
                with session_factory() as s:
                    kv_set(s, KV_BATCH_KEY, state)
                log.info("stance: submitted batch %s with %d posts in %d requests", state["batch_id"], sum(map(len, chunks)), len(chunks))
    elif posts:
        written += await _classify_sync(session_factory, client, gazetteer, sent.classifier_model, posts[:max_posts], attempts, sent.max_posts_per_batch, now, unclear)

    # 4. escalation of 'unclear' results
    written += await _escalate(session_factory, client, gazetteer, sent.escalation_model, unclear, sent.max_posts_per_batch, now)
    return written


def _pack(posts: Sequence[Post], attempts: Mapping[str, Any], per_request: int) -> list[list[Post]]:
    """Requests of ``per_request`` posts; a post whose earlier answer failed to parse goes alone."""
    retry = [p for p in posts if attempts.get(p.post_id)]
    fresh = [p for p in posts if not attempts.get(p.post_id)]
    return chunked(fresh, per_request) + [[p] for p in retry]


async def _classify_sync(
    session_factory: Callable[[], Any],
    client: Any,
    gazetteer: Gazetteer | None,
    model: str,
    posts: Sequence[Post],
    attempts: Mapping[str, Any],
    per_request: int,
    now: datetime,
    unclear: list[str],
) -> int:
    """Regular Messages API calls; appends 'unclear' post ids to ``unclear``. Returns rows written."""
    written = 0
    failed: list[str] = []
    succeeded: list[str] = []
    for chunk in _pack(posts, attempts, per_request):
        answered, text = await asyncio.to_thread(_call, client, build_params(chunk, gazetteer, model))
        if not answered:
            break  # API trouble: the rest waits for the next cycle and no attempt is counted
        results = parse_results(text, {p.post_id: list(p.race_ids or []) for p in chunk})
        if results is None:
            log.warning("stance: unparseable response for a chunk of %d posts; retrying next cycle", len(chunk))
            failed.extend(p.post_id for p in chunk)
            continue
        with session_factory() as s:
            written += _write_results(s, results, [p.post_id for p in chunk], model, now)
        succeeded.extend(p.post_id for p in chunk)
        unclear.extend(r.post_id for r in results if r.stance == "unclear")
    if failed or succeeded:
        with session_factory() as s:
            _note_attempts(s, failed, succeeded)
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
            written += _write_results(s, improved, list(dict.fromkeys(r.post_id for r in improved)), model, now, replace_unclear=True)
    return written
