"""Supabase Realtime consumer for order-book pushes (POST /realtime/token -> private broadcast channels).

Responsibilities (owner: sig workstream):
- Token refresh before the 3-hour expiry; reconnect with full rejoin of channels.
- Subscribe to `tournament:{tid}:market:{mid}` for up to config.limits.realtime_channels markets plus `user:{profileId}`.
- Parse `market_batch` (books + trades, delivery.revision, marketSettled, resyncRequired), `book_dirty`,
  `market_settled`, `account_batch` (fills). Detect revision gaps and emit a resync request.
- Expose an async iterator of normalized events: BookUpdate(exchange_id, bids, asks, ts, revision),
  FillEvent(FillInfo), SettledEvent(exchange_id), ResyncNeeded(exchange_id|None).
The trader's book_state consumes these; REST reconciliation covers dropped batches.

Protocol notes (reconstructed, see docs/research/platform.md section 2 and the competitor notes it cites):
- Supabase Realtime speaks the Phoenix channel protocol over one websocket:
  `wss://<project>/realtime/v1/websocket?apikey=<anonKey>&vsn=1.0.0`, JSON frames
  `{"topic","event","payload","ref"}` (vsn 2 uses `[join_ref, ref, topic, event, payload]`; both are decoded).
- A channel is joined with `phx_join` on topic `realtime:<name>` carrying `{"config": {"private": true, ...},
  "access_token": <token>}`; the server answers `phx_reply` with `status: ok|error`. Heartbeats go to topic
  `phoenix` every ~25 s; a refreshed token is pushed with an `access_token` event on every joined channel.
- Broadcast frames are `event: "broadcast"` with `payload: {"type": "broadcast", "event": <name>, "payload": {...}}`.
- Batches are delivered at most once, may arrive up to three times, and carry `delivery.revision` /
  `delivery.previousRevision`; a `previousRevision` above the last accepted revision means a gap.
- The server refuses joins beyond ~100 channels per connection ("ChannelRateLimitReached"); joins take 3.5-6 s
  each server-side, so they are sent in batches of ten with a 60 s timeout.
Every payload field is optional in the parser; message shapes are uncertain until observed live.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import quote

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import WebSocketException

from sigcup.settings import config
from sigcup.sig.client import RealtimeToken, parse_book, parse_dt, parse_fill, pick, to_bool, to_int
from sigcup.sig.schemas import FillInfo, Level

log = logging.getLogger(__name__)

KNOWN_EVENTS = ("market_batch", "book_dirty", "market_settled", "account_batch")
TOKEN_LIFETIME_S = 3 * 3600.0
DEFAULT_TOKEN_REFRESH_S = 2.5 * 3600.0


# ---------------------------------------------------------------------------------------------
# Normalized events
# ---------------------------------------------------------------------------------------------


@dataclass(slots=True)
class BookUpdate:
    """A full book snapshot pushed for one exchange (bids best-first descending, asks ascending)."""

    exchange_id: str
    bids: list[Level]
    asks: list[Level]
    ts: datetime | None
    revision: int | None
    sequence: int | None = None
    next_expiry_at: datetime | None = None  # a resting order expires then; expiries emit nothing
    market_id: str | None = None
    trades: list[dict] = field(default_factory=list)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None


@dataclass(slots=True)
class FillEvent:
    """A fill pushed on the user channel. Pushed fills may carry no id (then `fill.fill_id` is synthetic):
    treat the event as a hint to read /portfolio/fills rather than as the ledger entry itself."""

    fill: FillInfo
    revision: int | None = None


@dataclass(slots=True)
class SettledEvent:
    exchange_id: str | None
    market_id: str | None = None


@dataclass(slots=True)
class ResyncNeeded:
    """Reload this market's book over REST (exchange_id=None: reload every subscribed market)."""

    exchange_id: str | None
    market_id: str | None = None
    reason: str = ""


Event = BookUpdate | FillEvent | SettledEvent | ResyncNeeded
TokenProvider = Callable[[], Awaitable[RealtimeToken]]


# ---------------------------------------------------------------------------------------------
# Pure protocol helpers (fixture-tested)
# ---------------------------------------------------------------------------------------------


def market_topic(tournament_id: str, market_id: str) -> str:
    return f"tournament:{tournament_id}:market:{market_id}"


def market_id_from_topic(topic: str | None) -> str | None:
    """'realtime:tournament:T:market:M' -> 'M' (None for non-market topics)."""
    if not topic:
        return None
    t = topic[len("realtime:") :] if topic.startswith("realtime:") else topic
    marker = ":market:"
    if marker not in t:
        return None
    return t.split(marker, 1)[1].split(":", 1)[0] or None


def websocket_url(supabase_url: str, anon_key: str) -> str:
    """Supabase project URL -> realtime websocket URL (the access token travels in join payloads, not here)."""
    u = supabase_url.strip().rstrip("/")
    if u.startswith("https://"):
        u = "wss://" + u[len("https://") :]
    elif u.startswith("http://"):
        u = "ws://" + u[len("http://") :]
    elif not u.startswith(("ws://", "wss://")):
        u = "wss://" + u
    if "/realtime/v1" not in u:
        u += "/realtime/v1"
    if not u.endswith("/websocket"):
        u += "/websocket"
    return f"{u}?apikey={quote(anon_key, safe='')}&vsn=1.0.0"


def decode_frame(raw: str | bytes) -> tuple[str | None, str | None, Any, str | None]:
    """Phoenix frame -> (topic, event, payload, ref). Accepts the object (vsn 1) and array (vsn 2) encodings."""
    try:
        msg = json.loads(raw)
    except (TypeError, ValueError):
        return None, None, None, None
    if isinstance(msg, list):
        if len(msg) >= 5:
            _join_ref, ref, topic, event, payload = msg[:5]
            return _opt(topic), _opt(event), payload, _opt(ref)
        return None, None, None, None
    if isinstance(msg, Mapping):
        return _opt(msg.get("topic")), _opt(msg.get("event")), msg.get("payload"), _opt(msg.get("ref"))
    return None, None, None, None


def _opt(v: Any) -> str | None:
    return None if v is None else str(v)


def encode_frame(topic: str, event: str, payload: Any, ref: str | None) -> str:
    return json.dumps({"topic": topic, "event": event, "payload": payload, "ref": ref}, separators=(",", ":"))


def join_payload(access_token: str) -> dict[str, Any]:
    return {
        "config": {
            "broadcast": {"ack": False, "self": False},
            "presence": {"key": ""},
            "postgres_changes": [],
            "private": True,
        },
        "access_token": access_token,
    }


def unwrap_broadcast(event: str | None, payload: Any) -> tuple[str | None, Any]:
    """(inner event name, inner payload) for a broadcast frame; passes known events through unchanged."""
    if event == "broadcast" and isinstance(payload, Mapping):
        inner_event = pick(payload, "event", "type")
        inner = payload.get("payload")
        if inner is None:
            inner = {k: v for k, v in payload.items() if k not in ("event", "type")}
        return (_opt(inner_event), inner)
    return event, payload


class RevisionTracker:
    """Per-topic delivery revisions and per-exchange book versions (duplicate / gap detection)."""

    def __init__(self) -> None:
        self.revisions: dict[str, int] = {}
        self.versions: dict[str, tuple[int, float]] = {}

    def reset(self) -> None:
        self.revisions.clear()
        self.versions.clear()

    def classify(self, key: str, revision: int | None, previous: int | None) -> str:
        """'apply' | 'duplicate' | 'gap' (gap batches are still applied; the caller also requests a resync)."""
        last = self.revisions.get(key)
        if revision is None:
            return "apply"
        if last is not None:
            if revision <= last:
                return "duplicate"
            self.revisions[key] = revision
            if previous is not None and previous > last:
                return "gap"
            return "apply"
        self.revisions[key] = revision
        return "apply"

    def book_is_new(self, exchange_id: str, sequence: int | None, at: datetime | None) -> bool:
        """A book applies only if its (sequence, at) version is newer than the one held; never goes backwards."""
        if sequence is None and at is None:
            return True  # unversioned book: nothing to compare against, always apply
        version = (sequence if sequence is not None else -1, at.timestamp() if at is not None else 0.0)
        held = self.versions.get(exchange_id)
        if held is not None and version <= held:
            return False
        self.versions[exchange_id] = version
        return True


def _books_of(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    books = pick(payload, "books", "orderbooks", "orderBooks")
    if isinstance(books, list):
        return [b for b in books if isinstance(b, Mapping)]
    if isinstance(books, Mapping):
        return [
            {"exchangeId": k, **v} if isinstance(v, Mapping) and "exchangeId" not in v else v
            for k, v in books.items()
        ]
    single = pick(payload, "book", "orderbook", "orderBook")
    if isinstance(single, Mapping):
        return [single]
    if "bids" in payload or "asks" in payload:
        return [payload]
    return []


def parse_broadcast(
    event: str | None,
    payload: Any,
    *,
    topic: str | None = None,
    tracker: RevisionTracker | None = None,
    markets: Mapping[str, str | None] | None = None,
) -> list[Event]:
    """Normalize one broadcast (`market_batch`, `book_dirty`, `market_settled`, `account_batch`) into events.

    `markets` maps market id -> exchange id so topic-only events (book_dirty, market_settled) can name the
    exchange. `tracker` drops duplicate revisions and flags gaps; without it every batch is applied.
    """
    tracker = tracker or RevisionTracker()
    market_id = market_id_from_topic(topic)
    mapped_eid = markets.get(market_id) if markets and market_id else None
    body: Mapping[str, Any] = payload if isinstance(payload, Mapping) else {}
    name = (event or "").lower()
    out: list[Event] = []

    if name in ("book_dirty", "books_dirty"):
        eid = pick(body, "exchangeId", "exchange_id") or mapped_eid
        return [
            ResyncNeeded(_opt(eid), market_id=market_id or _opt(pick(body, "marketId")), reason="book_dirty")
        ]

    if name in ("market_settled", "settled"):
        eid = pick(body, "exchangeId", "exchange_id") or mapped_eid
        mid = market_id or _opt(pick(body, "marketId", "market_id"))
        return [SettledEvent(_opt(eid), market_id=mid)]

    if name in ("account_batch", "fills", "fill"):
        delivery = pick(body, "delivery", default={})
        revision = to_int(
            pick(delivery, "revision") if isinstance(delivery, Mapping) else pick(body, "revision")
        )
        previous = to_int(
            pick(delivery, "previousRevision", "previous_revision") if isinstance(delivery, Mapping) else None
        )
        verdict = tracker.classify(topic or "user", revision, previous)
        if verdict == "duplicate":
            return []
        if verdict == "gap":
            out.append(ResyncNeeded(None, reason="account_revision_gap"))
        if to_bool(pick(body, "resyncRequired", "resync_required")):
            out.append(ResyncNeeded(None, reason="account_resync_required"))
        fills = pick(body, "fills", "trades", "executions", default=[])
        if isinstance(fills, Mapping):
            fills = [fills]
        for raw in fills if isinstance(fills, list) else []:
            f = parse_fill(raw) if isinstance(raw, Mapping) else None
            if f is not None:
                out.append(FillEvent(f, revision=revision))
        return out

    if name in ("market_batch", "book", "books", "orderbook", "book_update"):
        delivery = pick(body, "delivery", default={})
        if isinstance(delivery, Mapping):
            revision = to_int(pick(delivery, "revision", "rev"))
            previous = to_int(pick(delivery, "previousRevision", "previous_revision", "prev"))
        else:
            revision = to_int(pick(body, "revision"))
            previous = to_int(pick(body, "previousRevision"))
        key = topic or f"market:{market_id or '?'}"
        verdict = tracker.classify(key, revision, previous)
        if verdict == "duplicate":
            return []
        books = _books_of(body)
        trades = pick(body, "trades", default=[])
        trades = [t for t in trades if isinstance(t, Mapping)] if isinstance(trades, list) else []
        eid_hint = _opt(pick(body, "exchangeId", "exchange_id")) or mapped_eid
        if verdict == "gap":
            out.append(ResyncNeeded(eid_hint, market_id=market_id, reason="revision_gap"))
        if to_bool(pick(body, "resyncRequired", "resync_required")):
            out.append(ResyncNeeded(eid_hint, market_id=market_id, reason="resync_required"))
        if any(pick(t, "sequence", "seq") is None for t in trades):
            out.append(ResyncNeeded(eid_hint, market_id=market_id, reason="trade_without_sequence"))
        for raw in books:
            book = parse_book(raw, eid_hint)
            if book is None:
                # a book we cannot attribute to an exchange: let REST (by market id) sort it out
                out.append(ResyncNeeded(None, market_id=market_id, reason="unknown_exchange"))
                continue
            seq = to_int(pick(raw, "asOf.sequence", "sequence", "seq", "version"))
            if not tracker.book_is_new(book.exchange_id, seq, book.ts):
                continue
            own_trades = [
                t for t in trades if _opt(pick(t, "exchangeId", "exchange_id")) in (None, book.exchange_id)
            ]
            out.append(
                BookUpdate(
                    exchange_id=book.exchange_id,
                    bids=book.bids,
                    asks=book.asks,
                    ts=book.ts,
                    revision=revision,
                    sequence=seq,
                    next_expiry_at=parse_dt(pick(raw, "nextExpiryAt", "next_expiry_at")),
                    market_id=market_id or _opt(pick(body, "marketId", "market_id")),
                    trades=[dict(t) for t in own_trades],
                )
            )
        if to_bool(pick(body, "marketSettled", "market_settled", "settled")):
            settled_eid = eid_hint or (books and _opt(pick(books[0], "exchangeId", "exchange_id"))) or None
            out.append(SettledEvent(settled_eid, market_id=market_id))
        return out

    log.debug("ignoring realtime event %r on %s", event, topic)
    return []


# ---------------------------------------------------------------------------------------------
# The feed
# ---------------------------------------------------------------------------------------------


class _Stall(Exception):
    pass


class RealtimeFeed:
    """Async iterator of normalized Realtime events with reconnect, rejoin, heartbeat and token refresh.

    Degrades gracefully: when the token call or the socket fails, `events()` simply yields nothing while it
    retries with exponential backoff (1 s -> 60 s); the trader keeps polling REST meanwhile. Each (re)connect
    yields `ResyncNeeded(None, reason="connected")` so the consumer reloads every subscribed book.
    """

    def __init__(
        self,
        token_provider: TokenProvider,
        tournament_id: str,
        *,
        markets: Mapping[str, str | None] | Iterable[str] | None = None,
        user_channel: str | None = None,
        max_channels: int | None = None,
        heartbeat_s: float = 25.0,
        token_refresh_s: float = DEFAULT_TOKEN_REFRESH_S,
        join_timeout_s: float = 60.0,
        join_batch: int = 10,
        join_retry_s: float = 30.0,
        stall_s: float = 90.0,
        tick_s: float = 1.0,
        reconnect_min_s: float = 1.0,
        reconnect_max_s: float = 60.0,
        connect: Callable[..., Any] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.token_provider = token_provider
        self.tournament_id = str(tournament_id)
        self.user_channel = user_channel
        self.max_channels = max_channels if max_channels is not None else config().limits.realtime_channels
        self.heartbeat_s = heartbeat_s
        self.token_refresh_s = token_refresh_s
        self.join_timeout_s = join_timeout_s
        self.join_batch = max(1, join_batch)
        self.join_retry_s = join_retry_s
        self.stall_s = stall_s
        self.tick_s = tick_s
        self.reconnect_min_s = reconnect_min_s
        self.reconnect_max_s = reconnect_max_s
        self._connect = connect or ws_connect
        self._clock = clock or time.monotonic
        # The handshake URL carries the anon key: keep the websockets library's DEBUG chatter off.
        wslog = logging.getLogger("websockets")
        if wslog.level == logging.NOTSET or wslog.level < logging.INFO:
            wslog.setLevel(logging.INFO)
        self._markets: dict[str, str | None] = {}
        self._stop = asyncio.Event()
        self._ws: Any = None
        self._refs = 0
        self.tracker = RevisionTracker()
        self.joined: set[str] = set()
        self.last_message_at: float | None = None
        self.connected = False
        self.connections = 0
        self.token: RealtimeToken | None = None
        self._markets_dirty = False
        if markets:
            self.set_markets(markets)

    # -- configuration -----------------------------------------------------------------------
    def set_markets(self, markets: Mapping[str, str | None] | Iterable[str]) -> None:
        """Markets to follow (market id -> exchange id); only the first `max_channels` are subscribed."""
        items = list(markets.items()) if isinstance(markets, Mapping) else [(str(m), None) for m in markets]
        cap = self.max_channels - (1 if self.user_channel else 0)
        if len(items) > cap:
            log.warning("realtime: %d markets requested, subscribing to the first %d", len(items), cap)
        self._markets = {str(k): (str(v) if v is not None else None) for k, v in items[: max(0, cap)]}
        self._markets_dirty = True

    @property
    def markets(self) -> dict[str, str | None]:
        return dict(self._markets)

    def topics(self) -> list[str]:
        out = [f"realtime:{market_topic(self.tournament_id, m)}" for m in self._markets]
        if self.user_channel:
            out.append(f"realtime:{self.user_channel}")
        return out

    async def stop(self) -> None:
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    # -- main loop ---------------------------------------------------------------------------
    async def events(self) -> AsyncIterator[Event]:
        """Yield normalized events forever (until `stop()`); reconnects with full rejoin on any failure."""
        self._stop.clear()
        backoff = self.reconnect_min_s
        while not self._stop.is_set():
            try:
                token = await self.token_provider()
            except Exception as e:  # noqa: BLE001
                log.warning("realtime token fetch failed (%s); retry in %.0fs", type(e).__name__, backoff)
                if await self._sleep(backoff):
                    break
                backoff = min(backoff * 2, self.reconnect_max_s)
                continue
            self.token = token
            if token.user_channel and not self.user_channel:
                self.user_channel = token.user_channel
            url = websocket_url(token.supabase_url, token.anon_key)
            try:
                async with self._connect(
                    url, open_timeout=20, close_timeout=5, max_size=8 * 1024 * 1024
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    self.connections += 1
                    self.joined.clear()
                    self.tracker.reset()
                    self.last_message_at = self._clock()
                    log.info("realtime connected (%d channels to join)", len(self.topics()))
                    yield ResyncNeeded(None, reason="connected" if self.connections == 1 else "reconnected")
                    backoff = self.reconnect_min_s
                    session = self._session(ws, token)
                    try:
                        async for ev in session:
                            yield ev
                    finally:
                        await session.aclose()
            except (WebSocketException, OSError, TimeoutError, _Stall) as e:
                log.warning("realtime connection ended: %s", _short(e))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                log.exception("realtime loop error: %s", _short(e))
            finally:
                self.connected = False
                self._ws = None
                self.joined.clear()
            if self._stop.is_set():
                break
            if await self._sleep(backoff):
                break
            backoff = min(backoff * 2, self.reconnect_max_s)

    async def _sleep(self, seconds: float) -> bool:
        """Sleep unless stopped; True when stop() was called."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return False

    def _next_ref(self) -> str:
        self._refs += 1
        return str(self._refs)

    async def _send(self, ws: Any, topic: str, event: str, payload: Any) -> str:
        ref = self._next_ref()
        await ws.send(encode_frame(topic, event, payload, ref))
        return ref

    async def _session(self, ws: Any, token: RealtimeToken) -> AsyncIterator[Event]:
        """Drive one connection: batched joins, heartbeats, token refresh, stall detection, event parsing."""
        now = self._clock()
        access_token = token.token
        pending: dict[str, tuple[str, float]] = {}  # ref -> (topic, sent_at)
        failed: dict[str, float] = {}  # topic -> retry_at
        failures: dict[str, int] = {}  # topic -> consecutive failed joins (retry delay doubles, capped)

        def schedule_retry(topic: str) -> None:
            failures[topic] = failures.get(topic, 0) + 1
            failed[topic] = self._clock() + min(self.join_retry_s * 2 ** (failures[topic] - 1), 600.0)

        queue: list[str] = self.topics()
        self._markets_dirty = False
        last_heartbeat = now
        refresh_at = now + min(self.token_refresh_s, max(60.0, token.seconds_left - 600.0))
        heartbeat_ref: str | None = None

        async def send_joins() -> None:
            while queue and len(pending) < self.join_batch:
                topic = queue.pop(0)
                if topic in self.joined:
                    continue
                ref = await self._send(ws, topic, "phx_join", join_payload(access_token))
                pending[ref] = (topic, self._clock())

        await send_joins()
        while not self._stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=self.tick_s)
            except TimeoutError:
                raw = None
            now = self._clock()
            if raw is not None:
                self.last_message_at = now
                topic, event, payload, ref = decode_frame(raw)
                if event == "phx_reply":
                    if ref is not None and ref in pending:
                        jt, _ = pending.pop(ref)
                        status = (
                            str(pick(payload, "status", default="")).lower()
                            if isinstance(payload, Mapping)
                            else ""
                        )
                        if status == "ok":
                            self.joined.add(jt)
                            failed.pop(jt, None)
                            failures.pop(jt, None)
                        else:
                            reason = pick(
                                payload, "response.reason", "response.message", "response", default=payload
                            )
                            log.log(
                                logging.WARNING if failures.get(jt, 0) < 2 else logging.DEBUG,
                                "realtime join failed for %s: %s",
                                _topic_label(jt),
                                _short(reason),
                            )
                            schedule_retry(jt)
                        await send_joins()
                    elif ref is not None and ref == heartbeat_ref:
                        heartbeat_ref = None
                elif event in ("phx_error", "phx_close"):
                    if topic in self.joined:
                        log.warning(
                            "realtime channel %s closed by server (%s); rejoining", _topic_label(topic), event
                        )
                        self.joined.discard(topic)
                        if topic not in queue:
                            queue.append(topic)
                        await send_joins()
                elif event in ("system", "presence_state", "presence_diff"):
                    log.debug("realtime %s on %s: %s", event, _topic_label(topic), _short(payload))
                else:
                    inner_event, inner_payload = unwrap_broadcast(event, payload)
                    if inner_event:
                        for ev in parse_broadcast(
                            inner_event,
                            inner_payload,
                            topic=topic,
                            tracker=self.tracker,
                            markets=self._markets,
                        ):
                            yield ev

            # -- housekeeping -------------------------------------------------------------------
            if now - last_heartbeat >= self.heartbeat_s:
                if heartbeat_ref is not None:
                    raise _Stall("previous heartbeat unanswered")
                heartbeat_ref = await self._send(ws, "phoenix", "heartbeat", {})
                last_heartbeat = now
            for ref, (jt, sent_at) in list(pending.items()):
                if now - sent_at > self.join_timeout_s:
                    pending.pop(ref, None)
                    log.warning("realtime join timed out for %s", _topic_label(jt))
                    schedule_retry(jt)
            for jt, retry_at in list(failed.items()):
                if now >= retry_at and jt not in queue and jt not in self.joined:
                    failed.pop(jt, None)
                    queue.append(jt)
            if self._markets_dirty:
                self._markets_dirty = False
                wanted = set(self.topics())
                for jt in list(self.joined):
                    if jt not in wanted:
                        self.joined.discard(jt)
                        try:
                            await self._send(ws, jt, "phx_leave", {})
                        except Exception:  # noqa: BLE001
                            pass
                queue = [t for t in queue if t in wanted]
                for jt in wanted:
                    if (
                        jt not in self.joined
                        and jt not in queue
                        and all(jt != p[0] for p in pending.values())
                    ):
                        queue.append(jt)
            if queue and len(pending) < self.join_batch:
                await send_joins()
            if self.joined and self.last_message_at is not None and now - self.last_message_at > self.stall_s:
                raise _Stall(f"no message for {self.stall_s:.0f}s")
            if now >= refresh_at:
                try:
                    token = await self.token_provider()
                except Exception as e:  # noqa: BLE001
                    log.warning("realtime token refresh failed (%s)", type(e).__name__)
                    if self.token is not None and self.token.seconds_left < 120:
                        raise _Stall("token about to expire and refresh failed") from e
                    refresh_at = now + 60.0
                else:
                    self.token = token
                    access_token = token.token
                    refresh_at = now + min(self.token_refresh_s, max(60.0, token.seconds_left - 600.0))
                    for jt in list(self.joined):
                        await self._send(ws, jt, "access_token", {"access_token": access_token})
                    log.info("realtime token refreshed on %d channels", len(self.joined))


def _short(v: Any, n: int = 160) -> str:
    s = str(v)
    return s if len(s) <= n else s[: n - 3] + "..."


def _topic_label(topic: str | None) -> str:
    return topic or "?"
