"""Live client for the documented SIG v1 API (https://sig.thesuper.market/api/v1/docs).

Responsibilities (owner: sig workstream):
- httpx.AsyncClient with Bearer auth, base URL from settings, read/write timeouts from config.limits.
- Two shared limiters (reads 90/min, writes 25/min) wrapping EVERY call; 429 -> freeze both for Retry-After.
- Records the server clock offset from the HTTP Date header (sigcup.common.time).
- Every write carries `tournamentId` and `idempotencyKey`; raises AuthError if the tournament id is unknown.
- Error classification per sigcup.sig.errors; OrderResult.uncertain=True on timeout/502.
- Cursor pagination helper for list endpoints (pagination.nextCursor / hasMore).
- Never logs headers or keys.
Implements sigcup.sig.adapter.ExchangeAdapter.

Payload shapes were reconstructed from the OpenAPI document and public client code (docs/research/platform.md
section 2), so every parser here is lenient: it tries several candidate keys, tolerates wrapped (`{"data": ...}`)
and bare bodies, and keeps unknown fields on the pydantic models (`extra="allow"`).

Usage::

    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min)   # ONE per process
    client = SigClient(api_key, budget, cfg=cfg)
    await client.get_tournament()                 # resolves + caches tournament_id
    res = await client.place_order(req, priority=PRIORITY_ARB)
    if res.uncertain: ...                         # reconcile with the same idempotency key
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from dateutil import parser as dateparser
from pydantic import Field

from sigcup.common.time import before_deadline, server_offset, set_server_offset_from_date_header, utc_now
from sigcup.settings import Limits, StrategyConfig, config, secrets
from sigcup.sig.errors import (
    AuthError,
    OrderStatusUnknown,
    RateLimited,
    RequestInFlight,
    SigError,
    TransientError,
    ValidationError,
)
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import (
    AccountInfo,
    ConstraintViolation,
    FillInfo,
    Leaderboard,
    LeaderboardEntry,
    Lenient,
    Level,
    MarketInfo,
    MultiLegResult,
    OrderBook,
    OrderInfo,
    OrderRequest,
    OrderResult,
    PositionInfo,
    PriceSnapshot,
    Relationship,
    TournamentInfo,
    round_tick,
)

log = logging.getLogger(__name__)

PRICES_CHUNK = 100  # GET /exchanges/prices accepts at most 100 ids
MULTI_LEG_MAX = 10
BATCH_MAX = 50
MAX_PAGES = 50
DEFAULT_RETRY_AFTER = 60.0

# Limiter priorities (lower numbers are served first). The strategy tiers mirror sigcup.trader.types.PRIORITY_*
# (not imported: the exchange layer does not depend on the trader).
PRIORITY_ARB, PRIORITY_THETA, PRIORITY_VALUE, PRIORITY_MM, PRIORITY_HOUSEKEEPING = 0, 1, 2, 3, 4
PRIORITY_CANCEL = 1  # pulling a resting quote reduces risk: ahead of value / mm placements
PRIORITY_RECONCILE = 1  # account, positions, open orders, fills: they resolve uncertain orders
PRIORITY_MARKET_DATA = 3  # bulk price and depth polling
_STRATEGY_PRIORITY = {
    "arb": PRIORITY_ARB,
    "election": PRIORITY_ARB,
    "theta": PRIORITY_THETA,
    "value": PRIORITY_VALUE,
    "convex": PRIORITY_VALUE,
    "fade": PRIORITY_VALUE,
    "follow": PRIORITY_VALUE,
    "mm": PRIORITY_MM,
}

# Server clock sampling: the HTTP Date header has 1 s resolution, so only a fresh (uncached, quick) response
# is sampled, a sample may always tighten the estimate upwards (each one is a lower bound on the true offset),
# and the estimate only moves down on a change bigger than the truncation + latency noise.
CLOCK_SAMPLE_MAX_RTT_S = 2.0
CLOCK_DECREASE_S = 3.0


def strategy_priority(strategy: Any) -> int:
    """Default write priority for an OrderRequest.strategy tag (manual / unknown -> housekeeping)."""
    s = str(getattr(strategy, "value", strategy) or "").lower()
    return _STRATEGY_PRIORITY.get(s, PRIORITY_HOUSEKEEPING)


class CancelAllIncomplete(SigError):
    """cancel-all did not confirm every cancel: `failed` lists the order ids that may still be live (empty when
    the exchange answered a partial result we could not read). Verify with get_open_orders()."""

    def __init__(self, message: str, status: int | None = None, *, cancelled: int = 0, failed: Sequence[str] = ()):
        super().__init__(message, status, "CANCEL_ALL_INCOMPLETE")
        self.cancelled = cancelled
        self.failed = list(failed)


_SHAPES_LOGGED: set[tuple[str, frozenset[str]]] = set()


def _warn_shape(kind: str, raw: Mapping[str, Any], message: str) -> None:
    """Log an unrecognised payload once per (kind, key set), with a raw sample, at WARNING."""
    key = (kind, frozenset(str(k) for k in raw.keys()))
    if key in _SHAPES_LOGGED:
        return
    _SHAPES_LOGGED.add(key)
    sample = str(dict(raw))
    log.warning("%s: keys=%s sample=%s", message, sorted(key[1]), sample[:400])

_PAGE_KEYS: tuple[str, ...] = (
    "data",
    "items",
    "results",
    "orders",
    "positions",
    "fills",
    "markets",
    "leaderboard",
    "relationships",
    "violations",
    "entries",
)


# ---------------------------------------------------------------------------------------------
# Lenient parsing helpers (shared with sigcup.sig.realtime)
# ---------------------------------------------------------------------------------------------


def pick(d: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """Return the first present, non-None value among candidate keys (dotted paths allowed)."""
    if not isinstance(d, Mapping):
        return default
    for key in keys:
        cur: Any = d
        ok = True
        for part in key.split("."):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur is not None:
            return cur
    return default


def to_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def to_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def to_bool(v: Any) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    return None


_FRAC_RE = re.compile(r"(\.\d{6})\d+")


def parse_dt(v: Any) -> datetime | None:
    """Parse ISO strings (any fractional precision, 'Z'), epoch seconds/milliseconds or datetimes to aware UTC."""
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.replace(tzinfo=UTC) if v.tzinfo is None else v.astimezone(UTC)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        secs = float(v)
        if secs > 1e11:  # milliseconds
            secs /= 1000.0
        try:
            return datetime.fromtimestamp(secs, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(v, str):
        s = _FRAC_RE.sub(r"\1", v.strip().replace("Z", "+00:00"))
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            try:
                dt = dateparser.parse(v)
            except (ValueError, OverflowError, TypeError):
                return None
        return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return None


def iso_ms(dt: datetime) -> str:
    """UTC ISO-8601 with millisecond precision and a trailing Z (the exchange's expirationDate format)."""
    dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def norm_side(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return "yes"
    if s in ("no", "n", "false", "0"):
        return "no"
    return s or None


def norm_action(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("buy", "bid", "b"):
        return "buy"
    if s in ("sell", "ask", "s"):
        return "sell"
    return s or None


def parse_level(raw: Any) -> Level | None:
    """A book level: {price, quantity|size|qty} or [price, qty]."""
    if isinstance(raw, Mapping):
        price = to_float(pick(raw, "price", "p", "priceLimit", "limit"))
        qty = to_int(pick(raw, "quantity", "size", "qty", "q", "shares", "amount"))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 2:
        price, qty = to_float(raw[0]), to_int(raw[1])
    else:
        return None
    if price is None or qty is None:
        return None
    return Level(price=price, quantity=qty)


def parse_levels(raw: Any, *, descending: bool) -> list[Level]:
    levels = [lv for lv in (parse_level(x) for x in (raw or [])) if lv is not None]
    levels.sort(key=lambda lv: lv.price, reverse=descending)
    return levels


def parse_book(raw: Mapping[str, Any], exchange_id: str | None = None) -> OrderBook | None:
    """Parse `{exchangeId, bids[], asks[], asOf{at,sequence}}` (or the market-detail flavour) into an OrderBook."""
    if not isinstance(raw, Mapping):
        return None
    inner = raw
    if "bids" not in raw and "asks" not in raw:
        # market-detail style: {"exchanges": [{"id", "bids", "asks"}]} or {"book": {...}} / {"orderbook": {...}}
        nested = pick(raw, "book", "orderbook", "orderBook")
        if isinstance(nested, Mapping):
            inner = nested
        else:
            exchanges = pick(raw, "exchanges", "books")
            if isinstance(exchanges, list) and exchanges:
                inner = exchanges[0]
                for ex in exchanges:
                    if (
                        exchange_id
                        and str(pick(ex, "exchangeId", "exchange_id", "id", default="")) == exchange_id
                    ):
                        inner = ex
                        break
    eid = (
        pick(inner, "exchangeId", "exchange_id", "id")
        or pick(raw, "exchangeId", "exchange_id")
        or exchange_id
    )
    if eid is None:
        return None
    ts = parse_dt(pick(inner, "asOf.at", "asOf", "at", "timestamp", "ts", "updatedAt", "updated_at", "time"))
    extra: dict[str, Any] = {}
    seq = to_int(pick(inner, "asOf.sequence", "sequence", "seq", "version"))
    if seq is not None:
        extra["sequence"] = seq
    nxt = parse_dt(pick(inner, "nextExpiryAt", "next_expiry_at"))
    if nxt is not None:
        extra["next_expiry_at"] = nxt
    return OrderBook(
        exchange_id=str(eid),
        bids=parse_levels(pick(inner, "bids", "buys", "yesBids", default=[]), descending=True),
        asks=parse_levels(pick(inner, "asks", "sells", "yesAsks", default=[]), descending=False),
        ts=ts,
        **extra,
    )


def synth_fill_id(d: Mapping[str, Any], order_id: str | None, exchange_id: str | None) -> str:
    """Deterministic id for fills the exchange pushes without one (realtime account_batch)."""
    parts = [
        str(order_id or ""),
        str(exchange_id or ""),
        str(pick(d, "executedAt", "filledAt", "createdAt", "timestamp", "at", "ts", "time", default="")),
        str(pick(d, "price", "fillPrice", default="")),
        str(pick(d, "quantity", "qty", "size", default="")),
        str(pick(d, "side", "option", default="")),
        str(pick(d, "action", default="")),
    ]
    return "synth:" + hashlib.sha256("|".join(parts).encode()).hexdigest()[:24]


def parse_fill(raw: Mapping[str, Any]) -> FillInfo | None:
    """Lenient fill parser shared by REST and realtime payloads. Returns None if essentials are missing."""
    if not isinstance(raw, Mapping):
        return None
    order_id = pick(raw, "orderId", "order_id", "order.id")
    exchange_id = pick(raw, "exchangeId", "exchange_id", "exchange.id", "marketExchangeId")
    price = to_float(pick(raw, "price", "fillPrice", "fill_price", "executionPrice", "avgPrice"))
    qty = to_int(pick(raw, "quantity", "qty", "size", "quantityTraded", "filledQuantity", "shares"))
    ts = parse_dt(
        pick(raw, "executedAt", "filledAt", "createdAt", "timestamp", "at", "ts", "time", "tradedAt", "date")
    )
    if exchange_id is None or price is None or qty is None:
        return None
    side = norm_side(pick(raw, "side", "option", "outcome", "outcomeSide"))
    if side not in ("yes", "no"):
        is_yes = to_bool(pick(raw, "isYes", "is_yes"))
        if is_yes is not None:
            side = "yes" if is_yes else "no"
    action = norm_action(pick(raw, "action", "orderType", "type", "direction"))
    if action not in ("buy", "sell"):
        is_buy = to_bool(pick(raw, "isBuy", "is_buy", "buy"))
        if is_buy is None:
            is_sell = to_bool(pick(raw, "isSell", "is_sell"))
            is_buy = None if is_sell is None else not is_sell
        if is_buy is not None:
            action = "buy" if is_buy else "sell"
    if qty < 0:  # signed quantity convention (+YES/-NO)
        qty = abs(qty)
        side = side or "no"
    fill_id = pick(raw, "fillId", "fill_id", "id", "tradeId", "trade_id", "executionId", "transactionId")
    extra = {
        k: v
        for k, v in raw.items()
        if k not in {"fill_id", "order_id", "exchange_id", "ts", "side", "action"}
    }
    if side is None or action is None:
        extra["side_or_action_missing"] = True
        _warn_shape("fill", raw, "fill without a recognisable side/action (defaulting to yes/buy)")
    return FillInfo(
        fill_id=str(fill_id) if fill_id is not None else synth_fill_id(raw, order_id, exchange_id),
        order_id=str(order_id) if order_id is not None else None,
        exchange_id=str(exchange_id),
        ts=ts or utc_now(),
        side=side or "yes",
        action=action or "buy",
        price=price,
        quantity=qty,
        fee=to_float(pick(raw, "fee", "fees", "commission")) or 0.0,
        **{k: v for k, v in extra.items() if k not in {"price", "quantity", "fee"}},
    )


def unwrap(payload: Any, *keys: str) -> Any:
    """Return payload['data'] (or another wrapper key) when the body is wrapped, else the payload itself."""
    if isinstance(payload, Mapping):
        for k in keys or ("data",):
            if k in payload and payload[k] is not None:
                return payload[k]
    return payload


def page_items(payload: Any, keys: Sequence[str] = _PAGE_KEYS) -> tuple[list[Any], str | None, bool]:
    """Split a list response into (items, next_cursor, has_more); tolerates bare lists and several wrappers,
    including a nested wrapper that carries its own pagination (`{"data": {"orders": [...], "pagination": ...}}`)."""
    if payload is None:
        return [], None, False
    items, cursor, has_more, _found = _page_split(payload, keys)
    if has_more is None:
        has_more = cursor is not None
    return items, cursor, bool(has_more)


def _page_split(payload: Any, keys: Sequence[str]) -> tuple[list[Any], str | None, bool | None, bool]:
    """(items, cursor, has_more or None when unstated, whether a list was found). Outer pagination wins over
    the pagination of the nested wrapper the items came from."""
    if isinstance(payload, list):
        return list(payload), None, None, True
    if not isinstance(payload, Mapping):
        return [], None, None, False
    items: list[Any] = []
    found = False
    inner_cursor: str | None = None
    inner_more: bool | None = None
    for k in keys:
        v = payload.get(k)
        if isinstance(v, list):
            items, found = v, True
            break
        if isinstance(v, Mapping):
            sub_items, sub_cursor, sub_more, sub_found = _page_split(v, keys)
            if sub_found:
                items, found, inner_cursor, inner_more = sub_items, True, sub_cursor, sub_more
                break
    pag = payload.get("pagination") if isinstance(payload.get("pagination"), Mapping) else payload
    cursor = pick(pag, "nextCursor", "next_cursor", "cursor", "next")
    has_more = to_bool(pick(pag, "hasMore", "has_more", "more"))
    cursor_s = str(cursor) if cursor is not None else inner_cursor
    if has_more is None:
        has_more = inner_more
    return items, cursor_s, has_more, found


def _error_info(resp: httpx.Response) -> tuple[str | None, str]:
    """(code, message) from an error body of the shape {error:{code,message}} / {code,message} / text."""
    code: str | None = None
    message = ""
    try:
        body = resp.json()
    except ValueError:
        body = None
    if isinstance(body, Mapping):
        err = body.get("error") if isinstance(body.get("error"), Mapping) else body
        code = pick(err, "code", "errorCode", "type")
        message = str(pick(err, "message", "detail", "details", "error", "msg", default="") or "")
        if not message and isinstance(body.get("error"), str):
            message = body["error"]
        if not message and isinstance(body.get("message"), str):
            message = body["message"]
    if not message:
        message = (resp.text or "")[:300]
    return (str(code) if code is not None else None), message or f"HTTP {resp.status_code}"


def _retry_after(resp: httpx.Response) -> float:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return DEFAULT_RETRY_AFTER
    try:
        return float(max(1.0, min(600.0, float(raw))))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        return float(max(1.0, min(600.0, (when - datetime.now(UTC)).total_seconds())))
    except (TypeError, ValueError):
        return DEFAULT_RETRY_AFTER


class RealtimeToken(Lenient):
    """POST /realtime/token response: Supabase project URL, anon key and a ~3-h access token."""

    supabase_url: str = Field(alias="supabaseUrl")
    anon_key: str = Field(alias="anonKey")
    token: str
    expires_at: datetime | None = Field(default=None, alias="expiresAt")
    user_channel: str | None = None  # e.g. "user:<profileId>" when the server names it

    @property
    def seconds_left(self) -> float:
        if self.expires_at is None:
            return 3 * 3600.0
        return (self.expires_at - utc_now()).total_seconds()


# ---------------------------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------------------------


class SigClient:
    """ExchangeAdapter over the documented SIG v1 REST API. One instance per process, sharing one Budget."""

    name = "sig"

    def __init__(
        self,
        api_key: str,
        budget: Budget,
        *,
        base_url: str | None = None,
        tournament_slug: str | None = None,
        tournament_id: str | None = None,
        cfg: StrategyConfig | None = None,
        limits: Limits | None = None,
        http: httpx.AsyncClient | None = None,
        enforce_deadline: bool = True,
    ) -> None:
        if not api_key:
            raise AuthError("SIG API key is empty (set SIG_API_KEY / SIG_READ_KEY)", 401, "NO_KEY")
        self.budget = budget
        self.cfg = cfg or config()
        self.limits = limits or self.cfg.limits
        self.base_url = (base_url or secrets().sig_base_url).rstrip("/")
        self.tournament_slug = (
            tournament_slug if tournament_slug is not None else secrets().sig_tournament_slug
        )
        self.tournament_id: str | None = tournament_id
        self.enforce_deadline = enforce_deadline
        self.profile_id: str | None = None
        self.username: str | None = None
        self._tournament: TournamentInfo | None = None
        self._fills_path: str | None = None
        self._positions_path: str | None = None
        self._clock_synced = False
        self._truncation_warned: set[str] = set()
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
            timeout=httpx.Timeout(self.limits.read_timeout_s),
        )

    # -- lifecycle ---------------------------------------------------------------------------
    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> SigClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- transport ---------------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        write: bool = False,
        priority: int = 2,
        extra_ok: Iterable[int] = (),
        deadline: bool = False,
        with_status: bool = False,
    ) -> Any:
        """One rate-limited HTTP call. Returns the parsed JSON body (None when empty), or (status, body) when
        `with_status`. `deadline=True` (order-creating writes) re-checks the competition deadline after the
        limiter wait, which can outlast a 429 freeze, so nothing is POSTed past the cutoff.

        Raises RateLimited (429, after freezing both budgets), RequestInFlight (409), AuthError (401/403),
        ValidationError (other 4xx), TransientError (503, or any 5xx/timeout on a read) and OrderStatusUnknown
        (502 / other 5xx / timeout / transport failure on a write: the request may have been applied).
        """
        limiter = self.budget.writes if write else self.budget.reads
        await limiter.wait(priority)
        if deadline:
            self._check_deadline()
        timeout = self.limits.write_timeout_s if write else self.limits.read_timeout_s
        params = {k: v for k, v in (params or {}).items() if v is not None}
        t0 = time.monotonic()
        try:
            resp = await self._http.request(method, path, params=params or None, json=json, timeout=timeout)
        except httpx.TimeoutException as e:
            log.warning("%s %s timed out after %.1fs", method, path, time.monotonic() - t0)
            if write:
                raise OrderStatusUnknown(f"timeout on {method} {path}", None, "TIMEOUT") from e
            raise TransientError(f"timeout on {method} {path}", None, "TIMEOUT") from e
        except httpx.TransportError as e:
            log.warning("%s %s transport error: %s", method, path, type(e).__name__)
            if write:
                raise OrderStatusUnknown(f"transport error on {method} {path}", None, "TRANSPORT") from e
            raise TransientError(f"transport error on {method} {path}", None, "TRANSPORT") from e
        rtt = time.monotonic() - t0
        self._sample_server_clock(resp, rtt)
        log.debug("%s %s -> %s in %.0f ms", method, path, resp.status_code, rtt * 1000)
        body = self._classify(resp, method, path, write, set(extra_ok))
        return (resp.status_code, body) if with_status else body

    def _sample_server_clock(self, resp: httpx.Response, rtt: float) -> None:
        """Update the server clock offset from a fresh response's Date header without 1-s truncation jitter."""
        date = resp.headers.get("Date")
        if not date or "Age" in resp.headers or rtt > CLOCK_SAMPLE_MAX_RTT_S:
            return  # absent, served from a cache (an old Date), or too slow to bound the offset tightly
        try:
            server = parsedate_to_datetime(date)
        except (TypeError, ValueError):
            return
        if server.tzinfo is None:
            server = server.replace(tzinfo=UTC)
        sample = (server - datetime.now(UTC)).total_seconds()
        current = server_offset().total_seconds()
        if not self._clock_synced or sample > current or sample < current - CLOCK_DECREASE_S:
            set_server_offset_from_date_header(date)
            self._clock_synced = True

    def _classify(self, resp: httpx.Response, method: str, path: str, write: bool, extra_ok: set[int]) -> Any:
        status = resp.status_code
        if 200 <= status < 300 or status in extra_ok:
            return _body(resp)
        code, message = _error_info(resp)
        if status == 429:
            ra = _retry_after(resp)
            self.budget.freeze_all(ra)
            log.warning("429 on %s %s: freezing all calls for %.0fs", method, path, ra)
            raise RateLimited(ra)
        if status == 409:
            raise RequestInFlight(message or "request in flight", status, code or "REQUEST_IN_FLIGHT")
        if status in (401, 403):
            raise AuthError(message, status, code or "AUTH")
        if status == 503:
            raise TransientError(message, status, code or "SERVICE_UNAVAILABLE")
        if 400 <= status < 500:
            raise ValidationError(message, status, code or "VALIDATION_ERROR")
        if write:
            raise OrderStatusUnknown(message, status, code or "ORDER_STATUS_UNKNOWN")
        raise TransientError(message, status, code or "SERVER_ERROR")

    async def _get(self, path: str, params: Mapping[str, Any] | None = None, *, priority: int = 2) -> Any:
        return await self._request("GET", path, params=params, priority=priority)

    async def _paginate(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        priority: int = 2,
        max_pages: int = MAX_PAGES,
        stop_when: Any = None,
    ) -> list[Any]:
        """Follow pagination.nextCursor / hasMore. `stop_when(items) -> bool` ends the walk early."""
        out: list[Any] = []
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(max_pages):
            p = dict(params or {})
            if cursor:
                p["cursor"] = cursor
            payload = await self._get(path, p, priority=priority)
            items, cursor, has_more = page_items(payload)
            out.extend(items)
            if stop_when is not None and stop_when(items):
                break
            if not has_more or not cursor or cursor in seen:
                limit = to_int(p.get("limit"))
                if limit and len(items) >= limit and not cursor and path not in self._truncation_warned:
                    self._truncation_warned.add(path)
                    log.warning(
                        "%s returned a full page (%d items) with no cursor: the list may be truncated (%s)",
                        path,
                        len(items),
                        str(payload)[:300],
                    )
                break
            seen.add(cursor)
        return out

    # -- tournament --------------------------------------------------------------------------
    async def get_tournament(self) -> TournamentInfo:
        """GET /tournaments/{slug}; caches `tournament_id` (and the tournament's own balance field)."""
        if not self.tournament_slug:
            raise AuthError("SIG_TOURNAMENT_SLUG is not set", 403, "NO_TOURNAMENT")
        raw = unwrap(await self._get(f"/tournaments/{self.tournament_slug}"), "data", "tournament")
        if not isinstance(raw, Mapping):
            raise ValidationError("unexpected tournament payload", None, "BAD_PAYLOAD")
        norm: dict[str, Any] = dict(raw)
        norm["id"] = str(pick(raw, "id", "tournamentId", "tournament_id", default=""))
        norm["slug"] = str(pick(raw, "slug", default=self.tournament_slug))
        for k in ("startDate", "endDate"):
            if k in norm:
                norm[k] = parse_dt(norm[k])
        if not norm["id"]:
            raise AuthError("tournament payload has no id", 403, "NO_TOURNAMENT")
        info = TournamentInfo.model_validate(norm)
        self.tournament_id = info.id
        self._tournament = info
        return info

    async def ensure_tournament_id(self) -> str:
        """The tournament id every order (and tournament-scoped read) must carry."""
        if self.tournament_id:
            return self.tournament_id
        await self.get_tournament()
        assert self.tournament_id is not None
        return self.tournament_id

    # -- market data -------------------------------------------------------------------------
    async def list_markets(self, status: str | None = None) -> list[MarketInfo]:
        """GET /tournaments/{slug}/markets (paginated); falls back to /markets?tournamentId=."""
        params: dict[str, Any] = {"limit": 100}
        if status:
            params["status"] = status
        rows: list[Any]
        if self.tournament_slug:
            try:
                rows = await self._paginate(f"/tournaments/{self.tournament_slug}/markets", params)
            except ValidationError as e:
                if e.status != 404:
                    raise
                rows = await self._paginate(
                    "/markets", {**params, "tournamentId": await self.ensure_tournament_id()}
                )
        else:
            rows = await self._paginate(
                "/markets", {**params, "tournamentId": await self.ensure_tournament_id()}
            )
        out: list[MarketInfo] = []
        for raw in rows:
            m = _parse_market(raw)
            if m is not None:
                out.append(m)
        return out

    async def get_market(self, market_id: str) -> MarketInfo | None:
        """GET /markets/{id} (detail incl. overround / hasArbitrageOpportunity); None on 404."""
        try:
            raw = unwrap(await self._get(f"/markets/{market_id}"), "data", "market")
        except ValidationError as e:
            if e.status == 404:
                return None
            raise
        return _parse_market(raw) if isinstance(raw, Mapping) else None

    async def get_prices(
        self, exchange_ids: list[str], *, priority: int = PRIORITY_MARKET_DATA
    ) -> dict[str, PriceSnapshot]:
        """GET /exchanges/prices?ids=... in chunks of 100 ids (one read per chunk)."""
        ids = [str(x) for x in dict.fromkeys(exchange_ids)]
        if not ids:
            return {}
        tid = await self.ensure_tournament_id()
        out: dict[str, PriceSnapshot] = {}
        for i in range(0, len(ids), PRICES_CHUNK):
            chunk = ids[i : i + PRICES_CHUNK]
            payload = await self._get(
                "/exchanges/prices", {"ids": ",".join(chunk), "tournamentId": tid}, priority=priority
            )
            for eid, snap in _parse_prices(payload).items():
                out[eid] = snap
        return out

    async def get_price(self, exchange_id: str) -> PriceSnapshot | None:
        """GET /exchanges/{id}/price."""
        payload = unwrap(await self._get(f"/exchanges/{exchange_id}/price"), "data", "price")
        if not isinstance(payload, Mapping):
            return None
        return _parse_price_row(payload, default_id=str(exchange_id))

    async def get_orderbook(
        self, exchange_id: str, depth: int = 20, *, priority: int = PRIORITY_MARKET_DATA
    ) -> OrderBook:
        """GET /exchanges/{id}/orderbook?depth=N (depth <= 200). Bids descending, asks ascending."""
        depth = max(1, min(200, int(depth)))
        tid = await self.ensure_tournament_id()
        payload = await self._get(
            f"/exchanges/{exchange_id}/orderbook", {"depth": depth, "tournamentId": tid}, priority=priority
        )
        raw = unwrap(payload, "data", "orderbook", "orderBook", "book")
        book = parse_book(raw, str(exchange_id)) if isinstance(raw, Mapping) else None
        if book is None:
            raise ValidationError(f"unexpected orderbook payload for {exchange_id}", None, "BAD_PAYLOAD")
        if book.ts is None:
            book.ts = utc_now()
        return book

    # -- account -----------------------------------------------------------------------------
    async def get_account(self, *, priority: int = PRIORITY_RECONCILE) -> AccountInfo:
        """GET /account; if it carries no balance, the tournament's `myBalance` is used (one extra read)."""
        raw = unwrap(await self._get("/account", priority=priority), "data", "account")
        raw = raw if isinstance(raw, Mapping) else {}
        pid = pick(raw, "profileId", "profile_id", "id", "userId", "user.id")
        if pid is not None:
            self.profile_id = str(pid)
        uname = pick(raw, "username", "handle", "name", "user.username")
        if uname is not None:
            self.username = str(uname)
        cash = to_float(
            pick(
                raw,
                "cash",
                "availableCash",
                "availableBalance",
                "balance",
                "myBalance",
                "cashBalance",
                "userBalance",
                "memberBalance",
                "tournamentBalance",
            )
        )
        extra = dict(raw)
        if cash is None and self.tournament_slug:
            t = await self.get_tournament()
            cash = t.my_balance
            extra["balance_source"] = "tournament.myBalance"
        if cash is None:
            raise ValidationError("account payload has no balance field", None, "BAD_PAYLOAD")
        positions_value = to_float(
            pick(
                raw, "positionsValue", "positions_value", "totalHoldingsValue", "holdingsValue", "marketValue"
            )
        )
        total = to_float(pick(raw, "total", "totalAccountValue", "equity", "netWorth", "accountValue"))
        if total is None and positions_value is not None:
            total = cash + positions_value
        for k in ("cash", "positions_value", "total"):
            extra.pop(k, None)
        return AccountInfo(cash=cash, positions_value=positions_value, total=total, **extra)

    async def get_positions(self, *, priority: int = PRIORITY_RECONCILE) -> list[PositionInfo]:
        """GET /tournaments/{slug}/portfolio/positions (fallback /portfolio/positions?tournamentId=)."""
        rows = await self._tournament_portfolio("positions", priority=priority)
        out: list[PositionInfo] = []
        for raw in rows:
            p = _parse_position(raw)
            if p is not None:
                out.append(p)
        return out

    async def get_open_orders(self, *, priority: int = PRIORITY_RECONCILE) -> list[OrderInfo]:
        """GET /orders?status=open&tournamentId=... (paginated)."""
        tid = await self.ensure_tournament_id()
        rows = await self._paginate(
            "/orders", {"status": "open", "tournamentId": tid, "limit": 100}, priority=priority
        )
        out: list[OrderInfo] = []
        for raw in rows:
            o = _parse_order(raw)
            if o is not None:
                out.append(o)
        return out

    async def get_order(self, order_id: str, *, priority: int = PRIORITY_RECONCILE) -> OrderInfo | None:
        """GET /orders/{id}; None when the exchange does not know the id."""
        try:
            raw = unwrap(await self._get(f"/orders/{order_id}", priority=priority), "data", "order")
        except ValidationError as e:
            if e.status == 404:
                return None
            raise
        return _parse_order(raw) if isinstance(raw, Mapping) else None

    async def get_order_fills(self, order_id: str, *, priority: int = PRIORITY_RECONCILE) -> list[FillInfo]:
        """GET /orders/{id}/fills."""
        rows = await self._paginate(f"/orders/{order_id}/fills", {"limit": 200}, priority=priority)
        return [f for f in (parse_fill(r) for r in rows if isinstance(r, Mapping)) if f is not None]

    async def get_fills(
        self, since: datetime | None = None, *, priority: int = PRIORITY_RECONCILE
    ) -> list[FillInfo]:
        """GET /tournaments/{slug}/portfolio/fills (paginated, newest first); stops once a whole page predates
        `since`. Fallback: /portfolio/fills?tournamentId=."""
        since_utc = since.replace(tzinfo=UTC) if since is not None and since.tzinfo is None else since

        def older_page(items: list[Any]) -> bool:
            """True when a newest-first page lies entirely before `since` (older pages cannot help)."""
            if since_utc is None or len(items) < 2:
                return False
            stamps = [
                parse_dt(pick(r, "executedAt", "filledAt", "createdAt", "timestamp", "at", "ts", "time"))
                for r in items
                if isinstance(r, Mapping)
            ]
            stamps = [s for s in stamps if s is not None]
            if len(stamps) < 2 or stamps[0] < stamps[-1]:  # oldest-first (or unknown) ordering: keep walking
                return False
            return all(s < since_utc for s in stamps)

        rows = await self._tournament_portfolio("fills", stop_when=older_page, priority=priority)
        fills = [f for f in (parse_fill(r) for r in rows if isinstance(r, Mapping)) if f is not None]
        if since_utc is not None:
            fills = [f for f in fills if f.ts >= since_utc]
        return fills

    async def _tournament_portfolio(self, kind: str, *, stop_when: Any = None, priority: int = 2) -> list[Any]:
        """Paginate a portfolio list, remembering which of the two documented paths works."""
        cache_attr = "_fills_path" if kind == "fills" else "_positions_path"
        path = getattr(self, cache_attr)
        params: dict[str, Any] = {"limit": 200}
        if path is None:
            candidates: list[tuple[str, dict[str, Any]]] = []
            if self.tournament_slug:
                candidates.append((f"/tournaments/{self.tournament_slug}/portfolio/{kind}", params))
            candidates.append(
                (f"/portfolio/{kind}", {**params, "tournamentId": await self.ensure_tournament_id()})
            )
            last: SigError | None = None
            for cand, p in candidates:
                try:
                    rows = await self._paginate(cand, p, stop_when=stop_when, priority=priority)
                except ValidationError as e:
                    if e.status != 404:
                        raise
                    last = e
                    continue
                setattr(self, cache_attr, cand)
                return rows
            raise last or ValidationError(f"no {kind} endpoint", 404, "NOT_FOUND")
        if path.startswith("/portfolio/"):
            params["tournamentId"] = await self.ensure_tournament_id()
        return await self._paginate(path, params, stop_when=stop_when, priority=priority)

    # -- orders ------------------------------------------------------------------------------
    def _order_body(self, req: OrderRequest, tid: str) -> dict[str, Any]:
        if req.quantity <= 0:
            raise ValidationError(f"quantity must be positive, got {req.quantity}", None, "BAD_QUANTITY")
        if not req.idempotency_key:
            raise ValidationError("idempotency_key is required", None, "NO_IDEMPOTENCY_KEY")
        body: dict[str, Any] = {
            "exchangeId": str(req.exchange_id),
            "side": req.side,
            "action": req.action,
            "quantity": int(req.quantity),
            "price": round_tick(req.price),
            "tournamentId": tid,
            "idempotencyKey": req.idempotency_key,
        }
        if req.expires_at is not None:
            body["expirationDate"] = iso_ms(req.expires_at)
        return body

    def _check_deadline(self) -> None:
        if self.enforce_deadline and not before_deadline(self.cfg.deadline_utc):
            raise ValidationError("past the competition deadline: no orders may be sent", None, "DEADLINE")

    async def place_order(self, req: OrderRequest, *, priority: int | None = None) -> OrderResult:
        """POST /orders. Never raises for an unknown outcome: returns OrderResult(uncertain=True) instead.
        The limiter priority defaults from `req.strategy` (arb 0, theta 1, value 2, mm 3, other 4)."""
        self._check_deadline()
        tid = await self.ensure_tournament_id()
        body = self._order_body(req, tid)
        prio = priority if priority is not None else strategy_priority(req.strategy)
        try:
            payload = await self._request(
                "POST", "/orders", json=body, write=True, priority=prio, deadline=True
            )
        except OrderStatusUnknown as e:
            log.warning("order %s on %s uncertain: %s", req.idempotency_key, req.exchange_id, e)
            return OrderResult(
                uncertain=True, status="unknown", idempotencyKey=req.idempotency_key, error=str(e)
            )
        res = _parse_order_result(unwrap(payload, "data", "order"), req.idempotency_key)
        if res.order_id is None and res.status is None:
            # A 2xx with no recognisable receipt: treat as unknown rather than "placed".
            res.uncertain = True
        return res

    async def place_multi_leg(
        self, legs: list[OrderRequest], *, priority: int | None = None
    ) -> MultiLegResult:
        """POST /orders/multi-leg (<= 10 legs, atomic placement, non-atomic fills).

        The response is handled atomically: per-leg receipts are matched by idempotency key, then `index`, then
        position; a group-level error rejects every leg; a body that does not account for every leg marks every
        leg uncertain (never a mix of one leg holding the envelope and padded unknowns)."""
        if not legs:
            return MultiLegResult(results=[], atomic_ok=True)
        if len(legs) > MULTI_LEG_MAX:
            raise ValidationError(f"multi-leg accepts at most {MULTI_LEG_MAX} legs", None, "TOO_MANY_LEGS")
        self._check_deadline()
        tid = await self.ensure_tournament_id()
        group_key = _group_key(legs)
        keys = [r.idempotency_key for r in legs]
        body = {
            "tournamentId": tid,
            "idempotencyKey": group_key,
            "legs": [self._order_body(r, tid) for r in legs],
        }
        prio = priority if priority is not None else min(strategy_priority(r.strategy) for r in legs)
        try:
            payload = await self._request(
                "POST", "/orders/multi-leg", json=body, write=True, priority=prio, deadline=True
            )
        except OrderStatusUnknown as e:
            log.warning("multi-leg %s uncertain: %s", group_key, e)
            return MultiLegResult(group_id=group_key, results=_all_uncertain(keys), atomic_ok=False)
        gid = (
            pick(unwrap(payload), "groupId", "group_id", "multiLegId", "id")
            if isinstance(payload, Mapping)
            else None
        )
        group_id = str(gid) if gid is not None else group_key
        items = _split_legs(payload, keys)
        if items is None:
            err = _error_envelope(payload)
            if err is not None:
                log.warning("multi-leg %s rejected as a group: %s", group_key, err[1])
                return MultiLegResult(group_id=group_id, results=_all_rejected(keys, err), atomic_ok=False)
            log.warning(
                "multi-leg %s: response cannot be split into %d legs; all legs unknown: %s",
                group_key,
                len(keys),
                str(payload)[:400],
            )
            return MultiLegResult(group_id=group_id, results=_all_uncertain(keys), atomic_ok=False)
        results = [_parse_order_result(it, k) for it, k in zip(items, keys, strict=True)]
        if not all(_has_receipt(r) for r in results):
            log.warning(
                "multi-leg %s: response does not account for every leg; all legs unknown: %s",
                group_key,
                str(payload)[:400],
            )
            return MultiLegResult(group_id=group_id, results=_all_uncertain(keys), atomic_ok=False)
        ok = all(
            not r.uncertain and (r.status or "").lower() not in ("rejected", "error", "failed")
            for r in results
        )
        return MultiLegResult(group_id=group_id, results=results, atomic_ok=ok)

    async def place_batch(self, reqs: list[OrderRequest], *, priority: int | None = None) -> list[OrderResult]:
        """POST /orders/batch (<= 50 independent orders; a 207 carries per-order results). An order whose result
        is missing or unrecognisable is reported uncertain."""
        if not reqs:
            return []
        if len(reqs) > BATCH_MAX:
            raise ValidationError(f"batch accepts at most {BATCH_MAX} orders", None, "TOO_MANY_ORDERS")
        self._check_deadline()
        tid = await self.ensure_tournament_id()
        keys = [r.idempotency_key for r in reqs]
        body = {
            "tournamentId": tid,
            "idempotencyKey": _group_key(reqs),
            "orders": [self._order_body(r, tid) for r in reqs],
        }
        prio = priority if priority is not None else min(strategy_priority(r.strategy) for r in reqs)
        try:
            payload = await self._request(
                "POST", "/orders/batch", json=body, write=True, priority=prio, extra_ok=(207,), deadline=True
            )
        except OrderStatusUnknown as e:
            log.warning("batch uncertain: %s", e)
            return _all_uncertain(keys)
        items = _split_legs(payload, keys)
        if items is None:
            err = _error_envelope(payload)
            if err is not None:
                return _all_rejected(keys, err)
            log.warning("batch: response cannot be split into %d orders: %s", len(keys), str(payload)[:400])
            return _all_uncertain(keys)
        results = [_parse_order_result(it, k) for it, k in zip(items, keys, strict=True)]
        for r in results:
            if not _has_receipt(r):
                r.uncertain = True
        return results

    async def cancel_order(self, order_id: str, *, priority: int = PRIORITY_CANCEL) -> bool:
        """DELETE /orders/{id}. True when cancelled; False ONLY when the exchange no longer knows the order
        (404/410: already filled, expired or cancelled).

        An unknown outcome is never reported as False: a timeout, transport failure, 502 or other 5xx raises
        OrderStatusUnknown and a 503 raises TransientError, because the order may still be live."""
        tid = self.tournament_id
        try:
            await self._request(
                "DELETE", f"/orders/{order_id}", params={"tournamentId": tid}, write=True, priority=priority
            )
        except ValidationError as e:
            if e.status in (404, 410):
                return False
            raise
        except OrderStatusUnknown as e:
            log.warning("cancel %s uncertain (order may still be live): %s", order_id, e)
            raise
        return True

    async def cancel_all(self, exchange_id: str | None = None, *, priority: int = 0) -> int:
        """POST /orders/cancel-all (priority 0: the kill switch). Returns how many orders were cancelled.

        A failure is never reported as "nothing to cancel". Raises CancelAllIncomplete when the per-order results
        (207/422) name orders that were not cancelled (`.failed`) or when a 207 cannot be read; ValidationError for
        an error envelope (e.g. a 422 with no per-order results); OrderStatusUnknown on a timeout / 5xx. Orders
        reported as already gone (not found, filled, expired) are not failures."""
        tid = await self.ensure_tournament_id()
        body: dict[str, Any] = {"tournamentId": tid}
        if exchange_id:
            body["exchangeId"] = str(exchange_id)
        status, payload = await self._request(
            "POST",
            "/orders/cancel-all",
            json=body,
            write=True,
            priority=priority,
            extra_ok=(207, 422),
            with_status=True,
        )
        cancelled, failed, recognised = _cancel_outcome(payload)
        if failed:
            log.error("cancel-all: %d cancelled, %d NOT cancelled: %s", cancelled, len(failed), failed[:20])
            raise CancelAllIncomplete(
                f"cancel-all left {len(failed)} orders uncancelled", status, cancelled=cancelled, failed=failed
            )
        if recognised:
            return cancelled
        err = _error_envelope(payload)
        if err is not None or status == 422:
            code, message = err or (None, f"cancel-all rejected: {str(payload)[:200]}")
            raise ValidationError(message, status, code or "VALIDATION_ERROR")
        if status == 207:
            raise CancelAllIncomplete(
                f"cancel-all returned an unreadable partial result: {str(payload)[:200]}", status
            )
        if payload is not None:
            log.warning("cancel-all: unrecognised %s body, count unknown: %s", status, str(payload)[:300])
        return 0

    # -- tournament-level reads --------------------------------------------------------------
    async def get_leaderboard(self, limit: int = 25) -> Leaderboard:
        """GET /tournaments/{slug}/leaderboard?period=all: entries, memberCount, our rank/total."""
        if not self.tournament_slug:
            raise AuthError("SIG_TOURNAMENT_SLUG is not set", 403, "NO_TOURNAMENT")
        payload = await self._get(
            f"/tournaments/{self.tournament_slug}/leaderboard", {"period": "all", "limit": limit}
        )
        return _parse_leaderboard(payload, me_username=self.username, me_profile=self.profile_id)

    async def get_relationships(self) -> list[Relationship]:
        """GET /relationships?tournamentId=... (mutually_exclusive groups with isExhaustive)."""
        tid = await self.ensure_tournament_id()
        rows = await self._paginate("/relationships", {"tournamentId": tid, "limit": 100})
        return [r for r in (_parse_relationship(x) for x in rows if isinstance(x, Mapping)) if r is not None]

    async def get_constraint_violations(
        self, min_violation: float | None = None
    ) -> list[ConstraintViolation]:
        """GET /relationships/constraints?violationsOnly=true (&minViolation=)."""
        tid = await self.ensure_tournament_id()
        params: dict[str, Any] = {"tournamentId": tid, "violationsOnly": "true"}
        if min_violation is not None:
            params["minViolation"] = min_violation
        rows = await self._paginate("/relationships/constraints", params)
        return [v for v in (_parse_violation(x) for x in rows if isinstance(x, Mapping)) if v is not None]

    async def realtime_token(self, *, priority: int = PRIORITY_HOUSEKEEPING) -> RealtimeToken:
        """POST /realtime/token -> supabaseUrl, anonKey, token, expiresAt (+ channels.user when given).

        It costs a write, so it queues behind every order and cancel (housekeeping priority); RealtimeFeed reuses
        one token while it is valid."""
        payload = unwrap(
            await self._request("POST", "/realtime/token", json={}, write=True, priority=priority)
        )
        if not isinstance(payload, Mapping):
            raise ValidationError("unexpected realtime token payload", None, "BAD_PAYLOAD")
        url = pick(payload, "supabaseUrl", "supabase_url", "url", "realtimeUrl")
        anon = pick(payload, "anonKey", "anon_key", "apikey", "apiKey")
        token = pick(payload, "token", "accessToken", "access_token", "jwt")
        if not (url and anon and token):
            raise ValidationError("realtime token payload is missing fields", None, "BAD_PAYLOAD")
        expires = parse_dt(pick(payload, "expiresAt", "expires_at", "expiry"))
        if expires is None:
            ttl = to_float(pick(payload, "expiresIn", "expires_in", "ttl"))
            expires = utc_now() + timedelta(seconds=ttl if ttl else 3 * 3600)
        user_channel = pick(payload, "channels.user", "userChannel", "user_channel")
        if user_channel is None and self.profile_id:
            user_channel = f"user:{self.profile_id}"
        extras = {
            k: v for k, v in payload.items() if k not in {"supabaseUrl", "anonKey", "token", "expiresAt"}
        }
        return RealtimeToken(
            supabaseUrl=str(url),
            anonKey=str(anon),
            token=str(token),
            expiresAt=expires,
            user_channel=str(user_channel) if user_channel else None,
            **{k: v for k, v in extras.items() if k != "user_channel"},
        )


# ---------------------------------------------------------------------------------------------
# Private parsers
# ---------------------------------------------------------------------------------------------


def _body(resp: httpx.Response) -> Any:
    if not resp.content:
        return None
    try:
        return resp.json()
    except ValueError:
        return {"raw": resp.text}


def _group_key(reqs: Sequence[OrderRequest]) -> str:
    """Deterministic idempotency key for a multi-leg / batch body: a retry resends the same key."""
    return hashlib.sha256("|".join(r.idempotency_key for r in reqs).encode()).hexdigest()[:32]


def _parse_market(raw: Any) -> MarketInfo | None:
    if not isinstance(raw, Mapping):
        return None
    mid = pick(raw, "id", "marketId", "market_id")
    title = pick(raw, "title", "name", "question")
    if mid is None or title is None:
        return None
    exchange_id = pick(raw, "exchangeId", "exchange_id")
    exchanges = pick(raw, "exchanges", default=[])
    if exchange_id is None and isinstance(exchanges, list) and exchanges:
        chosen = exchanges[0]
        for ex in exchanges:
            opt = str(pick(ex, "option", "side", "outcome", default="")).lower()
            if opt in ("yes", "y"):
                chosen = ex
                break
        exchange_id = pick(chosen, "id", "exchangeId", "exchange_id")
    cats = pick(raw, "categories")
    category = pick(raw, "category")
    if category is None and isinstance(cats, list) and cats:
        c0 = cats[0]
        category = pick(c0, "name", "slug", "id") if isinstance(c0, Mapping) else str(c0)
    norm: dict[str, Any] = dict(raw)
    norm.update(
        {
            "id": str(mid),
            "title": str(title),
            "exchangeId": str(exchange_id) if exchange_id is not None else None,
            "status": pick(raw, "status", "state"),
            "category": category,
            "settlementDate": parse_dt(
                pick(raw, "settlementDate", "settlement_date", "resolutionDate", "endDate")
            ),
            "resolution": pick(
                raw, "resolution", "resolutionCriteria", "resolutionRules", "rules", "description"
            ),
            "overround": to_float(pick(raw, "overround")),
            "hasArbitrageOpportunity": to_bool(
                pick(raw, "hasArbitrageOpportunity", "has_arbitrage_opportunity")
            ),
            "isMultiOutcome": to_bool(pick(raw, "isMultiOutcome", "is_multi_outcome")),
        }
    )
    norm.pop("exchange_id", None)
    return MarketInfo.model_validate(norm)


def _parse_price_row(row: Mapping[str, Any], default_id: str | None = None) -> PriceSnapshot | None:
    eid = pick(row, "exchangeId", "exchange_id", "id", "exchange.id") or default_id
    if eid is None:
        return None
    extra = {
        k: v
        for k, v in row.items()
        if k not in {"exchange_id", "best_bid", "best_ask", "last", "bid_size", "ask_size", "ts"}
    }
    return PriceSnapshot(
        exchange_id=str(eid),
        best_bid=to_float(pick(row, "bestBid", "best_bid", "bid", "bestBidPrice", "yesBid")),
        best_ask=to_float(pick(row, "bestAsk", "best_ask", "ask", "bestAskPrice", "yesAsk")),
        last=to_float(pick(row, "latestPrice", "lastPrice", "last", "lastTradePrice", "last_price", "price")),
        bid_size=to_int(pick(row, "bidSize", "bestBidSize", "bid_size", "bidQuantity", "bestBidQuantity")),
        ask_size=to_int(pick(row, "askSize", "bestAskSize", "ask_size", "askQuantity", "bestAskQuantity")),
        ts=parse_dt(pick(row, "asOf.at", "asOf", "updatedAt", "updated_at", "timestamp", "ts", "at"))
        or utc_now(),
        **extra,
    )


def _parse_prices(payload: Any) -> dict[str, PriceSnapshot]:
    out: dict[str, PriceSnapshot] = {}
    body = unwrap(payload, "data", "prices", "exchanges")
    if isinstance(body, Mapping) and not any(k in body for k in ("exchangeId", "bestBid", "bid", "best_bid")):
        # {"<id>": {...}, ...}
        for key, row in body.items():
            if isinstance(row, Mapping):
                snap = _parse_price_row(row, default_id=str(key))
                if snap is not None:
                    out[snap.exchange_id] = snap
        return out
    rows = body if isinstance(body, list) else [body]
    for row in rows:
        if isinstance(row, Mapping):
            snap = _parse_price_row(row)
            if snap is not None:
                out[snap.exchange_id] = snap
    return out


def _parse_position(raw: Any) -> PositionInfo | None:
    if not isinstance(raw, Mapping):
        return None
    eid = pick(raw, "exchangeId", "exchange_id", "exchange.id")
    qty = to_int(pick(raw, "quantity", "qty", "shares", "position", "netQuantity"))
    split = False
    if qty is None:  # {yesShares, noShares} flavour: net it (+YES / -NO)
        yes = to_int(pick(raw, "yesShares", "yes_shares", "yesQuantity", "yes_quantity"))
        no = to_int(pick(raw, "noShares", "no_shares", "noQuantity", "no_quantity"))
        if yes is not None or no is not None:
            qty, split = (yes or 0) - (no or 0), True
    if eid is None or qty is None:
        _warn_shape("position", raw, "dropping a position row with no recognisable exchange id / quantity")
        return None
    side = norm_side(pick(raw, "side", "option", "outcome"))
    if side == "no" and qty > 0 and not split:
        qty = -qty
    extra = {k: v for k, v in raw.items() if k not in {"exchange_id", "quantity", "avg_price", "mark"}}
    return PositionInfo(
        exchange_id=str(eid),
        quantity=qty,
        avg_price=to_float(
            pick(raw, "avgCost", "avgPrice", "averagePrice", "averagePricePaid", "avg_price", "costBasis")
        ),
        mark=to_float(pick(raw, "currentPrice", "mark", "markPrice", "latestPrice", "lastPrice")),
        **extra,
    )


def _parse_order(raw: Any) -> OrderInfo | None:
    if not isinstance(raw, Mapping):
        return None
    oid = pick(raw, "orderId", "order_id", "id")
    if oid is None:
        return None
    norm: dict[str, Any] = {
        k: v
        for k, v in raw.items()
        if k
        not in {
            "orderId",
            "order_id",
            "id",
            "exchangeId",
            "exchange_id",
            "createdAt",
            "expirationDate",
            "expiresAt",
        }
    }
    norm.update(
        {
            "orderId": str(oid),
            "exchangeId": _opt_str(pick(raw, "exchangeId", "exchange_id", "exchange.id")),
            "side": norm_side(pick(raw, "side", "option", "outcome")),
            "action": norm_action(pick(raw, "action", "orderType", "type", "direction")),
            "price": to_float(pick(raw, "price", "priceLimit", "limitPrice")),
            "quantity": to_int(pick(raw, "quantity", "qty", "size", "originalQuantity")),
            "filled": to_int(
                pick(raw, "filled", "filledQuantity", "quantityFilled", "quantityTraded", "executedQuantity")
            ),
            "status": _opt_str(pick(raw, "status", "state")),
            "idempotencyKey": _opt_str(pick(raw, "idempotencyKey", "idempotency_key", "clientOrderId")),
            "createdAt": parse_dt(pick(raw, "createdAt", "created_at", "timestamp", "placedAt")),
            "expirationDate": parse_dt(pick(raw, "expirationDate", "expiresAt", "expires_at", "expiration")),
        }
    )
    if norm["filled"] is None:
        remaining = to_int(pick(raw, "remainingQuantity", "remaining", "openQuantity"))
        if remaining is not None and norm["quantity"] is not None:
            norm["filled"] = max(0, norm["quantity"] - remaining)
    return OrderInfo.model_validate(norm)


def _opt_str(v: Any) -> str | None:
    return None if v is None else str(v)


_ORDER_RESULT_KEYS = {
    "orderId",
    "order_id",
    "status",
    "quantityTraded",
    "quantity_traded",
    "totalCost",
    "total_cost",
    "fillPrice",
    "fill_price",
    "uncertain",
    "idempotencyKey",
}


def _parse_order_result(raw: Any, idempotency_key: str | None = None) -> OrderResult:
    """One order receipt (`{orderId, status, quantityTraded, totalCost, fillPrice, ...}`), lenient.

    A missing receipt (None) is reported as uncertain; `idempotencyKey` is always attached so the OMS can
    match receipts back to intents.
    """
    if not isinstance(raw, Mapping):
        return OrderResult(
            uncertain=raw is None, status=None if raw is None else str(raw), idempotencyKey=idempotency_key
        )
    inner = raw
    if isinstance(raw.get("data"), Mapping):
        inner = {**raw, **raw["data"]}
    if isinstance(raw.get("order"), Mapping):
        inner = {**inner, **raw["order"]}
    err = inner.get("error")
    status = _opt_str(pick(inner, "status", "state"))
    if err and status is None:
        status = "rejected"
    norm: dict[str, Any] = {k: v for k, v in inner.items() if k not in _ORDER_RESULT_KEYS}
    norm.update(
        {
            "idempotencyKey": _opt_str(pick(inner, "idempotencyKey", "idempotency_key")) or idempotency_key,
            "orderId": _opt_str(pick(inner, "orderId", "order_id", "id")),
            "status": status,
            "quantityTraded": to_int(
                pick(inner, "quantityTraded", "quantity_traded", "filledQuantity", "filled")
            ),
            "totalCost": to_float(pick(inner, "totalCost", "total_cost", "cost")),
            "fillPrice": to_float(pick(inner, "fillPrice", "fill_price", "avgFillPrice", "averagePrice")),
        }
    )
    if isinstance(err, Mapping):
        norm["error"] = pick(err, "message", "code", default=str(err))
    return OrderResult.model_validate(norm)


def _item_key(it: Any) -> str | None:
    """The idempotency key a per-leg / per-order result echoes (top level, or inside data / order)."""
    if not isinstance(it, Mapping):
        return None
    for src in (it, it.get("data"), it.get("order")):
        if isinstance(src, Mapping):
            k = pick(src, "idempotencyKey", "idempotency_key", "clientOrderId")
            if k is not None:
                return str(k)
    return None


def _split_legs(payload: Any, keys: Sequence[str]) -> list[Any] | None:
    """Per-leg result objects in leg order, or None when the body cannot be split leg by leg.

    Shapes: {data:[{index,data}]}, {results|orders|legs|items:[...]}, a bare list, results keyed by idempotency key
    ({orders:{<key>:{...}}}) or by index, and a bare id list ({orderIds:[...]}). List items are matched to legs
    by idempotency key first, then by `index`, then by position (only when the counts agree). A leg with no
    matching item gets None (the caller treats it as unknown)."""
    n = len(keys)
    body = unwrap(payload, "data")
    if isinstance(body, list):
        return _match_items(body, keys)
    if not isinstance(body, Mapping):
        return None
    found = pick(body, "results", "orders", "legs", "items", "orderIds", "order_ids")
    if isinstance(found, list):
        return _match_items(found, keys)
    keyed = _keyed_results(found if isinstance(found, Mapping) else body, keys)
    if keyed is not None:
        return keyed
    if n == 1 and pick(body, "orderId", "order_id") is not None:
        return [body]  # a single-order receipt for a one-leg request
    return None


def _keyed_results(mapping: Mapping[str, Any], keys: Sequence[str]) -> list[Any] | None:
    if any(k in mapping for k in keys):
        return [mapping.get(k) for k in keys]
    if mapping and all(str(k).isdigit() for k in mapping):
        return [mapping.get(str(i)) for i in range(len(keys))]
    return None


def _match_items(items: Sequence[Any], keys: Sequence[str]) -> list[Any] | None:
    n = len(keys)
    item_keys = [_item_key(it) for it in items]
    if any(k in keys for k in item_keys if k is not None):
        by_key = {k: it for k, it in zip(item_keys, items, strict=True) if k is not None}
        return [by_key.get(k) for k in keys]
    if items and all(isinstance(it, Mapping) and to_int(it.get("index")) is not None for it in items):
        by_index = {to_int(it.get("index")): it for it in items}
        return [by_index.get(i) for i in range(n)]
    if len(items) != n:
        return None
    out: list[Any] = []
    for it in items:
        if isinstance(it, Mapping):
            out.append(it)
        elif isinstance(it, (str, int)) and not isinstance(it, bool):
            out.append({"orderId": str(it)})  # a bare order id
        else:
            out.append(None)
    return out


def _has_receipt(r: OrderResult) -> bool:
    return not r.uncertain and (r.order_id is not None or r.status is not None)


def _all_uncertain(keys: Sequence[str]) -> list[OrderResult]:
    return [OrderResult(uncertain=True, status="unknown", idempotencyKey=k) for k in keys]


def _all_rejected(keys: Sequence[str], err: tuple[str | None, str]) -> list[OrderResult]:
    code, message = err
    return [OrderResult(status="rejected", idempotencyKey=k, error=message, errorCode=code) for k in keys]


def _error_envelope(payload: Any) -> tuple[str | None, str] | None:
    """(code, message) when a body is a request-level failure ({success:false,...} / {error:{...}}), else None."""
    if not isinstance(payload, Mapping):
        return None
    data = payload.get("data")
    err = payload.get("error")
    failed = to_bool(payload.get("success")) is False or (
        isinstance(data, Mapping) and to_bool(data.get("success")) is False
    )
    if not (failed or err):
        return None
    if not isinstance(err, Mapping) and isinstance(data, Mapping) and isinstance(data.get("error"), Mapping):
        err = data["error"]
    src = err if isinstance(err, Mapping) else payload
    code = pick(src, "code", "errorCode")
    message = pick(src, "message", "detail", "details", "msg")
    if message is None and isinstance(err, str):
        message = err
    return (str(code) if code is not None else None), str(message or "request failed")


_GONE_HINTS = (
    "not_found",
    "notfound",
    "not found",
    "already",
    "gone",
    "not_open",
    "not open",
    "filled",
    "expired",
    "closed",
    "inactive",
)
_CANCEL_COUNT_KEYS = ("cancelled", "canceled", "cancelledCount", "canceledCount", "count")
_CANCEL_FAILED_KEYS = ("failed", "failures", "errors", "notCancelled", "not_cancelled")
_CANCEL_RESULT_KEYS = ("results", "data", "orders", "cancelledOrders", "cancelled_orders", "canceledOrders")


def _cancel_item(it: Any) -> str:
    """'ok' | 'gone' | 'failed' for one per-order cancel result (a bare id counts as cancelled)."""
    if not isinstance(it, Mapping):
        return "ok" if it is not None else "failed"
    st = str(pick(it, "status", "state", default="")).lower()
    err = it.get("error")
    ok = to_bool(pick(it, "ok", "success", "cancelled", "canceled"))
    if st in ("filled", "expired"):
        return "gone"
    if not (err or st in ("error", "failed", "rejected", "not_found") or ok is False):
        return "ok"
    if isinstance(err, Mapping):
        err_text = " ".join(str(err.get(k, "")) for k in ("code", "message"))
    else:
        err_text = str(err or "")
    detail = f"{st} {err_text} {pick(it, 'reason', 'message', default='')}".lower()
    return "gone" if any(h in detail for h in _GONE_HINTS) else "failed"


def _order_id_of(it: Any) -> str:
    if isinstance(it, Mapping):
        return str(pick(it, "orderId", "order_id", "id", default="?"))
    return str(it)


def _tally(items: Sequence[Any], *, failed_list: bool = False) -> tuple[int, list[str]]:
    """(cancelled, failed ids) over per-order results; in a `failed` list every non-gone item is a failure."""
    cancelled = 0
    failed: list[str] = []
    for it in items:
        verdict = _cancel_item(it)
        if verdict == "gone":
            continue
        if failed_list or verdict == "failed":
            failed.append(_order_id_of(it))
        else:
            cancelled += 1
    return cancelled, failed


def _cancel_outcome(payload: Any) -> tuple[int, list[str], bool]:
    """(cancelled count, ids NOT cancelled, whether the body was recognised) for a cancel-all response."""
    if isinstance(payload, list):
        cancelled, failed = _tally(payload)
        return cancelled, failed, True
    if not isinstance(payload, Mapping):
        return 0, [], False
    scopes: list[Mapping[str, Any]] = [payload]
    if isinstance(payload.get("data"), Mapping):
        scopes.append(payload["data"])
    count: int | None = None
    failed: list[str] = []
    recognised = False
    for scope in scopes:
        scope_count: int | None = None
        for key in _CANCEL_COUNT_KEYS:
            v = scope.get(key)
            if isinstance(v, list):
                c, f = _tally(v)
                scope_count, recognised = c, True
                failed.extend(f)
                break
            if v is not None and not isinstance(v, bool) and to_int(v) is not None:
                scope_count, recognised = to_int(v), True
                break
        for key in _CANCEL_FAILED_KEYS:
            v = scope.get(key)
            if isinstance(v, list):
                recognised = True
                failed.extend(_tally(v, failed_list=True)[1])
                break
        for key in _CANCEL_RESULT_KEYS:
            v = scope.get(key)
            if isinstance(v, list):
                c, f = _tally(v)
                recognised = True
                failed.extend(f)
                if scope_count is None:
                    scope_count = c
                break
        if count is None and scope_count is not None:
            count = scope_count
    return (count or 0), list(dict.fromkeys(failed)), recognised


def _parse_leaderboard(payload: Any, *, me_username: str | None, me_profile: str | None) -> Leaderboard:
    body = payload if isinstance(payload, Mapping) else {}
    data = body.get("data") if isinstance(body.get("data"), Mapping) else None
    merged: dict[str, Any] = {**body, **(data or {})}
    rows, _, _ = page_items(
        merged, ("leaderboard", "entries", "data", "items", "results", "rows", "rankings")
    )
    entries: list[LeaderboardEntry] = []
    my_rank = to_int(pick(merged, "myRank", "my_rank", "me.rank", "currentUser.rank", "userRank"))
    my_total = to_float(
        pick(
            merged,
            "myTotal",
            "my_total",
            "myValue",
            "me.finalTotalValue",
            "me.total",
            "currentUser.finalTotalValue",
        )
    )
    member_count = to_int(
        pick(merged, "memberCount", "member_count", "totalMembers", "participants", "totalCount", "total")
    )
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        rank = to_int(pick(raw, "rank", "position", "place"))
        total = to_float(
            pick(raw, "finalTotalValue", "totalValue", "total", "value", "balance", "equity", "netWorth")
        )
        if rank is None or total is None:
            continue
        user = pick(raw, "username", "user", "name", "displayName", "handle", "profileId", "userId")
        is_me = to_bool(pick(raw, "isMe", "isCurrentUser", "me", "isSelf", "isYou")) or False
        pid = pick(raw, "profileId", "userId", "id")
        if not is_me and me_username and user is not None and str(user) == me_username:
            is_me = True
        if not is_me and me_profile and pid is not None and str(pid) == me_profile:
            is_me = True
        entry = LeaderboardEntry(
            rank=rank,
            total=total,
            is_me=is_me,
            user=str(user) if user is not None else None,
            **{k: v for k, v in raw.items() if k not in {"rank", "total", "is_me", "user"}},
        )
        entries.append(entry)
        if is_me:
            my_rank = my_rank if my_rank is not None else rank
            my_total = my_total if my_total is not None else total
    entries.sort(key=lambda e: e.rank)
    return Leaderboard(entries=entries, member_count=member_count, my_rank=my_rank, my_total=my_total)


def _exchange_ids_from(raw: Mapping[str, Any]) -> list[str]:
    ids: list[str] = []
    direct = pick(raw, "exchangeIds", "exchange_ids", "memberExchangeIds")
    if isinstance(direct, list):
        ids.extend(str(x) for x in direct if x is not None)
    for key in (
        "exchanges",
        "members",
        "markets",
        "legs",
        "nodes",
        "observedPrices",
        "suggestedCorrectiveTrades",
    ):
        items = raw.get(key)
        if isinstance(items, list):
            for it in items:
                if isinstance(it, Mapping):
                    eid = pick(it, "exchangeId", "exchange_id", "exchange.id") or (
                        pick(it, "id") if key in ("exchanges",) else None
                    )
                    if eid is not None:
                        ids.append(str(eid))
                elif it is not None and key in ("exchanges",):
                    ids.append(str(it))
    return list(dict.fromkeys(ids))


def _parse_relationship(raw: Mapping[str, Any]) -> Relationship | None:
    kind = pick(raw, "type", "kind", "relationshipType")
    extra = {k: v for k, v in raw.items() if k not in {"id", "kind", "isExhaustive", "exchange_ids"}}
    return Relationship(
        id=_opt_str(pick(raw, "id", "relationshipId")),
        kind=str(kind).lower() if kind is not None else None,
        isExhaustive=to_bool(pick(raw, "isExhaustive", "is_exhaustive", "exhaustive")),
        exchange_ids=_exchange_ids_from(raw),
        **extra,
    )


def _parse_violation(raw: Mapping[str, Any]) -> ConstraintViolation | None:
    trades = pick(raw, "suggestedCorrectiveTrades", "suggested_trades", "correctiveTrades", default=[])
    extra = {
        k: v
        for k, v in raw.items()
        if k not in {"relationship_id", "exchange_ids", "kind", "magnitude", "suggestedCorrectiveTrades"}
    }
    return ConstraintViolation(
        relationship_id=_opt_str(pick(raw, "relationshipId", "relationship_id", "relationship.id", "id")),
        exchange_ids=_exchange_ids_from(raw),
        kind=_opt_str(pick(raw, "type", "kind", "violationType")),
        magnitude=to_float(pick(raw, "violationAmount", "magnitude", "amount", "violation")),
        suggestedCorrectiveTrades=[t for t in trades if isinstance(t, Mapping)]
        if isinstance(trades, list)
        else [],
        **extra,
    )
