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

from sigcup.common.time import before_deadline, set_server_offset_from_date_header, utc_now
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
    action = norm_action(pick(raw, "action", "orderType", "type", "direction"))
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
    """Split a list response into (items, next_cursor, has_more); tolerates bare lists and several wrappers."""
    if payload is None:
        return [], None, False
    if isinstance(payload, list):
        return list(payload), None, False
    if not isinstance(payload, Mapping):
        return [], None, False
    items: list[Any] = []
    for k in keys:
        v = payload.get(k)
        if isinstance(v, list):
            items = v
            break
        if isinstance(v, Mapping):
            inner_items, _, _ = page_items(v, keys)
            if inner_items:
                items = inner_items
                break
    pag = payload.get("pagination") if isinstance(payload.get("pagination"), Mapping) else payload
    cursor = pick(pag, "nextCursor", "next_cursor", "cursor", "next")
    has_more = to_bool(pick(pag, "hasMore", "has_more", "more"))
    if has_more is None:
        has_more = cursor is not None
    return items, (str(cursor) if cursor is not None else None), bool(has_more)


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
    ) -> Any:
        """One rate-limited HTTP call. Returns the parsed JSON body (None when empty).

        Raises RateLimited (429, after freezing both budgets), RequestInFlight (409), AuthError (401/403),
        ValidationError (other 4xx), TransientError (503, or any 5xx/timeout on a read) and OrderStatusUnknown
        (502 / other 5xx / timeout / transport failure on a write: the request may have been applied).
        """
        limiter = self.budget.writes if write else self.budget.reads
        await limiter.wait(priority)
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
        set_server_offset_from_date_header(resp.headers.get("Date"))
        log.debug("%s %s -> %s in %.0f ms", method, path, resp.status_code, (time.monotonic() - t0) * 1000)
        return self._classify(resp, method, path, write, set(extra_ok))

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

    async def get_prices(self, exchange_ids: list[str], *, priority: int = 2) -> dict[str, PriceSnapshot]:
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

    async def get_orderbook(self, exchange_id: str, depth: int = 20, *, priority: int = 2) -> OrderBook:
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
    async def get_account(self) -> AccountInfo:
        """GET /account; if it carries no balance, the tournament's `myBalance` is used (one extra read)."""
        raw = unwrap(await self._get("/account"), "data", "account")
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

    async def get_positions(self) -> list[PositionInfo]:
        """GET /tournaments/{slug}/portfolio/positions (fallback /portfolio/positions?tournamentId=)."""
        rows = await self._tournament_portfolio("positions")
        out: list[PositionInfo] = []
        for raw in rows:
            p = _parse_position(raw)
            if p is not None:
                out.append(p)
        return out

    async def get_open_orders(self) -> list[OrderInfo]:
        """GET /orders?status=open&tournamentId=... (paginated)."""
        tid = await self.ensure_tournament_id()
        rows = await self._paginate("/orders", {"status": "open", "tournamentId": tid, "limit": 100})
        out: list[OrderInfo] = []
        for raw in rows:
            o = _parse_order(raw)
            if o is not None:
                out.append(o)
        return out

    async def get_order(self, order_id: str) -> OrderInfo | None:
        """GET /orders/{id}; None when the exchange does not know the id."""
        try:
            raw = unwrap(await self._get(f"/orders/{order_id}"), "data", "order")
        except ValidationError as e:
            if e.status == 404:
                return None
            raise
        return _parse_order(raw) if isinstance(raw, Mapping) else None

    async def get_order_fills(self, order_id: str) -> list[FillInfo]:
        """GET /orders/{id}/fills."""
        rows = await self._paginate(f"/orders/{order_id}/fills", {"limit": 200})
        return [f for f in (parse_fill(r) for r in rows if isinstance(r, Mapping)) if f is not None]

    async def get_fills(self, since: datetime | None = None) -> list[FillInfo]:
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

        rows = await self._tournament_portfolio("fills", stop_when=older_page)
        fills = [f for f in (parse_fill(r) for r in rows if isinstance(r, Mapping)) if f is not None]
        if since_utc is not None:
            fills = [f for f in fills if f.ts >= since_utc]
        return fills

    async def _tournament_portfolio(self, kind: str, *, stop_when: Any = None) -> list[Any]:
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
                    rows = await self._paginate(cand, p, stop_when=stop_when)
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
        return await self._paginate(path, params, stop_when=stop_when)

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

    async def place_order(self, req: OrderRequest, *, priority: int = 2) -> OrderResult:
        """POST /orders. Never raises for an unknown outcome: returns OrderResult(uncertain=True) instead."""
        self._check_deadline()
        tid = await self.ensure_tournament_id()
        body = self._order_body(req, tid)
        try:
            payload = await self._request("POST", "/orders", json=body, write=True, priority=priority)
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

    async def place_multi_leg(self, legs: list[OrderRequest], *, priority: int = 0) -> MultiLegResult:
        """POST /orders/multi-leg (<= 10 legs, atomic placement, non-atomic fills)."""
        if not legs:
            return MultiLegResult(results=[], atomic_ok=True)
        if len(legs) > MULTI_LEG_MAX:
            raise ValidationError(f"multi-leg accepts at most {MULTI_LEG_MAX} legs", None, "TOO_MANY_LEGS")
        self._check_deadline()
        tid = await self.ensure_tournament_id()
        group_key = _group_key(legs)
        body = {
            "tournamentId": tid,
            "idempotencyKey": group_key,
            "legs": [self._order_body(r, tid) for r in legs],
        }
        try:
            payload = await self._request(
                "POST", "/orders/multi-leg", json=body, write=True, priority=priority
            )
        except OrderStatusUnknown as e:
            log.warning("multi-leg %s uncertain: %s", group_key, e)
            return MultiLegResult(
                group_id=group_key,
                results=[
                    OrderResult(uncertain=True, status="unknown", idempotencyKey=r.idempotency_key)
                    for r in legs
                ],
                atomic_ok=False,
            )
        items = _leg_results(payload, len(legs))
        results = [_parse_order_result(it, r.idempotency_key) for it, r in zip(items, legs, strict=True)]
        ok = all(
            not r.uncertain and (r.status or "").lower() not in ("rejected", "error", "failed")
            for r in results
        )
        gid = (
            pick(unwrap(payload), "groupId", "group_id", "multiLegId", "id")
            if isinstance(payload, Mapping)
            else None
        )
        return MultiLegResult(
            group_id=str(gid) if gid is not None else group_key, results=results, atomic_ok=ok
        )

    async def place_batch(self, reqs: list[OrderRequest], *, priority: int = 2) -> list[OrderResult]:
        """POST /orders/batch (<= 50 independent orders; a 207 carries per-order results)."""
        if not reqs:
            return []
        if len(reqs) > BATCH_MAX:
            raise ValidationError(f"batch accepts at most {BATCH_MAX} orders", None, "TOO_MANY_ORDERS")
        self._check_deadline()
        tid = await self.ensure_tournament_id()
        body = {
            "tournamentId": tid,
            "idempotencyKey": _group_key(reqs),
            "orders": [self._order_body(r, tid) for r in reqs],
        }
        try:
            payload = await self._request(
                "POST", "/orders/batch", json=body, write=True, priority=priority, extra_ok=(207,)
            )
        except OrderStatusUnknown as e:
            log.warning("batch uncertain: %s", e)
            return [
                OrderResult(uncertain=True, status="unknown", idempotencyKey=r.idempotency_key) for r in reqs
            ]
        items = _leg_results(payload, len(reqs))
        return [_parse_order_result(it, r.idempotency_key) for it, r in zip(items, reqs, strict=True)]

    async def cancel_order(self, order_id: str, *, priority: int = 2) -> bool:
        """DELETE /orders/{id}. False when the exchange no longer knows the order or the outcome is unknown."""
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
            log.warning("cancel %s uncertain: %s", order_id, e)
            return False
        return True

    async def cancel_all(self, exchange_id: str | None = None, *, priority: int = 0) -> int:
        """POST /orders/cancel-all (priority 0: the kill switch). Accepts 207/422 partial results."""
        tid = await self.ensure_tournament_id()
        body: dict[str, Any] = {"tournamentId": tid}
        if exchange_id:
            body["exchangeId"] = str(exchange_id)
        payload = await self._request(
            "POST", "/orders/cancel-all", json=body, write=True, priority=priority, extra_ok=(207, 422)
        )
        return _cancelled_count(payload)

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

    async def realtime_token(self, *, priority: int = 1) -> RealtimeToken:
        """POST /realtime/token -> supabaseUrl, anonKey, token, expiresAt (+ channels.user when given)."""
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
    if eid is None or qty is None:
        return None
    side = norm_side(pick(raw, "side", "option", "outcome"))
    if side == "no" and qty > 0:
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


def _leg_results(payload: Any, n: int) -> list[Any]:
    """Per-leg results from a multi-leg/batch body: {data:[{index,data}]}, {results:[...]}, {orders:[...]}, [...]"""
    body = unwrap(payload, "data", "results", "orders", "legs", "items")
    if isinstance(body, Mapping):
        body = pick(body, "results", "orders", "legs", "items", default=[body])
    items = list(body) if isinstance(body, list) else [body]
    if items and all(isinstance(it, Mapping) and "index" in it for it in items):
        items.sort(key=lambda it: to_int(it.get("index")) or 0)
    while len(items) < n:
        items.append(None)
    return items[:n]


def _cancelled_count(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, list):
        return sum(1 for it in payload if _cancel_ok(it))
    if isinstance(payload, Mapping):
        n = to_int(
            pick(
                payload, "cancelled", "canceled", "cancelledCount", "canceledCount", "count", "data.cancelled"
            )
        )
        if n is not None:
            return n
        items = pick(payload, "results", "data", "orders", "cancelledOrders", "cancelled_orders")
        if isinstance(items, list):
            return sum(1 for it in items if _cancel_ok(it))
    return 0


def _cancel_ok(it: Any) -> bool:
    if isinstance(it, Mapping):
        st = str(pick(it, "status", "state", default="")).lower()
        if "error" in it or st in ("error", "failed", "rejected", "not_found"):
            return False
        ok = to_bool(pick(it, "ok", "success", "cancelled", "canceled"))
        return True if ok is None else ok
    return it is not None


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
