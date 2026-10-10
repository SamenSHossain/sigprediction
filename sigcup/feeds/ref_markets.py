"""Reference prices: Kalshi (public market data, no key) and Polymarket (Gamma + CLOB) for the crosswalked
markets; PredictIt if alive. Writes RefPrice rows. Cadence feeds.ref_cadence_s normally,
election_night.ref_cadence_s during election night (the cadence switch lives in sigcup.feeds.loop).

RefPrice conventions (what sigcup.trader.loop reads back):
  source='kalshi'      key = Market.kalshi_ticker             prices from integer cents (yes_bid / yes_ask / last_price)
  source='polymarket'  key = Market.polymarket_condition_id   Gamma outcomePrices / bestBid / bestAsk, CLOB /midpoint + /price
  source='predictit'   key = Market.predictit_id (contract id) bestSellYesCost (bid) / bestBuyYesCost (ask) / lastTradePrice
  liquidity            source-specific: Kalshi volume (contracts, else open interest); Polymarket volumeNum ($,
                       else liquidityNum); PredictIt has none.

Every parser is a pure function over the decoded JSON (lenient: tolerates wrapped/bare bodies, renamed
fields, string numbers) so it is testable from fixtures; the fetchers are thin and tolerate any failure.
Collector contract: ``async def run_once(session_factory, cfg, secrets) -> int`` (rows written).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx
from sqlalchemy import select

from sigcup.common.time import utc_now
from sigcup.db.models import Market, RefPrice
from sigcup.settings import Secrets, StrategyConfig

log = logging.getLogger(__name__)

PREDICTIT_ALL_URL = "https://www.predictit.org/api/marketdata/all/"
KALSHI_CHUNK = 50  # tickers per GET /markets?tickers=
GAMMA_CHUNK = 20  # condition ids per GET /markets?condition_ids=
CLOB_MAX_PER_CYCLE = 40  # CLOB calls are per token: cap them per run (Gamma already carries bid/ask)
HTTP_TIMEOUT_S = 15.0
CLOSED_STATUSES = {"settled", "resolved", "closed", "finalized", "cancelled", "canceled", "voided"}


# ---------------------------------------------------------------------------------------------
# Lenient helpers (deliberately not imported from sigcup.sig.client, which is written concurrently)
# ---------------------------------------------------------------------------------------------


def pick(d: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    """First present, non-None value among candidate keys (dotted paths allowed)."""
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
    if v is None or v == "" or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def json_list(v: Any) -> list[Any]:
    """Gamma encodes some arrays as JSON strings ('["0.66", "0.34"]'); accept either form."""
    if v is None:
        return []
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
        except ValueError:
            return [x.strip() for x in v.strip("[]").split(",") if x.strip()]
        return parsed if isinstance(parsed, list) else [parsed]
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def clamp01(p: float | None) -> float | None:
    if p is None:
        return None
    return min(1.0, max(0.0, p))


# ---------------------------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------------------------


@dataclass
class RefQuote:
    """One reference-market snapshot in probability units."""

    source: str  # kalshi | polymarket | predictit
    key: str  # ticker / condition id / contract id
    yes_bid: float | None = None
    yes_ask: float | None = None
    yes_mid: float | None = None
    last: float | None = None
    liquidity: float | None = None
    title: str | None = None
    status: str | None = None
    market_id: str | None = None  # SIG market id, attached when writing
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.yes_bid is not None and self.yes_ask is not None and self.yes_bid > self.yes_ask:
            self.yes_bid, self.yes_ask = self.yes_ask, self.yes_bid  # never trust side semantics blindly
        if self.yes_mid is None:
            if self.yes_bid is not None and self.yes_ask is not None:
                self.yes_mid = (self.yes_bid + self.yes_ask) / 2
            elif self.last is not None:
                self.yes_mid = self.last
            elif self.yes_bid is not None:
                self.yes_mid = self.yes_bid
            elif self.yes_ask is not None:
                self.yes_mid = self.yes_ask

    @property
    def is_open(self) -> bool:
        return (self.status or "").lower() not in CLOSED_STATUSES

    def to_row(self, ts: datetime, market_id: str | None = None) -> RefPrice:
        return RefPrice(
            ts=ts,
            source=self.source,
            key=self.key,
            market_id=market_id or self.market_id,
            yes_bid=self.yes_bid,
            yes_ask=self.yes_ask,
            yes_mid=self.yes_mid,
            liquidity=self.liquidity,
        )


@dataclass
class GammaMarket:
    """The subset of a Gamma market record we use."""

    condition_id: str
    gamma_id: str | None = None
    slug: str | None = None
    question: str | None = None
    event_title: str | None = None
    event_slug: str | None = None
    yes_token_id: str | None = None
    no_token_id: str | None = None
    yes_price: float | None = None
    best_bid: float | None = None
    best_ask: float | None = None
    last: float | None = None
    liquidity: float | None = None
    volume: float | None = None
    active: bool | None = None
    closed: bool | None = None
    end_date: str | None = None

    @property
    def title(self) -> str:
        return self.question or self.event_title or self.slug or self.condition_id

    def to_quote(self) -> RefQuote:
        status = "closed" if self.closed else ("active" if self.active is not False else "inactive")
        return RefQuote(
            source="polymarket",
            key=self.condition_id,
            yes_bid=clamp01(self.best_bid),
            yes_ask=clamp01(self.best_ask),
            yes_mid=clamp01(self.yes_price),
            last=clamp01(self.last),
            liquidity=self.volume if self.volume is not None else self.liquidity,
            title=self.title,
            status=status,
            extra={
                "gamma_id": self.gamma_id,
                "slug": self.slug,
                "yes_token_id": self.yes_token_id,
                "book_liquidity": self.liquidity,
            },
        )


# ---------------------------------------------------------------------------------------------
# Kalshi parsing (prices are integer cents; *_dollars fields are decimal strings on newer payloads)
# ---------------------------------------------------------------------------------------------


def kalshi_price(m: Mapping[str, Any], cents_key: str, dollars_key: str | None = None) -> float | None:
    """A Kalshi price in [0, 1]: integer cents (0-100) in `cents_key`, else decimal dollars in `dollars_key`."""
    v = to_float(m.get(cents_key))
    if v is not None:
        return clamp01(v / 100.0)
    if dollars_key:
        v = to_float(m.get(dollars_key))
        if v is not None:
            return clamp01(v)
    return None


def parse_kalshi_market(m: Mapping[str, Any]) -> RefQuote | None:
    """One Kalshi market record -> RefQuote (None without a ticker)."""
    if not isinstance(m, Mapping):
        return None
    ticker = pick(m, "ticker", "market_ticker", "id")
    if not ticker:
        return None
    bid = kalshi_price(m, "yes_bid", "yes_bid_dollars")
    ask = kalshi_price(m, "yes_ask", "yes_ask_dollars")
    last = kalshi_price(m, "last_price", "last_price_dollars")
    if ask is None:
        no_bid = kalshi_price(m, "no_bid", "no_bid_dollars")
        if no_bid is not None:
            ask = clamp01(1.0 - no_bid)
    volume = to_float(pick(m, "volume", "volume_fp"))
    oi = to_float(pick(m, "open_interest", "open_interest_fp"))
    liquidity = volume if volume else oi
    return RefQuote(
        source="kalshi",
        key=str(ticker),
        yes_bid=bid,
        yes_ask=ask,
        last=last,
        liquidity=liquidity,
        title=pick(m, "title", "yes_sub_title", "subtitle"),
        status=pick(m, "status"),
        extra={
            "event_ticker": pick(m, "event_ticker"),
            "open_interest": oi,
            "volume": volume,
            "volume_24h": to_float(pick(m, "volume_24h", "volume_24h_fp")),
            "close_time": pick(m, "close_time", "expected_expiration_time"),
            "result": pick(m, "result"),
        },
    )


def parse_kalshi_markets(payload: Any) -> dict[str, RefQuote]:
    """``{"markets": [...]}``, ``{"market": {...}}``, a bare list or a single record -> {ticker: RefQuote}."""
    if payload is None:
        return {}
    rows: list[Any]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        if isinstance(payload.get("markets"), list):
            rows = payload["markets"]
        elif isinstance(payload.get("market"), Mapping):
            rows = [payload["market"]]
        elif isinstance(payload.get("data"), (list, Mapping)):
            return parse_kalshi_markets(payload["data"])
        else:
            rows = [payload]
    else:
        return {}
    out: dict[str, RefQuote] = {}
    for r in rows:
        q = parse_kalshi_market(r) if isinstance(r, Mapping) else None
        if q is not None:
            out[q.key] = q
    return out


# ---------------------------------------------------------------------------------------------
# Polymarket parsing
# ---------------------------------------------------------------------------------------------


def _yes_index(m: Mapping[str, Any]) -> int:
    outcomes = [str(x).strip().lower() for x in json_list(pick(m, "outcomes"))]
    if "yes" in outcomes:
        return outcomes.index("yes")
    return 0


def parse_gamma_market(m: Mapping[str, Any], *, event: Mapping[str, Any] | None = None) -> GammaMarket | None:
    """One Gamma market record -> GammaMarket (None without a condition id)."""
    if not isinstance(m, Mapping):
        return None
    cid = pick(m, "conditionId", "condition_id", "conditionID")
    if not cid:
        return None
    yi = _yes_index(m)
    prices = [to_float(x) for x in json_list(pick(m, "outcomePrices", "outcome_prices"))]
    tokens = [str(x) for x in json_list(pick(m, "clobTokenIds", "clob_token_ids", "tokenIds"))]
    yes_price = prices[yi] if len(prices) > yi else None
    yes_token = tokens[yi] if len(tokens) > yi else None
    no_token = next((t for i, t in enumerate(tokens) if i != yi), None)
    ev = event if isinstance(event, Mapping) else None
    ev_list = pick(m, "events")
    if ev is None and isinstance(ev_list, list) and ev_list and isinstance(ev_list[0], Mapping):
        ev = ev_list[0]
    return GammaMarket(
        condition_id=str(cid),
        gamma_id=str(pick(m, "id")) if pick(m, "id") is not None else None,
        slug=pick(m, "slug", "marketSlug"),
        question=pick(m, "question", "title", "name"),
        event_title=pick(ev, "title", "name") if ev else None,
        event_slug=pick(ev, "slug") if ev else None,
        yes_token_id=yes_token,
        no_token_id=no_token,
        yes_price=clamp01(yes_price),
        best_bid=clamp01(to_float(pick(m, "bestBid", "best_bid"))),
        best_ask=clamp01(to_float(pick(m, "bestAsk", "best_ask"))),
        last=clamp01(to_float(pick(m, "lastTradePrice", "last_trade_price"))),
        liquidity=to_float(pick(m, "liquidityNum", "liquidity", "liquidityClob")),
        volume=to_float(pick(m, "volumeNum", "volume", "volumeClob")),
        active=pick(m, "active"),
        closed=pick(m, "closed"),
        end_date=pick(m, "endDate", "end_date_iso", "endDateIso"),
    )


def parse_gamma_markets(payload: Any) -> list[GammaMarket]:
    """A Gamma /markets response (list, ``{"data": [...]}`` or a single record) -> GammaMarket list."""
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        if isinstance(payload.get("data"), list):
            payload = payload["data"]
        elif isinstance(payload.get("markets"), list):
            payload = payload["markets"]
        else:
            gm = parse_gamma_market(payload)
            return [gm] if gm else []
    if not isinstance(payload, list):
        return []
    out = []
    for m in payload:
        gm = parse_gamma_market(m) if isinstance(m, Mapping) else None
        if gm:
            out.append(gm)
    return out


def parse_gamma_search(payload: Any) -> list[GammaMarket]:
    """Gamma /public-search -> every market under every returned event (event title attached)."""
    if not isinstance(payload, Mapping):
        return parse_gamma_markets(payload)
    out: list[GammaMarket] = []
    events = payload.get("events")
    if isinstance(events, list):
        for ev in events:
            if not isinstance(ev, Mapping):
                continue
            for m in ev.get("markets") or []:
                gm = parse_gamma_market(m, event=ev) if isinstance(m, Mapping) else None
                if gm:
                    out.append(gm)
    markets = payload.get("markets")
    if isinstance(markets, list):
        out.extend(parse_gamma_markets(markets))
    return out


def parse_clob_midpoint(payload: Any) -> float | None:
    """``{"mid": "0.655"}`` -> 0.655."""
    return clamp01(to_float(pick(payload, "mid", "midpoint", "price")))


def parse_clob_price(payload: Any) -> float | None:
    """``{"price": "0.66"}`` -> 0.66."""
    return clamp01(to_float(pick(payload, "price", "mid")))


# ---------------------------------------------------------------------------------------------
# PredictIt parsing
# ---------------------------------------------------------------------------------------------


def parse_predictit_all(payload: Any) -> dict[str, RefQuote]:
    """/api/marketdata/all/ -> {contract id: RefQuote}. bestSellYesCost = bid, bestBuyYesCost = ask."""
    markets = pick(payload, "markets", "Markets") if isinstance(payload, Mapping) else payload
    if not isinstance(markets, list):
        return {}
    out: dict[str, RefQuote] = {}
    for mk in markets:
        if not isinstance(mk, Mapping):
            continue
        mname = pick(mk, "shortName", "name")
        mid = pick(mk, "id")
        for c in pick(mk, "contracts", "Contracts", default=[]) or []:
            if not isinstance(c, Mapping):
                continue
            cid = pick(c, "id")
            if cid is None:
                continue
            bid = clamp01(to_float(pick(c, "bestSellYesCost")))
            ask = clamp01(to_float(pick(c, "bestBuyYesCost")))
            if ask is None:
                no_bid = to_float(pick(c, "bestSellNoCost"))
                if no_bid is not None:
                    ask = clamp01(1.0 - no_bid)
            cname = pick(c, "shortName", "name")
            out[str(cid)] = RefQuote(
                source="predictit",
                key=str(cid),
                yes_bid=bid,
                yes_ask=ask,
                last=clamp01(to_float(pick(c, "lastTradePrice"))),
                liquidity=None,
                title=f"{mname} / {cname}" if mname and cname else (cname or mname),
                status=pick(c, "status") or pick(mk, "status"),
                extra={"market_id": mid, "market_name": mname, "url": pick(mk, "url")},
            )
    return out


# ---------------------------------------------------------------------------------------------
# Fetchers (thin; every failure returns an empty result and is logged)
# ---------------------------------------------------------------------------------------------


def _client(http: httpx.AsyncClient | None) -> tuple[httpx.AsyncClient, bool]:
    if http is not None:
        return http, False
    return httpx.AsyncClient(timeout=HTTP_TIMEOUT_S, headers={"User-Agent": "sigcup/0.1 (+ref feed)"}), True


async def _get_json(http: httpx.AsyncClient, url: str, params: Any = None) -> Any:
    """GET -> decoded JSON; None on any HTTP/transport/decoding failure (logged at warning)."""
    try:
        resp = await http.get(url, params=params)
    except httpx.HTTPError as e:
        log.warning("GET %s failed: %s", url, type(e).__name__)
        return None
    if resp.status_code != 200:
        log.warning("GET %s -> %s", url, resp.status_code)
        return None
    try:
        return resp.json()
    except ValueError:
        log.warning("GET %s: non-JSON body", url)
        return None


async def fetch_kalshi_market(http: httpx.AsyncClient, base: str, ticker: str) -> RefQuote | None:
    """GET {base}/markets/{ticker}; None on 404 or failure."""
    payload = await _get_json(http, f"{base.rstrip('/')}/markets/{ticker}")
    quotes = parse_kalshi_markets(payload)
    return quotes.get(ticker) or (next(iter(quotes.values())) if quotes else None)


async def fetch_kalshi(
    http: httpx.AsyncClient, base: str, tickers: Iterable[str], *, chunk: int = KALSHI_CHUNK
) -> dict[str, RefQuote]:
    """GET {base}/markets?tickers=a,b,... in chunks; falls back to per-ticker reads for a failed chunk."""
    tickers = [t for t in dict.fromkeys(t for t in tickers if t)]
    out: dict[str, RefQuote] = {}
    url = f"{base.rstrip('/')}/markets"
    for i in range(0, len(tickers), chunk):
        part = tickers[i : i + chunk]
        payload = await _get_json(http, url, {"tickers": ",".join(part), "limit": max(len(part), 1)})
        got = parse_kalshi_markets(payload) if payload is not None else {}
        if payload is None:
            for t in part:
                q = await fetch_kalshi_market(http, base, t)
                if q:
                    got[q.key] = q
        out.update({k: v for k, v in got.items() if k in part})
        for k, v in got.items():
            out.setdefault(k, v)
    return out


async def fetch_gamma_markets(
    http: httpx.AsyncClient, gamma: str, condition_ids: Iterable[str], *, chunk: int = GAMMA_CHUNK
) -> dict[str, GammaMarket]:
    """GET {gamma}/markets?condition_ids=...(repeated) in chunks -> {condition id: GammaMarket}."""
    ids = [c for c in dict.fromkeys(c for c in condition_ids if c)]
    out: dict[str, GammaMarket] = {}
    url = f"{gamma.rstrip('/')}/markets"
    for i in range(0, len(ids), chunk):
        part = ids[i : i + chunk]
        params = [("condition_ids", c) for c in part] + [("limit", str(len(part)))]
        payload = await _get_json(http, url, params)
        for gm in parse_gamma_markets(payload):
            out[gm.condition_id] = gm
    return out


async def fetch_gamma_by_slug(http: httpx.AsyncClient, gamma: str, slug: str) -> GammaMarket | None:
    """GET {gamma}/markets?slug=... -> the market or None."""
    payload = await _get_json(http, f"{gamma.rstrip('/')}/markets", {"slug": slug})
    found = parse_gamma_markets(payload)
    return found[0] if found else None


async def fetch_gamma_search(http: httpx.AsyncClient, gamma: str, query: str) -> list[GammaMarket]:
    """GET {gamma}/public-search?q=... -> markets of the returned events."""
    payload = await _get_json(http, f"{gamma.rstrip('/')}/public-search", {"q": query, "limit_per_type": 10})
    return parse_gamma_search(payload)


async def fetch_clob_quote(
    http: httpx.AsyncClient, clob: str, token_id: str
) -> tuple[float | None, float | None, float | None]:
    """(bid, ask, mid) for a CLOB token via /price?side=sell, /price?side=buy and /midpoint."""
    base = clob.rstrip("/")
    sell = parse_clob_price(await _get_json(http, f"{base}/price", {"token_id": token_id, "side": "sell"}))
    buy = parse_clob_price(await _get_json(http, f"{base}/price", {"token_id": token_id, "side": "buy"}))
    mid = parse_clob_midpoint(await _get_json(http, f"{base}/midpoint", {"token_id": token_id}))
    bid, ask = (sell, buy)
    if bid is not None and ask is not None and bid > ask:
        bid, ask = ask, bid
    return bid, ask, mid


async def fetch_predictit(http: httpx.AsyncClient, url: str = PREDICTIT_ALL_URL) -> dict[str, RefQuote]:
    """PredictIt's public JSON dump (empty when unreachable or discontinued)."""
    payload = await _get_json(http, url)
    return parse_predictit_all(payload) if payload is not None else {}


# ---------------------------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RefLeg:
    """The crosswalk ids of one SIG market."""

    market_id: str
    kalshi_ticker: str | None = None
    polymarket_condition_id: str | None = None
    polymarket_token_id: str | None = None
    predictit_id: str | None = None


def legs_from_markets(markets: Iterable[Market]) -> list[RefLeg]:
    out = []
    for m in markets:
        if (m.status or "open").lower() in CLOSED_STATUSES:
            continue
        if not (m.kalshi_ticker or m.polymarket_condition_id or m.predictit_id):
            continue
        out.append(
            RefLeg(
                market_id=m.market_id,
                kalshi_ticker=m.kalshi_ticker or None,
                polymarket_condition_id=m.polymarket_condition_id or None,
                polymarket_token_id=m.polymarket_token_id or None,
                predictit_id=m.predictit_id or None,
            )
        )
    return out


async def collect(
    http: httpx.AsyncClient,
    cfg: StrategyConfig,
    legs: Sequence[RefLeg],
    *,
    sources: Iterable[str] = ("kalshi", "polymarket", "predictit"),
    clob_max: int = CLOB_MAX_PER_CYCLE,
) -> list[RefQuote]:
    """Fetch every source for the given legs and return quotes with `market_id` attached."""
    sources = set(sources)
    out: list[RefQuote] = []
    feeds = cfg.feeds

    if "kalshi" in sources:
        by_ticker = {leg.kalshi_ticker: leg for leg in legs if leg.kalshi_ticker}
        try:
            quotes = await fetch_kalshi(http, feeds.kalshi_base, by_ticker)
        except Exception as e:  # noqa: BLE001
            log.warning("kalshi fetch failed: %s", e)
            quotes = {}
        for ticker, leg in by_ticker.items():
            q = quotes.get(ticker)
            if q is not None:
                q.market_id = leg.market_id
                out.append(q)

    if "polymarket" in sources:
        by_cid = {leg.polymarket_condition_id: leg for leg in legs if leg.polymarket_condition_id}
        try:
            gms = await fetch_gamma_markets(http, feeds.polymarket_gamma, by_cid)
        except Exception as e:  # noqa: BLE001
            log.warning("polymarket gamma fetch failed: %s", e)
            gms = {}
        clob_calls = 0
        for cid, leg in by_cid.items():
            gm = gms.get(cid)
            q = gm.to_quote() if gm is not None else None
            token = leg.polymarket_token_id or (gm.yes_token_id if gm else None)
            needs_clob = q is None or q.yes_bid is None or q.yes_ask is None
            if needs_clob and token and clob_calls < clob_max:
                clob_calls += 1
                try:
                    bid, ask, mid = await fetch_clob_quote(http, feeds.polymarket_clob, token)
                except Exception as e:  # noqa: BLE001
                    log.warning("polymarket clob fetch failed for %s: %s", cid, e)
                    bid = ask = mid = None
                if q is None:
                    if bid is None and ask is None and mid is None:
                        continue
                    q = RefQuote(source="polymarket", key=cid, yes_bid=bid, yes_ask=ask, yes_mid=mid)
                else:
                    q.yes_bid = q.yes_bid if q.yes_bid is not None else bid
                    q.yes_ask = q.yes_ask if q.yes_ask is not None else ask
                    if mid is not None:
                        q.yes_mid = mid
                    elif q.yes_bid is not None and q.yes_ask is not None:
                        q.yes_mid = (q.yes_bid + q.yes_ask) / 2
            if q is None:
                continue
            q.market_id = leg.market_id
            out.append(q)

    if "predictit" in sources:
        by_pid = {leg.predictit_id: leg for leg in legs if leg.predictit_id}
        if by_pid:
            try:
                quotes = await fetch_predictit(http)
            except Exception as e:  # noqa: BLE001
                log.warning("predictit fetch failed: %s", e)
                quotes = {}
            for pid, leg in by_pid.items():
                q = quotes.get(pid)
                if q is not None:
                    q.market_id = leg.market_id
                    out.append(q)
    return out


def load_legs(session_factory: Callable[[], Any]) -> list[RefLeg]:
    with session_factory() as s:
        rows = s.execute(select(Market)).scalars().all()
        return legs_from_markets(rows)


def write_quotes(session_factory: Callable[[], Any], quotes: Iterable[RefQuote], ts: datetime) -> int:
    """Persist quotes as RefPrice rows (one row per quote; nothing is deduplicated: the trader reads the latest)."""
    rows = [q.to_row(ts) for q in quotes if q.yes_mid is not None or q.yes_bid is not None or q.yes_ask is not None]
    if not rows:
        return 0
    with session_factory() as s:
        s.add_all(rows)
    return len(rows)


async def run_once(
    session_factory: Callable[[], Any],
    cfg: StrategyConfig,
    secrets: Secrets | None = None,
    *,
    http: httpx.AsyncClient | None = None,
    sources: Iterable[str] = ("kalshi", "polymarket", "predictit"),
    now: datetime | None = None,
) -> int:
    """Collector entry point: one pass over every Market with a crosswalk id; returns RefPrice rows written."""
    legs = load_legs(session_factory)
    if not legs:
        log.info("ref_markets: no crosswalked markets yet (run scripts/bootstrap_races.py)")
        return 0
    client, owns = _client(http)
    try:
        quotes = await collect(client, cfg, legs, sources=sources)
    finally:
        if owns:
            await client.aclose()
    n = write_quotes(session_factory, quotes, now or utc_now())
    by_source: dict[str, int] = {}
    for q in quotes:
        by_source[q.source] = by_source.get(q.source, 0) + 1
    log.info("ref_markets: %d legs -> %d rows %s", len(legs), n, by_source)
    return n
