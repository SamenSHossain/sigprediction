"""In-memory view of the exchange: best bid/ask/last per exchange_id, depth cache with timestamps,
our positions/open orders, cash and reservation. Updated by Realtime pushes and REST polls; the OMS
and the signal engine read from here, never from the DB, during a cycle. Writes Quote rows on change.

Usage::

    book = BookState(cfg)
    book.load_meta()                      # Race/Market metadata from the DB
    changed = book.update_prices(await adapter.get_prices(ids))   # PriceSnapshot dicts
    book.update_book(await adapter.get_orderbook(eid))            # depth
    book.update_realtime(event)           # duck-typed: exchange_id, bids, asks, ts
    views = book.views(positions, open_orders)                    # dict[exchange_id, MarketView]

Quote rows are queued whenever the top of book (bid/ask/sizes/last) changes and flushed either
automatically (``persist=True``, the default) after each bulk update or explicitly via ``flush_quotes``.
Every update path tolerates missing or renamed fields.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from sigcup.common.log import get_logger
from sigcup.common.time import ensure_utc, parse_iso, utc_now
from sigcup.db import engine as dbengine
from sigcup.db.models import IntentState, Market, OrderIntent, Quote, Race
from sigcup.settings import StrategyConfig
from sigcup.sig.schemas import AccountInfo, Level, OrderBook, PositionInfo
from sigcup.trader.types import MarketView, RaceInfo

log = get_logger(__name__)

_MISSING = object()


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Lenient attribute/key getter: returns the first present candidate name."""
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
            continue
        val = getattr(obj, name, _MISSING)
        if val is not _MISSING and val is not None:
            return val
    return default


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return None


def _to_dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return ensure_utc(v)
    if isinstance(v, int | float):
        secs = float(v) / (1000.0 if v > 1e11 else 1.0)  # epoch seconds or milliseconds
        return datetime.fromtimestamp(secs, tz=UTC)
    if isinstance(v, str):
        try:
            return parse_iso(v)
        except ValueError:
            return None
    return None


def _levels(raw: Any) -> list[Level]:
    """Normalise a list of levels given as Level objects, dicts, or (price, qty) pairs."""
    out: list[Level] = []
    for item in raw or []:
        if isinstance(item, Level):
            out.append(item)
            continue
        if isinstance(item, Mapping):
            p = _to_float(_get(item, "price", "p", "px"))
            q = _to_int(_get(item, "quantity", "qty", "size", "q", default=0))
        elif isinstance(item, list | tuple) and len(item) >= 2:
            p, q = _to_float(item[0]), _to_int(item[1])
        else:
            p = _to_float(_get(item, "price", "p"))
            q = _to_int(_get(item, "quantity", "qty", "size", default=0))
        if p is None:
            continue
        out.append(Level(price=p, quantity=q or 0))
    return out


@dataclass
class Top:
    """Top-of-book snapshot for one market."""

    exchange_id: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    ts: datetime | None = None
    source: str = "rest"

    def same_as(self, other: Top | None) -> bool:
        if other is None:
            return False
        return (
            self.bid == other.bid
            and self.ask == other.ask
            and self.last == other.last
            and self.bid_size == other.bid_size
            and self.ask_size == other.ask_size
        )

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last


@dataclass
class MarketMeta:
    """Race/market metadata joined from the Race and Market tables."""

    exchange_id: str
    market_id: str
    race_id: str
    party: str
    chamber: str
    state: str | None
    status: str = "open"
    title: str = ""
    kalshi_ticker: str | None = None
    polymarket_condition_id: str | None = None


@dataclass
class PositionState:
    quantity: int = 0
    avg_price: float | None = None
    mark: float | None = None


@dataclass
class BookState:
    """Top-of-book + depth cache keyed by exchange_id, plus our positions, open intents, cash and
    reservation. ``persist`` controls whether Quote rows are written on change."""

    cfg: StrategyConfig
    persist: bool = True
    tops: dict[str, Top] = field(default_factory=dict)
    books: dict[str, OrderBook] = field(default_factory=dict)
    book_ts: dict[str, datetime] = field(default_factory=dict)
    positions: dict[str, PositionState] = field(default_factory=dict)
    open_intents: dict[str, OrderIntent] = field(default_factory=dict)
    meta: dict[str, MarketMeta] = field(default_factory=dict)
    race_meta: dict[str, RaceInfo] = field(default_factory=dict)
    cash: float = 0.0
    reserved: float = 0.0
    positions_value: float = 0.0
    last_bulk_at: datetime | None = None
    _pending_quotes: list[Quote] = field(default_factory=list, repr=False)

    # -- metadata ----------------------------------------------------------------------------
    def load_meta(self, session: Session | None = None) -> int:
        """Load Race/Market rows into ``meta``/``race_meta``. Returns the number of markets mapped."""
        if session is not None:
            return self._load_meta(session)
        with dbengine.session() as s:
            return self._load_meta(s)

    def _load_meta(self, s: Session) -> int:
        races = {r.race_id: r for r in s.execute(select(Race)).scalars().all()}
        markets = s.execute(select(Market)).scalars().all()
        self.meta.clear()
        self.race_meta.clear()
        for r in races.values():
            self.race_meta[r.race_id] = RaceInfo(
                race_id=r.race_id,
                chamber=r.chamber,
                state=r.state,
                label=r.label,
                exclusive=r.exclusive,
                exhaustive=r.exhaustive,
                pvi=r.pvi,
                incumbent_party=r.incumbent_party,
                settlement_note=r.settlement_note,
            )
        for m in markets:
            race = races.get(m.race_id)
            self.meta[m.exchange_id] = MarketMeta(
                exchange_id=m.exchange_id,
                market_id=m.market_id,
                race_id=m.race_id,
                party=m.party,
                chamber=race.chamber if race else "unknown",
                state=race.state if race else None,
                status=m.status or "open",
                title=m.title,
                kalshi_ticker=m.kalshi_ticker,
                polymarket_condition_id=m.polymarket_condition_id,
            )
            if race is not None:
                self.race_meta[m.race_id].markets[m.party] = m.exchange_id
        return len(self.meta)

    def races(self) -> dict[str, RaceInfo]:
        """RaceInfo per race_id (party -> exchange_id map included)."""
        return dict(self.race_meta)

    def exchange_ids(self) -> list[str]:
        """Every exchange id we know about (mapped markets first, then unmapped quotes)."""
        ids = list(self.meta.keys())
        ids.extend(eid for eid in self.tops if eid not in self.meta)
        return ids

    # -- updates -----------------------------------------------------------------------------
    def update_prices(
        self,
        snaps: Mapping[str, Any] | Iterable[Any],
        source: str = "rest",
        ts: datetime | None = None,
    ) -> list[str]:
        """Apply a bulk price update (PriceSnapshot objects or dicts). Returns the exchange ids whose
        top of book changed. Quote rows are queued for those and flushed when ``persist`` is set."""
        items = snaps.values() if isinstance(snaps, Mapping) else snaps
        changed: list[str] = []
        now = ts or utc_now()
        for snap in items:
            eid = _get(snap, "exchange_id", "exchangeId", "id")
            if not eid:
                continue
            eid = str(eid)
            prev = self.tops.get(eid)
            top = Top(
                exchange_id=eid,
                bid=_to_float(_get(snap, "best_bid", "bestBid", "bid")),
                ask=_to_float(_get(snap, "best_ask", "bestAsk", "ask")),
                last=_to_float(_get(snap, "last", "latestPrice", "last_price", "lastPrice")),
                bid_size=_to_int(_get(snap, "bid_size", "bidSize", "bestBidSize")),
                ask_size=_to_int(_get(snap, "ask_size", "askSize", "bestAskSize")),
                ts=_to_dt(_get(snap, "ts", "timestamp", "updatedAt")) or now,
                source=source,
            )
            if top.last is None and prev is not None:
                top.last = prev.last
            if top.same_as(prev):
                prev.ts = top.ts  # type: ignore[union-attr]
                continue
            self.tops[eid] = top
            changed.append(eid)
            self._queue_quote(top)
        if source == "rest":
            self.last_bulk_at = now
        if self.persist and changed:
            self.flush_quotes()
        return changed

    def update_book(self, book: OrderBook, source: str = "rest") -> bool:
        """Cache a depth snapshot and refresh the derived top of book. Returns True if the top changed."""
        eid = book.exchange_id
        ts = _to_dt(book.ts) or utc_now()
        self.books[eid] = book
        self.book_ts[eid] = ts
        prev = self.tops.get(eid)
        bids = sorted(book.bids, key=lambda lv: -lv.price)
        asks = sorted(book.asks, key=lambda lv: lv.price)
        top = Top(
            exchange_id=eid,
            bid=bids[0].price if bids else None,
            ask=asks[0].price if asks else None,
            last=prev.last if prev else None,
            bid_size=bids[0].quantity if bids else None,
            ask_size=asks[0].quantity if asks else None,
            ts=ts,
            source=source,
        )
        if top.same_as(prev):
            prev.ts = ts  # type: ignore[union-attr]
            return False
        self.tops[eid] = top
        self._queue_quote(top)
        if self.persist and source == "rest":
            self.flush_quotes()
        return True

    def update_realtime(self, upd: Any) -> bool:
        """Apply a realtime BookUpdate-like object (duck-typed: exchange_id, bids, asks, ts).
        Returns True if the top of book changed. Quotes are queued, not flushed (call ``flush_quotes``)."""
        eid = _get(upd, "exchange_id", "exchangeId", "market_id", "marketId")
        if not eid:
            return False
        bids = _levels(_get(upd, "bids", "buys", default=[]))
        asks = _levels(_get(upd, "asks", "sells", default=[]))
        ts = _to_dt(_get(upd, "ts", "timestamp")) or utc_now()
        book = OrderBook(exchange_id=str(eid), bids=bids, asks=asks, ts=ts)
        changed = self.update_book(book, source="realtime")
        last = _to_float(_get(upd, "last", "last_price", "lastPrice"))
        if last is not None:
            top = self.tops[str(eid)]
            if top.last != last:
                top.last = last
                if not changed:
                    self._queue_quote(top)
                changed = True
        if self.persist and len(self._pending_quotes) >= 50:
            self.flush_quotes()
        return changed

    def set_positions(self, positions: Iterable[Any] | Mapping[str, Any]) -> None:
        """Replace the position cache from PositionInfo objects (or dicts)."""
        items = positions.values() if isinstance(positions, Mapping) else positions
        self.positions.clear()
        for p in items:
            eid = _get(p, "exchange_id", "exchangeId")
            if not eid:
                continue
            self.positions[str(eid)] = PositionState(
                quantity=_to_int(_get(p, "quantity", "qty", default=0)) or 0,
                avg_price=_to_float(_get(p, "avg_price", "avgPrice", "averagePrice")),
                mark=_to_float(_get(p, "mark", "markPrice")),
            )

    def set_account(self, account: AccountInfo | None = None, *, cash: float | None = None, reserved: float | None = None) -> None:
        """Update cash / positions value from an AccountInfo and/or explicit values."""
        if account is not None:
            self.cash = float(_get(account, "cash", "balance", default=self.cash))
            pv = _to_float(_get(account, "positions_value", "positionsValue"))
            if pv is not None:
                self.positions_value = pv
        if cash is not None:
            self.cash = float(cash)
        if reserved is not None:
            self.reserved = float(reserved)

    def set_open_intents(self, intents: Iterable[OrderIntent]) -> None:
        """Replace the open-intent cache (ACKED intents only are exposed in views)."""
        self.open_intents = {i.id: i for i in intents}

    # -- reads -------------------------------------------------------------------------------
    def top(self, exchange_id: str) -> Top | None:
        return self.tops.get(exchange_id)

    def book(self, exchange_id: str) -> OrderBook | None:
        return self.books.get(exchange_id)

    def mid(self, exchange_id: str) -> float | None:
        t = self.tops.get(exchange_id)
        return t.mid if t else None

    @property
    def equity(self) -> float:
        return self.cash + self.positions_value

    def position(self, exchange_id: str) -> int:
        p = self.positions.get(exchange_id)
        return p.quantity if p else 0

    def stale_books(self, max_age_s: float, now: datetime | None = None) -> list[str]:
        """Exchange ids whose depth cache is older than ``max_age_s`` (or missing)."""
        now = now or utc_now()
        out: list[str] = []
        for eid in self.exchange_ids():
            ts = self.book_ts.get(eid)
            if ts is None or (now - ts).total_seconds() > max_age_s:
                out.append(eid)
        return out

    def hot_markets(self, limit: int) -> list[str]:
        """Markets worth a realtime channel: those with positions or open orders first, then the
        most two-sided / closest-to-50 books."""
        scored: list[tuple[float, str]] = []
        with_orders = {i.exchange_id for i in self.open_intents.values()}
        for eid in self.exchange_ids():
            score = 0.0
            if self.position(eid) != 0:
                score += 2.0
            if eid in with_orders:
                score += 2.0
            m = self.mid(eid)
            if m is not None:
                score += 1.0 - abs(m - 0.5) * 2
            scored.append((score, eid))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [eid for _, eid in scored[:limit]]

    def views(
        self,
        positions: Iterable[Any] | Mapping[str, Any] | None = None,
        open_orders: Iterable[OrderIntent] | None = None,
    ) -> dict[str, MarketView]:
        """Build MarketView objects joined with race metadata. ``positions``/``open_orders`` override the
        cached ones when given. Markets without metadata are included with chamber ``unknown``."""
        if positions is not None:
            self.set_positions(positions)
        if open_orders is not None:
            self.set_open_intents(open_orders)
        intents = list(self.open_intents.values())
        by_market: dict[str, list[OrderIntent]] = {}
        for i in intents:
            if str(i.state) == IntentState.ACKED.value:
                by_market.setdefault(i.exchange_id, []).append(i)
        out: dict[str, MarketView] = {}
        for eid in self.exchange_ids():
            m = self.meta.get(eid)
            t = self.tops.get(eid)
            pos = self.positions.get(eid)
            out[eid] = MarketView(
                exchange_id=eid,
                race_id=m.race_id if m else eid,
                party=m.party if m else "?",
                chamber=m.chamber if m else "unknown",
                state=m.state if m else None,
                bid=t.bid if t else None,
                ask=t.ask if t else None,
                last=t.last if t else None,
                bid_size=t.bid_size if t else None,
                ask_size=t.ask_size if t else None,
                book=self.books.get(eid),
                position=pos.quantity if pos else 0,
                avg_price=pos.avg_price if pos else None,
                open_orders=by_market.get(eid, []),
                ts=t.ts if t else None,
                status=m.status if m else "open",
            )
        return out

    # -- persistence -------------------------------------------------------------------------
    def _queue_quote(self, top: Top) -> None:
        self._pending_quotes.append(
            Quote(
                ts=top.ts or utc_now(),
                exchange_id=top.exchange_id,
                bid=top.bid,
                ask=top.ask,
                last=top.last,
                bid_size=top.bid_size,
                ask_size=top.ask_size,
                source=top.source,
            )
        )

    def flush_quotes(self, session: Session | None = None) -> int:
        """Write queued Quote rows. Returns the number written (0 on DB failure, which is logged)."""
        if not self._pending_quotes:
            return 0
        rows, self._pending_quotes = self._pending_quotes, []
        try:
            if session is not None:
                session.add_all(rows)
                session.flush()
            else:
                with dbengine.session() as s:
                    s.add_all(rows)
        except Exception:  # noqa: BLE001 - quotes are best-effort telemetry
            log.exception("failed to persist %d quote rows", len(rows))
            return 0
        return len(rows)

    @property
    def pending_quotes(self) -> int:
        return len(self._pending_quotes)


def load_positions_from_db(session: Session) -> dict[str, PositionInfo]:
    """Convenience: Position rows -> PositionInfo (used when the adapter is unavailable)."""
    from sigcup.db.models import Position

    rows = session.execute(select(Position)).scalars().all()
    return {
        r.exchange_id: PositionInfo(exchange_id=r.exchange_id, quantity=r.quantity, avg_price=r.avg_price, mark=r.mark)
        for r in rows
    }
