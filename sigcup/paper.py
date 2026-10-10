"""PaperAdapter: implements ExchangeAdapter without touching the exchange.

- Wraps a read-only live client (or recorded quotes from the DB) for market data.
- Orders rest in memory; a buy fills when its price >= best ask (sell when price <= best bid) at the
  resting price; partial fills honour displayed size; expiry honoured.
- Tracks cash, positions, fills with ids `paper:<n>`; cancel/cancel_all supported.
- Persists intents/fills with paper=True so reports can separate paper from live (done by the OMS,
  which is constructed with ``paper=True``; the adapter itself keeps everything in memory).

Price convention follows the exchange: ``price`` is the price of the *side* named in the order (a buy
of NO at 0.10 pays 0.10 per share). Internally every order is mapped to its YES-equivalent for matching
against the YES-quoted book: buy NO @p == sell YES @1-p, sell NO @p == buy YES @1-p. Unowned sells fill as
the opposite buy, exactly like the exchange.

Market data source, in order of preference: the ``source`` adapter (its ``get_prices`` /
``get_orderbook`` / ``list_markets`` ... are delegated and the tops cached for matching), else the latest
Quote rows in the DB, else whatever ``set_top`` was told. Resting orders are re-evaluated on every data
refresh and on every account/positions/orders/fills read.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import select

from sigcup.common.log import get_logger
from sigcup.common.time import ensure_utc, utc_now
from sigcup.db import engine as dbengine
from sigcup.db.models import Market
from sigcup.db.repo import latest_quotes
from sigcup.settings import StrategyConfig
from sigcup.sig.adapter import ExchangeAdapter
from sigcup.sig.errors import RequestInFlight, ValidationError
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import (
    AccountInfo,
    ConstraintViolation,
    FillInfo,
    Leaderboard,
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
    is_on_tick,
    round_tick,
)

log = get_logger(__name__)


@dataclass
class _Top:
    bid: float | None = None
    ask: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    last: float | None = None
    ts: datetime | None = None


@dataclass
class _Order:
    order_id: str
    req: OrderRequest
    created_at: datetime
    expires_at: datetime | None
    filled: int = 0
    status: str = "open"  # open | filled | cancelled | expired | rejected
    total_cost: float = 0.0
    fills: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, int(self.req.quantity) - self.filled)

    @property
    def yes_action(self) -> str:
        """buy | sell in YES-equivalent terms."""
        long_yes = (self.req.side == "yes") == (self.req.action == "buy")
        return "buy" if long_yes else "sell"

    @property
    def yes_price(self) -> float:
        return self.req.price if self.req.side == "yes" else round_tick(1.0 - self.req.price)


@dataclass
class _Position:
    quantity: int = 0  # signed +YES / -NO
    cost: float = 0.0  # cash paid for the current net position (basis)

    @property
    def avg_price(self) -> float | None:
        q = abs(self.quantity)
        return self.cost / q if q else None


class PaperAdapter:
    """Dry-run exchange. Satisfies ``sigcup.sig.adapter.ExchangeAdapter``."""

    name = "paper"

    def __init__(
        self,
        cfg: StrategyConfig,
        *,
        source: ExchangeAdapter | None = None,
        cash: float | None = None,
        clock: Callable[[], datetime] | None = None,
        budget: Budget | None = None,
        use_db_quotes: bool = True,
        tournament_id: str | None = "paper",
    ) -> None:
        self.cfg = cfg
        self.source = source
        self.clock = clock or utc_now
        self.budget = budget
        self.use_db_quotes = use_db_quotes and source is None
        self.tournament_id = getattr(source, "tournament_id", None) or tournament_id
        self.cash = float(cfg.bankroll_start if cash is None else cash)
        self.tops: dict[str, _Top] = {}
        self.books: dict[str, OrderBook] = {}
        self.orders: dict[str, _Order] = {}
        self.by_key: dict[str, str] = {}
        self.fills: list[FillInfo] = []
        self.positions: dict[str, _Position] = {}
        self.writes: int = 0
        self.reads: int = 0
        self._ids = itertools.count(1)
        self._fill_ids = itertools.count(1)
        self.lease_s = 90.0
        self._in_flight: dict[str, datetime] = {}

    # -- market data helpers -------------------------------------------------------------------
    def set_top(
        self,
        exchange_id: str,
        bid: float | None,
        ask: float | None,
        bid_size: int | None = None,
        ask_size: int | None = None,
        last: float | None = None,
    ) -> None:
        """Set the best quotes for a market (tests / replay) and re-evaluate resting orders."""
        self.tops[exchange_id] = _Top(bid, ask, bid_size, ask_size, last, self.clock())
        self._match_all()

    def _refresh_from_db(self) -> None:
        try:
            with dbengine.session() as s:
                rows = latest_quotes(s)
        except Exception as e:  # noqa: BLE001
            log.warning("paper: cannot read quotes from the DB: %s", e)
            return
        for eid, q in rows.items():
            self.tops[eid] = _Top(q.bid, q.ask, q.bid_size, q.ask_size, q.last, ensure_utc(q.ts) if q.ts else None)

    def _snapshot(self, eid: str) -> PriceSnapshot:
        t = self.tops.get(eid, _Top())
        return PriceSnapshot(
            exchange_id=eid, best_bid=t.bid, best_ask=t.ask, last=t.last, bid_size=t.bid_size, ask_size=t.ask_size, ts=t.ts
        )

    async def _read(self) -> None:
        self.reads += 1
        if self.budget is not None:
            async with self.budget.reads.acquire(priority=2):
                pass

    async def _write(self, priority: int = 2) -> None:
        """Take a write slot exactly like the live client would (blocking on the shared limiter)."""
        self.writes += 1
        if self.budget is not None:
            async with self.budget.writes.acquire(priority=priority):
                pass

    # -- matching ------------------------------------------------------------------------------
    def _expire(self, now: datetime) -> None:
        for o in self.orders.values():
            if o.status == "open" and o.expires_at is not None and ensure_utc(o.expires_at) <= now:
                o.status = "expired"

    def _match_all(self) -> None:
        now = self.clock()
        self._expire(now)
        for o in sorted(self.orders.values(), key=lambda o: o.created_at):
            if o.status == "open":
                self._match(o, now)

    def _match(self, o: _Order, now: datetime) -> None:
        top = self.tops.get(o.req.exchange_id)
        if top is None or o.remaining <= 0:
            return
        if o.yes_action == "buy":
            if top.ask is None or o.yes_price < top.ask - 1e-9:
                return
            avail = top.ask_size if top.ask_size is not None else o.remaining
        else:
            if top.bid is None or o.yes_price > top.bid + 1e-9:
                return
            avail = top.bid_size if top.bid_size is not None else o.remaining
        qty = min(o.remaining, max(0, int(avail)))
        if qty <= 0:
            return
        self._fill(o, qty, now)
        # consume displayed size so a second order cannot fill against the same liquidity
        if o.yes_action == "buy" and top.ask_size is not None:
            top.ask_size = max(0, top.ask_size - qty)
        elif o.yes_action == "sell" and top.bid_size is not None:
            top.bid_size = max(0, top.bid_size - qty)

    def _fill(self, o: _Order, qty: int, now: datetime) -> FillInfo:
        price = float(o.req.price)  # at the resting price, in the order's side terms
        yes_price = o.yes_price
        eid = o.req.exchange_id
        pos = self.positions.setdefault(eid, _Position())
        if o.yes_action == "buy":
            self._apply_delta(pos, +qty, yes_price)
        else:
            self._apply_delta(pos, -qty, yes_price)
        o.filled += qty
        o.total_cost += price * qty
        if o.remaining == 0:
            o.status = "filled"
        fill = FillInfo(
            fill_id=f"paper:{next(self._fill_ids)}",
            order_id=o.order_id,
            exchange_id=eid,
            ts=now,
            side=o.req.side,
            action=o.req.action,
            price=price,
            quantity=qty,
            fee=0.0,
        )
        self.fills.append(fill)
        o.fills.append(fill.fill_id)
        top = self.tops.get(eid)
        if top is not None:
            top.last = yes_price
        return fill

    def _apply_delta(self, pos: _Position, delta_yes: int, yes_price: float) -> None:
        """Apply a signed YES-share delta. Buying YES pays p; selling owned YES receives p; selling
        unowned YES is buying NO at 1-p (cash out), and vice versa."""
        q = pos.quantity
        remaining = delta_yes
        if q > 0 and remaining < 0:  # selling owned YES
            close = min(q, -remaining)
            basis = pos.cost / q
            self.cash += yes_price * close
            pos.quantity -= close
            pos.cost -= basis * close
            remaining += close
        elif q < 0 and remaining > 0:  # buying back owned NO (sell NO at 1 - yes_price)
            close = min(-q, remaining)
            basis = pos.cost / -q
            self.cash += (1.0 - yes_price) * close
            pos.quantity += close
            pos.cost -= basis * close
            remaining -= close
        if remaining > 0:  # new long YES
            self.cash -= yes_price * remaining
            pos.cost += yes_price * remaining
            pos.quantity += remaining
        elif remaining < 0:  # new NO
            self.cash -= (1.0 - yes_price) * -remaining
            pos.cost += (1.0 - yes_price) * -remaining
            pos.quantity += remaining
        if pos.quantity == 0:
            pos.cost = 0.0

    def _mark(self, eid: str) -> float | None:
        t = self.tops.get(eid)
        if t is None:
            return None
        if t.bid is not None and t.ask is not None:
            return (t.bid + t.ask) / 2
        return t.last

    def _positions_value(self) -> float:
        total = 0.0
        for eid, p in self.positions.items():
            if p.quantity == 0:
                continue
            m = self._mark(eid)
            if m is None:
                m = p.avg_price if p.quantity > 0 else 1.0 - (p.avg_price or 0.5)
                m = m if m is not None else 0.5
            total += p.quantity * m if p.quantity > 0 else -p.quantity * (1.0 - m)
        return total

    def _cash_needed(self, req: OrderRequest) -> float:
        pos = self.positions.get(req.exchange_id, _Position()).quantity
        long_yes = (req.side == "yes") == (req.action == "buy")
        yes_price = req.price if req.side == "yes" else 1.0 - req.price
        if long_yes:
            uncovered = max(0, req.quantity - max(0, -pos))
            return yes_price * uncovered
        uncovered = max(0, req.quantity - max(0, pos))
        return (1.0 - yes_price) * uncovered

    def _reserved(self) -> float:
        return sum(self._cash_needed(o.req) * (o.remaining / max(1, o.req.quantity)) for o in self.orders.values() if o.status == "open")

    # -- ExchangeAdapter: market data ----------------------------------------------------------
    async def get_tournament(self) -> TournamentInfo:
        if self.source is not None:
            return await self.source.get_tournament()
        await self._read()
        return TournamentInfo(id="paper", slug="paper", name="paper", my_balance=self.cash)

    async def list_markets(self) -> list[MarketInfo]:
        if self.source is not None:
            return await self.source.list_markets()
        await self._read()
        with dbengine.session() as s:
            rows = s.execute(select(Market)).scalars().all()
        return [MarketInfo(id=m.market_id, exchange_id=m.exchange_id, title=m.title, status=m.status) for m in rows]

    async def get_prices(self, exchange_ids: list[str]) -> dict[str, PriceSnapshot]:
        if self.source is not None:
            snaps = await self.source.get_prices(exchange_ids)
            for eid, s in snaps.items():
                self.tops[eid] = _Top(s.best_bid, s.best_ask, s.bid_size, s.ask_size, s.last, s.ts)
            self._match_all()
            return snaps
        await self._read()
        if self.use_db_quotes:
            self._refresh_from_db()
        self._match_all()
        return {eid: self._snapshot(eid) for eid in exchange_ids if eid in self.tops}

    async def get_orderbook(self, exchange_id: str, depth: int = 20) -> OrderBook:
        if self.source is not None:
            book = await self.source.get_orderbook(exchange_id, depth)
            self.books[exchange_id] = book
            self.tops[exchange_id] = _Top(
                book.best_bid, book.best_ask,
                book.bids[0].quantity if book.bids else None,
                book.asks[0].quantity if book.asks else None,
                self.tops.get(exchange_id, _Top()).last, book.ts,
            )
            self._match_all()
            return book
        await self._read()
        if self.use_db_quotes:
            self._refresh_from_db()
        t = self.tops.get(exchange_id, _Top())
        bids = [Level(price=t.bid, quantity=t.bid_size or 0)] if t.bid is not None else []
        asks = [Level(price=t.ask, quantity=t.ask_size or 0)] if t.ask is not None else []
        return OrderBook(exchange_id=exchange_id, bids=bids, asks=asks, ts=t.ts)

    async def get_leaderboard(self) -> Leaderboard:
        if self.source is not None:
            return await self.source.get_leaderboard()
        await self._read()
        total = self.cash + self._positions_value()
        return Leaderboard(entries=[], member_count=None, my_rank=None, my_total=total)

    async def get_relationships(self) -> list[Relationship]:
        if self.source is not None:
            return await self.source.get_relationships()
        await self._read()
        return []

    async def get_constraint_violations(self) -> list[ConstraintViolation]:
        if self.source is not None:
            return await self.source.get_constraint_violations()
        await self._read()
        return []

    @property
    def has_realtime(self) -> bool:
        return self.source is not None and hasattr(self.source, "realtime_token")

    async def realtime_token(self, **kwargs: Any) -> Any:
        """Pass-through to the wrapped live client (read scope is enough for Realtime)."""
        if not self.has_realtime:
            raise RuntimeError("paper adapter has no realtime-capable source")
        return await self.source.realtime_token(**kwargs)  # type: ignore[union-attr]

    # -- ExchangeAdapter: account --------------------------------------------------------------
    async def get_account(self) -> AccountInfo:
        await self._read()
        self._match_all()
        pv = self._positions_value()
        return AccountInfo(cash=self.cash, positions_value=pv, total=self.cash + pv)

    async def get_positions(self) -> list[PositionInfo]:
        await self._read()
        self._match_all()
        return [
            PositionInfo(exchange_id=eid, quantity=p.quantity, avg_price=p.avg_price, mark=self._mark(eid))
            for eid, p in self.positions.items()
            if p.quantity != 0
        ]

    async def get_open_orders(self) -> list[OrderInfo]:
        await self._read()
        self._match_all()
        return [self._info(o) for o in self.orders.values() if o.status == "open"]

    async def get_fills(self, since: datetime | None = None) -> list[FillInfo]:
        await self._read()
        self._match_all()
        if since is None:
            return list(self.fills)
        since = ensure_utc(since)
        return [f for f in self.fills if ensure_utc(f.ts) >= since]

    def _info(self, o: _Order) -> OrderInfo:
        return OrderInfo(
            order_id=o.order_id,
            exchange_id=o.req.exchange_id,
            side=o.req.side,
            action=o.req.action,
            price=o.req.price,
            quantity=o.req.quantity,
            filled=o.filled,
            status=o.status,
            idempotency_key=o.req.idempotency_key,
            created_at=o.created_at,
            expires_at=o.expires_at,
        )

    # -- ExchangeAdapter: writes ---------------------------------------------------------------
    def _validate(self, req: OrderRequest) -> None:
        if req.quantity <= 0:
            raise ValidationError("quantity must be positive", 400, "VALIDATION_ERROR")
        if not is_on_tick(round(req.price, 3)):
            raise ValidationError(f"price {req.price} is off tick", 400, "VALIDATION_ERROR")
        if req.side not in ("yes", "no") or req.action not in ("buy", "sell"):
            raise ValidationError("bad side/action", 400, "VALIDATION_ERROR")
        if self._cash_needed(req) > self.cash - self._reserved() + 1e-9:
            raise ValidationError("insufficient funds", 400, "INSUFFICIENT_FUNDS")

    def _result(self, o: _Order) -> OrderResult:
        return OrderResult(
            order_id=o.order_id,
            status=o.status,
            quantity_traded=o.filled,
            total_cost=o.total_cost,
            fill_price=(o.total_cost / o.filled) if o.filled else None,
        )

    async def place_order(self, req: OrderRequest) -> OrderResult:
        await self._write()
        now = self.clock()
        if req.idempotency_key in self._in_flight:
            if (now - self._in_flight[req.idempotency_key]).total_seconds() < self.lease_s:
                raise RequestInFlight("request in flight", 409, "REQUEST_IN_FLIGHT")
            del self._in_flight[req.idempotency_key]
        if req.idempotency_key in self.by_key:  # idempotent replay returns the stored order
            return self._result(self.orders[self.by_key[req.idempotency_key]])
        self._expire(now)
        self._validate(req)
        o = _Order(
            order_id=f"paper-o{next(self._ids)}",
            req=req,
            created_at=now,
            expires_at=ensure_utc(req.expires_at) if req.expires_at else None,
        )
        self.orders[o.order_id] = o
        self.by_key[req.idempotency_key] = o.order_id
        self._match(o, now)
        log.info(
            "paper: %s %s %s @%.3f x%d -> %s (filled %d)",
            req.action, req.side, req.exchange_id, req.price, req.quantity, o.status, o.filled,
        )
        return self._result(o)

    async def place_multi_leg(self, legs: list[OrderRequest]) -> MultiLegResult:
        await self._write()
        if len(legs) > self.cfg.arb.max_legs:
            raise ValidationError(f"too many legs ({len(legs)} > {self.cfg.arb.max_legs})", 400, "VALIDATION_ERROR")
        for leg in legs:  # atomic placement: validate everything first
            if leg.idempotency_key not in self.by_key:
                self._validate(leg)
        results: list[OrderResult] = []
        now = self.clock()
        for leg in legs:
            if leg.idempotency_key in self.by_key:
                results.append(self._result(self.orders[self.by_key[leg.idempotency_key]]))
                continue
            o = _Order(
                order_id=f"paper-o{next(self._ids)}",
                req=leg,
                created_at=now,
                expires_at=ensure_utc(leg.expires_at) if leg.expires_at else None,
            )
            self.orders[o.order_id] = o
            self.by_key[leg.idempotency_key] = o.order_id
            self._match(o, now)
            results.append(self._result(o))
        return MultiLegResult(group_id=f"paper-g{next(self._ids)}", results=results, atomic_ok=True)

    async def cancel_order(self, order_id: str) -> bool:
        await self._write()
        o = self.orders.get(order_id)
        if o is None or o.status != "open":
            return False
        o.status = "cancelled"
        return True

    async def cancel_all(self) -> int:
        await self._write(priority=0)
        n = 0
        for o in self.orders.values():
            if o.status == "open":
                o.status = "cancelled"
                n += 1
        return n

    # -- introspection -------------------------------------------------------------------------
    @property
    def equity(self) -> float:
        return self.cash + self._positions_value()

    def open_orders(self) -> list[_Order]:
        return [o for o in self.orders.values() if o.status == "open"]
