"""In-memory ExchangeAdapter for the execution tests. Deterministic, scriptable failure modes, and a
record of every write attempt (so rate-limit tests can count writes per window).

Scripting: ``fake.script[idempotency_key] = ["uncertain", "lost", RequestInFlight(...), ...]`` pops one
behaviour per POST with that key; ``fake.default_script`` applies to keys without a script.

- "uncertain": the order IS recorded but the response is lost (OrderResult.uncertain=True).
- "lost": the request never reached the exchange (nothing recorded, uncertain response).
- an Exception instance: raised as-is.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from sigcup.common.time import ensure_utc, utc_now
from sigcup.sig.errors import RateLimited, ValidationError
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import (
    AccountInfo,
    ConstraintViolation,
    FillInfo,
    Leaderboard,
    LeaderboardEntry,
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
)


@dataclass
class FakeOrder:
    order_id: str
    req: OrderRequest
    created_at: datetime
    status: str = "open"
    filled: int = 0
    fills: list[FillInfo] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.req.quantity - self.filled)


class FakeAdapter:
    name = "fake"

    def __init__(
        self,
        *,
        budget: Budget | None = None,
        clock: Callable[[], datetime] | None = None,
        cash: float = 100_000.0,
        tournament_id: str | None = "t-fake",
    ) -> None:
        self.budget = budget
        self.clock = clock or utc_now
        self.cash = cash
        self.tournament_id = tournament_id
        self.prices: dict[str, PriceSnapshot] = {}
        self.books: dict[str, OrderBook] = {}
        self.markets: list[MarketInfo] = []
        self.orders: dict[str, FakeOrder] = {}
        self.by_key: dict[str, str] = {}
        self.fills: list[FillInfo] = []
        self.positions: dict[str, PositionInfo] = {}
        self.writes: list[tuple[datetime, str]] = []  # every write attempt, even if it raised
        self.reads: list[str] = []
        self.script: dict[str, list] = {}
        self.default_script: list = []
        self.cancel_script: list = []
        self.leaderboard = Leaderboard(entries=[], member_count=10, my_rank=5, my_total=cash)
        self.relationships: list[Relationship] = []
        self._ids = itertools.count(1)
        self._fill_ids = itertools.count(1)
        self.validate_cash = False

    # -- test helpers --------------------------------------------------------------------------
    def set_top(self, eid: str, bid: float | None, ask: float | None, bid_size: int = 20, ask_size: int = 20, last: float | None = None) -> None:
        self.prices[eid] = PriceSnapshot(exchange_id=eid, best_bid=bid, best_ask=ask, last=last, bid_size=bid_size, ask_size=ask_size, ts=self.clock())
        bids = [Level(price=bid, quantity=bid_size)] if bid is not None else []
        asks = [Level(price=ask, quantity=ask_size)] if ask is not None else []
        self.books[eid] = OrderBook(exchange_id=eid, bids=bids, asks=asks, ts=self.clock())

    def fill(self, order_id: str, qty: int | None = None, price: float | None = None) -> FillInfo:
        """Simulate a counterparty hitting our resting order."""
        o = self.orders[order_id]
        qty = o.remaining if qty is None else min(qty, o.remaining)
        price = o.req.price if price is None else price
        o.filled += qty
        if o.remaining == 0:
            o.status = "filled"
        f = FillInfo(
            fill_id=f"fake:{next(self._fill_ids)}",
            order_id=order_id,
            exchange_id=o.req.exchange_id,
            ts=self.clock(),
            side=o.req.side,
            action=o.req.action,
            price=price,
            quantity=qty,
        )
        self.fills.append(f)
        o.fills.append(f)
        long_yes = (o.req.side == "yes") == (o.req.action == "buy")
        yes_price = price if o.req.side == "yes" else 1 - price
        delta = qty if long_yes else -qty
        pos = self.positions.get(o.req.exchange_id)
        q = (pos.quantity if pos else 0) + delta
        self.positions[o.req.exchange_id] = PositionInfo(exchange_id=o.req.exchange_id, quantity=q, avg_price=yes_price, mark=yes_price)
        self.cash -= price * qty
        return f

    def expire_all(self) -> int:
        now = self.clock()
        n = 0
        for o in self.orders.values():
            if o.status == "open" and o.req.expires_at is not None and ensure_utc(o.req.expires_at) <= now:
                o.status = "expired"
                n += 1
        return n

    def open_orders(self) -> list[FakeOrder]:
        return [o for o in self.orders.values() if o.status == "open"]

    def _write(self, kind: str) -> None:
        self.writes.append((self.clock(), kind))
        if self.budget is not None and not self.budget.writes.try_acquire():
            raise RateLimited(60.0)

    def _read(self, kind: str) -> None:
        self.reads.append(kind)

    def _info(self, o: FakeOrder) -> OrderInfo:
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
            expires_at=o.req.expires_at,
        )

    def _result(self, o: FakeOrder) -> OrderResult:
        return OrderResult(order_id=o.order_id, status=o.status, quantity_traded=o.filled)

    # -- adapter: reads ------------------------------------------------------------------------
    async def get_tournament(self) -> TournamentInfo:
        self._read("tournament")
        return TournamentInfo(id=self.tournament_id or "t", slug="fake", my_balance=self.cash)

    async def list_markets(self) -> list[MarketInfo]:
        self._read("markets")
        if self.markets:
            return list(self.markets)
        return [MarketInfo(id=eid, exchange_id=eid, title=eid, status="open") for eid in self.prices]

    async def get_prices(self, exchange_ids: list[str]) -> dict[str, PriceSnapshot]:
        self._read("prices")
        return {eid: self.prices[eid] for eid in exchange_ids if eid in self.prices}

    async def get_orderbook(self, exchange_id: str, depth: int = 20) -> OrderBook:
        self._read("book")
        return self.books.get(exchange_id, OrderBook(exchange_id=exchange_id))

    async def get_account(self) -> AccountInfo:
        self._read("account")
        pv = sum(p.quantity * (p.mark or 0.5) if p.quantity > 0 else -p.quantity * (1 - (p.mark or 0.5)) for p in self.positions.values())
        return AccountInfo(cash=self.cash, positions_value=pv, total=self.cash + pv)

    async def get_positions(self) -> list[PositionInfo]:
        self._read("positions")
        return [p for p in self.positions.values() if p.quantity != 0]

    async def get_open_orders(self) -> list[OrderInfo]:
        self._read("orders")
        self.expire_all()
        return [self._info(o) for o in self.orders.values() if o.status == "open"]

    async def get_fills(self, since: datetime | None = None) -> list[FillInfo]:
        self._read("fills")
        if since is None:
            return list(self.fills)
        return [f for f in self.fills if ensure_utc(f.ts) >= ensure_utc(since)]

    async def get_leaderboard(self) -> Leaderboard:
        self._read("leaderboard")
        return self.leaderboard

    async def get_relationships(self) -> list[Relationship]:
        self._read("relationships")
        return list(self.relationships)

    async def get_constraint_violations(self) -> list[ConstraintViolation]:
        self._read("violations")
        return []

    # -- adapter: writes -----------------------------------------------------------------------
    def _behaviour(self, key: str):
        q = self.script.get(key)
        if q:
            return q.pop(0)
        if self.default_script:
            return self.default_script.pop(0)
        return None

    async def place_order(self, req: OrderRequest) -> OrderResult:
        self._write("place")
        b = self._behaviour(req.idempotency_key)
        if isinstance(b, BaseException):
            raise b
        if req.idempotency_key in self.by_key:
            return self._result(self.orders[self.by_key[req.idempotency_key]])
        if req.tournament_id is None:
            raise ValidationError("tournamentId required", 400, "VALIDATION_ERROR")
        if not is_on_tick(round(req.price, 3)) or req.quantity <= 0:
            raise ValidationError("off tick / bad qty", 400, "VALIDATION_ERROR")
        if self.validate_cash and req.action == "buy" and req.price * req.quantity > self.cash:
            raise ValidationError("insufficient funds", 400, "INSUFFICIENT_FUNDS")
        if b == "lost":
            return OrderResult(uncertain=True)
        o = FakeOrder(order_id=f"o{next(self._ids)}", req=req, created_at=self.clock())
        self.orders[o.order_id] = o
        self.by_key[req.idempotency_key] = o.order_id
        if b == "uncertain":
            return OrderResult(uncertain=True)
        return self._result(o)

    async def place_multi_leg(self, legs: list[OrderRequest]) -> MultiLegResult:
        self._write("multi")
        results = []
        for leg in legs:
            if leg.idempotency_key in self.by_key:
                results.append(self._result(self.orders[self.by_key[leg.idempotency_key]]))
                continue
            o = FakeOrder(order_id=f"o{next(self._ids)}", req=leg, created_at=self.clock())
            self.orders[o.order_id] = o
            self.by_key[leg.idempotency_key] = o.order_id
            results.append(self._result(o))
        return MultiLegResult(group_id=f"g{next(self._ids)}", results=results, atomic_ok=True)

    async def cancel_order(self, order_id: str) -> bool:
        self._write("cancel")
        if self.cancel_script:
            b = self.cancel_script.pop(0)
            if isinstance(b, BaseException):
                raise b
        o = self.orders.get(order_id)
        if o is None or o.status != "open":
            return False
        o.status = "cancelled"
        return True

    async def cancel_all(self) -> int:
        self._write("cancel_all")
        n = 0
        for o in self.orders.values():
            if o.status == "open":
                o.status = "cancelled"
                n += 1
        return n


def entries(*totals: float, me_rank: int | None = None) -> list[LeaderboardEntry]:
    return [LeaderboardEntry(rank=i + 1, total=t, is_me=(me_rank == i + 1)) for i, t in enumerate(totals)]


class FakeClock:
    """One clock for everything: ``mono`` feeds the Budget limiters, ``now`` the OMS / adapter."""

    def __init__(self, start: datetime | None = None) -> None:
        from datetime import UTC

        self.start = start or datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
        self.t = 0.0

    def mono(self) -> float:
        return 1000.0 + self.t

    def now(self) -> datetime:
        from datetime import timedelta

        return self.start + timedelta(seconds=self.t)

    def advance(self, seconds: float) -> None:
        self.t += seconds

    async def sleep(self, seconds: float) -> None:
        """Drop-in for asyncio.sleep that advances the fake clock instead of waiting."""
        self.advance(seconds)
