"""Normalized exchange data types. Parsers tolerate unknown/renamed fields: the official OpenAPI document
lives at https://sig.thesuper.market/api/v1/docs and these shapes were reconstructed from it and from
public client code, so every parser must be lenient (`extra='allow'`) and every field that may be missing
is Optional."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Side = Literal["yes", "no"]
Action = Literal["buy", "sell"]

TICK = 0.005
PRICE_MIN = 0.005
PRICE_MAX = 0.995


def round_tick(price: float) -> float:
    """Round to the exchange tick and clamp to the legal range."""
    p = round(round(price / TICK) * TICK, 3)
    return min(PRICE_MAX, max(PRICE_MIN, p))


def is_on_tick(price: float) -> bool:
    return PRICE_MIN <= price <= PRICE_MAX and abs(price / TICK - round(price / TICK)) < 1e-6


class Lenient(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class TournamentInfo(Lenient):
    id: str
    slug: str
    name: str | None = None
    start_date: datetime | None = Field(default=None, alias="startDate")
    end_date: datetime | None = Field(default=None, alias="endDate")
    member_count: int | None = Field(default=None, alias="memberCount")
    market_count: int | None = Field(default=None, alias="marketCount")
    my_balance: float | None = Field(default=None, alias="myBalance")


class MarketInfo(Lenient):
    id: str
    exchange_id: str | None = Field(default=None, alias="exchangeId")
    title: str
    status: str | None = None
    category: str | None = None
    settlement_date: datetime | None = Field(default=None, alias="settlementDate")
    resolution: str | None = None
    overround: float | None = None
    has_arbitrage_opportunity: bool | None = Field(default=None, alias="hasArbitrageOpportunity")
    is_multi_outcome: bool | None = Field(default=None, alias="isMultiOutcome")


class PriceSnapshot(Lenient):
    exchange_id: str
    best_bid: float | None = None
    best_ask: float | None = None
    last: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    ts: datetime | None = None

    @property
    def mid(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return self.last

    @property
    def spread(self) -> float | None:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


class Level(Lenient):
    price: float
    quantity: int


class OrderBook(Lenient):
    exchange_id: str
    bids: list[Level] = Field(default_factory=list)  # YES bids, best first (descending price)
    asks: list[Level] = Field(default_factory=list)  # YES asks, best first (ascending price)
    ts: datetime | None = None

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None


class OrderRequest(BaseModel):
    exchange_id: str
    side: Side
    action: Action
    price: float
    quantity: int
    idempotency_key: str
    expires_at: datetime | None = None
    strategy: str = "manual"
    tournament_id: str | None = None  # filled by the client; always sent


class OrderResult(Lenient):
    order_id: str | None = Field(default=None, alias="orderId")
    status: str | None = None
    quantity_traded: int | None = Field(default=None, alias="quantityTraded")
    total_cost: float | None = Field(default=None, alias="totalCost")
    fill_price: float | None = Field(default=None, alias="fillPrice")
    uncertain: bool = False  # True when the exchange answered 502 ORDER_STATUS_UNKNOWN / timeout


class MultiLegResult(Lenient):
    group_id: str | None = None
    results: list[OrderResult] = Field(default_factory=list)
    atomic_ok: bool = True


class OrderInfo(Lenient):
    order_id: str = Field(alias="orderId")
    exchange_id: str | None = Field(default=None, alias="exchangeId")
    side: str | None = None
    action: str | None = None
    price: float | None = None
    quantity: int | None = None
    filled: int | None = None
    status: str | None = None
    idempotency_key: str | None = Field(default=None, alias="idempotencyKey")
    created_at: datetime | None = Field(default=None, alias="createdAt")
    expires_at: datetime | None = Field(default=None, alias="expirationDate")


class FillInfo(Lenient):
    fill_id: str
    order_id: str | None = None
    exchange_id: str
    ts: datetime
    side: str
    action: str
    price: float
    quantity: int
    fee: float = 0.0


class PositionInfo(Lenient):
    exchange_id: str
    quantity: int  # signed: +YES / -NO
    avg_price: float | None = None
    mark: float | None = None


class AccountInfo(Lenient):
    cash: float
    positions_value: float | None = None
    total: float | None = None


class LeaderboardEntry(Lenient):
    rank: int
    total: float
    is_me: bool = False
    user: str | None = None


class Leaderboard(Lenient):
    entries: list[LeaderboardEntry] = Field(default_factory=list)
    member_count: int | None = None
    my_rank: int | None = None
    my_total: float | None = None


class Relationship(Lenient):
    id: str | None = None
    kind: str | None = None  # e.g. mutually_exclusive
    is_exhaustive: bool | None = Field(default=None, alias="isExhaustive")
    exchange_ids: list[str] = Field(default_factory=list)


class ConstraintViolation(Lenient):
    relationship_id: str | None = None
    exchange_ids: list[str] = Field(default_factory=list)
    kind: str | None = None
    magnitude: float | None = None
    suggested_trades: list[dict] = Field(default_factory=list, alias="suggestedCorrectiveTrades")
