"""Shared data types and the function contracts between trader modules. FROZEN during parallel
implementation: every workstream codes against these names.

Contracts (module -> callable):
  fairvalue.compute_all(session, cfg, markets, now) -> dict[exchange_id, FairValue]
  fairvalue.load_races(session) -> dict[race_id, RaceInfo]
  consistency.scan(ctx) -> list[ArbOpportunity]
  signals.generate(ctx, arbs) -> list[DesiredOrder]          # pre-risk
  risk.apply(ctx, desired) -> list[DesiredOrder]             # scaled / filtered; may drop orders
  risk.kill_switch_triggered(ctx, kill_path) -> str | None   # reason or None
  endgame.sleeve_fraction(cfg, now, leaderboard, equity) -> float
  election_night.is_active(cfg, now) -> bool
  election_night.taker_orders(ctx, ref_prices) -> list[DesiredOrder]
  oms.OMS(adapter, cfg, limiter).startup_reconcile() / sync(desired) / reconcile() / kill(reason)
  book_state.BookState.views(positions, open_orders) -> dict[exchange_id, MarketView]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from sigcup.db.models import Strategy
from sigcup.settings import StrategyConfig
from sigcup.sig.schemas import OrderBook

Side = Literal["yes", "no"]
Action = Literal["buy", "sell"]

PRIORITY_ARB = 0
PRIORITY_THETA = 1
PRIORITY_VALUE = 2
PRIORITY_MM = 3
PRIORITY_HOUSEKEEPING = 4


@dataclass
class RaceInfo:
    race_id: str
    chamber: str  # senate | governor | house | control
    state: str | None
    label: str
    exclusive: bool | None  # from relationships API only
    exhaustive: bool | None
    markets: dict[str, str] = field(default_factory=dict)  # party -> exchange_id
    pvi: float | None = None
    incumbent_party: str | None = None
    settlement_note: str | None = None


@dataclass
class MarketView:
    exchange_id: str
    race_id: str
    party: str
    chamber: str
    state: str | None
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    bid_size: int | None = None
    ask_size: int | None = None
    book: OrderBook | None = None
    position: int = 0  # signed shares: +YES / -NO
    avg_price: float | None = None
    open_orders: list = field(default_factory=list)  # OrderIntent rows (ACKED) on this market
    ts: datetime | None = None
    status: str = "open"

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


@dataclass
class FairValue:
    exchange_id: str
    fair: float  # P(YES)
    sd: float  # 1-sigma uncertainty in probability units
    components: dict[str, float] = field(default_factory=dict)  # ref / polls / ratings
    weights: dict[str, float] = field(default_factory=dict)
    ts: datetime | None = None
    ref_liquidity: float | None = None


@dataclass
class DesiredOrder:
    exchange_id: str
    side: Side
    action: Action
    price: float
    quantity: int
    strategy: Strategy
    priority: int = PRIORITY_VALUE
    group_id: str | None = None  # legs sharing a group are placed atomically (multi-leg)
    fair: float | None = None
    edge: float | None = None
    reason: str = ""
    take: bool = False  # allowed to cross the spread (arb legs, election-night takes)

    @property
    def is_long_yes(self) -> bool:
        """Economic direction: buying YES or selling NO is long YES."""
        return (self.side == "yes") == (self.action == "buy")

    @property
    def notional(self) -> float:
        """Cash needed if this fills as a buy (sell of unowned fills as the opposite buy)."""
        p = self.price if self.action == "buy" else 1 - self.price
        return p * self.quantity


@dataclass
class ArbOpportunity:
    kind: str  # yes_no | race_yes_sum | race_no_sum | control
    legs: list[DesiredOrder]
    profit_per_set: float  # SUSQies per one share of each leg
    max_sets: int  # bounded by displayed depth
    race_id: str | None = None


@dataclass
class ActiveEvent:
    race_id: str
    kind: str  # soft | hard | unknown
    ts: datetime
    description: str
    confidence: float


@dataclass
class TraderContext:
    cfg: StrategyConfig
    now: datetime
    markets: dict[str, MarketView]
    races: dict[str, RaceInfo]
    fair: dict[str, FairValue]
    cash: float
    reserved: float  # cash committed to resting buys (internal reservation)
    equity: float  # cash + marked positions
    bankroll_start: float
    events: list[ActiveEvent] = field(default_factory=list)
    sleeve_frac: float = 0.15
    election_night: bool = False
    hours_to_deadline: float = 1e9
    ref_prices: dict[str, float] = field(default_factory=dict)  # exchange_id -> best real-money YES mid
    daily_pnl: float = 0.0
    my_rank: int | None = None
