"""Builders shared by the model tests (no tests in this module)."""

from __future__ import annotations

from datetime import UTC, datetime

from sigcup.common.time import hours_to
from sigcup.settings import StrategyConfig
from sigcup.sig.schemas import Level, OrderBook
from sigcup.trader.types import ActiveEvent, FairValue, MarketView, RaceInfo, TraderContext

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def race(
    race_id: str,
    chamber: str = "senate",
    state: str | None = None,
    parties: tuple[str, ...] = ("D", "R"),
    exclusive: bool | None = True,
    exhaustive: bool | None = True,
    **kw,
) -> RaceInfo:
    info = RaceInfo(
        race_id=race_id,
        chamber=chamber,
        state=state,
        label=kw.pop("label", race_id),
        exclusive=exclusive,
        exhaustive=exhaustive,
        **kw,
    )
    for p in parties:
        info.markets[p] = f"{race_id}-{p}"
    return info


def view(
    info: RaceInfo,
    party: str,
    bid: float | None = None,
    ask: float | None = None,
    last: float | None = None,
    bid_size: int | None = None,
    ask_size: int | None = None,
    position: int = 0,
    status: str = "open",
    book: OrderBook | None = None,
) -> MarketView:
    return MarketView(
        exchange_id=info.markets[party],
        race_id=info.race_id,
        party=party,
        chamber=info.chamber,
        state=info.state,
        bid=bid,
        ask=ask,
        last=last,
        bid_size=bid_size,
        ask_size=ask_size,
        book=book,
        position=position,
        ts=NOW,
        status=status,
    )


def book(exchange_id: str, bids: list[tuple[float, int]], asks: list[tuple[float, int]]) -> OrderBook:
    return OrderBook(
        exchange_id=exchange_id,
        bids=[Level(price=p, quantity=q) for p, q in bids],
        asks=[Level(price=p, quantity=q) for p, q in asks],
        ts=NOW,
    )


def fair(exchange_id: str, p: float, sd: float = 0.03, ts: datetime | None = NOW) -> FairValue:
    return FairValue(exchange_id=exchange_id, fair=p, sd=sd, components={"ref": p}, weights={"ref": 1.0}, ts=ts)


def event(race_id: str, kind: str, ts: datetime, description: str = "event", confidence: float = 0.9) -> ActiveEvent:
    return ActiveEvent(race_id=race_id, kind=kind, ts=ts, description=description, confidence=confidence)


def ctx(
    cfg: StrategyConfig,
    races: list[RaceInfo],
    views: list[MarketView],
    fairs: list[FairValue] = (),
    now: datetime = NOW,
    **kw,
) -> TraderContext:
    kw.setdefault("cash", 100_000.0)
    kw.setdefault("reserved", 0.0)
    kw.setdefault("equity", 100_000.0)
    kw.setdefault("bankroll_start", cfg.bankroll_start)
    kw.setdefault("hours_to_deadline", hours_to(cfg.deadline_utc, now))
    return TraderContext(
        cfg=cfg,
        now=now,
        markets={v.exchange_id: v for v in views},
        races={r.race_id: r for r in races},
        fair={f.exchange_id: f for f in fairs},
        **kw,
    )
