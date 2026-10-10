"""Turns fair values, books, positions and config into the desired order set, tagged by strategy:
arb (from consistency), theta (favourites by yield), value (|fair-mid| > min_edge inside the band),
convex (toss-up/control sleeve), mm (optional), fade/follow (only when sentiment.enable_fade_follow and an
Event of the right kind is active), election (election-night taker rule). Returns DesiredOrder list with
priority and size BEFORE risk scaling.

Rules
-----
* Arb legs come first (``take=True`` groups from ``consistency.scan``); informational opportunities
  (``take=False`` / zero size) are not orders.
* Theta: for every leg with ``fair >= theta.min_fair`` the bid is the yield price
  ``round_tick(min(ask or 1, min(fair, fair_cap) / (1 + min_yield)))`` (never rounded *up* past the
  yield bound). The mirror on the other leg (``fair <= 1 - min_fair`` -> buy NO, priced the same way in
  NO terms with the NO ask ``1 - YES bid``) is the same economic position, so only the better-yielding
  candidate per race is emitted. Yield = expected payoff per SUSQie locked = ``min(fair, cap) / price - 1``.
* Value: inside ``value.band`` when ``|fair - mid| > value.min_edge`` -> resting order at
  ``fair -/+ quote_offset`` (buy YES if fair > mid else sell YES), capped at the touch so it never goes
  *through* the spread. Skipped while an Event(kind='hard') for the race is < 24 h old and the fair
  value's inputs predate it (``FairValue.ts``).
* Convex: toss-up (``0.3 <= fair <= 0.7``) or control markets with ``|fair - mid| > convex_sleeve.min_edge``
  -> strategy CONVEX, placeholder quantity 100 (risk sizes it from the sleeve); only while the sleeve
  fraction is positive.
* MM: only when ``mm.enabled`` and ``spread >= mm.min_spread`` -> two-sided at ``mid -/+ half_spread``
  (a side that would quote through our own fair value is dropped), size ``mm.size``.
* Fade / follow: only when ``sentiment.enable_fade_follow`` and an ActiveEvent exists for the race.
  Soft -> fade with a resting order ``FADE_OFFSET`` (4.5c, the middle of the 3-6c range) from the
  pre-event price on the far side of the move; hard -> shift the pre-event price by the move divided by
  the ~0.64 contemporaneous pass-through and quote toward it (follow). The pre-event price is the
  latest Quote before the event, else the (stale) fair value.
* Default quantities before risk scaling: theta 500, value 200, convex 100, mm ``mm.size``, fade /
  follow 100. Every price passes through ``round_tick``; nothing crosses the spread unless ``take``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select

from sigcup.common.log import get_logger
from sigcup.common.time import ensure_utc
from sigcup.db import engine as dbengine
from sigcup.db.models import Quote, Strategy
from sigcup.settings import StrategyConfig
from sigcup.sig.schemas import PRICE_MAX, PRICE_MIN, TICK, round_tick
from sigcup.trader.types import (
    PRIORITY_MM,
    PRIORITY_THETA,
    PRIORITY_VALUE,
    ActiveEvent,
    ArbOpportunity,
    DesiredOrder,
    FairValue,
    MarketView,
    TraderContext,
)

__all__ = [
    "CONVEX_QTY",
    "EVENT_QTY",
    "THETA_QTY",
    "VALUE_QTY",
    "generate",
    "theta_candidates",
    "theta_price",
    "theta_yield",
]

log = get_logger(__name__)

THETA_QTY = 500
VALUE_QTY = 200
CONVEX_QTY = 100
EVENT_QTY = 100
HARD_EVENT_WINDOW = timedelta(hours=24)
FADE_OFFSET = 0.045  # resting fade order this far from the pre-event price (3-6c in the literature)
FADE_MIN_MOVE = 0.03  # ignore smaller moves: indistinguishable from noise / the DMM spread
HARD_PASS_THROUGH = 0.64  # Angelini & De Angelis (2026): contemporaneous pass-through on Kalshi
EVENT_MIN_CONFIDENCE = 0.5
TOSSUP_BAND = (0.30, 0.70)
_EPS = 1e-9
_OPEN_STATUSES = {"open", "active", "trading", ""}


# -- helpers --------------------------------------------------------------------------------------
def _is_open(v: MarketView) -> bool:
    return (v.status or "open").lower() in _OPEN_STATUSES


def _cap_buy(price: float, ask: float | None) -> float:
    """A resting buy never goes through the ask (at the ask is allowed: it fills at the ask)."""
    if ask is not None and price > ask:
        price = ask
    return round_tick(price)


def _cap_sell(price: float, bid: float | None) -> float:
    if bid is not None and price < bid:
        price = bid
    return round_tick(price)


def theta_price(fair: float | None, ask: float | None, cfg: StrategyConfig) -> float | None:
    """Yield-priced favourite bid: ``round_tick(min(ask or 1, min(fair, fair_cap) / (1 + min_yield)))``.

    None when ``fair`` is below ``theta.min_fair`` or no tick satisfies the yield bound. Rounding never
    rounds up past the bound (steps one tick down instead).
    """
    if fair is None or fair < cfg.theta.min_fair - _EPS:
        return None
    bound = min(float(fair), float(cfg.theta.fair_cap)) / (1.0 + float(cfg.theta.min_yield))
    target = bound if ask is None else min(float(ask), bound)
    if target < PRICE_MIN - _EPS:
        return None
    px = round_tick(target)
    if px > target + _EPS:
        px = round_tick(px - TICK)
    if px > target + _EPS or px < PRICE_MIN - _EPS:
        return None
    return px


def theta_yield(fair: float, price: float, cfg: StrategyConfig) -> float:
    """Expected payoff per SUSQie locked: ``min(fair, fair_cap) / price - 1``."""
    if price <= 0:
        return 0.0
    return min(float(fair), float(cfg.theta.fair_cap)) / float(price) - 1.0


def theta_candidates(ctx: TraderContext, race_views: list[MarketView]) -> list[tuple[float, DesiredOrder]]:
    """(yield, order) for every theta candidate on a race: buy YES on favourites, buy NO on longshots."""
    cfg = ctx.cfg
    out: list[tuple[float, DesiredOrder]] = []
    for v in race_views:
        fv = ctx.fair.get(v.exchange_id)
        if fv is None or not _is_open(v):
            continue
        fair = float(fv.fair)
        # YES side: favourite
        px = theta_price(fair, v.ask, cfg)
        if px is not None:
            y = theta_yield(fair, px, cfg)
            if y >= cfg.theta.min_yield - _EPS:
                out.append(
                    (
                        y,
                        DesiredOrder(
                            exchange_id=v.exchange_id,
                            side="yes",
                            action="buy",
                            price=px,
                            quantity=THETA_QTY,
                            strategy=Strategy.THETA,
                            priority=PRIORITY_THETA,
                            fair=fair,
                            edge=fair - px,
                            reason=f"theta yes fair={fair:.3f} yield={y:.3f}",
                        ),
                    )
                )
        # NO side: longshot (NO ask == 1 - YES bid)
        fair_no = 1.0 - fair
        no_ask = None if v.bid is None else 1.0 - float(v.bid)
        px_no = theta_price(fair_no, no_ask, cfg)
        if px_no is not None:
            y = theta_yield(fair_no, px_no, cfg)
            if y >= cfg.theta.min_yield - _EPS:
                out.append(
                    (
                        y,
                        DesiredOrder(
                            exchange_id=v.exchange_id,
                            side="no",
                            action="buy",
                            price=px_no,
                            quantity=THETA_QTY,
                            strategy=Strategy.THETA,
                            priority=PRIORITY_THETA,
                            fair=fair,
                            edge=fair_no - px_no,
                            reason=f"theta no fair_no={fair_no:.3f} yield={y:.3f}",
                        ),
                    )
                )
    return out


def _hard_event_blocks(ctx: TraderContext, race_id: str, fv: FairValue) -> ActiveEvent | None:
    """The hard event (< 24 h old, confident) whose timestamp the fair value's inputs do not cover."""
    for e in ctx.events:
        if e.race_id != race_id or (e.kind or "").lower() != "hard":
            continue
        if float(e.confidence or 0.0) < EVENT_MIN_CONFIDENCE:
            continue
        ts = ensure_utc(e.ts)
        if ctx.now - ts >= HARD_EVENT_WINDOW or ts > ctx.now + timedelta(minutes=5):
            continue
        if fv.ts is None or ensure_utc(fv.ts) < ts:
            return e
    return None


def _quote_toward(v: MarketView, fair: float, mid: float, offset: float) -> tuple[str, float] | None:
    """('buy'|'sell', price) resting ``offset`` inside the fair value, never through the spread."""
    if fair > mid:
        px = _cap_buy(fair - offset, v.ask)
        if px >= fair - _EPS:
            return None
        return "buy", px
    px = _cap_sell(fair + offset, v.bid)
    if px <= fair + _EPS:
        return None
    return "sell", px


def _pre_event_price(exchange_id: str, ts: datetime) -> float | None:
    """Latest stored top-of-book mid before ``ts`` (None when unavailable)."""
    try:
        with dbengine.session() as s:
            q = (
                select(Quote)
                .where(Quote.exchange_id == exchange_id, Quote.ts < ts)
                .order_by(Quote.ts.desc(), Quote.id.desc())
                .limit(1)
            )
            row = s.execute(q).scalars().first()
    except Exception as e:  # noqa: BLE001 - optional input
        log.debug("pre-event quote lookup failed for %s: %s", exchange_id, e)
        return None
    if row is None:
        return None
    if row.bid is not None and row.ask is not None:
        return (float(row.bid) + float(row.ask)) / 2
    return float(row.last) if row.last is not None else None


# -- strategies -----------------------------------------------------------------------------------
def _arb_orders(arbs: list[ArbOpportunity]) -> list[DesiredOrder]:
    out: list[DesiredOrder] = []
    for a in arbs or []:
        if a.max_sets <= 0 or not a.legs or not all(leg.take for leg in a.legs):
            continue
        out.extend(leg for leg in a.legs if leg.quantity > 0)
    return out


def _by_race(ctx: TraderContext) -> dict[str, list[MarketView]]:
    out: dict[str, list[MarketView]] = {}
    for race_id, race in ctx.races.items():
        for _party, eid in race.markets.items():
            v = ctx.markets.get(eid)
            if v is not None:
                out.setdefault(race_id, []).append(v)
    for v in ctx.markets.values():
        if v.race_id not in out or all(x.exchange_id != v.exchange_id for x in out[v.race_id]):
            out.setdefault(v.race_id, []).append(v)
    return out


def _theta(ctx: TraderContext, by_race: dict[str, list[MarketView]]) -> list[DesiredOrder]:
    out: list[DesiredOrder] = []
    for race_id in sorted(by_race):
        cands = theta_candidates(ctx, by_race[race_id])
        if not cands:
            continue
        cands.sort(key=lambda t: (-t[0], t[1].exchange_id, t[1].side))
        out.append(cands[0][1])
    return out


def _value_and_convex(ctx: TraderContext) -> list[DesiredOrder]:
    cfg = ctx.cfg
    lo, hi = cfg.value.band
    out: list[DesiredOrder] = []
    for eid in sorted(ctx.markets):
        v = ctx.markets[eid]
        fv = ctx.fair.get(eid)
        if fv is None or not _is_open(v) or v.mid is None:
            continue
        fair, mid = float(fv.fair), float(v.mid)
        edge = fair - mid
        blocking = _hard_event_blocks(ctx, v.race_id, fv)
        # value
        if lo < fair < hi and abs(edge) > cfg.value.min_edge + _EPS:
            if blocking is not None:
                log.debug("value %s skipped: hard event %r predates fair inputs", eid, blocking.description[:60])
            else:
                q = _quote_toward(v, fair, mid, cfg.value.quote_offset)
                if q is not None:
                    action, px = q
                    out.append(
                        DesiredOrder(
                            exchange_id=eid,
                            side="yes",
                            action=action,  # type: ignore[arg-type]
                            price=px,
                            quantity=VALUE_QTY,
                            strategy=Strategy.VALUE,
                            priority=PRIORITY_VALUE,
                            fair=fair,
                            edge=edge,
                            reason=f"value fair={fair:.3f} mid={mid:.3f} sd={fv.sd:.3f}",
                        )
                    )
        # convex sleeve
        tossup = TOSSUP_BAND[0] <= fair <= TOSSUP_BAND[1]
        if ctx.sleeve_frac > 0 and (tossup or v.chamber == "control") and abs(edge) > cfg.convex_sleeve.min_edge + _EPS:
            if blocking is not None:
                continue
            q = _quote_toward(v, fair, mid, cfg.value.quote_offset)
            if q is not None:
                action, px = q
                out.append(
                    DesiredOrder(
                        exchange_id=eid,
                        side="yes",
                        action=action,  # type: ignore[arg-type]
                        price=px,
                        quantity=CONVEX_QTY,
                        strategy=Strategy.CONVEX,
                        priority=PRIORITY_VALUE,
                        fair=fair,
                        edge=edge,
                        reason=f"convex fair={fair:.3f} mid={mid:.3f} sleeve={ctx.sleeve_frac:.2f}",
                    )
                )
    return out


def _mm(ctx: TraderContext) -> list[DesiredOrder]:
    cfg = ctx.cfg
    if not cfg.mm.enabled:
        return []
    out: list[DesiredOrder] = []
    for eid in sorted(ctx.markets):
        v = ctx.markets[eid]
        if not _is_open(v) or v.spread is None or v.spread < cfg.mm.min_spread - _EPS or v.mid is None:
            continue
        mid = float(v.mid)
        fv = ctx.fair.get(eid)
        fair = float(fv.fair) if fv is not None else None
        bid = _cap_buy(mid - cfg.mm.half_spread, v.ask)
        ask = _cap_sell(mid + cfg.mm.half_spread, v.bid)
        if bid >= ask - _EPS:
            continue
        if fair is None or bid <= fair + _EPS:
            out.append(
                DesiredOrder(
                    exchange_id=eid,
                    side="yes",
                    action="buy",
                    price=bid,
                    quantity=cfg.mm.size,
                    strategy=Strategy.MM,
                    priority=PRIORITY_MM,
                    fair=fair,
                    edge=None if fair is None else fair - bid,
                    reason=f"mm bid spread={v.spread:.3f}",
                )
            )
        if fair is None or ask >= fair - _EPS:
            out.append(
                DesiredOrder(
                    exchange_id=eid,
                    side="yes",
                    action="sell",
                    price=ask,
                    quantity=cfg.mm.size,
                    strategy=Strategy.MM,
                    priority=PRIORITY_MM,
                    fair=fair,
                    edge=None if fair is None else ask - fair,
                    reason=f"mm ask spread={v.spread:.3f}",
                )
            )
    return out


def _event_orders(ctx: TraderContext, by_race: dict[str, list[MarketView]]) -> list[DesiredOrder]:
    cfg = ctx.cfg
    if not cfg.sentiment.enable_fade_follow:
        return []
    out: list[DesiredOrder] = []
    seen: set[str] = set()
    for e in sorted(ctx.events, key=lambda x: ensure_utc(x.ts), reverse=True):
        kind = (e.kind or "").lower()
        if kind not in ("soft", "hard") or e.race_id in seen:
            continue
        if float(e.confidence or 0.0) < EVENT_MIN_CONFIDENCE:
            continue
        ets = ensure_utc(e.ts)
        if ctx.now - ets >= HARD_EVENT_WINDOW:
            continue
        seen.add(e.race_id)
        best: tuple[float, MarketView, float] | None = None
        for v in by_race.get(e.race_id, []):
            if not _is_open(v) or v.mid is None:
                continue
            pre = _pre_event_price(v.exchange_id, ets)
            if pre is None:
                fv = ctx.fair.get(v.exchange_id)
                if fv is not None and fv.ts is not None and ensure_utc(fv.ts) < ets:
                    pre = float(fv.fair)
            if pre is None:
                continue
            move = float(v.mid) - pre
            if best is None or abs(move) > abs(best[0]):
                best = (move, v, pre)
        if best is None or abs(best[0]) < FADE_MIN_MOVE:
            continue
        move, v, pre = best
        mid = float(v.mid)  # type: ignore[arg-type]
        if kind == "soft":
            if move > 0:
                px = _cap_sell(pre + FADE_OFFSET, v.bid)
                action = "sell"
            else:
                px = _cap_buy(pre - FADE_OFFSET, v.ask)
                action = "buy"
            out.append(
                DesiredOrder(
                    exchange_id=v.exchange_id,
                    side="yes",
                    action=action,  # type: ignore[arg-type]
                    price=px,
                    quantity=EVENT_QTY,
                    strategy=Strategy.FADE,
                    priority=PRIORITY_VALUE,
                    fair=pre,
                    edge=abs(px - pre),
                    reason=f"fade soft event: pre={pre:.3f} mid={mid:.3f} ({e.description[:60]})",
                )
            )
        else:
            shifted = min(PRICE_MAX, max(PRICE_MIN, pre + move / HARD_PASS_THROUGH))
            if abs(shifted - mid) <= cfg.value.min_edge:
                continue
            q = _quote_toward(v, shifted, mid, cfg.value.quote_offset)
            if q is None:
                continue
            action, px = q
            out.append(
                DesiredOrder(
                    exchange_id=v.exchange_id,
                    side="yes",
                    action=action,  # type: ignore[arg-type]
                    price=px,
                    quantity=EVENT_QTY,
                    strategy=Strategy.FOLLOW,
                    priority=PRIORITY_VALUE,
                    fair=shifted,
                    edge=shifted - mid,
                    reason=f"follow hard event: pre={pre:.3f} mid={mid:.3f} target={shifted:.3f} ({e.description[:60]})",
                )
            )
    return out


# -- entry point ----------------------------------------------------------------------------------
def generate(ctx: TraderContext, arbs: list[ArbOpportunity]) -> list[DesiredOrder]:
    """The pre-risk desired order set, arb legs first, then by priority (stable)."""
    by_race = _by_race(ctx)
    out: list[DesiredOrder] = []
    out.extend(_arb_orders(arbs))
    out.extend(_theta(ctx, by_race))
    out.extend(_value_and_convex(ctx))
    out.extend(_mm(ctx))
    out.extend(_event_orders(ctx, by_race))
    if ctx.hours_to_deadline <= 0:
        return []
    for d in out:
        d.price = round_tick(d.price)
    out.sort(key=lambda d: (d.priority, 0 if d.take else 1))
    return out
