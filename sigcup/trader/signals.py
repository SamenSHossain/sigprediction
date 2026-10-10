"""Turns fair values, books, positions and config into the desired order set, tagged by strategy:
arb (from consistency), theta (favourites by yield), value (|fair-mid| > min_edge inside the band),
convex (toss-up/control sleeve), mm (optional), fade/follow (only when sentiment.enable_fade_follow and an
Event of the right kind is active), election (election-night taker rule). Returns DesiredOrder list with
priority and size BEFORE risk scaling.

Rules
-----
* Arb legs come first (``take=True`` groups from ``consistency.scan``); informational opportunities
  (``take=False`` / zero size) are not orders.
* Election night (``ctx.election_night``): only the arb legs. Every other strategy here is a resting bet
  of the model against SIG, and on election night results move the real-money price first; the taker rule
  (``election_night.taker_orders``, added by the loop) is the election-night strategy. Working orders of
  the other strategies are no longer desired, so the OMS cancels them.
* Trade fair: the fair value the resting strategies act on. Normally ``FairValue.fair`` and ``.sd``. While
  a confident hard Event for the race is < 24 h old the slow components (polls, ratings) cannot have seen
  it: the fair is the reference component alone, used only when ``FairValue.ts`` (the freshest input,
  in practice the 2-minute reference) is not older than the event; with no post-event input or no
  reference component there is no opinion and theta / value / convex / mm skip the market.
* Book: edges and mids are measured against the *outside* book (``consistency.outside_view``: our own
  resting orders removed), so an order of ours never moves the mid it is measured against; caps that keep
  an order from crossing use the full book, ours included.
* Theta: for every leg with ``fair >= theta.min_fair`` the bid is the yield price
  ``round_tick(min(ask or 1, min(fair, fair_cap) / (1 + min_yield)))`` (never rounded *up* past the
  yield bound). The mirror on the other leg (``fair <= 1 - min_fair`` -> buy NO, priced the same way in
  NO terms with the NO ask ``1 - YES bid``) is the same economic position, so only the better-yielding
  candidate per race is emitted. Yield = expected payoff per SUSQie locked = ``min(fair, cap) / price - 1``.
  Hysteresis (every switch is a cancel plus a new order at the back of the queue): a leg we already work
  (an ACKED theta order) or hold is kept unless another beats its yield by more than
  ``THETA_SWITCH_MARGIN``; otherwise candidates within that margin of the best go to a YES leg (the
  favourite's own market).
* Value: inside ``value.band`` when ``|fair - mid| > max(value.min_edge, VALUE_SD_Z * sd)`` -> resting
  order at ``fair -/+ quote_offset`` (buy YES if fair > mid else sell YES), capped at the touch so it never
  goes *through* the spread. An edge inside the fair value's own uncertainty is not an edge.
* Convex: toss-up (``0.3 <= fair <= 0.7``) or control markets with
  ``|fair - mid| > max(convex_sleeve.min_edge, VALUE_SD_Z * sd)`` -> strategy CONVEX, only while the sleeve
  fraction is positive. Sized here from the sleeve (``risk.apply`` does not read it): each order is
  ``kelly * convex_sleeve.kelly_fraction * sleeve`` SUSQies with ``sleeve = ctx.sleeve_frac * bankroll``,
  all scaled down together so resting convex orders + convex positions (net convex fills, capped by the
  market position) + these orders stay within the sleeve.
* MM: only when ``mm.enabled`` and ``spread >= mm.min_spread`` -> two-sided at ``mid -/+ half_spread``
  (a side that would quote through our own fair value is dropped), size ``mm.size``.
* Fade / follow: only when ``sentiment.enable_fade_follow`` and an ActiveEvent exists for the race.
  Soft -> fade with a resting order ``FADE_OFFSET`` (4.5c, the middle of the 3-6c range) from the
  pre-event price on the far side of the move; hard -> shift the pre-event price by the move divided by
  the ~0.64 contemporaneous pass-through and quote toward it (follow). The pre-event price is the
  latest Quote before the event, else the (stale) fair value.
* Default quantities before risk scaling: theta 500, value 200, convex from the sleeve (``CONVEX_QTY``
  only names the old placeholder), mm ``mm.size``, fade / follow 100. Every price passes through
  ``round_tick``; nothing crosses the spread unless ``take``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import select

from sigcup.common.log import get_logger
from sigcup.common.time import ensure_utc
from sigcup.db import engine as dbengine
from sigcup.db.models import Fill, Quote, Strategy
from sigcup.settings import StrategyConfig
from sigcup.sig.schemas import PRICE_MAX, PRICE_MIN, TICK, round_tick
from sigcup.trader.consistency import outside_view
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
    "THETA_SWITCH_MARGIN",
    "VALUE_QTY",
    "VALUE_SD_Z",
    "generate",
    "theta_candidates",
    "theta_price",
    "theta_yield",
    "trade_fair",
]

log = get_logger(__name__)

THETA_QTY = 500
VALUE_QTY = 200
CONVEX_QTY = 100  # former placeholder; convex orders are now sized from the sleeve (_size_convex)
EVENT_QTY = 100
VALUE_SD_Z = 1.0  # value / convex need |fair - mid| > max(min_edge, VALUE_SD_Z * FairValue.sd)
THETA_SWITCH_MARGIN = 0.005  # yield a theta alternative must add to replace the working leg (~ one tick)
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
        tf = trade_fair(ctx, v.race_id, fv)
        if tf is None:
            continue
        fair = tf[0]
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


def _active_hard_event(ctx: TraderContext, race_id: str) -> ActiveEvent | None:
    """The latest confident hard event for the race that is less than ``HARD_EVENT_WINDOW`` old."""
    found: ActiveEvent | None = None
    for e in ctx.events:
        if e.race_id != race_id or (e.kind or "").lower() != "hard":
            continue
        if float(e.confidence or 0.0) < EVENT_MIN_CONFIDENCE:
            continue
        ts = ensure_utc(e.ts)
        if ctx.now - ts >= HARD_EVENT_WINDOW or ts > ctx.now + timedelta(minutes=5):
            continue
        if found is None or ts > ensure_utc(found.ts):
            found = e
    return found


def trade_fair(ctx: TraderContext, race_id: str, fv: FairValue | None) -> tuple[float, float] | None:
    """``(fair, sd)`` the resting strategies act on, or None for "no opinion" (see the module docstring).

    After a hard event the stale polls / ratings would hold the blend near its pre-event value (and the
    strategies would sell the winner), so only the reference component counts, and only when the fair
    value has an input newer than the event.
    """
    if fv is None:
        return None
    e = _active_hard_event(ctx, race_id)
    if e is None:
        return float(fv.fair), float(fv.sd)
    if fv.ts is None or ensure_utc(fv.ts) < ensure_utc(e.ts):
        log.debug("%s: hard event %r predates the fair value's inputs", fv.exchange_id, (e.description or "")[:60])
        return None
    ref = (fv.components or {}).get("ref")
    if ref is None:
        log.debug("%s: hard event %r and no reference component", fv.exchange_id, (e.description or "")[:60])
        return None
    return min(PRICE_MAX, max(PRICE_MIN, float(ref))), 0.0


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


def _theta_held(views: list[MarketView]) -> set[tuple[str, str]]:
    """(exchange_id, side) legs we already work with a theta buy or hold a position in."""
    held: set[tuple[str, str]] = set()
    for v in views:
        for i in v.open_orders or []:
            strategy = str(getattr(getattr(i, "strategy", ""), "value", getattr(i, "strategy", ""))).lower()
            if strategy == Strategy.THETA.value and str(getattr(i, "action", "buy")).lower() == "buy":
                held.add((v.exchange_id, str(getattr(i, "side", "yes")).lower()))
        if v.position > 0:
            held.add((v.exchange_id, "yes"))
        elif v.position < 0:
            held.add((v.exchange_id, "no"))
    return held


def _pick_theta(cands: list[tuple[float, DesiredOrder]], held: set[tuple[str, str]]) -> DesiredOrder:
    """Best yield with hysteresis toward a held / worked leg and a deterministic YES tie-break."""
    cands = sorted(cands, key=lambda t: (-t[0], t[1].exchange_id, t[1].side))
    best = cands[0][0]
    kept = [c for c in cands if (c[1].exchange_id, c[1].side) in held]
    if kept and best - kept[0][0] <= THETA_SWITCH_MARGIN + _EPS:
        return kept[0][1]
    near_yes = [c for c in cands if c[1].side == "yes" and best - c[0] <= THETA_SWITCH_MARGIN + _EPS]
    return (near_yes or cands)[0][1]


def _theta(ctx: TraderContext, by_race: dict[str, list[MarketView]]) -> list[DesiredOrder]:
    out: list[DesiredOrder] = []
    for race_id in sorted(by_race):
        cands = theta_candidates(ctx, by_race[race_id])
        if not cands:
            continue
        out.append(_pick_theta(cands, _theta_held(by_race[race_id])))
    return out


def _value_and_convex(ctx: TraderContext, outside: dict[str, MarketView]) -> list[DesiredOrder]:
    cfg = ctx.cfg
    lo, hi = cfg.value.band
    out: list[DesiredOrder] = []
    convex: list[DesiredOrder] = []
    for eid in sorted(ctx.markets):
        v = ctx.markets[eid]
        ov = outside.get(eid, v)
        fv = ctx.fair.get(eid)
        if fv is None or not _is_open(v) or ov.mid is None:
            continue
        tf = trade_fair(ctx, v.race_id, fv)
        if tf is None:
            continue
        fair, sd = tf
        mid = float(ov.mid)
        edge = fair - mid
        # value
        if lo < fair < hi and abs(edge) > max(cfg.value.min_edge, VALUE_SD_Z * sd) + _EPS:
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
                        reason=f"value fair={fair:.3f} mid={mid:.3f} sd={sd:.3f}",
                    )
                )
        # convex sleeve
        tossup = TOSSUP_BAND[0] <= fair <= TOSSUP_BAND[1]
        if (
            ctx.sleeve_frac > 0
            and (tossup or v.chamber == "control")
            and abs(edge) > max(cfg.convex_sleeve.min_edge, VALUE_SD_Z * sd) + _EPS
        ):
            q = _quote_toward(v, fair, mid, cfg.value.quote_offset)
            if q is not None:
                action, px = q
                convex.append(
                    DesiredOrder(
                        exchange_id=eid,
                        side="yes",
                        action=action,  # type: ignore[arg-type]
                        price=px,
                        quantity=0,  # sized from the sleeve below
                        strategy=Strategy.CONVEX,
                        priority=PRIORITY_VALUE,
                        fair=fair,
                        edge=edge,
                        reason=f"convex fair={fair:.3f} mid={mid:.3f} sleeve={ctx.sleeve_frac:.2f}",
                    )
                )
    out.extend(_size_convex(ctx, convex))
    return out


# -- convex sleeve sizing -------------------------------------------------------------------------
def _kelly(fair: float, price: float, long: bool) -> float:
    """Kelly bankroll fraction for a binary bought (``long``) or sold at YES price ``price``."""
    q = min(1.0, max(0.0, float(fair)))
    p = min(1.0, max(0.0, float(price)))
    if long:
        return max(0.0, (q - p) / (1.0 - p)) if p < 1.0 else 0.0
    return max(0.0, (p - q) / p) if p > 0.0 else 0.0


def _strategy_of(intent: object) -> str:
    s = getattr(intent, "strategy", "")
    return str(getattr(s, "value", s)).lower()


def _value_prob(ctx: TraderContext, v: MarketView) -> float:
    fv = ctx.fair.get(v.exchange_id)
    if fv is not None:
        return min(1.0, max(0.0, float(fv.fair)))
    if v.mid is not None:
        return float(v.mid)
    return float(v.avg_price) if v.avg_price is not None else 0.5


def _convex_fill_qty(exchange_ids: list[str]) -> dict[str, float] | None:
    """Net signed YES quantity of our convex fills per market (None when the ledger cannot be read)."""
    try:
        with dbengine.session() as s:
            rows = s.execute(
                select(Fill.exchange_id, Fill.side, Fill.action, Fill.quantity).where(
                    Fill.strategy == Strategy.CONVEX.value, Fill.exchange_id.in_(exchange_ids)
                )
            ).all()
    except Exception as e:  # noqa: BLE001 - attribution is best effort; the caller falls back conservatively
        log.warning("convex sleeve: fill ledger unavailable (%s); counting whole positions", e)
        return None
    out: dict[str, float] = {}
    for eid, side, action, qty in rows:
        long_yes = (str(side).lower() == "yes") == (str(action).lower() == "buy")
        out[eid] = out.get(eid, 0.0) + (float(qty or 0) if long_yes else -float(qty or 0))
    return out


def _convex_deployed(ctx: TraderContext, replaced: set[tuple[str, str, str]]) -> float:
    """SUSQies the convex sleeve already uses: resting convex orders (except those this cycle re-desires,
    keyed ``(exchange_id, side, action)``) plus convex positions valued at the fair value."""
    used = 0.0
    for eid, v in ctx.markets.items():
        for i in v.open_orders or []:
            if _strategy_of(i) != Strategy.CONVEX.value:
                continue
            side = str(getattr(i, "side", "yes")).lower()
            action = str(getattr(i, "action", "buy")).lower()
            if (eid, side, action) in replaced:
                continue
            try:
                remaining = int(getattr(i, "quantity", 0) or 0) - int(getattr(i, "filled_qty", 0) or 0)
                price = float(i.price)  # type: ignore[attr-defined]
            except (AttributeError, TypeError, ValueError):
                continue
            if remaining > 0:
                used += remaining * (price if action == "buy" else 1.0 - price)
    held = {eid: v for eid, v in ctx.markets.items() if v.position}
    if held:
        net = _convex_fill_qty(sorted(held))
        for eid, v in held.items():
            pos = float(v.position)
            if net is None:
                q = pos  # attribution unknown: count the whole position against the sleeve
            else:
                c = net.get(eid, 0.0)
                q = (min(abs(c), abs(pos)) if pos > 0 else -min(abs(c), abs(pos))) if c * pos > 0 else 0.0
            if q:
                p = _value_prob(ctx, v)
                used += q * p if q > 0 else -q * (1.0 - p)
    return used


def _size_convex(ctx: TraderContext, cands: list[DesiredOrder]) -> list[DesiredOrder]:
    """Fractional-Kelly quantities from the sleeve, scaled so the sleeve budget holds."""
    if not cands:
        return []
    bankroll = ctx.bankroll_start if ctx.bankroll_start > 0 else max(0.0, ctx.equity)
    sleeve = max(0.0, float(ctx.sleeve_frac)) * bankroll
    if sleeve <= 0:
        return []
    room = max(0.0, sleeve - _convex_deployed(ctx, {(d.exchange_id, d.side, d.action) for d in cands}))
    kf = float(ctx.cfg.convex_sleeve.kelly_fraction)
    wants: list[tuple[DesiredOrder, float, float]] = []
    for d in cands:
        cost = d.price if d.action == "buy" else 1.0 - d.price
        cash = _kelly(d.fair if d.fair is not None else d.price, d.price, d.action == "buy") * kf * sleeve
        if cash > 0 and cost > 0:
            wants.append((d, cash, cost))
    total = sum(cash for _, cash, _ in wants)
    if total <= 0 or room <= 0:
        return []
    scale = min(1.0, room / total)
    out: list[DesiredOrder] = []
    for d, cash, cost in wants:
        qty = int(cash * scale / cost + _EPS)
        if qty > 0:
            d.quantity = qty
            out.append(d)
    return out


def _mm(ctx: TraderContext, outside: dict[str, MarketView]) -> list[DesiredOrder]:
    cfg = ctx.cfg
    if not cfg.mm.enabled:
        return []
    out: list[DesiredOrder] = []
    for eid in sorted(ctx.markets):
        v = ctx.markets[eid]
        ov = outside.get(eid, v)
        if not _is_open(v) or ov.spread is None or ov.spread < cfg.mm.min_spread - _EPS or ov.mid is None:
            continue
        mid = float(ov.mid)
        fv = ctx.fair.get(eid)
        fair: float | None = None
        if fv is not None:
            tf = trade_fair(ctx, v.race_id, fv)
            if tf is None:
                continue  # a hard event the fair value has not caught up with: do not quote into it
            fair = tf[0]
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
                    reason=f"mm bid spread={ov.spread:.3f}",
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
                    reason=f"mm ask spread={ov.spread:.3f}",
                )
            )
    return out


def _event_orders(ctx: TraderContext, by_race: dict[str, list[MarketView]], outside: dict[str, MarketView]) -> list[DesiredOrder]:
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
            if not _is_open(v) or outside.get(v.exchange_id, v).mid is None:
                continue
            pre = _pre_event_price(v.exchange_id, ets)
            if pre is None:
                fv = ctx.fair.get(v.exchange_id)
                if fv is not None and fv.ts is not None and ensure_utc(fv.ts) < ets:
                    pre = float(fv.fair)
            if pre is None:
                continue
            move = float(outside.get(v.exchange_id, v).mid) - pre  # type: ignore[arg-type]
            if best is None or abs(move) > abs(best[0]):
                best = (move, v, pre)
        if best is None or abs(best[0]) < FADE_MIN_MOVE:
            continue
        move, v, pre = best
        mid = float(outside.get(v.exchange_id, v).mid)  # type: ignore[arg-type]
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
    if ctx.hours_to_deadline <= 0:
        return []
    out: list[DesiredOrder] = []
    out.extend(_arb_orders(arbs))
    if not ctx.election_night:
        by_race = _by_race(ctx)
        outside = {eid: outside_view(v) for eid, v in ctx.markets.items()}
        out.extend(_theta(ctx, by_race))
        out.extend(_value_and_convex(ctx, outside))
        out.extend(_mm(ctx, outside))
        out.extend(_event_orders(ctx, by_race, outside))
    for d in out:
        d.price = round_tick(d.price)
    out.sort(key=lambda d: (d.priority, 0 if d.take else 1))
    return out
