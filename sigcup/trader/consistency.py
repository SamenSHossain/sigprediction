"""Dutch-book scanner: YES+NO on one market, YES-sum / NO-sum across the legs of a race, and control vs
seat implied bounds. Only races whose exclusivity+exhaustiveness is confirmed by the relationships API
(Race.exclusive and Race.exhaustive True) are eligible when arb.require_api_exclusivity is set. Returns
ArbOpportunity(legs=[DesiredOrder...], profit_per_set, max_sets_by_depth).

Notes
-----
* On this exchange YES and NO of one market are the *same* book (buy NO @p == sell YES @1-p), so a
  "YES+NO" check on a single market can never find anything the best bid/ask does not already show;
  the ``yes_no`` kind is kept in the type for completeness but never emitted.
* ``race_no_sum``: the best bids over a race's party legs sum to more than ``1 + arb.min_profit`` ->
  sell YES (== buy NO) on every leg at its bid. The set pays exactly one leg's worth at settlement only
  if at most one leg resolves YES (exclusive); if none does it pays more. ``race_yes_sum``: the best asks
  sum to less than ``1 - arb.min_profit`` -> buy YES on every leg at its ask; pays at least 1 only if at
  least one leg resolves YES (exhaustive). Both are gated on ``exclusive and exhaustive`` from the
  relationships API (never from titles): fusion tickets (RI), independents (NE/MT Senate, RI Gov) and
  December runoffs (GA) all break one of the two.
* ``max_sets`` is bounded by the displayed size at the best level of every leg (top-of-book size, else
  the cached depth). A leg with unknown size bounds the set at zero (no opportunity) because an
  unfilled leg is a directional position, not an arbitrage.
* Legs share a ``group_id`` (placed atomically by the OMS), carry ``take=True`` and priority 0.
* ``control``: the control market's bid/ask is compared with the Frechet bounds implied by the seat
  markets' fair values (``simulate.frechet_bounds``), only when every seat race of the chamber is
  present (``simulate.EXPECTED_RACES``) and has a fair value or a mid. Violations are reported as
  *informational* opportunities (``take=False``, zero quantity) since the bounds are wide and the true
  value depends on the correlation structure, not on the marginals alone.
"""

from __future__ import annotations

from sigcup.common.ids import new_idempotency_key
from sigcup.common.log import get_logger
from sigcup.db.models import Strategy
from sigcup.settings import StrategyConfig
from sigcup.sig.schemas import round_tick
from sigcup.trader import simulate as _sim
from sigcup.trader.fairvalue import control_chamber
from sigcup.trader.types import (
    PRIORITY_ARB,
    ArbOpportunity,
    DesiredOrder,
    MarketView,
    RaceInfo,
    TraderContext,
)

__all__ = ["control_checks", "displayed_size", "exclusivity_ok", "race_legs", "scan"]

log = get_logger(__name__)

_EPS = 1e-9
_OPEN_STATUSES = {"open", "active", "trading", ""}


# -- helpers --------------------------------------------------------------------------------------
def _is_open(v: MarketView) -> bool:
    return (v.status or "open").lower() in _OPEN_STATUSES


def exclusivity_ok(race: RaceInfo, cfg: StrategyConfig) -> bool:
    """True when the race may be treated as mutually exclusive AND exhaustive.

    With ``arb.require_api_exclusivity`` both flags must be *True* (None = unknown = rejected).
    """
    if cfg.arb.require_api_exclusivity:
        return race.exclusive is True and race.exhaustive is True
    return race.exclusive is not False and race.exhaustive is not False


def race_legs(ctx: TraderContext, race: RaceInfo) -> list[MarketView]:
    """Open market views of a race's party legs (one per party), ordered by party label."""
    views: dict[str, MarketView] = {}
    for party, eid in race.markets.items():
        v = ctx.markets.get(eid)
        if v is not None:
            views[party] = v
    if not views:
        for v in ctx.markets.values():
            if v.race_id == race.race_id:
                views.setdefault(v.party, v)
    return [views[p] for p in sorted(views) if _is_open(views[p])]


def displayed_size(v: MarketView, side: str) -> int:
    """Shares displayed at the best level on ``side`` ('bid' | 'ask'); 0 when unknown."""
    size = v.bid_size if side == "bid" else v.ask_size
    if size is not None:
        return max(0, int(size))
    book = v.book
    if book is not None:
        levels = book.bids if side == "bid" else book.asks
        price = v.bid if side == "bid" else v.ask
        if levels:
            best = max(levels, key=lambda lv: lv.price) if side == "bid" else min(levels, key=lambda lv: lv.price)
            if price is None or abs(best.price - price) < _EPS:
                return max(0, int(best.quantity))
    return 0


def _group_id(kind: str, race_id: str) -> str:
    return f"{kind}:{race_id}:{new_idempotency_key()[:8]}"[:32]


# -- race-level sums ------------------------------------------------------------------------------
def _race_no_sum(race: RaceInfo, legs: list[MarketView], cfg: StrategyConfig) -> ArbOpportunity | None:
    if any(v.bid is None for v in legs):
        return None
    total = sum(float(v.bid) for v in legs)  # type: ignore[arg-type]
    profit = total - 1.0
    if profit <= cfg.arb.min_profit + _EPS:
        return None
    max_sets = min(displayed_size(v, "bid") for v in legs)
    if max_sets <= 0:
        return None
    gid = _group_id("no_sum", race.race_id)
    orders = [
        DesiredOrder(
            exchange_id=v.exchange_id,
            side="yes",
            action="sell",
            price=round_tick(float(v.bid)),  # type: ignore[arg-type]
            quantity=max_sets,
            strategy=Strategy.ARB,
            priority=PRIORITY_ARB,
            group_id=gid,
            edge=profit,
            reason=f"race_no_sum {race.race_id}: bids sum {total:.3f}",
            take=True,
        )
        for v in legs
    ]
    return ArbOpportunity(kind="race_no_sum", legs=orders, profit_per_set=profit, max_sets=max_sets, race_id=race.race_id)


def _race_yes_sum(race: RaceInfo, legs: list[MarketView], cfg: StrategyConfig) -> ArbOpportunity | None:
    if any(v.ask is None for v in legs):
        return None
    total = sum(float(v.ask) for v in legs)  # type: ignore[arg-type]
    profit = 1.0 - total
    if profit <= cfg.arb.min_profit + _EPS:
        return None
    max_sets = min(displayed_size(v, "ask") for v in legs)
    if max_sets <= 0:
        return None
    gid = _group_id("yes_sum", race.race_id)
    orders = [
        DesiredOrder(
            exchange_id=v.exchange_id,
            side="yes",
            action="buy",
            price=round_tick(float(v.ask)),  # type: ignore[arg-type]
            quantity=max_sets,
            strategy=Strategy.ARB,
            priority=PRIORITY_ARB,
            group_id=gid,
            edge=profit,
            reason=f"race_yes_sum {race.race_id}: asks sum {total:.3f}",
            take=True,
        )
        for v in legs
    ]
    return ArbOpportunity(kind="race_yes_sum", legs=orders, profit_per_set=profit, max_sets=max_sets, race_id=race.race_id)


# -- control vs seats -----------------------------------------------------------------------------
def _seat_prob_dem(ctx: TraderContext, race: RaceInfo) -> float | None:
    """P(Democrat wins) for a seat from the fair values (D leg, else 1 - R leg), else the book mid."""
    d_eid = next((eid for p, eid in race.markets.items() if (p or "").upper().startswith("D")), None)
    r_eid = next((eid for p, eid in race.markets.items() if (p or "").upper().startswith("R")), None)
    if d_eid is not None:
        fv = ctx.fair.get(d_eid)
        if fv is not None:
            return fv.fair
    if r_eid is not None:
        fv = ctx.fair.get(r_eid)
        if fv is not None:
            return 1.0 - fv.fair
    if d_eid is not None:
        mv = ctx.markets.get(d_eid)
        if mv is not None and mv.mid is not None:
            return mv.mid
    if r_eid is not None:
        mv = ctx.markets.get(r_eid)
        if mv is not None and mv.mid is not None:
            return 1.0 - mv.mid
    return None


def control_checks(ctx: TraderContext) -> list[ArbOpportunity]:
    """Informational opportunities where a control market trades outside the seat-implied bounds."""
    out: list[ArbOpportunity] = []
    cfg = ctx.cfg
    reqs = _sim.control_requirements(ctx.races)
    for race_id in sorted(ctx.races):
        race = ctx.races[race_id]
        if race.chamber != "control":
            continue
        chamber = control_chamber(race)
        if chamber not in reqs:
            continue
        k, seat_ids = reqs[chamber]
        if _sim.coverage(chamber, len(seat_ids)) < 1.0 or k > len(seat_ids):
            log.debug("control check %s skipped: %d of %s seat races present", chamber, len(seat_ids), _sim.EXPECTED_RACES.get(chamber))
            continue
        probs: list[float] = []
        complete = True
        for rid in seat_ids:
            p = _seat_prob_dem(ctx, ctx.races[rid])
            if p is None:
                complete = False
                break
            probs.append(p)
        if not complete or not probs:
            continue
        lo, hi = _sim.frechet_bounds(probs, k)
        for party, eid in race.markets.items():
            v = ctx.markets.get(eid)
            if v is None or not _is_open(v):
                continue
            role = (party or "").upper()[:1]
            if role == "D":
                b_lo, b_hi = lo, hi
            elif role == "R":
                b_lo, b_hi = 1.0 - hi, 1.0 - lo
            else:
                continue
            leg: DesiredOrder | None = None
            if v.ask is not None and v.ask < b_lo - cfg.arb.min_profit:
                leg = DesiredOrder(
                    exchange_id=eid,
                    side="yes",
                    action="buy",
                    price=round_tick(v.ask),
                    quantity=0,
                    strategy=Strategy.ARB,
                    priority=PRIORITY_ARB,
                    fair=b_lo,
                    edge=b_lo - v.ask,
                    reason=f"control {chamber} {party}: ask {v.ask:.3f} below Frechet lower bound {b_lo:.3f}",
                    take=False,
                )
                profit = b_lo - v.ask
            elif v.bid is not None and v.bid > b_hi + cfg.arb.min_profit:
                leg = DesiredOrder(
                    exchange_id=eid,
                    side="yes",
                    action="sell",
                    price=round_tick(v.bid),
                    quantity=0,
                    strategy=Strategy.ARB,
                    priority=PRIORITY_ARB,
                    fair=b_hi,
                    edge=v.bid - b_hi,
                    reason=f"control {chamber} {party}: bid {v.bid:.3f} above Frechet upper bound {b_hi:.3f}",
                    take=False,
                )
                profit = v.bid - b_hi
            if leg is not None:
                log.info("consistency: %s", leg.reason)
                out.append(ArbOpportunity(kind="control", legs=[leg], profit_per_set=profit, max_sets=0, race_id=race_id))
    return out


# -- entry point ----------------------------------------------------------------------------------
def scan(ctx: TraderContext) -> list[ArbOpportunity]:
    """All current Dutch books (executable, ``take=True``) plus informational control checks."""
    cfg = ctx.cfg
    out: list[ArbOpportunity] = []
    for race_id in sorted(ctx.races):
        race = ctx.races[race_id]
        if race.chamber == "control":
            continue
        legs = race_legs(ctx, race)
        if len(legs) < 2 or len(legs) > cfg.arb.max_legs:
            continue
        if not exclusivity_ok(race, cfg):
            continue
        for opp in (_race_no_sum(race, legs, cfg), _race_yes_sum(race, legs, cfg)):
            if opp is not None:
                log.info("consistency: %s %s profit=%.3f sets=%d", opp.kind, race_id, opp.profit_per_set, opp.max_sets)
                out.append(opp)
    try:
        out.extend(control_checks(ctx))
    except Exception as e:  # noqa: BLE001 - informational only
        log.warning("control consistency check failed: %s", e)
    return out
