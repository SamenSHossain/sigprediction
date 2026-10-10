"""Election-night mode (from election_night.start_utc until the deadline): the taker rule
(real-money reference >= take_when_ref_ge and SIG lags by >= min_lag -> take SIG liquidity, priority 0),
reserve deployment policy, and settlement edge cases (runoffs, fusion, independents, N/A refunds).

Taker rule (``taker_orders``), evaluated only against the *reference* prices passed in (never a SIG
print)::

    ref >= take_when_ref_ge  and  SIG ask <= ref - min_lag        -> buy YES at the ask
    ref <= 1 - take_when_ref_ge  and  SIG bid >= ref + min_lag    -> buy NO at 1 - bid (== sell YES at bid)

(the NO side is the mirror: the NO reference is ``1 - ref`` and the NO ask is ``1 - YES bid``, so
``NO ask <= NO ref - min_lag`` is ``bid >= ref + min_lag``). Sizes are bounded by the displayed size at
the touch (unknown size -> no order) and by the deployable reserve, ``reserve_deploy_frac`` of the
liquid reserve (``risk.liquid_reserve_frac_final72h`` of the bankroll in the final window), never more
than the free cash. Takes are allocated best edge first.

Freshness: ``ref_prices`` carries no timestamps (and the loop reads the newest stored row at any age),
so every take must be *confirmed* by the fair value: ``ctx.fair[eid]`` has a ``ref`` component, its
input time is within ``fairvalue.ref_max_age`` of now (a few collector cadences on election night) and
that fresh reference is within ``REF_CONFIRM_TOL_LAGS * min_lag`` of the price passed in. A stalled
collector, or a source contradicted by fresher data, never triggers a take.

The touch is the *outside* book (``consistency.outside_view``: our own resting orders removed), and a
take whose side of the book holds one of our own resting orders at that price or better is skipped (it
would match our own order first); the OMS cancels such stale orders in the same cycle.

Settlement edge cases: a race whose ``settlement_note`` mentions a runoff, fusion ticket, independent
candidate or N/A refund is sized at ``CAVEAT_SIZE_MULTIPLIER`` because the real-money contract may
resolve on different terms (e.g. Kalshi's December runoff contract vs a party-winner market that waits
for the official result).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from sigcup.common.log import get_logger
from sigcup.common.time import ensure_utc
from sigcup.db.models import Strategy
from sigcup.settings import StrategyConfig
from sigcup.sig.schemas import round_tick
from sigcup.trader.consistency import displayed_size, outside_view, self_trade_risk
from sigcup.trader.fairvalue import ref_max_age
from sigcup.trader.types import PRIORITY_ARB, DesiredOrder, RaceInfo, TraderContext

__all__ = [
    "CAVEAT_SIZE_MULTIPLIER",
    "REF_CONFIRM_TOL_LAGS",
    "confirmed_ref",
    "deployable_cash",
    "is_active",
    "settlement_caveats",
    "taker_orders",
]

log = get_logger(__name__)

CAVEAT_SIZE_MULTIPLIER = 0.5
REF_CONFIRM_TOL_LAGS = 0.5  # the fresh reference may differ from the passed one by half the min_lag
_CAVEAT_WORDS = ("runoff", "run-off", "fusion", "independent", "n/a", "refund", "recount")
_EPS = 1e-9
_OPEN_STATUSES = {"open", "active", "trading", ""}


def is_active(cfg: StrategyConfig, now: datetime) -> bool:
    """True from ``election_night.start_utc`` (inclusive) until the deadline (exclusive)."""
    now = ensure_utc(now)
    return ensure_utc(cfg.election_night.start_utc) <= now < ensure_utc(cfg.deadline_utc)


def deployable_cash(ctx: TraderContext) -> float:
    """Cash the taker rule may spend this cycle: ``reserve_deploy_frac`` of the liquid reserve, capped
    by the free cash (cash minus the internal reservation for resting buys)."""
    cfg = ctx.cfg
    frac = cfg.risk.liquid_reserve_frac_final72h if ctx.hours_to_deadline <= cfg.endgame.final_hours else cfg.risk.liquid_reserve_frac
    reserve = max(0.0, frac * float(ctx.bankroll_start))
    budget = cfg.election_night.reserve_deploy_frac * reserve
    free = max(0.0, float(ctx.cash) - float(ctx.reserved))
    return max(0.0, min(budget, free))


def settlement_caveats(race: RaceInfo | None) -> list[str]:
    """Words in the race's settlement note that flag a resolution edge case."""
    if race is None:
        return []
    note = (race.settlement_note or "").lower()
    found = [w for w in _CAVEAT_WORDS if w in note]
    if race.exclusive is False or race.exhaustive is False:
        found.append("not exclusive/exhaustive")
    return found


def confirmed_ref(ctx: TraderContext, exchange_id: str, ref: float) -> float | None:
    """The fresh reference price in ``ctx.fair`` that confirms ``ref``, else None (see the module docstring)."""
    fv = ctx.fair.get(exchange_id)
    if fv is None or fv.ts is None:
        return None
    fresh = fv.components.get("ref")
    if fresh is None:
        return None
    if ensure_utc(ctx.now) - ensure_utc(fv.ts) > ref_max_age(ctx.cfg, ctx.now):
        return None
    if abs(float(fresh) - float(ref)) > REF_CONFIRM_TOL_LAGS * float(ctx.cfg.election_night.min_lag) + _EPS:
        return None
    return float(fresh)


def taker_orders(ctx: TraderContext, ref_prices: Mapping[str, float]) -> list[DesiredOrder]:
    """Election-night takes (priority 0, ``take=True``) from the reference prices only."""
    cfg = ctx.cfg
    if not is_active(cfg, ctx.now) or ctx.hours_to_deadline <= 0:
        return []
    thr = float(cfg.election_night.take_when_ref_ge)
    lag = float(cfg.election_night.min_lag)
    cands: list[tuple[float, DesiredOrder, float]] = []  # (edge, order, caveat multiplier)
    for eid in sorted(ref_prices):
        v = ctx.markets.get(eid)
        if v is None or (v.status or "open").lower() not in _OPEN_STATUSES:
            continue
        try:
            ref = float(ref_prices[eid])
        except (TypeError, ValueError):
            continue
        if not 0.0 <= ref <= 1.0:
            continue
        if confirmed_ref(ctx, eid, ref) is None:
            log.debug("election: %s reference %.3f not confirmed by a fresh fair value; no take", eid, ref)
            continue
        race = ctx.races.get(v.race_id)
        mult = CAVEAT_SIZE_MULTIPLIER if settlement_caveats(race) else 1.0
        ov = outside_view(v)
        if ref >= thr and ov.ask is not None and float(ov.ask) <= ref - lag + _EPS:
            size = displayed_size(ov, "ask")
            if size <= 0 or self_trade_risk(v, "ask", ov.ask):
                continue
            px = round_tick(float(ov.ask))
            cands.append(
                (
                    ref - px,
                    DesiredOrder(
                        exchange_id=eid,
                        side="yes",
                        action="buy",
                        price=px,
                        quantity=size,
                        strategy=Strategy.ELECTION,
                        priority=PRIORITY_ARB,
                        fair=ref,
                        edge=ref - px,
                        reason=f"election take yes: ref={ref:.3f} ask={px:.3f}",
                        take=True,
                    ),
                    mult,
                )
            )
        elif ref <= 1.0 - thr and ov.bid is not None and float(ov.bid) >= ref + lag - _EPS:
            size = displayed_size(ov, "bid")
            if size <= 0 or self_trade_risk(v, "bid", ov.bid):
                continue
            px_no = round_tick(1.0 - float(ov.bid))
            edge = (1.0 - ref) - px_no
            cands.append(
                (
                    edge,
                    DesiredOrder(
                        exchange_id=eid,
                        side="no",
                        action="buy",
                        price=px_no,
                        quantity=size,
                        strategy=Strategy.ELECTION,
                        priority=PRIORITY_ARB,
                        fair=ref,
                        edge=edge,
                        reason=f"election take no: ref={ref:.3f} bid={float(ov.bid):.3f}",
                        take=True,
                    ),
                    mult,
                )
            )
    cands.sort(key=lambda t: (-t[0], t[1].exchange_id))
    budget = deployable_cash(ctx)
    out: list[DesiredOrder] = []
    for edge, d, mult in cands:
        if edge <= 0:
            continue
        qty = int(d.quantity * mult)
        affordable = int(budget / d.price + _EPS) if d.price > 0 else 0
        qty = min(qty, affordable)
        if qty <= 0:
            continue
        d.quantity = qty
        budget -= qty * d.price
        log.info("election: %s x%d", d.reason, qty)
        out.append(d)
    return out
