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
from sigcup.trader.consistency import displayed_size
from sigcup.trader.types import PRIORITY_ARB, DesiredOrder, RaceInfo, TraderContext

__all__ = ["CAVEAT_SIZE_MULTIPLIER", "deployable_cash", "is_active", "settlement_caveats", "taker_orders"]

log = get_logger(__name__)

CAVEAT_SIZE_MULTIPLIER = 0.5
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
        race = ctx.races.get(v.race_id)
        mult = CAVEAT_SIZE_MULTIPLIER if settlement_caveats(race) else 1.0
        if ref >= thr and v.ask is not None and float(v.ask) <= ref - lag + _EPS:
            size = displayed_size(v, "ask")
            if size <= 0:
                continue
            px = round_tick(float(v.ask))
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
        elif ref <= 1.0 - thr and v.bid is not None and float(v.bid) >= ref + lag - _EPS:
            size = displayed_size(v, "bid")
            if size <= 0:
                continue
            px_no = round_tick(1.0 - float(v.bid))
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
                        reason=f"election take no: ref={ref:.3f} bid={float(v.bid):.3f}",
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
