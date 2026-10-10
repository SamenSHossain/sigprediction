"""Sizing and limits.

- kelly_fraction(fair, price, long) for a binary contract; fractional multipliers per strategy.
- Caps: per race (netted across party legs: +YES on D == -YES on R economically), per state, reserve
  (liquid_reserve_frac, higher in the final 72 h), max position shares.
- Stress: uniform national swing of risk.national_swing_pts through the T-distribution fair-value map must
  cost <= max_swing_loss_frac of bankroll; a candidate order set is rejected or scaled down otherwise.
- Theta pricing: bid <= min(ask, min(fair, theta.fair_cap) / (1 + theta.min_yield)).
- Daily loss limit evaluation and the kill-switch predicate.

Exposure convention (no entry-price ambiguity): a leg's cash at risk is its *current* value,
``q * p`` for long YES and ``|q| * (1 - p)`` for NO, where ``p`` is the fair value when known, else the
book mid, else the order price. Per-race exposure is the worst-case loss over the race's outcomes
("party X wins" for every party leg, plus "none of the listed parties"), which makes buy-YES-on-D and
buy-NO-on-R add up (same bet) while buy-YES-on-D and buy-YES-on-R offset (a hedge).

``apply(ctx, desired)`` is the entry point used by the trader loop; everything else is a pure helper.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

from sigcup.common.log import get_logger
from sigcup.db.models import Strategy
from sigcup.settings import StrategyConfig
from sigcup.sig.schemas import PRICE_MIN, TICK, round_tick
from sigcup.trader.probmath import shift_probability
from sigcup.trader.types import DesiredOrder, TraderContext

log = get_logger(__name__)

_EPS = 1e-9
_OTHER = "__other__"  # outcome: none of the listed party legs wins (independent / N-A)
_OPEN_STATUSES = {"open", "active", "trading", ""}


# -- Kelly and pricing ------------------------------------------------------------------------
def kelly_fraction(fair: float, price: float, long: bool = True) -> float:
    """Optimal bankroll fraction for a binary bought at YES-price ``price`` with belief ``fair`` (P(YES)).

    long  (buy YES / sell NO): (q - p) / (1 - p)
    short (sell YES / buy NO): (p - q) / p
    Returns 0 when there is no edge or the inputs are degenerate.
    """
    q = min(1.0, max(0.0, fair))
    p = min(1.0, max(0.0, price))
    if long:
        if p >= 1.0:
            return 0.0
        f = (q - p) / (1.0 - p)
    else:
        if p <= 0.0:
            return 0.0
        f = (p - q) / p
    return max(0.0, f)


def strategy_multiplier(strategy: Strategy | str, cfg: StrategyConfig) -> float:
    """Fractional-Kelly multiplier for a strategy tag."""
    s = str(getattr(strategy, "value", strategy)).lower()
    if s in (Strategy.ARB.value, Strategy.ELECTION.value):
        return cfg.risk.kelly_fraction_arb
    if s == Strategy.THETA.value:
        return cfg.risk.kelly_fraction_theta
    if s == Strategy.CONVEX.value:
        return cfg.convex_sleeve.kelly_fraction
    return cfg.risk.kelly_fraction_default


def kelly_size(
    fair: float,
    price: float,
    long: bool,
    strategy: Strategy | str,
    bankroll: float,
    cfg: StrategyConfig,
) -> int:
    """Shares under fractional Kelly: stake = f * multiplier * bankroll; shares = stake / cost per share
    (``price`` for long YES, ``1 - price`` for short). ``price`` is the YES price."""
    f = kelly_fraction(fair, price, long) * strategy_multiplier(strategy, cfg)
    if f <= 0 or bankroll <= 0:
        return 0
    cost = price if long else 1.0 - price
    if cost <= 0:
        return 0
    return int(f * bankroll / cost + 1e-6)


def theta_bid(fair: float | None, ask: float | None, cfg: StrategyConfig) -> float | None:
    """Yield-priced favourite bid: ``round_tick(min(ask, min(fair, fair_cap) / (1 + min_yield)))``.

    Returns None when ``fair`` is below ``theta.min_fair`` or no bid on the tick grid still offers the
    minimum yield. Rounding never rounds *up* past the yield bound (it steps one tick down instead).
    """
    if fair is None or fair < cfg.theta.min_fair - _EPS:
        return None
    capped = min(fair, cfg.theta.fair_cap)
    bound = capped / (1.0 + cfg.theta.min_yield)
    target = bound if ask is None else min(ask, bound)
    bid = round_tick(target)
    if bid > target + _EPS:
        bid = round_tick(bid - TICK)
    if bid < PRICE_MIN - _EPS or bid > target + _EPS:
        return None
    return bid


# -- reserve / daily loss / kill switch --------------------------------------------------------
def reserve_fraction(cfg: StrategyConfig, hours_to_deadline: float) -> float:
    """Liquid reserve as a fraction of the starting bankroll (higher in the final ``endgame.final_hours``)."""
    if hours_to_deadline <= cfg.endgame.final_hours:
        return cfg.risk.liquid_reserve_frac_final72h
    return cfg.risk.liquid_reserve_frac


def reserve_amount(ctx: TraderContext) -> float:
    """Cash that must stay liquid right now (election night may deploy ``reserve_deploy_frac`` of it)."""
    amt = reserve_fraction(ctx.cfg, ctx.hours_to_deadline) * ctx.bankroll_start
    if ctx.election_night:
        amt *= 1.0 - ctx.cfg.election_night.reserve_deploy_frac
    return max(0.0, amt)


def daily_loss_breached(ctx: TraderContext) -> bool:
    """True when today's P&L is at or below -daily_loss_limit_frac * bankroll_start."""
    limit = ctx.cfg.risk.daily_loss_limit_frac * ctx.bankroll_start
    return limit > 0 and ctx.daily_pnl <= -limit


def kill_switch_triggered(ctx: TraderContext, kill_path: Path | str | None) -> str | None:
    """Reason the kill switch should fire (kill file present, daily loss limit) or None."""
    if kill_path is not None and Path(kill_path).exists():
        return f"kill file present: {kill_path}"
    if daily_loss_breached(ctx):
        return (
            f"daily loss limit: pnl={ctx.daily_pnl:.0f} <= "
            f"-{ctx.cfg.risk.daily_loss_limit_frac:.0%} of bankroll"
        )
    return None


# -- exposure model ---------------------------------------------------------------------------
@dataclass
class Leg:
    """One market's net economic position used by the exposure and stress calculations."""

    exchange_id: str
    race_id: str
    party: str
    state: str | None
    chamber: str
    yes_qty: float  # signed +YES / -NO shares
    p: float  # probability used to value the leg


def _market_prob(ctx: TraderContext, exchange_id: str, fallback: float | None = None) -> float:
    fv = ctx.fair.get(exchange_id)
    if fv is not None and fv.fair is not None:
        return min(1.0, max(0.0, fv.fair))
    mv = ctx.markets.get(exchange_id)
    if mv is not None and mv.mid is not None:
        return min(1.0, max(0.0, mv.mid))
    if fallback is not None:
        return min(1.0, max(0.0, fallback))
    return 0.5


def _order_yes_qty(d: DesiredOrder, qty: float | None = None) -> float:
    q = float(d.quantity if qty is None else qty)
    return q if d.is_long_yes else -q


def _yes_price(d: DesiredOrder) -> float:
    return d.price if d.side == "yes" else 1.0 - d.price


def position_legs(ctx: TraderContext) -> dict[str, Leg]:
    """Current positions as legs."""
    legs: dict[str, Leg] = {}
    for eid, mv in ctx.markets.items():
        if not mv.position:
            continue
        legs[eid] = Leg(
            exchange_id=eid,
            race_id=mv.race_id,
            party=mv.party,
            state=mv.state,
            chamber=mv.chamber,
            yes_qty=float(mv.position),
            p=_market_prob(ctx, eid, mv.avg_price),
        )
    return legs


def resting_yes_qty(ctx: TraderContext) -> dict[str, float]:
    """Signed YES quantity of resting (ACKED) intents per market, as if they filled."""
    out: dict[str, float] = {}
    for eid, mv in ctx.markets.items():
        for intent in mv.open_orders or []:
            remaining = max(0, int(getattr(intent, "quantity", 0)) - int(getattr(intent, "filled_qty", 0) or 0))
            if remaining <= 0:
                continue
            long_yes = (getattr(intent, "side", "yes") == "yes") == (getattr(intent, "action", "buy") == "buy")
            out[eid] = out.get(eid, 0.0) + (remaining if long_yes else -remaining)
    return out


def _add_leg(ctx: TraderContext, legs: dict[str, Leg], eid: str, yes_qty: float, price: float | None) -> None:
    if eid in legs:
        legs[eid].yes_qty += yes_qty
        return
    mv = ctx.markets.get(eid)
    legs[eid] = Leg(
        exchange_id=eid,
        race_id=mv.race_id if mv else eid,
        party=mv.party if mv else "?",
        state=mv.state if mv else None,
        chamber=mv.chamber if mv else "unknown",
        yes_qty=yes_qty,
        p=_market_prob(ctx, eid, price),
    )


def combined_legs(ctx: TraderContext, candidates: Iterable[tuple[DesiredOrder, float]] = ()) -> dict[str, Leg]:
    """Positions + resting orders + candidate orders (each scaled by its factor), netted per market."""
    legs = {eid: replace(leg) for eid, leg in position_legs(ctx).items()}
    for eid, q in resting_yes_qty(ctx).items():
        _add_leg(ctx, legs, eid, q, None)
    for d, f in candidates:
        if f <= 0:
            continue
        _add_leg(ctx, legs, d.exchange_id, _order_yes_qty(d) * f, _yes_price(d))
    return legs


def race_exposure(legs: Iterable[Leg], exhaustive: bool = False, parties: Iterable[str] = ()) -> float:
    """Worst-case loss across the race's outcomes (current-value convention, gains ignored).

    Outcomes are "party X wins" for every party in ``parties`` (the race's known markets) plus the
    legs' own parties, and "none of them wins" unless the relationships API verified the race as
    exhaustive (``RaceInfo.exhaustive``) and at least two parties are known; it is never inferred."""
    legs = list(legs)
    if not legs:
        return 0.0
    outcome_set = {leg.party for leg in legs} | set(parties)
    outcomes = [*outcome_set]
    if not exhaustive or len(outcome_set) < 2:
        outcomes.append(_OTHER)
    worst = 0.0
    for outcome in outcomes:
        loss = 0.0
        for leg in legs:
            if leg.yes_qty == 0:
                continue
            wins_yes = leg.party == outcome
            if leg.yes_qty > 0 and not wins_yes:
                loss += leg.yes_qty * leg.p
            elif leg.yes_qty < 0 and wins_yes:
                loss += -leg.yes_qty * (1.0 - leg.p)
        worst = max(worst, loss)
    return worst


def _exhaustive(ctx: TraderContext, race_id: str) -> bool:
    race = ctx.races.get(race_id)
    return bool(race is not None and race.exhaustive)


def _race_parties(ctx: TraderContext, race_id: str) -> set[str]:
    race = ctx.races.get(race_id)
    parties = set(race.markets) if race is not None else set()
    parties |= {mv.party for mv in ctx.markets.values() if mv.race_id == race_id}
    return parties


def _race_exposure_in(ctx: TraderContext, race_id: str, legs: Iterable[Leg]) -> float:
    return race_exposure(legs, _exhaustive(ctx, race_id), _race_parties(ctx, race_id))


def exposures_by_race(legs: Iterable[Leg], ctx: TraderContext | None = None) -> dict[str, float]:
    """Per-race netted exposure (``ctx`` supplies the verified-exhaustive flags and party lists)."""
    by_race: dict[str, list[Leg]] = {}
    for leg in legs:
        by_race.setdefault(leg.race_id, []).append(leg)
    if ctx is None:
        return {rid: race_exposure(ls) for rid, ls in by_race.items()}
    return {rid: _race_exposure_in(ctx, rid, ls) for rid, ls in by_race.items()}


def _party_sign(party: str) -> float:
    p = (party or "").upper()
    if p.startswith("D"):
        return 1.0
    if p.startswith("R"):
        return -1.0
    return 0.0


def swing_loss(legs: Iterable[Leg], cfg: StrategyConfig, swing_pts: float | None = None) -> float:
    """Loss (positive = money lost) under a uniform national swing of ``swing_pts`` in the direction that
    hurts most (toward D or toward R), mapped through probmath.shift_probability per chamber."""
    pts = cfg.risk.national_swing_pts if swing_pts is None else swing_pts
    legs = list(legs)
    worst = 0.0
    for direction in (1.0, -1.0):
        pnl = 0.0
        for leg in legs:
            sign = _party_sign(leg.party)
            if leg.yes_qty == 0 or sign == 0.0:
                continue
            p_after = shift_probability(leg.p, direction * sign * pts, leg.chamber, cfg)
            pnl += leg.yes_qty * (p_after - leg.p)
        worst = max(worst, -pnl)
    return worst


def portfolio_swing_loss(ctx: TraderContext, candidates: Iterable[DesiredOrder] = ()) -> float:
    """Swing-stress loss for positions + resting orders + ``candidates`` as if filled."""
    return swing_loss(combined_legs(ctx, [(d, 1.0) for d in candidates]).values(), ctx.cfg)


def cash_required(d: DesiredOrder, position: int = 0, quantity: int | None = None) -> float:
    """Cash a fill would consume. Sells of owned shares need none; unowned sells fill as the opposite
    buy at ``1 - price`` (the exchange's rule)."""
    q = d.quantity if quantity is None else quantity
    if q <= 0:
        return 0.0
    if d.action == "buy":
        return d.price * q
    owned = max(0, position) if d.side == "yes" else max(0, -position)
    uncovered = max(0, q - owned)
    return (1.0 - d.price) * uncovered


# -- apply ------------------------------------------------------------------------------------
@dataclass
class _Cand:
    order: DesiredOrder
    seq: int
    factor: float = 1.0
    reducer: bool = False  # closes or shrinks an existing position: exempt from scaling
    dropped: str | None = None

    @property
    def quantity(self) -> int:
        return int(self.order.quantity * self.factor + 1e-9)

    @property
    def alive(self) -> bool:
        return self.dropped is None and self.quantity > 0


def _settle_groups(cands: list[_Cand], reason: str) -> None:
    """Groups are kept or dropped whole and scaled by their smallest leg factor."""
    for c in cands:
        if c.dropped is None and c.quantity <= 0:
            c.dropped = reason
    groups: dict[str, list[_Cand]] = {}
    for c in cands:
        if c.order.group_id:
            groups.setdefault(c.order.group_id, []).append(c)
    for legs in groups.values():
        if all(leg.dropped is None for leg in legs):
            fmin = min(leg.factor for leg in legs)
            for leg in legs:
                leg.factor = fmin
        if any(leg.dropped is not None or leg.quantity <= 0 for leg in legs):
            for leg in legs:
                leg.dropped = leg.dropped or reason


def _scale(cands: list[_Cand], factor: float, reason: str) -> None:
    """Scale every live non-reducer candidate by ``factor``."""
    if factor < 1.0 - _EPS:
        for c in cands:
            if c.dropped is None and not c.reducer:
                c.factor *= factor
    _settle_groups(cands, reason)


def _bisect_factor(
    cands: list[_Cand],
    measure: Callable[[list[tuple[DesiredOrder, float]]], float],
    limit: float,
    iters: int = 14,
) -> float:
    """Largest factor in [0, 1] for the scalable candidates such that ``measure`` <= ``limit``."""

    def at(f: float) -> float:
        pairs = [(c.order, c.factor * (1.0 if c.reducer else f)) for c in cands if c.alive]
        return measure(pairs)

    if at(1.0) <= limit + _EPS:
        return 1.0
    if at(0.0) > limit + _EPS:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if at(mid) <= limit + _EPS:
            lo = mid
        else:
            hi = mid
    return lo


def _position(ctx: TraderContext, exchange_id: str) -> int:
    mv = ctx.markets.get(exchange_id)
    return mv.position if mv else 0


def _race_of(ctx: TraderContext, exchange_id: str) -> str:
    mv = ctx.markets.get(exchange_id)
    return mv.race_id if mv else exchange_id


def _state_of(ctx: TraderContext, exchange_id: str) -> str | None:
    mv = ctx.markets.get(exchange_id)
    if mv is None:
        return None
    if mv.state:
        return mv.state
    race = ctx.races.get(mv.race_id)
    return race.state if race else None


def apply(ctx: TraderContext, desired: list[DesiredOrder]) -> list[DesiredOrder]:
    """Scale / filter the desired order set against every limit. Honours ``priority`` (lower first) and
    ``take`` (first within a priority), keeps or drops multi-leg ``group_id`` sets whole, never touches
    existing positions, and exempts position-reducing orders from scaling. Returns new DesiredOrder
    objects with tick-rounded prices, sorted by priority."""
    cfg = ctx.cfg
    bankroll = ctx.bankroll_start if ctx.bankroll_start > 0 else max(ctx.equity, 1.0)
    resting = resting_yes_qty(ctx)
    cands: list[_Cand] = []
    for seq, d in enumerate(desired):
        d2 = replace(d, price=round_tick(d.price), quantity=int(d.quantity))
        c = _Cand(order=d2, seq=seq)
        mv = ctx.markets.get(d2.exchange_id)
        pos = mv.position if mv else 0
        if d2.quantity <= 0:
            c.dropped = "zero quantity"
        elif mv is not None and (mv.status or "open").lower() not in _OPEN_STATUSES:
            c.dropped = f"market status {mv.status}"
        elif ctx.hours_to_deadline <= 0:
            c.dropped = "past deadline"
        c.reducer = (pos > 0 and not d2.is_long_yes and d2.quantity <= pos) or (
            pos < 0 and d2.is_long_yes and d2.quantity <= -pos
        )
        cands.append(c)
    if daily_loss_breached(ctx):
        for c in cands:
            if c.dropped is None and not c.reducer:
                c.dropped = "daily loss limit"

    # 1. per-market position cap
    for c in cands:
        if c.dropped is not None or c.reducer:
            continue
        held = abs(_position(ctx, c.order.exchange_id)) + abs(resting.get(c.order.exchange_id, 0.0))
        room = cfg.risk.max_position_shares - held
        if room <= 0:
            c.dropped = "max_position_shares"
        elif c.quantity > room:
            c.factor *= room / c.quantity
    _settle_groups(cands, "max_position_shares")

    # 2. per-race cap (netted across legs)
    race_cap = cfg.risk.per_race_cap_frac * bankroll
    for race_id in sorted({_race_of(ctx, c.order.exchange_id) for c in cands if c.alive}):
        members = [c for c in cands if c.alive and _race_of(ctx, c.order.exchange_id) == race_id]

        def measure_race(pairs: list[tuple[DesiredOrder, float]], rid: str = race_id) -> float:
            legs = combined_legs(ctx, pairs)
            return _race_exposure_in(ctx, rid, (leg for leg in legs.values() if leg.race_id == rid))

        f = _bisect_factor(members, measure_race, race_cap)
        if f < 1.0:
            log.info("risk: race %s over per-race cap -> scale %.3f", race_id, f)
            _scale(members, f, "per_race_cap")
            _settle_groups(cands, "per_race_cap")

    # 3. per-state cap
    state_cap = cfg.risk.per_state_cap_frac * bankroll
    for state in sorted({s for s in (_state_of(ctx, c.order.exchange_id) for c in cands if c.alive) if s}):
        members = [c for c in cands if c.alive and _state_of(ctx, c.order.exchange_id) == state]

        def measure_state(pairs: list[tuple[DesiredOrder, float]], st: str = state) -> float:
            legs = combined_legs(ctx, pairs)
            return sum(exposures_by_race((lg for lg in legs.values() if lg.state == st), ctx).values())

        f = _bisect_factor(members, measure_state, state_cap)
        if f < 1.0:
            log.info("risk: state %s over per-state cap -> scale %.3f", state, f)
            _scale(members, f, "per_state_cap")
            _settle_groups(cands, "per_state_cap")

    # 4. swing stress on the whole portfolio + candidates
    live = [c for c in cands if c.alive]
    if live:
        limit = cfg.risk.max_swing_loss_frac * bankroll

        def measure_swing(pairs: list[tuple[DesiredOrder, float]]) -> float:
            return swing_loss(combined_legs(ctx, pairs).values(), cfg)

        f = _bisect_factor(live, measure_swing, limit)
        if f < 1.0:
            log.info("risk: swing stress over limit -> scale %.3f", f)
            _scale(live, f, "swing_stress")
            _settle_groups(cands, "swing_stress")

    # 5. reserve / free cash, walking in priority order (take first within a priority)
    ranked = sorted(
        (c for c in cands if c.alive),
        key=lambda c: (c.order.priority, 0 if c.order.take else 1, c.seq),
    )
    free = ctx.cash - ctx.reserved - reserve_amount(ctx)
    seen_groups: set[str] = set()
    for c in ranked:
        if c.dropped is not None:
            continue
        gid = c.order.group_id
        if gid:
            if gid in seen_groups:
                continue
            seen_groups.add(gid)
            legs = [x for x in ranked if x.order.group_id == gid and x.alive]
        else:
            legs = [c]
        need = sum(cash_required(x.order, _position(ctx, x.order.exchange_id), x.quantity) for x in legs)
        if need <= free + _EPS:
            free -= need
            continue
        if gid:
            for x in legs:
                x.dropped = "reserve"
            continue
        per_share = cash_required(c.order, _position(ctx, c.order.exchange_id), 1)
        fit = int(max(0.0, free) / per_share + _EPS) if per_share > 0 else c.quantity
        if fit <= 0:
            c.dropped = "reserve"
            continue
        c.factor *= fit / c.quantity
        free -= cash_required(c.order, _position(ctx, c.order.exchange_id), c.quantity)

    out: list[DesiredOrder] = []
    for c in cands:
        if c.dropped is not None:
            log.debug(
                "risk: drop %s %s %s %s @%.3f x%d (%s)",
                c.order.strategy, c.order.exchange_id, c.order.action, c.order.side,
                c.order.price, c.order.quantity, c.dropped,
            )
            continue
        if c.quantity <= 0:
            continue
        out.append(replace(c.order, quantity=c.quantity))
    out.sort(key=lambda d: (d.priority, 0 if d.take else 1))
    return out
