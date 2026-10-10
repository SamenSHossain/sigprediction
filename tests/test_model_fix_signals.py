"""Regression tests for review findings on signals.py: F1 (stale blend after a decisive reference move),
F4 (theta leg flip-flop), F5 (our own resting order counted as the market) and F8 (convex sleeve size)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sigcup.db.models import IntentState, OrderIntent, Strategy
from sigcup.trader import signals as SG
from sigcup.trader.types import PRIORITY_ARB, ArbOpportunity, DesiredOrder, FairValue
from tests.test_model_support import NOW, book, ctx, event, fair, race, view

NIGHT = datetime(2026, 11, 4, 2, 0, tzinfo=UTC)


def intent(eid: str, side: str, action: str, price: float, qty: int, strategy: Strategy, filled: int = 0) -> OrderIntent:
    """An ACKED resting order as BookState puts it in ``MarketView.open_orders``."""
    return OrderIntent(
        id=f"{eid}-{side}-{action}-{strategy.value}"[:32],
        exchange_id=eid,
        side=side,
        action=action,
        price=price,
        quantity=qty,
        strategy=strategy.value,
        state=IntentState.ACKED.value,
        filled_qty=filled,
        created_at=NOW,
        updated_at=NOW,
        expires_at=NOW + timedelta(seconds=900),
    )


def _blended(eid: str, fair_: float, ref: float, slow: float, sd: float, ts: datetime) -> FairValue:
    """A blended fair value held near mid-band by stale polls / ratings while the reference has moved."""
    return FairValue(
        exchange_id=eid,
        fair=fair_,
        sd=sd,
        components={"ref": ref, "polls": slow, "ratings": slow},
        weights={"ref": 0.6, "polls": 0.3, "ratings": 0.1},
        ts=ts,
    )


def _decided_race(now: datetime, sd: float = 0.236):
    """F1 scenario: toss-up race decided for D on the real-money market; SIG lags at 0.92/0.94."""
    info = race("SEN-IA", "senate", "IA")
    views = [
        view(info, "D", bid=0.92, ask=0.94, bid_size=500, ask_size=500),
        view(info, "R", bid=0.05, ask=0.07, bid_size=500, ask_size=500),
    ]
    ts = now - timedelta(minutes=1)
    fairs = [_blended("SEN-IA-D", 0.788, 0.98, 0.5, sd, ts), _blended("SEN-IA-R", 0.212, 0.02, 0.5, sd, ts)]
    return info, views, fairs


def _bets_against_reference(orders) -> list[DesiredOrder]:
    """Orders that are short the D leg or long the R leg (the side the reference says has lost)."""
    bad = []
    for d in orders:
        if d.exchange_id == "SEN-IA-D" and not d.is_long_yes:
            bad.append(d)
        if d.exchange_id == "SEN-IA-R" and d.is_long_yes:
            bad.append(d)
    return bad


# -- F1 ---------------------------------------------------------------------------------------------
def test_f1_election_night_emits_only_arb_legs(cfg):
    info, views, fairs = _decided_race(NIGHT)
    legs = [
        DesiredOrder("SEN-IA-D", "yes", "sell", 0.92, 10, Strategy.ARB, PRIORITY_ARB, group_id="g", take=True),
        DesiredOrder("SEN-IA-R", "yes", "sell", 0.10, 10, Strategy.ARB, PRIORITY_ARB, group_id="g", take=True),
    ]
    arb = ArbOpportunity(kind="race_no_sum", legs=legs, profit_per_set=0.02, max_sets=10, race_id="SEN-IA")
    c = ctx(cfg, [info], views, fairs, now=NIGHT, election_night=True)
    out = SG.generate(c, [arb])
    assert out == legs  # value / convex / theta / mm are resting bets of the slow model: none on election night
    assert _bets_against_reference(SG.generate(c, [])) == []


def test_f1_hard_event_trades_on_the_reference_only(cfg):
    # small sd so the sd gate alone does not hide the problem
    info, views, fairs = _decided_race(NOW, sd=0.03)
    hard = event("SEN-IA", "hard", NOW - timedelta(minutes=30), "candidate withdrew")
    c = ctx(cfg, [info], views, fairs, events=[hard])
    out = SG.generate(c, [])
    assert _bets_against_reference(out) == []
    assert [d for d in out if d.strategy in (Strategy.VALUE, Strategy.CONVEX)] == []  # ref 0.98 is outside the band
    # a fair whose inputs all predate the event is still blocked outright
    stale = [_blended(f.exchange_id, f.fair, f.components["ref"], 0.5, 0.03, NOW - timedelta(hours=2)) for f in fairs]
    c = ctx(cfg, [info], views, stale, events=[hard])
    assert [d for d in SG.generate(c, []) if d.strategy in (Strategy.VALUE, Strategy.CONVEX, Strategy.THETA)] == []


def test_f1_value_edge_must_exceed_fair_value_sd(cfg):
    info, views, fairs = _decided_race(NOW, sd=0.236)
    c = ctx(cfg, [info], views, fairs)
    out = SG.generate(c, [])
    assert [d for d in out if d.strategy in (Strategy.VALUE, Strategy.CONVEX)] == []
    # the same edge with a confident fair value still trades
    info, views, fairs = _decided_race(NOW, sd=0.03)
    c = ctx(cfg, [info], views, fairs)
    assert [d for d in SG.generate(c, []) if d.strategy == Strategy.VALUE]


# -- F4 ---------------------------------------------------------------------------------------------
def _theta_choice(cfg, views, d_fair, r_fair):
    info = race("SEN-DE", "senate", "DE")
    c = ctx(cfg, [info], views, [fair("SEN-DE-D", d_fair), fair("SEN-DE-R", r_fair)])
    orders = [d for d in SG.generate(c, []) if d.strategy == Strategy.THETA]
    assert len(orders) == 1
    return orders[0]


def test_f4_theta_leg_choice_is_stable_under_fair_jitter(cfg):
    info = race("SEN-DE", "senate", "DE")
    # both candidates capped by the yield bound at the same tick (0.85): the D ask and the R NO ask
    # (1 - R bid) are above it
    views = [view(info, "D", bid=0.80, ask=0.90), view(info, "R", bid=0.05, ask=0.12)]
    jitter = [(0.9010, 0.1005), (0.8995, 0.1000), (0.9002, 0.0999), (0.8990, 0.0985)]
    chosen = []
    for d_fair, r_fair in jitter:
        d = _theta_choice(cfg, views, d_fair, r_fair)
        chosen.append((d.exchange_id, d.side))
        # the OMS acks the order: it is now a working theta order on that leg
        for v in views:
            v.open_orders = [intent(d.exchange_id, d.side, "buy", d.price, d.quantity, Strategy.THETA)] if v.exchange_id == d.exchange_id else []
    assert len(set(chosen)) == 1, chosen
    # without any working order the near-tie is broken deterministically toward the favourite's YES leg
    for v in views:
        v.open_orders = []
    d = _theta_choice(cfg, views, 0.8995, 0.1000)
    assert (d.exchange_id, d.side) == ("SEN-DE-D", "yes")
    # a clearly better alternative still wins over the working order
    views[0].open_orders = [intent("SEN-DE-D", "yes", "buy", 0.85, 500, Strategy.THETA)]
    views[1].bid = 0.10  # R NO ask 0.90 -> bound 0.95/1.055 = 0.9005 -> 0.90 ...
    d = _theta_choice(cfg, views, 0.90, 0.02)  # ... and NO fair 0.98 -> R NO yield >> D YES yield
    assert (d.exchange_id, d.side) == ("SEN-DE-R", "no")


# -- F5 ---------------------------------------------------------------------------------------------
def test_f5_value_order_is_not_cancelled_by_its_own_quote(cfg):
    info = race("SEN-TX", "senate", "TX")
    first = view(info, "D", bid=0.40, ask=0.62, bid_size=50, ask_size=50)
    c = ctx(cfg, [info], [first], [fair("SEN-TX-D", 0.60)], sleeve_frac=0.0)
    placed = [d for d in SG.generate(c, []) if d.strategy == Strategy.VALUE]
    assert [(d.action, d.price) for d in placed] == [("buy", 0.58)]
    # next cycle: our 0.58 bid is now the best bid of the exchange's book
    second = view(
        info,
        "D",
        bid=0.58,
        ask=0.62,
        bid_size=200,
        ask_size=50,
        book=book("SEN-TX-D", bids=[(0.58, 200), (0.40, 50)], asks=[(0.62, 50)]),
    )
    second.open_orders = [intent("SEN-TX-D", "yes", "buy", 0.58, 200, Strategy.VALUE)]
    c = ctx(cfg, [info], [second], [fair("SEN-TX-D", 0.60)], sleeve_frac=0.0)
    again = [d for d in SG.generate(c, []) if d.strategy == Strategy.VALUE]
    assert [(d.action, d.price) for d in again] == [("buy", 0.58)]  # same key and price: the OMS keeps it
    # also when the bulk feed carries no size: the cached depth shows the level is ours
    second.bid_size = None
    assert [(d.action, d.price) for d in SG.generate(c, []) if d.strategy == Strategy.VALUE] == [("buy", 0.58)]


# -- F8 ---------------------------------------------------------------------------------------------
def _tossups(n: int = 30):
    races, views, fairs = [], [], []
    for i in range(n):
        info = race(f"GOV-X{i:02d}", "governor", f"X{i:02d}", parties=("D",))
        races.append(info)
        views.append(view(info, "D", bid=0.40, ask=0.50))
        fairs.append(fair(info.markets["D"], 0.60))
    return races, views, fairs


def _convex_notional(orders) -> float:
    return sum(d.notional for d in orders if d.strategy == Strategy.CONVEX)


def test_f8_convex_size_follows_the_sleeve_fraction_and_respects_it(cfg):
    races, views, fairs = _tossups()
    small = SG.generate(ctx(cfg, races, views, fairs, sleeve_frac=0.15), [])
    large = SG.generate(ctx(cfg, races, views, fairs, sleeve_frac=0.30), [])
    n_small, n_large = _convex_notional(small), _convex_notional(large)
    assert n_large > n_small > 0
    assert n_small <= 0.15 * cfg.bankroll_start + 1e-6
    assert n_large <= 0.30 * cfg.bankroll_start + 1e-6
    # resting convex orders elsewhere count against the sleeve
    ctl = race("CONTROL-HOUSE", "control", None, label="House control")
    held = view(ctl, "D", bid=0.70, ask=0.80)
    held.open_orders = [intent("CONTROL-HOUSE-D", "yes", "buy", 0.50, 20_000, Strategy.CONVEX)]  # 10k resting
    out = SG.generate(ctx(cfg, [*races, ctl], [*views, held], fairs, sleeve_frac=0.15), [])
    assert _convex_notional(out) <= 0.15 * cfg.bankroll_start - 10_000 + 1e-6
