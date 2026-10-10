"""A2: theta orders are ranked by yield and value / convex orders by |edge| (best first) inside their priority
group, in signals.generate and in risk.apply's allocation of cash, so when cash or writes bind the best
opportunity is served first rather than the alphabetically first race."""

from __future__ import annotations

from sigcup.db.models import Strategy
from sigcup.settings import Risk, StrategyConfig
from sigcup.trader import risk
from sigcup.trader import signals as SG
from sigcup.trader.types import PRIORITY_THETA, PRIORITY_VALUE, DesiredOrder
from tests.test_model_support import ctx, fair, race, view


def _favourites():
    """SEN-AA: fair 0.92, ask 0.88 -> bid min(0.88, 0.92 / 1.055) = 0.870, yield 5.7%.
    SEN-ZZ: fair 0.97, ask 0.90 -> bid 0.900, yield 7.8% (better, but alphabetically last)."""
    aa, zz = race("SEN-AA", "senate", "AA"), race("SEN-ZZ", "senate", "ZZ")
    views = [view(aa, "D", bid=0.84, ask=0.88), view(zz, "D", bid=0.86, ask=0.90)]
    fairs = [fair("SEN-AA-D", 0.92), fair("SEN-ZZ-D", 0.97)]
    return [aa, zz], views, fairs


def test_generate_ranks_theta_by_yield_not_race_id(cfg):
    races, views, fairs = _favourites()
    c = ctx(cfg, races, views, fairs)
    theta = [d for d in SG.generate(c, []) if d.strategy == Strategy.THETA]
    assert [d.exchange_id for d in theta] == ["SEN-ZZ-D", "SEN-AA-D"]
    yields = [SG.theta_yield(d.fair, d.price, cfg) for d in theta]
    assert yields == sorted(yields, reverse=True)


def test_generate_ranks_value_and_convex_by_abs_edge_within_their_priority(cfg):
    aa, zz = race("SEN-AA", "senate", "AA"), race("SEN-ZZ", "senate", "ZZ")
    # both toss-ups at fair 0.50: AA's outside mid is 0.44 (edge 6c), ZZ's is 0.40 (edge 10c)
    views = [view(aa, "D", bid=0.43, ask=0.45), view(zz, "D", bid=0.39, ask=0.41)]
    fairs = [fair("SEN-AA-D", 0.50), fair("SEN-ZZ-D", 0.50)]
    c = ctx(cfg, [aa, zz], views, fairs)
    out = SG.generate(c, [])
    group = [d for d in out if d.priority == PRIORITY_VALUE]
    assert {d.strategy for d in group} == {Strategy.VALUE, Strategy.CONVEX}
    edges = [abs(d.edge) for d in group]
    assert edges == sorted(edges, reverse=True)
    assert group[0].exchange_id == "SEN-ZZ-D"
    # priority groups stay intact: nothing of priority VALUE before a THETA order, etc.
    prios = [d.priority for d in out]
    assert prios == sorted(prios)


def test_limited_cash_funds_the_highest_yield_favourite_first(cfg):
    races, views, fairs = _favourites()
    # free cash above the 20k reserve is 1 000 SUSQies: enough for one favourite, not both
    c = ctx(cfg, races, views, fairs, cash=21_000.0)
    out = risk.apply(c, SG.generate(c, []))
    theta = [d for d in out if d.strategy == Strategy.THETA]
    assert [d.exchange_id for d in theta] == ["SEN-ZZ-D"]
    assert theta[0].quantity * theta[0].price <= 1_000 + 1e-6


def _theta(eid: str, price: float, fair_: float) -> DesiredOrder:
    return DesiredOrder(
        exchange_id=eid, side="yes", action="buy", price=price, quantity=500, strategy=Strategy.THETA,
        priority=PRIORITY_THETA, fair=fair_, edge=fair_ - price,
    )


def test_risk_allocates_cash_by_yield_even_if_the_input_is_unsorted():
    cfg = StrategyConfig(risk=Risk(liquid_reserve_frac=0.0, liquid_reserve_frac_final72h=0.0))
    races, views, fairs = _favourites()
    c = ctx(cfg, races, views, fairs, cash=1_000.0)
    desired = [_theta("SEN-AA-D", 0.87, 0.92), _theta("SEN-ZZ-D", 0.90, 0.97)]  # race-id order
    out = risk.apply(c, desired)
    assert [d.exchange_id for d in out] == ["SEN-ZZ-D"]
    # with cash for both, the output is best-first too (the OMS preserves order within a priority)
    c = ctx(cfg, races, views, fairs, cash=100_000.0)
    out = risk.apply(c, desired)
    assert [d.exchange_id for d in out] == ["SEN-ZZ-D", "SEN-AA-D"]


def test_risk_ranks_value_orders_by_abs_edge():
    cfg = StrategyConfig(risk=Risk(liquid_reserve_frac=0.0, liquid_reserve_frac_final72h=0.0))
    aa, zz = race("SEN-AA", "senate", "AA"), race("SEN-ZZ", "senate", "ZZ")
    views = [view(aa, "D", bid=0.43, ask=0.45), view(zz, "D", bid=0.39, ask=0.41)]
    fairs = [fair("SEN-AA-D", 0.50), fair("SEN-ZZ-D", 0.50)]
    c = ctx(cfg, [aa, zz], views, fairs, cash=200.0)
    desired = [
        DesiredOrder(exchange_id="SEN-AA-D", side="yes", action="buy", price=0.45, quantity=200, strategy=Strategy.VALUE, fair=0.50, edge=0.06),
        DesiredOrder(exchange_id="SEN-ZZ-D", side="yes", action="buy", price=0.41, quantity=200, strategy=Strategy.VALUE, fair=0.50, edge=0.10),
    ]
    out = risk.apply(c, desired)
    assert [d.exchange_id for d in out] == ["SEN-ZZ-D"]
