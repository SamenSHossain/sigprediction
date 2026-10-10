"""PaperAdapter fill rules: crossing the best quote fills at the resting price, displayed size bounds the
fill, unowned sells become NO positions, expiry/cancel/idempotency, cash and positions accounting."""

from __future__ import annotations

from datetime import timedelta

import pytest

from sigcup.paper import PaperAdapter
from sigcup.sig.errors import ValidationError
from sigcup.sig.schemas import OrderRequest
from tests.fakes_exec import FakeClock


def req(eid: str, price: float, qty: int, side="yes", action="buy", key: str | None = None, expires_in: float | None = None, clock: FakeClock | None = None) -> OrderRequest:
    exp = clock.now() + timedelta(seconds=expires_in) if (expires_in is not None and clock is not None) else None
    return OrderRequest(exchange_id=eid, side=side, action=action, price=price, quantity=qty, idempotency_key=key or f"k-{eid}-{side}-{action}-{price}-{qty}", expires_at=exp)


@pytest.fixture()
def paper(cfg, db):
    clock = FakeClock()
    p = PaperAdapter(cfg, cash=10_000.0, clock=clock.now, use_db_quotes=False)
    p.set_top("m1", bid=0.50, ask=0.58, bid_size=20, ask_size=20)
    return p, clock


async def test_buy_crossing_ask_fills_at_resting_price_bounded_by_size(paper):
    p, _ = paper
    res = await p.place_order(req("m1", 0.60, 50))
    assert res.order_id and res.quantity_traded == 20 and res.status == "open"
    assert res.fill_price == pytest.approx(0.60)  # at the resting price, not the ask
    assert p.cash == pytest.approx(10_000 - 0.60 * 20)
    (pos,) = await p.get_positions()
    assert pos.quantity == 20 and pos.avg_price == pytest.approx(0.60)
    fills = await p.get_fills()
    assert [f.fill_id for f in fills] == ["paper:1"] and fills[0].quantity == 20
    # more liquidity arrives: the remainder fills
    p.set_top("m1", bid=0.50, ask=0.55, ask_size=100)
    open_orders = await p.get_open_orders()
    assert open_orders == []
    (pos,) = await p.get_positions()
    assert pos.quantity == 50
    assert p.cash == pytest.approx(10_000 - 0.60 * 50)


async def test_buy_below_ask_rests(paper):
    p, _ = paper
    res = await p.place_order(req("m1", 0.52, 10))
    assert res.quantity_traded == 0 and res.status == "open"
    orders = await p.get_open_orders()
    assert len(orders) == 1 and orders[0].idempotency_key == "k-m1-yes-buy-0.52-10" and orders[0].filled == 0
    assert p.cash == 10_000.0 and await p.get_positions() == []


async def test_sell_crossing_bid_without_position_creates_no_position(paper):
    p, _ = paper
    res = await p.place_order(req("m1", 0.45, 10, action="sell"))
    assert res.quantity_traded == 10
    (pos,) = await p.get_positions()
    assert pos.quantity == -10  # short YES == long NO
    assert p.cash == pytest.approx(10_000 - (1 - 0.45) * 10)  # filled as buying NO at 0.55
    assert pos.avg_price == pytest.approx(0.55)


async def test_buy_no_is_sell_yes_equivalent(paper):
    p, _ = paper
    # buy NO @0.50 == sell YES @0.50; best bid 0.50 -> fills
    res = await p.place_order(req("m1", 0.50, 10, side="no"))
    assert res.quantity_traded == 10 and res.fill_price == pytest.approx(0.50)
    (pos,) = await p.get_positions()
    assert pos.quantity == -10 and p.cash == pytest.approx(10_000 - 0.50 * 10)
    # buy NO @0.45 == sell YES @0.55 > bid: rests
    res = await p.place_order(req("m1", 0.45, 10, side="no"))
    assert res.quantity_traded == 0


async def test_sell_owned_returns_cash_and_flattens(paper):
    p, _ = paper
    await p.place_order(req("m1", 0.60, 20))  # fills 20 @0.60
    p.set_top("m1", bid=0.70, ask=0.75, bid_size=50)
    res = await p.place_order(req("m1", 0.65, 20, action="sell"))
    assert res.quantity_traded == 20
    assert await p.get_positions() == []
    assert p.cash == pytest.approx(10_000 - 0.60 * 20 + 0.65 * 20)
    acct = await p.get_account()
    assert acct.cash == pytest.approx(p.cash) and acct.positions_value == 0


async def test_expiry_and_cancel(paper):
    p, clock = paper
    r1 = await p.place_order(req("m1", 0.52, 10, expires_in=60, clock=clock))
    r2 = await p.place_order(req("m1", 0.51, 10))
    assert len(await p.get_open_orders()) == 2
    clock.advance(61)
    orders = await p.get_open_orders()
    assert [o.order_id for o in orders] == [r2.order_id]
    assert p.orders[r1.order_id].status == "expired"
    assert await p.cancel_order(r2.order_id) is True
    assert await p.cancel_order(r2.order_id) is False
    assert await p.get_open_orders() == []
    await p.place_order(req("m1", 0.50, 5))
    await p.place_order(req("m1", 0.49, 5))
    assert await p.cancel_all() == 2


async def test_idempotent_replay_and_validation(paper):
    p, _ = paper
    r1 = await p.place_order(req("m1", 0.52, 10, key="same"))
    r2 = await p.place_order(req("m1", 0.52, 10, key="same"))
    assert r1.order_id == r2.order_id and len(p.orders) == 1
    with pytest.raises(ValidationError):
        await p.place_order(req("m1", 0.5012, 10))
    with pytest.raises(ValidationError):
        await p.place_order(req("m1", 0.50, 0))
    with pytest.raises(ValidationError):  # insufficient funds: 0.5 * 100000 > cash
        await p.place_order(req("m1", 0.50, 100_000))
    assert len(p.orders) == 1


async def test_resting_orders_reserve_against_insufficient_funds(paper):
    p, _ = paper
    await p.place_order(req("m1", 0.50, 19_000))  # rests, needs 9 500 of 10 000
    with pytest.raises(ValidationError):
        await p.place_order(req("m1", 0.50, 2_000))
    await p.place_order(req("m1", 0.50, 1_000))  # 500 still fits


async def test_multi_leg_is_atomic_on_validation(paper):
    p, _ = paper
    legs = [req("m1", 0.52, 10, key="l1"), req("m1", 0.5012, 10, key="l2")]
    with pytest.raises(ValidationError):
        await p.place_multi_leg(legs)
    assert p.orders == {}
    res = await p.place_multi_leg([req("m1", 0.52, 10, key="l1"), req("m1", 0.51, 10, key="l3")])
    assert len(res.results) == 2 and res.atomic_ok and len(p.orders) == 2


async def test_paper_uses_db_quotes_when_no_source(cfg, db):
    from sigcup.db import engine as dbengine
    from sigcup.db.models import Quote

    clock = FakeClock()
    with dbengine.session() as s:
        s.add(Quote(ts=clock.now(), exchange_id="m9", bid=0.40, ask=0.44, bid_size=10, ask_size=30))
    p = PaperAdapter(cfg, cash=1000.0, clock=clock.now)
    snaps = await p.get_prices(["m9", "missing"])
    assert set(snaps) == {"m9"} and snaps["m9"].best_ask == 0.44
    res = await p.place_order(req("m9", 0.45, 40))
    assert res.quantity_traded == 30
    assert p.tops["m9"].ask_size == 0  # displayed size consumed until the next quote refresh
    book = await p.get_orderbook("m9")
    assert book.best_bid == 0.40 and book.best_ask == 0.44


async def test_paper_wraps_source_adapter(cfg, db):
    from tests.fakes_exec import FakeAdapter

    clock = FakeClock()
    src = FakeAdapter(clock=clock.now)
    src.set_top("m1", 0.30, 0.34, ask_size=5)
    p = PaperAdapter(cfg, source=src, cash=1000.0, clock=clock.now)
    assert p.tournament_id == src.tournament_id
    snaps = await p.get_prices(["m1"])
    assert snaps["m1"].best_ask == 0.34
    res = await p.place_order(req("m1", 0.35, 10))
    assert res.quantity_traded == 5
    assert src.writes == [], "paper never writes to the source"
    lb = await p.get_leaderboard()
    assert lb.my_rank == src.leaderboard.my_rank
