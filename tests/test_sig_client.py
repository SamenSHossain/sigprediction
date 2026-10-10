"""SigClient behaviour against a mocked exchange (respx). Nothing here touches the network."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import httpx
import pytest
import respx
from freezegun import freeze_time

from sigcup.common import time as ctime
from sigcup.sig.adapter import ExchangeAdapter
from sigcup.sig.client import RealtimeToken, SigClient, page_items, parse_dt, parse_fill
from sigcup.sig.errors import (
    AuthError,
    OrderStatusUnknown,
    RateLimited,
    RequestInFlight,
    TransientError,
    ValidationError,
)
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import OrderRequest

BASE = "https://sig.test/api/v1"
KEY = "test-key-SECRET-do-not-log"
FIX = Path(__file__).parent / "fixtures" / "sig"


def fixture(name: str) -> dict:
    return json.loads((FIX / name).read_text())


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def router():
    with respx.mock(base_url=BASE, assert_all_called=False) as r:
        yield r


@pytest.fixture()
def tournament_route(router):
    return router.get("/tournaments/cup").mock(
        return_value=httpx.Response(200, json=fixture("tournament.json"))
    )


@pytest.fixture()
def server_offset_guard():
    saved = ctime._server_offset
    yield
    ctime._server_offset = saved


def make_client(cfg, budget: Budget | None = None, **kw) -> SigClient:
    return SigClient(KEY, budget or Budget(90, 25), base_url=BASE, tournament_slug="cup", cfg=cfg, **kw)


def order(**kw) -> OrderRequest:
    base = dict(exchange_id="1076", side="yes", action="buy", price=0.845, quantity=10, idempotency_key="k-1")
    base.update(kw)
    return OrderRequest(**base)


# ---------------------------------------------------------------------------------------------
# Protocol conformance and order bodies
# ---------------------------------------------------------------------------------------------


async def test_implements_exchange_adapter(cfg):
    client = make_client(cfg)
    assert isinstance(client, ExchangeAdapter)
    assert client.name == "sig"
    assert client.tournament_id is None
    await client.aclose()


async def test_every_order_body_carries_tournament_id_and_idempotency_key(cfg, router, tournament_route):
    single = router.post("/orders").mock(return_value=httpx.Response(200, json=fixture("order_result.json")))
    multi = router.post("/orders/multi-leg").mock(
        return_value=httpx.Response(200, json=fixture("multi_leg_result.json"))
    )
    batch = router.post("/orders/batch").mock(
        return_value=httpx.Response(207, json=fixture("batch_result_207.json"))
    )
    client = make_client(cfg)
    expires = datetime(2026, 10, 10, 12, 15, 0, 250_000, tzinfo=UTC)

    res = await client.place_order(order(price=0.8463, expires_at=expires))
    body = json.loads(single.calls.last.request.content)
    assert body["tournamentId"] == "tour-2026-midterms"
    assert body["idempotencyKey"] == "k-1"
    assert body["price"] == 0.845  # rounded to the 0.005 tick before sending
    assert body["quantity"] == 10 and body["side"] == "yes" and body["action"] == "buy"
    assert body["exchangeId"] == "1076"
    assert body["expirationDate"] == "2026-10-10T12:15:00.250Z"
    assert res.order_id == "o-100" and res.status == "open" and not res.uncertain
    assert single.calls.last.request.headers["Authorization"] == f"Bearer {KEY}"

    legs = [order(idempotency_key="leg-a"), order(exchange_id="1077", idempotency_key="leg-b", price=0.08)]
    ml = await client.place_multi_leg(legs)
    body = json.loads(multi.calls.last.request.content)
    assert body["tournamentId"] == "tour-2026-midterms"
    assert body["idempotencyKey"]  # deterministic group key
    assert [leg["idempotencyKey"] for leg in body["legs"]] == ["leg-a", "leg-b"]
    assert all(leg["tournamentId"] == "tour-2026-midterms" for leg in body["legs"])
    assert ml.group_id == "ml-7" and ml.atomic_ok
    # results come back sorted by index and keyed to the legs' idempotency keys
    assert [r.order_id for r in ml.results] == ["o-201", "o-202"]
    assert [r.idempotencyKey for r in ml.results] == ["leg-a", "leg-b"]
    assert ml.results[1].quantity_traded == 10 and ml.results[1].fill_price == 0.08

    reqs = [order(idempotency_key="b-1"), order(idempotency_key="b-2", quantity=99999)]
    results = await client.place_batch(reqs)
    body = json.loads(batch.calls.last.request.content)
    assert body["tournamentId"] == "tour-2026-midterms" and body["idempotencyKey"]
    assert [o["idempotencyKey"] for o in body["orders"]] == ["b-1", "b-2"]
    assert all(o["tournamentId"] == "tour-2026-midterms" for o in body["orders"])
    assert results[0].order_id == "o-301" and not results[0].uncertain
    assert results[1].order_id is None and results[1].status == "rejected"
    assert "insufficient funds" in str(results[1].error)

    # the tournament id was resolved once and cached
    assert tournament_route.call_count == 1
    assert client.tournament_id == "tour-2026-midterms"
    await client.aclose()


async def test_multi_leg_same_legs_reuse_the_same_group_key(cfg, router, tournament_route):
    multi = router.post("/orders/multi-leg").mock(
        return_value=httpx.Response(200, json=fixture("multi_leg_result.json"))
    )
    client = make_client(cfg)
    legs = [order(idempotency_key="leg-a"), order(exchange_id="1077", idempotency_key="leg-b")]
    await client.place_multi_leg(legs)
    await client.place_multi_leg(legs)
    keys = [json.loads(c.request.content)["idempotencyKey"] for c in multi.calls]
    assert keys[0] == keys[1]
    await client.aclose()


async def test_order_without_tournament_slug_raises_auth_error(cfg, router):
    client = SigClient(KEY, Budget(90, 25), base_url=BASE, tournament_slug="", cfg=cfg)
    with pytest.raises(AuthError):
        await client.place_order(order())
    await client.aclose()


async def test_invalid_quantity_is_rejected_locally(cfg, router, tournament_route):
    posts = router.post("/orders")
    client = make_client(cfg)
    with pytest.raises(ValidationError):
        await client.place_order(order(quantity=0))
    assert posts.call_count == 0
    await client.aclose()


async def test_deadline_guard_blocks_orders(cfg, router, tournament_route, server_offset_guard):
    ctime._server_offset = timedelta(0)
    posts = router.post("/orders").mock(return_value=httpx.Response(200, json=fixture("order_result.json")))
    client = make_client(cfg)
    with freeze_time("2026-11-04T17:00:01Z"):
        with pytest.raises(ValidationError) as ei:
            await client.place_order(order())
    assert ei.value.code == "DEADLINE"
    assert posts.call_count == 0
    with freeze_time("2026-11-04T16:00:00Z"):
        res = await client.place_order(order())
    assert res.order_id == "o-100"
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------------------------


async def test_429_freezes_both_budgets_and_raises(cfg, router, tournament_route):
    router.post("/orders").mock(
        return_value=httpx.Response(
            429, headers={"Retry-After": "30"}, json={"error": {"code": "RATE_LIMITED"}}
        )
    )
    clock = FakeClock()
    budget = Budget(90, 25, clock=clock)
    client = make_client(cfg, budget)
    with pytest.raises(RateLimited) as ei:
        await client.place_order(order())
    assert ei.value.retry_after == 30
    assert budget.reads.available() == 0 and budget.writes.available() == 0
    assert budget.writes.frozen_for() == pytest.approx(30)
    clock.t += 31
    assert budget.reads.available() > 0 and budget.writes.available() > 0
    await client.aclose()


async def test_429_without_retry_after_freezes_60s(cfg, router):
    router.get("/account").mock(return_value=httpx.Response(429))
    clock = FakeClock()
    budget = Budget(90, 25, clock=clock)
    client = make_client(cfg, budget)
    with pytest.raises(RateLimited) as ei:
        await client.get_account()
    assert ei.value.retry_after == 60
    assert budget.writes.frozen_for() == pytest.approx(60)
    await client.aclose()


async def test_409_request_in_flight(cfg, router, tournament_route):
    router.post("/orders").mock(
        return_value=httpx.Response(
            409, json={"error": {"code": "REQUEST_IN_FLIGHT", "message": "lease held"}}
        )
    )
    client = make_client(cfg)
    with pytest.raises(RequestInFlight) as ei:
        await client.place_order(order())
    assert ei.value.status == 409 and ei.value.code == "REQUEST_IN_FLIGHT"
    await client.aclose()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(502, json={"error": {"code": "ORDER_STATUS_UNKNOWN", "message": "unknown"}}),
        httpx.Response(500, text="boom"),
        httpx.Response(504, text="gateway timeout"),
    ],
)
async def test_5xx_on_write_returns_uncertain_never_raises(cfg, router, tournament_route, response):
    router.post("/orders").mock(return_value=response)
    client = make_client(cfg)
    res = await client.place_order(order())
    assert res.uncertain is True
    assert res.order_id is None
    assert res.idempotencyKey == "k-1"
    await client.aclose()


async def test_timeout_on_write_returns_uncertain(cfg, router, tournament_route):
    router.post("/orders").mock(side_effect=httpx.ReadTimeout("slow engine"))
    router.post("/orders/multi-leg").mock(side_effect=httpx.ConnectError("reset"))
    client = make_client(cfg)
    res = await client.place_order(order())
    assert res.uncertain is True
    ml = await client.place_multi_leg([order(idempotency_key="a"), order(idempotency_key="b")])
    assert ml.atomic_ok is False
    assert len(ml.results) == 2 and all(r.uncertain for r in ml.results)
    assert [r.idempotencyKey for r in ml.results] == ["a", "b"]
    await client.aclose()


async def test_write_timeout_uses_config_write_timeout(cfg, router, tournament_route):
    route = router.post("/orders").mock(return_value=httpx.Response(200, json=fixture("order_result.json")))
    client = make_client(cfg)
    await client.place_order(order())
    assert route.calls.last.request.extensions["timeout"]["read"] == pytest.approx(cfg.limits.write_timeout_s)
    assert tournament_route.calls.last.request.extensions["timeout"]["read"] == pytest.approx(
        cfg.limits.read_timeout_s
    )
    await client.aclose()


async def test_503_is_transient_and_timeout_on_read_is_transient(cfg, router, tournament_route):
    router.post("/orders").mock(return_value=httpx.Response(503, json={"error": {"code": "TX_CONFLICT"}}))
    router.get("/account").mock(side_effect=httpx.ReadTimeout("slow"))
    router.get("/orders").mock(return_value=httpx.Response(502, text="bad gateway"))
    client = make_client(cfg)
    with pytest.raises(TransientError) as ei:
        await client.place_order(order())
    assert ei.value.status == 503 and ei.value.code == "TX_CONFLICT"
    with pytest.raises(TransientError):
        await client.get_account()
    with pytest.raises(TransientError):
        await client.get_open_orders()
    await client.aclose()


@pytest.mark.parametrize("status", [401, 403])
async def test_401_403_are_auth_errors(cfg, router, status):
    router.get("/account").mock(
        return_value=httpx.Response(status, json={"message": "accept the Terms first"})
    )
    client = make_client(cfg)
    with pytest.raises(AuthError) as ei:
        await client.get_account()
    assert ei.value.status == status and "Terms" in str(ei.value)
    await client.aclose()


async def test_other_4xx_is_validation_error_with_server_message(cfg, router, tournament_route):
    router.post("/orders").mock(
        return_value=httpx.Response(
            400, json={"error": {"code": "VALIDATION_ERROR", "message": "insufficient funds", "details": {}}}
        )
    )
    client = make_client(cfg)
    with pytest.raises(ValidationError) as ei:
        await client.place_order(order())
    assert str(ei.value) == "insufficient funds"
    assert ei.value.status == 400 and ei.value.code == "VALIDATION_ERROR"
    await client.aclose()


async def test_cancel_all_raises_order_status_unknown_on_timeout(cfg, router, tournament_route):
    router.post("/orders/cancel-all").mock(side_effect=httpx.ReadTimeout("slow"))
    client = make_client(cfg)
    with pytest.raises(OrderStatusUnknown):
        await client.cancel_all()
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# Clock offset and logging hygiene
# ---------------------------------------------------------------------------------------------


async def test_date_header_sets_server_offset(cfg, router, server_offset_guard):
    future = datetime.now(UTC) + timedelta(seconds=120)
    router.get("/account").mock(
        return_value=httpx.Response(
            200,
            headers={"Date": format_datetime(future, usegmt=True)},
            json=fixture("account_with_cash.json"),
        )
    )
    client = make_client(cfg)
    await client.get_account()
    assert 118 <= ctime.server_offset().total_seconds() <= 121
    await client.aclose()


async def test_key_and_headers_never_logged(cfg, router, tournament_route, caplog):
    router.post("/orders").mock(return_value=httpx.Response(502, text="unknown"))
    router.get("/account").mock(return_value=httpx.Response(401, json={"message": "nope"}))
    client = make_client(cfg)
    with caplog.at_level(logging.DEBUG):
        await client.place_order(order())
        with pytest.raises(AuthError):
            await client.get_account()
    text = caplog.text
    assert KEY not in text
    assert "Authorization" not in text and "Bearer" not in text
    assert "POST /orders" in text  # but the call itself is logged
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------------------------


async def test_get_tournament_parses_and_caches(cfg, router, tournament_route):
    client = make_client(cfg)
    t = await client.get_tournament()
    assert t.id == "tour-2026-midterms" and t.member_count == 1234 and t.my_balance == 98765.5
    assert t.start_date == datetime(2026, 10, 1, 16, 0, tzinfo=UTC)
    assert client.tournament_id == "tour-2026-midterms"
    await client.aclose()


async def test_get_prices_chunks_by_100_and_parses_variants(cfg, router, tournament_route):
    def respond(request: httpx.Request) -> httpx.Response:
        ids = request.url.params["ids"].split(",")
        assert request.url.params["tournamentId"] == "tour-2026-midterms"
        rows = [{"exchangeId": i, "bestBid": 0.4, "bestAsk": 0.45, "latestPrice": 0.42} for i in ids]
        return httpx.Response(200, json={"data": rows})

    route = router.get("/exchanges/prices").mock(side_effect=respond)
    client = make_client(cfg)
    ids = [str(1000 + i) for i in range(250)]
    snaps = await client.get_prices(ids)
    assert route.call_count == 3
    sizes = [len(c.request.url.params["ids"].split(",")) for c in route.calls]
    assert sizes == [100, 100, 50]
    assert len(snaps) == 250 and snaps["1249"].mid == pytest.approx(0.425)

    route.mock(return_value=httpx.Response(200, json=fixture("prices.json")))
    snaps = await client.get_prices(["1076", "1077", "1078", "1079"])
    assert snaps["1076"].best_bid == 0.845 and snaps["1076"].best_ask == 0.88 and snaps["1076"].last == 0.85
    assert snaps["1076"].ts == datetime(2026, 10, 10, 12, 0, 0, 123456, tzinfo=UTC)
    assert snaps["1076"].spread == pytest.approx(0.035)
    assert snaps["1077"].best_bid == 0.10 and snaps["1077"].bid_size == 20 and snaps["1077"].ask_size == 35
    assert snaps["1078"].best_bid == 0.5 and snaps["1078"].last == 0.51 and snaps["1078"].ask_size == 4000
    assert snaps["1079"].best_bid is None and snaps["1079"].best_ask == 0.995 and snaps["1079"].mid is None
    assert snaps["1076"].model_extra["spread"] == 0.035  # unknown fields are kept
    assert await client.get_prices([]) == {}
    await client.aclose()


async def test_get_orderbook_passes_depth_and_sorts_levels(cfg, router, tournament_route):
    route = router.get("/exchanges/1076/orderbook").mock(
        return_value=httpx.Response(200, json=fixture("orderbook.json"))
    )
    client = make_client(cfg)
    book = await client.get_orderbook("1076", depth=50)
    assert route.calls.last.request.url.params["depth"] == "50"
    assert route.calls.last.request.url.params["tournamentId"] == "tour-2026-midterms"
    assert [lv.price for lv in book.bids] == [0.845, 0.84, 0.82]
    assert [lv.price for lv in book.asks] == [0.88, 0.9]
    assert book.best_bid == 0.845 and book.best_ask == 0.88
    assert book.bids[0].quantity == 20
    assert book.ts == datetime(2026, 10, 10, 12, 0, 1, 500000, tzinfo=UTC)
    assert book.model_extra["sequence"] == 992
    await client.aclose()


async def test_get_positions_paginates_and_signs_quantities(cfg, router, tournament_route):
    route = router.get("/tournaments/cup/portfolio/positions").mock(
        side_effect=[
            httpx.Response(200, json=fixture("positions_page1.json")),
            httpx.Response(200, json=fixture("positions_page2.json")),
        ]
    )
    client = make_client(cfg)
    positions = await client.get_positions()
    assert route.call_count == 2
    assert "cursor" not in route.calls[0].request.url.params
    assert route.calls[1].request.url.params["cursor"] == "c2"
    by_id = {p.exchange_id: p for p in positions}
    assert by_id["1076"].quantity == 300 and by_id["1076"].avg_price == 0.84 and by_id["1076"].mark == 0.86
    assert by_id["1077"].quantity == -150  # option NO -> negative
    assert by_id["1078"].quantity == -40 and by_id["1078"].avg_price == 0.48
    await client.aclose()


async def test_get_positions_falls_back_to_portfolio_path(cfg, router, tournament_route):
    router.get("/tournaments/cup/portfolio/positions").mock(
        return_value=httpx.Response(404, json={"message": "no"})
    )
    fallback = router.get("/portfolio/positions").mock(
        return_value=httpx.Response(200, json=fixture("positions_page2.json"))
    )
    client = make_client(cfg)
    positions = await client.get_positions()
    assert [p.exchange_id for p in positions] == ["1078"]
    assert fallback.calls.last.request.url.params["tournamentId"] == "tour-2026-midterms"
    await client.get_positions()
    assert fallback.call_count == 2  # the working path is remembered
    await client.aclose()


async def test_get_open_orders_paginates_and_normalises(cfg, router, tournament_route):
    route = router.get("/orders").mock(
        side_effect=[
            httpx.Response(200, json=fixture("orders_page1.json")),
            httpx.Response(200, json=fixture("orders_page2.json")),
        ]
    )
    client = make_client(cfg)
    orders = await client.get_open_orders()
    assert route.call_count == 2
    p0 = route.calls[0].request.url.params
    assert p0["status"] == "open" and p0["tournamentId"] == "tour-2026-midterms"
    assert route.calls[1].request.url.params["cursor"] == "abc"
    assert [o.order_id for o in orders] == ["o-1", "o-2", "o-3"]
    o1, o2 = orders[0], orders[1]
    assert (
        o1.filled == 20
        and o1.idempotency_key == "k-1"
        and o1.expires_at == datetime(2026, 10, 10, 12, 14, tzinfo=UTC)
    )
    assert (
        o2.side == "no"
        and o2.action == "sell"
        and o2.price == 0.155
        and o2.filled == 0
        and o2.status == "open"
    )
    await client.aclose()


async def test_get_fills_filters_by_since_and_stops_paging_early(cfg, router, tournament_route):
    route = router.get("/tournaments/cup/portfolio/fills").mock(
        side_effect=[
            httpx.Response(200, json=fixture("fills_page1.json")),
            httpx.Response(200, json=fixture("fills_page2.json")),
            httpx.Response(200, json={"data": [], "pagination": {"hasMore": False}}),
        ]
    )
    client = make_client(cfg)
    fills = await client.get_fills(since=datetime(2026, 10, 9, 12, 0, tzinfo=UTC))
    assert route.call_count == 2  # page 2 is entirely older than `since`: no third read
    assert [f.fill_id for f in fills] == ["f-3", "f-2"]
    assert fills[0].order_id == "o-1" and fills[0].quantity == 20 and fills[0].price == 0.845
    assert fills[0].ts.tzinfo is not None

    route.side_effect = [
        httpx.Response(200, json=fixture("fills_page1.json")),
        httpx.Response(200, json=fixture("fills_page2.json")),
        httpx.Response(200, json={"data": [], "pagination": {"hasMore": False}}),
    ]
    all_fills = await client.get_fills()
    assert [f.fill_id for f in all_fills] == ["f-3", "f-2", "f-1", "f-0"]
    await client.aclose()


async def test_get_account_uses_tournament_balance_when_account_has_none(cfg, router, tournament_route):
    router.get("/account").mock(return_value=httpx.Response(200, json=fixture("account.json")))
    client = make_client(cfg)
    acct = await client.get_account()
    assert acct.cash == 98765.5
    assert client.profile_id == "prof-42" and client.username == "samen"
    assert tournament_route.call_count == 1

    router.get("/account").mock(return_value=httpx.Response(200, json=fixture("account_with_cash.json")))
    acct = await client.get_account()
    assert acct.cash == 50000.25 and acct.positions_value == 40000.0 and acct.total == 90000.25
    assert tournament_route.call_count == 1
    await client.aclose()


async def test_get_leaderboard_parses_member_count_and_our_rank(cfg, router):
    route = router.get("/tournaments/cup/leaderboard").mock(
        return_value=httpx.Response(200, json=fixture("leaderboard.json"))
    )
    client = make_client(cfg)
    lb = await client.get_leaderboard()
    assert route.calls.last.request.url.params["period"] == "all"
    assert lb.member_count == 1234 and lb.my_rank == 17 and lb.my_total == 104000.0
    assert [e.rank for e in lb.entries] == [1, 2, 3, 17]
    assert lb.entries[2].total == 140000.0 and lb.entries[2].user == "gamma"
    assert lb.entries[3].is_me and not lb.entries[0].is_me

    # without explicit isMe / myRank the entry is matched on our username
    body = fixture("leaderboard.json")
    body.pop("myRank")
    body["leaderboard"][3].pop("isMe")
    route.mock(return_value=httpx.Response(200, json=body))
    client.username = "samen"
    lb = await client.get_leaderboard()
    assert lb.my_rank == 17 and lb.entries[3].is_me
    await client.aclose()


async def test_relationships_and_constraint_violations(cfg, router, tournament_route):
    rel = router.get("/relationships").mock(
        return_value=httpx.Response(200, json=fixture("relationships.json"))
    )
    con = router.get("/relationships/constraints").mock(
        return_value=httpx.Response(200, json=fixture("constraints.json"))
    )
    client = make_client(cfg)
    rels = await client.get_relationships()
    assert rel.calls.last.request.url.params["tournamentId"] == "tour-2026-midterms"
    assert rels[0].id == "rel-1" and rels[0].kind == "mutually_exclusive" and rels[0].is_exhaustive is True
    assert rels[0].exchange_ids == ["1076", "1077"]
    assert rels[1].id == "rel-2" and rels[1].kind == "mutually_exclusive" and rels[1].is_exhaustive is False
    assert rels[1].exchange_ids == ["2001", "2002", "2003"]

    viols = await client.get_constraint_violations()
    params = con.calls.last.request.url.params
    assert params["violationsOnly"] == "true" and params["tournamentId"] == "tour-2026-midterms"
    assert len(viols) == 1
    v = viols[0]
    assert v.relationship_id == "rel-1" and v.magnitude == 0.03 and v.kind == "sum_of_yes_bids_gt_1"
    assert v.exchange_ids == ["1076", "1077"]
    assert len(v.suggested_trades) == 2 and v.suggested_trades[0]["action"] == "sell"
    await client.aclose()


async def test_list_markets_picks_yes_exchange_and_keeps_extras(cfg, router):
    router.get("/tournaments/cup/markets").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "387",
                        "title": "Will the Democratic Party win the Rhode Island Senate race?",
                        "status": "open",
                        "settlementDate": "2026-11-04T17:00:00Z",
                        "categories": [{"name": "election-outcome"}],
                        "exchanges": [{"id": "1076", "option": "YES", "latestPrice": 0.85}],
                        "overround": 1.01,
                        "hasArbitrageOpportunity": False,
                    },
                    {"id": "388", "title": "Will the Republican Party win ...?", "exchangeId": "1077"},
                    {"id": "bad"},
                ],
                "pagination": {"hasMore": False},
            },
        )
    )
    client = make_client(cfg)
    markets = await client.list_markets()
    assert [m.id for m in markets] == ["387", "388"]
    assert markets[0].exchange_id == "1076" and markets[0].category == "election-outcome"
    assert markets[0].settlement_date == datetime(2026, 11, 4, 17, 0, tzinfo=UTC)
    assert markets[0].overround == 1.01 and markets[0].has_arbitrage_opportunity is False
    assert markets[0].model_extra["exchanges"][0]["latestPrice"] == 0.85
    assert markets[1].exchange_id == "1077"
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# Cancels, token
# ---------------------------------------------------------------------------------------------


async def test_cancel_order_true_on_2xx_false_on_404(cfg, router, tournament_route):
    route = router.delete("/orders/o-1").mock(return_value=httpx.Response(204))
    router.delete("/orders/gone").mock(return_value=httpx.Response(404, json={"message": "not found"}))
    client = make_client(cfg)
    await client.ensure_tournament_id()
    assert await client.cancel_order("o-1") is True
    assert route.calls.last.request.url.params["tournamentId"] == "tour-2026-midterms"
    assert await client.cancel_order("gone") is False
    await client.aclose()


async def test_cancel_all_accepts_207_partial_results(cfg, router, tournament_route):
    route = router.post("/orders/cancel-all").mock(
        return_value=httpx.Response(207, json=fixture("cancel_all_207.json"))
    )
    client = make_client(cfg)
    assert await client.cancel_all() == 2
    body = json.loads(route.calls.last.request.content)
    assert body == {"tournamentId": "tour-2026-midterms"}
    route.mock(return_value=httpx.Response(200, json={"cancelled": 5}))
    assert await client.cancel_all(exchange_id="1076") == 5
    assert json.loads(route.calls.last.request.content)["exchangeId"] == "1076"
    await client.aclose()


async def test_cancel_all_uses_priority_zero(cfg, router, tournament_route):
    router.post("/orders/cancel-all").mock(return_value=httpx.Response(200, json={"cancelled": 0}))
    client = make_client(cfg)
    seen: list[int] = []
    orig = client.budget.writes.wait

    async def spy(priority: int = 2) -> None:
        seen.append(priority)
        await orig(priority)

    client.budget.writes.wait = spy  # type: ignore[method-assign]
    await client.cancel_all()
    assert seen == [0]
    await client.aclose()


async def test_realtime_token_parses_fields(cfg, router):
    route = router.post("/realtime/token").mock(
        return_value=httpx.Response(200, json=fixture("realtime_token.json"))
    )
    client = make_client(cfg)
    tok = await client.realtime_token()
    assert isinstance(tok, RealtimeToken)
    assert tok.supabase_url == "https://abcdefgh.supabase.co"
    assert tok.anon_key == "anon-public-key" and tok.token == "eyJ.realtime.jwt"
    assert tok.expires_at == datetime(2026, 10, 10, 15, 0, tzinfo=UTC)
    assert tok.user_channel == "user:prof-42"
    assert route.call_count == 1
    # an expiresIn-style body still yields an expiry
    route.mock(
        return_value=httpx.Response(
            200,
            json={"supabaseUrl": "https://x.supabase.co", "anonKey": "a", "token": "t", "expiresIn": 10800},
        )
    )
    tok = await client.realtime_token()
    assert tok.expires_at is not None and tok.seconds_left > 10000
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------------------------


async def test_every_call_goes_through_the_budget(cfg, router, tournament_route):
    router.get("/account").mock(return_value=httpx.Response(200, json=fixture("account_with_cash.json")))
    router.post("/orders").mock(return_value=httpx.Response(200, json=fixture("order_result.json")))
    budget = Budget(90, 25)
    client = make_client(cfg, budget)
    await client.get_account()
    assert budget.reads.used() == 1 and budget.writes.used() == 0
    await client.place_order(order())
    assert budget.reads.used() == 2  # + tournament lookup
    assert budget.writes.used() == 1
    await client.aclose()


async def test_sliding_window_blocks_the_26th_write_in_a_minute(cfg, router, tournament_route):
    router.post("/orders").mock(return_value=httpx.Response(200, json=fixture("order_result.json")))
    clock = FakeClock()
    budget = Budget(90, 25, clock=clock)
    client = make_client(cfg, budget)
    for i in range(25):
        res = await client.place_order(order(idempotency_key=f"k-{i}"))
        assert not res.uncertain
    assert budget.writes.available() == 0
    assert budget.reads.available() > 0  # reads are a separate window

    task = asyncio.create_task(client.place_order(order(idempotency_key="k-26")))
    done, _ = await asyncio.wait({task}, timeout=0.3)
    assert not done  # the 26th write is parked, not sent
    assert len(router.routes[-1].calls) == 25 or router.post("/orders").call_count == 25

    clock.t += 60.1  # the window slides
    await asyncio.wait_for(task, timeout=3)
    assert task.result().order_id == "o-100"
    assert router.post("/orders").call_count == 26
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# Helper behaviour
# ---------------------------------------------------------------------------------------------


def test_parse_dt_variants():
    assert parse_dt("2026-10-10T12:00:00.1234567Z") == datetime(2026, 10, 10, 12, 0, 0, 123456, tzinfo=UTC)
    assert parse_dt("2026-10-10T12:00:00+02:00") == datetime(2026, 10, 10, 10, 0, tzinfo=UTC)
    assert parse_dt(1791633600) == datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    assert parse_dt(1791633600000) == datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    assert parse_dt(datetime(2026, 1, 1)) == datetime(2026, 1, 1, tzinfo=UTC)
    assert parse_dt(None) is None and parse_dt("garbage") is None


def test_page_items_shapes():
    assert page_items([1, 2]) == ([1, 2], None, False)
    assert page_items({"data": [1], "pagination": {"nextCursor": "x", "hasMore": True}}) == ([1], "x", True)
    assert page_items({"orders": [1], "nextCursor": None}) == ([1], None, False)
    assert page_items({"data": {"items": [3]}, "pagination": {"cursor": "c"}}) == ([3], "c", True)
    assert page_items(None) == ([], None, False)


def test_parse_fill_synthesises_ids_and_tolerates_missing_fields():
    f = parse_fill({"exchangeId": "1", "price": 0.5, "quantity": 3, "executedAt": "2026-10-10T00:00:00Z"})
    assert f is not None and f.fill_id.startswith("synth:") and f.side == "yes" and f.action == "buy"
    assert f.model_extra.get("side_or_action_missing") is True
    again = parse_fill({"exchangeId": "1", "price": 0.5, "quantity": 3, "executedAt": "2026-10-10T00:00:00Z"})
    assert again is not None and again.fill_id == f.fill_id
    assert parse_fill({"price": 0.5}) is None
