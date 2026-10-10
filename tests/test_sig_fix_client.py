"""Regression tests for review findings on SigClient (sigcup/sig/client.py). Mocked with respx: no network."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
import respx

from sigcup.common import time as ctime
from sigcup.sig.client import CancelAllIncomplete, SigClient, page_items, parse_fill
from sigcup.sig.errors import OrderStatusUnknown, SigError, TransientError, ValidationError
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import OrderRequest

BASE = "https://sig.test/api/v1"
TID = "tour-2026-midterms"


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
def server_offset_guard():
    saved = ctime._server_offset
    yield
    ctime._server_offset = saved


def make_client(cfg, budget: Budget | None = None, **kw) -> SigClient:
    kw.setdefault("tournament_id", TID)
    return SigClient("k", budget or Budget(90, 25), base_url=BASE, tournament_slug="cup", cfg=cfg, **kw)


def leg(key: str, eid: str, **kw) -> OrderRequest:
    base = dict(exchange_id=eid, side="no", action="buy", price=0.9, quantity=10, idempotency_key=key)
    base.update(kw)
    return OrderRequest(**base)


def spy_priorities(limiter) -> list[int]:
    seen: list[int] = []
    orig = limiter.wait

    async def spy(priority: int = 2) -> None:
        seen.append(priority)
        await orig(priority)

    limiter.wait = spy
    return seen


# ---------------------------------------------------------------------------------------------
# F1: an uncertain DELETE must not look like "already gone"
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mock",
    [
        {"return_value": httpx.Response(502, json={"error": {"code": "ORDER_STATUS_UNKNOWN"}})},
        {"return_value": httpx.Response(500, text="boom")},
        {"side_effect": httpx.ReadTimeout("slow engine")},
        {"side_effect": httpx.ConnectError("reset")},
    ],
)
async def test_cancel_order_uncertain_outcome_raises_instead_of_false(cfg, router, mock):
    router.delete("/orders/o-9").mock(**mock)
    client = make_client(cfg)
    with pytest.raises(OrderStatusUnknown):
        await client.cancel_order("o-9")
    await client.aclose()


async def test_cancel_order_false_only_when_the_order_is_gone(cfg, router):
    router.delete("/orders/gone").mock(return_value=httpx.Response(404, json={"message": "not found"}))
    router.delete("/orders/old").mock(return_value=httpx.Response(410, json={"message": "gone"}))
    router.delete("/orders/busy").mock(return_value=httpx.Response(503, json={"error": {"code": "TX_CONFLICT"}}))
    client = make_client(cfg)
    assert await client.cancel_order("gone") is False
    assert await client.cancel_order("old") is False
    with pytest.raises(TransientError):
        await client.cancel_order("busy")
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# F3: multi-leg responses are handled atomically
# ---------------------------------------------------------------------------------------------

LEGS = [leg("a", "1"), leg("b", "2"), leg("c", "3")]


async def test_multi_leg_group_error_rejects_every_leg(cfg, router):
    router.post("/orders/multi-leg").mock(
        return_value=httpx.Response(
            200, json={"success": False, "error": {"code": "INSUFFICIENT_FUNDS", "message": "insufficient funds"}}
        )
    )
    client = make_client(cfg)
    res = await client.place_multi_leg(LEGS)
    assert res.atomic_ok is False
    assert [r.uncertain for r in res.results] == [False, False, False]
    assert [r.status for r in res.results] == ["rejected"] * 3
    assert [r.idempotencyKey for r in res.results] == ["a", "b", "c"]
    assert all("insufficient funds" in str(r.error) for r in res.results)
    await client.aclose()


async def test_multi_leg_results_keyed_by_idempotency_key(cfg, router):
    router.post("/orders/multi-leg").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "groupId": "g1",
                    "orders": {
                        "c": {"orderId": "o3", "status": "open"},
                        "a": {"orderId": "o1", "status": "open"},
                        "b": {"orderId": "o2", "status": "filled", "quantityTraded": 10},
                    },
                }
            },
        )
    )
    client = make_client(cfg)
    res = await client.place_multi_leg(LEGS)
    assert [r.order_id for r in res.results] == ["o1", "o2", "o3"]
    assert [r.idempotencyKey for r in res.results] == ["a", "b", "c"]
    assert res.atomic_ok is True and res.group_id == "g1"
    await client.aclose()


async def test_multi_leg_list_results_matched_by_leg_key_not_position(cfg, router):
    router.post("/orders/multi-leg").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"idempotencyKey": "b", "orderId": "o2", "status": "open"},
                    {"idempotencyKey": "c", "orderId": "o3", "status": "open"},
                    {"idempotencyKey": "a", "orderId": "o1", "status": "open"},
                ]
            },
        )
    )
    client = make_client(cfg)
    res = await client.place_multi_leg(LEGS)
    assert [r.order_id for r in res.results] == ["o1", "o2", "o3"]
    await client.aclose()


async def test_multi_leg_order_id_list_maps_positionally(cfg, router):
    router.post("/orders/multi-leg").mock(
        return_value=httpx.Response(200, json={"groupId": "g1", "orderIds": ["o1", "o2", "o3"]})
    )
    client = make_client(cfg)
    res = await client.place_multi_leg(LEGS)
    assert [r.order_id for r in res.results] == ["o1", "o2", "o3"]
    assert not any(r.uncertain for r in res.results) and res.group_id == "g1"
    await client.aclose()


@pytest.mark.parametrize(
    "body",
    [
        {"data": {"groupId": "g1", "status": "accepted"}},  # a group receipt with no per-leg results
        {"data": {"id": "ml-9", "status": "open"}},  # looks like one order: must not become leg 0's receipt
        {"results": [{"orderId": "o1", "status": "open"}]},  # fewer results than legs
        {"results": [{"orderId": "o1"}, {"orderId": "o2"}, {"foo": 1}]},  # a leg with neither id nor status
    ],
)
async def test_multi_leg_unsplittable_response_marks_every_leg_uncertain(cfg, router, body):
    router.post("/orders/multi-leg").mock(return_value=httpx.Response(200, json=body))
    client = make_client(cfg)
    res = await client.place_multi_leg(LEGS)
    assert res.atomic_ok is False
    assert [r.uncertain for r in res.results] == [True, True, True]
    assert [r.order_id for r in res.results] == [None, None, None]
    assert [r.idempotencyKey for r in res.results] == ["a", "b", "c"]
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# F4: a failed cancel-all is never reported as "nothing to cancel"
# ---------------------------------------------------------------------------------------------


async def test_cancel_all_422_error_envelope_raises(cfg, router):
    router.post("/orders/cancel-all").mock(
        return_value=httpx.Response(
            422, json={"error": {"code": "VALIDATION_ERROR", "message": "tournamentId is invalid"}}
        )
    )
    client = make_client(cfg)
    with pytest.raises(SigError) as ei:
        await client.cancel_all()
    assert ei.value.status == 422 and "tournamentId is invalid" in str(ei.value)
    await client.aclose()


async def test_cancel_all_partial_failure_raises_with_failed_ids(cfg, router):
    route = router.post("/orders/cancel-all").mock(
        return_value=httpx.Response(
            207,
            json={
                "results": [
                    {"orderId": "o-1", "status": "cancelled"},
                    {"orderId": "o-2", "status": "error", "error": {"code": "TX_CONFLICT"}},
                    {"orderId": "o-3", "status": "error", "error": {"code": "NOT_FOUND"}},
                ]
            },
        )
    )
    client = make_client(cfg)
    with pytest.raises(CancelAllIncomplete) as ei:
        await client.cancel_all()
    assert ei.value.failed == ["o-2"] and ei.value.cancelled == 1
    # the nested {data: {cancelled: [...], failed: [...]}} flavour too
    route.mock(
        return_value=httpx.Response(
            207, json={"data": {"cancelled": ["o-1", "o-4"], "failed": [{"orderId": "o-5", "error": "locked"}]}}
        )
    )
    with pytest.raises(CancelAllIncomplete) as ei:
        await client.cancel_all()
    assert ei.value.failed == ["o-5"] and ei.value.cancelled == 2
    await client.aclose()


async def test_cancel_all_counts_nested_lists_and_unrecognised_partial_raises(cfg, router):
    route = router.post("/orders/cancel-all").mock(
        return_value=httpx.Response(200, json={"data": {"cancelled": ["o-1", "o-2", "o-3"], "failed": []}})
    )
    client = make_client(cfg)
    assert await client.cancel_all() == 3
    route.mock(return_value=httpx.Response(207, json={"data": {"message": "partial"}}))
    with pytest.raises(CancelAllIncomplete):
        await client.cancel_all()
    route.mock(return_value=httpx.Response(200, json={"success": False, "error": {"message": "engine paused"}}))
    with pytest.raises(SigError):
        await client.cancel_all()
    # a 422 where every "failure" is an order that is already gone: nothing was left live
    route.mock(
        return_value=httpx.Response(
            422, json={"results": [{"orderId": "o-1", "status": "error", "error": {"code": "NOT_FOUND"}}]}
        )
    )
    assert await client.cancel_all() == 0
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# F5: the deadline is re-checked after the rate-limiter wait
# ---------------------------------------------------------------------------------------------


def _set_now(iso: str) -> None:
    ctime._server_offset = datetime.fromisoformat(iso).replace(tzinfo=UTC) - datetime.now(UTC)


@pytest.mark.parametrize("kind", ["single", "multi", "batch"])
async def test_no_order_post_after_deadline_even_when_admitted_before_it(
    cfg, router, server_offset_guard, kind
):
    single = router.post("/orders").mock(return_value=httpx.Response(200, json={"orderId": "o1", "status": "open"}))
    multi = router.post("/orders/multi-leg").mock(return_value=httpx.Response(200, json={"results": []}))
    batch = router.post("/orders/batch").mock(return_value=httpx.Response(200, json={"results": []}))
    clock = FakeClock()
    client = make_client(cfg, Budget(90, 25, clock=clock))
    client.budget.freeze_all(60)  # a 429 just froze everything
    _set_now("2026-11-04T16:59:10")  # before the 16:59:30 cutoff
    if kind == "single":
        coro = client.place_order(leg("k", "1"))
    elif kind == "multi":
        coro = client.place_multi_leg([leg("a", "1"), leg("b", "2")])
    else:
        coro = client.place_batch([leg("a", "1"), leg("b", "2")])
    task = asyncio.create_task(coro)
    await asyncio.sleep(0.1)
    assert not task.done()  # parked in the frozen limiter
    _set_now("2026-11-04T17:00:06")
    clock.t += 61
    with pytest.raises(ValidationError) as ei:
        await asyncio.wait_for(task, 5)
    assert ei.value.code == "DEADLINE"
    assert single.call_count == multi.call_count == batch.call_count == 0
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# F7: nested pagination is followed
# ---------------------------------------------------------------------------------------------


def test_page_items_reads_cursor_from_a_nested_wrapper():
    nested = {"data": {"orders": [1, 2], "pagination": {"nextCursor": "x", "hasMore": True}}}
    assert page_items(nested) == ([1, 2], "x", True)
    assert page_items({"data": {"items": [1], "nextCursor": "y"}}) == ([1], "y", True)
    assert page_items({"data": {"items": [1], "pagination": {"hasMore": False}}}) == ([1], None, False)
    # outer pagination still wins when present
    assert page_items({"data": {"items": [3]}, "pagination": {"cursor": "c"}}) == ([3], "c", True)


async def test_get_open_orders_follows_nested_pagination(cfg, router):
    def order_row(i: int) -> dict:
        return {"id": f"o-{i}", "exchangeId": "1", "side": "yes", "action": "buy", "price": 0.5, "quantity": 1}

    route = router.get("/orders").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "data": {
                        "orders": [order_row(i) for i in range(100)],
                        "pagination": {"nextCursor": "p2", "hasMore": True},
                    }
                },
            ),
            httpx.Response(
                200,
                json={"data": {"orders": [order_row(i) for i in range(100, 140)], "pagination": {"hasMore": False}}},
            ),
        ]
    )
    client = make_client(cfg)
    orders = await client.get_open_orders()
    assert len(orders) == 140
    assert route.call_count == 2 and route.calls[1].request.url.params["cursor"] == "p2"
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# F10 (minor): fills / positions in unrecognised shapes
# ---------------------------------------------------------------------------------------------


def test_parse_fill_reads_boolean_direction_keys():
    f = parse_fill({"exchangeId": "1", "price": 0.5, "quantity": 3, "isBuy": False, "outcome": "YES"})
    assert f is not None and f.side == "yes" and f.action == "sell"
    assert not f.model_extra.get("side_or_action_missing")
    g = parse_fill({"exchangeId": "1", "price": 0.5, "quantity": 3, "is_buy": "true", "isYes": False})
    assert g is not None and g.side == "no" and g.action == "buy"


def test_parse_fill_logs_unrecognised_shape_once(caplog):
    row = {"exchangeId": "1", "price": 0.5, "quantity": 3, "weirdDirectionKey": "x9"}
    with caplog.at_level(logging.WARNING, logger="sigcup.sig.client"):
        parse_fill(row)
        parse_fill(dict(row, price=0.6))
    hits = [r for r in caplog.records if "weirdDirectionKey" in r.getMessage()]
    assert len(hits) == 1


async def test_positions_with_yes_no_share_keys_are_kept(cfg, router, caplog):
    router.get("/tournaments/cup/portfolio/positions").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"exchangeId": "1", "yesShares": 30, "noShares": 0},
                    {"exchangeId": "2", "yesShares": 0, "noShares": 12},
                    {"exchangeId": "3", "holdingsMysteryKey": 4},
                ]
            },
        )
    )
    client = make_client(cfg)
    with caplog.at_level(logging.WARNING, logger="sigcup.sig.client"):
        positions = await client.get_positions()
    assert {p.exchange_id: p.quantity for p in positions} == {"1": 30, "2": -12}
    assert any("holdingsMysteryKey" in r.getMessage() for r in caplog.records)  # the dropped row is logged
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# F11 (minor): the server clock offset does not jitter or follow cached responses
# ---------------------------------------------------------------------------------------------


async def test_server_offset_ignores_truncation_jitter_and_cached_dates(cfg, router, server_offset_guard):
    base = datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=120)
    dates = iter([base, base - timedelta(seconds=1), base - timedelta(seconds=600)])

    def respond(request: httpx.Request) -> httpx.Response:
        when = next(dates)
        headers = {"Date": format_datetime(when, usegmt=True)}
        if request.url.params.get("cached"):
            headers["Age"] = "540"
        return httpx.Response(200, headers=headers, json={"cash": 1.0})

    router.get("/account").mock(side_effect=respond)
    client = make_client(cfg)
    ctime._server_offset = timedelta(0)
    await client.get_account()
    first = ctime.server_offset()
    assert 118 <= first.total_seconds() <= 121
    await client.get_account()  # one second "earlier": the Date header's truncation, not a clock change
    assert ctime.server_offset() >= first
    await client._get("/account", {"cached": 1})  # an edge-cached response with an old Date
    assert ctime.server_offset() >= first
    await client.aclose()


async def test_server_offset_follows_a_real_clock_change(cfg, router, server_offset_guard):
    now = datetime.now(UTC).replace(microsecond=0)
    dates = iter([now + timedelta(seconds=120), now + timedelta(seconds=20)])
    router.get("/account").mock(
        side_effect=lambda req: httpx.Response(
            200, headers={"Date": format_datetime(next(dates), usegmt=True)}, json={"cash": 1.0}
        )
    )
    client = make_client(cfg)
    ctime._server_offset = timedelta(0)
    await client.get_account()
    await client.get_account()
    assert 17 <= ctime.server_offset().total_seconds() <= 21
    await client.aclose()


# ---------------------------------------------------------------------------------------------
# F12 (minor): priorities reach the limiter
# ---------------------------------------------------------------------------------------------


async def test_order_priority_defaults_from_strategy(cfg, router):
    router.post("/orders").mock(return_value=httpx.Response(200, json={"orderId": "o1", "status": "open"}))
    router.post("/orders/multi-leg").mock(
        return_value=httpx.Response(200, json={"results": [{"orderId": "o1", "status": "open"}] * 2})
    )
    router.delete("/orders/o1").mock(return_value=httpx.Response(204))
    client = make_client(cfg)
    seen = spy_priorities(client.budget.writes)
    for strategy in ("arb", "theta", "value", "mm", "manual"):
        await client.place_order(leg("k-" + strategy, "1", strategy=strategy))
    await client.place_order(leg("k-x", "1", strategy="mm"), priority=0)  # an explicit priority wins
    await client.place_multi_leg([leg("a", "1", strategy="arb"), leg("b", "2", strategy="arb")])
    await client.cancel_order("o1")
    assert seen == [0, 1, 2, 3, 4, 0, 0, 1]
    await client.aclose()


async def test_reconciliation_reads_outrank_bulk_market_data(cfg, router):
    router.get("/account").mock(return_value=httpx.Response(200, json={"cash": 1.0}))
    router.get("/orders").mock(return_value=httpx.Response(200, json={"data": []}))
    router.get("/tournaments/cup/portfolio/positions").mock(return_value=httpx.Response(200, json={"data": []}))
    router.get("/tournaments/cup/portfolio/fills").mock(return_value=httpx.Response(200, json={"data": []}))
    router.get("/exchanges/prices").mock(return_value=httpx.Response(200, json={"data": []}))
    router.get("/exchanges/1/orderbook").mock(
        return_value=httpx.Response(200, json={"exchangeId": "1", "bids": [], "asks": []})
    )
    client = make_client(cfg)
    seen = spy_priorities(client.budget.reads)
    await client.get_account()
    await client.get_open_orders()
    await client.get_positions()
    await client.get_fills()
    await client.get_prices(["1"])
    await client.get_orderbook("1")
    assert seen == [1, 1, 1, 1, 3, 3]
    await client.aclose()


async def test_realtime_token_uses_housekeeping_priority(cfg, router):
    router.post("/realtime/token").mock(
        return_value=httpx.Response(200, json={"supabaseUrl": "https://x.supabase.co", "anonKey": "a", "token": "t"})
    )
    client = make_client(cfg)
    seen = spy_priorities(client.budget.writes)
    await client.realtime_token()
    assert seen == [4]
    await client.aclose()


def test_cancel_all_incomplete_is_a_sig_error():
    e = CancelAllIncomplete("x", 207, cancelled=1, failed=["o-2"])
    assert isinstance(e, SigError) and e.code == "CANCEL_ALL_INCOMPLETE" and json.dumps(e.failed)
