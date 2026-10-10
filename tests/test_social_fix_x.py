"""Regression tests for the X collector: one failing race must not stall the whole round-robin (F1)."""

from __future__ import annotations

from typing import Any

import httpx
import respx

from sigcup.db import engine as dbengine
from sigcup.db.repo import kv_get, kv_set
from sigcup.feeds import social_x
from sigcup.settings import Secrets
from tests.test_social_helpers import NOW, make_gazetteer

# make_gazetteer() races with a query, in run order: GOV-GA, HOUSE-NY-03, HOUSE-PA-08, SEN-AR, SEN-GA, SEN-TX
FIRST_RACE = "GOV-GA"


def official() -> Secrets:
    return Secrets(_env_file=None, x_vendor="official", x_vendor_key="tok")


def empty_ok() -> httpx.Response:
    return httpx.Response(200, json={"data": [], "meta": {"result_count": 0}})


def queried_races(seen: list[httpx.Request]) -> list[str]:
    g = make_gazetteer()
    by_query = {social_x.build_query(g, rid): rid for rid in g.race_ids()}
    return [by_query[r.url.params["query"]] for r in seen]


@respx.mock
async def test_x_stale_since_id_400_is_cleared_and_the_queue_moves_on(db, cfg):
    """X answers 400 'Invalid since_id' once a stored since_id falls out of the 7-day window."""
    seen: list[httpx.Request] = []

    def side(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "since_id" in request.url.params:
            return httpx.Response(400, json={"title": "Invalid Request", "detail": "Invalid since_id"})
        return empty_ok()

    respx.get(host="api.x.com").mock(side_effect=side)
    with dbengine.session() as s:
        kv_set(s, social_x.KV_SINCE_PREFIX + FIRST_RACE, "1000000000000000001")

    for _ in range(3):
        await social_x.run_once(dbengine.session, cfg, official(), gazetteer=make_gazetteer(), now=NOW, max_queries=2)

    races = queried_races(seen)
    assert set(races) >= {FIRST_RACE, "HOUSE-NY-03", "HOUSE-PA-08", "SEN-AR", "SEN-GA", "SEN-TX"}
    with dbengine.session() as s:
        assert kv_get(s, social_x.KV_CURSOR) == 0  # 3 runs x 2 races over 6 races: wrapped once
        assert kv_get(s, social_x.KV_SINCE_PREFIX + FIRST_RACE) is None  # the stale since_id was dropped
    # the stale since_id is retried without it in the same run, never sent again
    assert sum(1 for r in seen if "since_id" in r.url.params) == 1


@respx.mock
async def test_x_persistently_rejected_query_is_skipped_not_head_of_line(db, cfg):
    seen: list[httpx.Request] = []

    def side(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "Burt Jones" in request.url.params["query"]:  # GOV-GA's query is rejected every time
            return httpx.Response(400, json={"title": "Invalid Request"})
        return empty_ok()

    respx.get(host="api.x.com").mock(side_effect=side)
    for _ in range(3):
        await social_x.run_once(dbengine.session, cfg, official(), gazetteer=make_gazetteer(), now=NOW, max_queries=2)

    races = queried_races(seen)
    assert races.count(FIRST_RACE) == 1
    assert set(races) == {FIRST_RACE, "HOUSE-NY-03", "HOUSE-PA-08", "SEN-AR", "SEN-GA", "SEN-TX"}


@respx.mock
async def test_x_auth_and_server_errors_still_stop_the_cycle(db, cfg):
    for status in (401, 403, 500, 503):
        route = respx.get(host="api.x.com").mock(return_value=httpx.Response(status))
        before = route.call_count
        assert await social_x.run_once(dbengine.session, cfg, official(), gazetteer=make_gazetteer(), now=NOW) == 0
        assert route.call_count - before == 1, status
        with dbengine.session() as s:
            assert kv_get(s, social_x.KV_CURSOR) == 0


@respx.mock
async def test_x_network_error_stops_the_cycle(db, cfg):
    calls: list[Any] = []

    def boom(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ConnectError("blocked")

    respx.get(host="api.x.com").mock(side_effect=boom)
    assert await social_x.run_once(dbengine.session, cfg, official(), gazetteer=make_gazetteer(), now=NOW) == 0
    assert len(calls) == 1
