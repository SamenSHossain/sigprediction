"""GDELT circuit breaker: when GDELT (or the path to it) is unreachable, a run stops after
``BREAKER_FAILURES`` (3) consecutive connection / proxy / timeout failures instead of sending every race's
query 5.2 s apart (minutes of dead time that delayed every later collector). Logged once per run; HTTP
answers (429 / 500 / bad query) are GDELT being reachable and do not count. respx only, no network."""

from __future__ import annotations

import logging

import httpx
import pytest
import respx

from sigcup.db import engine as dbengine
from sigcup.db.repo import kv_get
from sigcup.feeds import news_gdelt
from tests.test_social_helpers import NOW, load_json, make_gazetteer

GDELT_HOST, GDELT_PATH = "api.gdeltproject.org", "/api/v2/doc/doc"
RACES_WITH_NAMES = 6  # the helper gazetteer: six races carry candidate names


def _raise(exc_type):
    def side(request: httpx.Request) -> httpx.Response:
        raise exc_type("unreachable", request=request)

    return side


@respx.mock
@pytest.mark.parametrize("exc_type", [httpx.ConnectError, httpx.ConnectTimeout, httpx.ProxyError, httpx.ReadTimeout])
async def test_gdelt_run_stops_after_three_consecutive_connection_failures(db, cfg, caplog, exc_type):
    route = respx.get(host=GDELT_HOST, path=GDELT_PATH).mock(side_effect=_raise(exc_type))
    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    with caplog.at_level(logging.INFO, logger="sigcup.feeds.news_gdelt"):
        n = await news_gdelt.run_once(dbengine.session, cfg, None, gazetteer=make_gazetteer(), sleep=sleep, now=NOW, max_requests=10)
    assert n == 0
    assert route.call_count == 3, "the run must stop once GDELT is clearly unreachable"
    assert len(sleeps) == 2  # no 5.2 s waits for queries that are never sent
    assert news_gdelt.BREAKER_FAILURES == 3
    tripped = [r for r in caplog.records if "circuit breaker" in r.getMessage()]
    assert len(tripped) == 1 and tripped[0].levelno == logging.WARNING
    with dbengine.session() as s:
        # the races never queried come first next run; nothing is recorded as covered
        assert kv_get(s, news_gdelt.KV_CURSOR) == 3
        assert not kv_get(s, news_gdelt.KV_COVERED_SINCE)


@respx.mock
async def test_gdelt_breaker_counts_only_consecutive_failures(db, cfg):
    art = load_json("gdelt_artlist.json")
    seen: list[int] = []

    def side(request: httpx.Request) -> httpx.Response:
        seen.append(len(seen))
        if len(seen) == 3:  # the third query answers: the failure streak resets
            return httpx.Response(200, json=art)
        raise httpx.ConnectError("flaky", request=request)

    respx.get(host=GDELT_HOST, path=GDELT_PATH).mock(side_effect=side)

    async def sleep(_: float) -> None:
        return None

    await news_gdelt.run_once(dbengine.session, cfg, None, gazetteer=make_gazetteer(), sleep=sleep, now=NOW, max_requests=10)
    assert len(seen) == RACES_WITH_NAMES  # 2 failures, 1 success, 3 failures: trips only on the last race
    with dbengine.session() as s:
        assert kv_get(s, news_gdelt.KV_CURSOR) == 0  # every race was tried: the rotation wraps


@respx.mock
@pytest.mark.parametrize("response", [httpx.Response(500), httpx.Response(429, text="rate limited")])
async def test_gdelt_http_errors_do_not_trip_the_breaker(db, cfg, response):
    route = respx.get(host=GDELT_HOST, path=GDELT_PATH).mock(return_value=response)

    async def sleep(_: float) -> None:
        return None

    assert await news_gdelt.run_once(dbengine.session, cfg, None, gazetteer=make_gazetteer(), sleep=sleep, now=NOW, max_requests=10) == 0
    assert route.call_count == RACES_WITH_NAMES
