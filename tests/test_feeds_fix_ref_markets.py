"""Regression tests for the reference-price collector review findings.

F2: Kalshi's 0-cent bid / 100-cent (or 0) ask mark an *empty* side, not a price; crossed books are dropped
    rather than swapped; closed / determined markets are not quoted from their (empty) books, and a
    determined market with a ``result`` is quoted at its settlement value.
F3: one slow or rate-limited source must not black out the others: sources run under their own time
    budget and each writes as soon as it finishes; a failed Kalshi chunk only fans out into per-ticker
    reads (capped) when the failure was caused by the request, never on a timeout or a 429.
Nothing touches the network (httpx.MockTransport / respx)."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Market, Race, RefPrice
from sigcup.feeds import ref_markets as RM

KALSHI = "https://kalshi.test/trade-api/v2"
GAMMA = "https://gamma.test"
CLOB = "https://clob.test"
NOW = datetime(2026, 11, 4, 3, 0, tzinfo=UTC)
CID = "0xabc0000000000000000000000000000000000000000000000000000000000001"


# ---------------------------------------------------------------------------------------------
# F2: empty / one-sided / crossed / determined Kalshi books
# ---------------------------------------------------------------------------------------------


def test_kalshi_empty_book_is_not_a_half_price():
    q = RM.parse_kalshi_market({"ticker": "W", "status": "active", "yes_bid": 0, "yes_ask": 100, "last_price": 97, "volume": 50000})
    assert q is not None
    assert q.yes_bid is None and q.yes_ask is None
    assert q.yes_mid == pytest.approx(0.97)  # falls back to the last trade, not (0 + 1) / 2


def test_kalshi_zero_ask_is_an_empty_side_not_a_crossed_book():
    q = RM.parse_kalshi_market({"ticker": "W", "status": "active", "yes_bid": 99, "yes_ask": 0})
    assert q is not None
    assert q.yes_bid == pytest.approx(0.99) and q.yes_ask is None
    assert q.yes_mid == pytest.approx(0.99)  # never swapped into bid 0.0 / ask 0.99 / mid 0.495
    with_last = RM.parse_kalshi_market({"ticker": "W", "status": "active", "yes_bid": 99, "yes_ask": 0, "last_price": 98})
    assert with_last is not None and with_last.yes_mid == pytest.approx(0.98)  # one-sided: last trade wins


def test_kalshi_empty_sides_in_dollar_fields_and_no_bid():
    q = RM.parse_kalshi_market(
        {"ticker": "W", "status": "active", "yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000", "last_price_dollars": "0.9700"}
    )
    assert q is not None and q.yes_bid is None and q.yes_ask is None and q.yes_mid == pytest.approx(0.97)
    # an empty NO bid must not become a YES ask of 1.0
    q = RM.parse_kalshi_market({"ticker": "W", "status": "active", "yes_bid": 40, "no_bid": 0})
    assert q is not None and q.yes_ask is None and q.yes_mid == pytest.approx(0.40)


def test_kalshi_crossed_book_is_dropped_not_swapped():
    q = RM.parse_kalshi_market({"ticker": "W", "status": "active", "yes_bid": 70, "yes_ask": 60, "last_price": 66})
    assert q is not None and q.yes_bid is None and q.yes_ask is None
    assert q.yes_mid == pytest.approx(0.66)


@pytest.mark.parametrize("status", ["determined", "inactive", "disputed", "amended", "closed", "settled", "finalized"])
def test_kalshi_non_trading_statuses_are_not_open(status):
    q = RM.parse_kalshi_market({"ticker": "W", "status": status, "yes_bid": 0, "yes_ask": 0, "last_price": 99})
    assert q is not None and not q.is_open


def test_kalshi_determined_market_uses_its_result():
    yes = RM.parse_kalshi_market({"ticker": "W", "status": "determined", "result": "yes", "yes_bid": 0, "yes_ask": 0, "last_price": 99, "volume": 50000})
    assert yes is not None and yes.yes_mid == 1.0 and yes.yes_bid is None and yes.yes_ask is None
    no = RM.parse_kalshi_market({"ticker": "L", "status": "settled", "result": "no", "yes_bid": 0, "yes_ask": 100, "last_price": 1})
    assert no is not None and no.yes_mid == 0.0
    pending = RM.parse_kalshi_market({"ticker": "P", "status": "determined", "result": "", "yes_bid": 0, "yes_ask": 0, "last_price": 99})
    assert pending is not None and pending.yes_mid != 1.0 and pending.extra.get("settled_value") is None


def _legs() -> list[RM.RefLeg]:
    return [
        RM.RefLeg(market_id="1", kalshi_ticker="K-WIN"),
        RM.RefLeg(market_id="2", kalshi_ticker="K-CLOSED"),
        RM.RefLeg(market_id="3", kalshi_ticker="K-EMPTY"),
    ]


async def test_collect_skips_closed_kalshi_books_and_quotes_results(cfg):
    cfg.feeds.kalshi_base = KALSHI
    payload = {
        "markets": [
            {"ticker": "K-WIN", "status": "determined", "result": "yes", "yes_bid": 0, "yes_ask": 0, "last_price": 99, "volume": 50000},
            {"ticker": "K-CLOSED", "status": "closed", "result": "", "yes_bid": 0, "yes_ask": 100, "last_price": 97, "volume": 40000},
            {"ticker": "K-EMPTY", "status": "active", "yes_bid": 0, "yes_ask": 100, "last_price": 97, "volume": 30000},
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        quotes = await RM.collect(http, cfg, _legs(), sources=("kalshi",))
    by = {q.key: q for q in quotes}
    assert set(by) == {"K-WIN", "K-EMPTY"}  # the closed, undetermined book is not quoted
    assert by["K-WIN"].yes_mid == 1.0 and by["K-WIN"].market_id == "1"
    assert by["K-EMPTY"].yes_mid == pytest.approx(0.97) and by["K-EMPTY"].yes_bid is None


# ---------------------------------------------------------------------------------------------
# F3: no per-ticker fan-out on timeouts / 429s; capped fallback otherwise
# ---------------------------------------------------------------------------------------------

TICKERS = [f"SENATEX{i}-26-D" for i in range(120)]


async def test_kalshi_timeout_does_not_fan_out_into_per_ticker_reads(monkeypatch):
    monkeypatch.setattr(RM, "KALSHI_FALLBACK_PAUSE_S", 0.0, raising=False)
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        raise httpx.ReadTimeout("slow", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        out = await RM.fetch_kalshi(http, KALSHI, TICKERS)
    assert out == {}
    assert all(p.endswith("/markets") for p in paths), paths[:5]  # no GET /markets/{ticker}
    assert len(paths) <= 3  # at most one call per 50-ticker chunk (here: it stops at the first timeout)


async def test_kalshi_429_does_not_fan_out(monkeypatch):
    monkeypatch.setattr(RM, "KALSHI_FALLBACK_PAUSE_S", 0.0, raising=False)
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        out = await RM.fetch_kalshi(http, KALSHI, TICKERS)
    assert out == {}
    assert paths == ["/trade-api/v2/markets"]  # rate limited: stop, do not multiply the requests


async def test_kalshi_bad_request_fallback_is_capped(monkeypatch):
    monkeypatch.setattr(RM, "KALSHI_FALLBACK_PAUSE_S", 0.0, raising=False)
    per_ticker: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/markets"):
            return httpx.Response(400)  # e.g. one malformed ticker poisons the batch
        t = request.url.path.rsplit("/", 1)[-1]
        per_ticker.append(t)
        return httpx.Response(200, json={"market": {"ticker": t, "status": "active", "yes_bid": 40, "yes_ask": 42}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        out = await RM.fetch_kalshi(http, KALSHI, TICKERS)
    assert 0 < len(per_ticker) <= 10  # capped (was: one GET per ticker, 120 of them)
    assert set(out) == set(per_ticker)


async def test_kalshi_fallback_stops_at_the_first_per_ticker_timeout(monkeypatch):
    monkeypatch.setattr(RM, "KALSHI_FALLBACK_PAUSE_S", 0.0, raising=False)
    per_ticker: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/markets"):
            return httpx.Response(503)
        per_ticker.append(request.url.path)
        raise httpx.ConnectTimeout("down", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        out = await RM.fetch_kalshi(http, KALSHI, TICKERS[:5])
    assert out == {} and len(per_ticker) == 1


# ---------------------------------------------------------------------------------------------
# F3: a hung source does not stop the healthy ones from being written
# ---------------------------------------------------------------------------------------------


def _seed() -> None:
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="Texas Senate"))
        s.add(
            Market(
                market_id="101", exchange_id="1101", race_id="SEN-TX", party="D", title="TX D", status="open",
                kalshi_ticker="SENATETX-26-D", polymarket_condition_id=CID, polymarket_token_id="111",
            )
        )


GAMMA_TX = [
    {
        "id": "501234", "question": "Will the Democratic Party win the Texas Senate race in 2026?", "conditionId": CID,
        "outcomes": "[\"Yes\", \"No\"]", "outcomePrices": "[\"0.665\", \"0.335\"]", "clobTokenIds": "[\"111\", \"222\"]",
        "bestBid": 0.66, "bestAsk": 0.67, "volumeNum": 1830000.25, "active": True, "closed": False,
    }
]


async def test_hung_kalshi_does_not_black_out_polymarket(db, cfg, monkeypatch):
    """Kalshi hangs: the loop's run timeout used to cancel the whole pass before anything was written."""
    _seed()
    cfg.feeds.kalshi_base = KALSHI
    cfg.feeds.polymarket_gamma = GAMMA
    cfg.feeds.polymarket_clob = CLOB
    monkeypatch.setattr(RM, "MIN_SOURCE_BUDGET_S", 0.2, raising=False)
    monkeypatch.setattr(RM, "SOURCE_BUDGET_FRACTION", 0.0, raising=False)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "kalshi.test":
            await asyncio.sleep(30)  # an election-night Kalshi under load
            return httpx.Response(200, json={"markets": []})
        if request.url.host == "gamma.test":
            return httpx.Response(200, json=GAMMA_TX)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        try:
            # stands in for loop.run_collector's wait_for(run_timeout_s(...))
            await asyncio.wait_for(RM.run_once(dbengine.session, cfg, None, http=http, now=NOW), timeout=2.0)
        except TimeoutError:
            pass
    with dbengine.session() as s:
        rows = s.execute(select(RefPrice)).scalars().all()
    assert [(r.source, r.key) for r in rows] == [("polymarket", CID)]
    assert rows[0].yes_mid == pytest.approx(0.665)


def test_source_budget_follows_the_ref_cadence(cfg):
    normal = RM.source_budget_s(cfg, datetime(2026, 10, 10, 12, 0, tzinfo=UTC))
    night = RM.source_budget_s(cfg, NOW)
    assert night <= normal
    assert normal < 2 * cfg.feeds.ref_cadence_s  # always inside the loop's run timeout
    assert night >= RM.MIN_SOURCE_BUDGET_S
