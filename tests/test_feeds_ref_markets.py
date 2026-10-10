"""Reference-price collector: Kalshi / Polymarket / PredictIt payload parsing from fixtures and a
respx-mocked run_once writing RefPrice rows. Nothing touches the network."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy import select

from sigcup.common.time import ensure_utc
from sigcup.db import engine as dbengine
from sigcup.db.models import Market, Race, RefPrice
from sigcup.feeds import ref_markets as RM

FIX = Path(__file__).parent / "fixtures" / "feeds"
KALSHI = "https://kalshi.test/trade-api/v2"
GAMMA = "https://gamma.test"
CLOB = "https://clob.test"
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def fixture(name: str):
    return json.loads((FIX / name).read_text())


# ---------------------------------------------------------------------------------------------
# parsers
# ---------------------------------------------------------------------------------------------


def test_parse_kalshi_markets_cents_and_dollars():
    quotes = RM.parse_kalshi_markets(fixture("kalshi_markets.json"))
    assert set(quotes) == {"SENATETX-26-D", "SENATETX-26-R", "GOVPARTYGA-26-R", "SENATEGA-24-D"}
    d = quotes["SENATETX-26-D"]
    assert (d.yes_bid, d.yes_ask, d.last, d.yes_mid) == (0.66, 0.68, 0.67, 0.67)
    assert d.liquidity == 184233 and d.title == "Texas Senate: Democratic" and d.is_open
    r = quotes["SENATETX-26-R"]
    assert r.liquidity == 90000  # volume 0 -> open interest
    ga = quotes["GOVPARTYGA-26-R"]
    assert (ga.yes_bid, ga.yes_ask, ga.last) == (0.55, 0.57, 0.56)
    assert ga.liquidity == 12000.0
    assert not quotes["SENATEGA-24-D"].is_open


def test_parse_kalshi_single_and_bare_shapes():
    single = RM.parse_kalshi_markets(fixture("kalshi_market_single.json"))
    assert list(single) == ["CONTROLS-2026-D"] and single["CONTROLS-2026-D"].yes_mid == pytest.approx(0.635)
    bare = RM.parse_kalshi_markets([{"ticker": "X-1", "yes_bid": 10}, {"no_ticker": 1}])
    assert list(bare) == ["X-1"] and bare["X-1"].yes_mid == 0.10
    assert RM.parse_kalshi_markets(None) == {} and RM.parse_kalshi_markets("junk") == {}
    noask = RM.parse_kalshi_market({"ticker": "Y", "yes_bid": 40, "no_bid": 55})
    assert noask is not None and noask.yes_ask == pytest.approx(0.45)


def test_parse_gamma_markets_json_strings_and_reversed_outcomes():
    gms = RM.parse_gamma_markets(fixture("gamma_markets.json"))
    assert [g.gamma_id for g in gms] == ["501234", "501235"]
    tx = gms[0]
    assert tx.condition_id.startswith("0xabc") and tx.yes_token_id == "111" and tx.no_token_id == "222"
    assert tx.yes_price == 0.665 and tx.best_bid == 0.66 and tx.best_ask == 0.67 and tx.volume == 1830000.25
    assert tx.event_title == "Texas Senate Election Winner 2026"
    q = tx.to_quote()
    assert q.source == "polymarket" and q.key == tx.condition_id and q.yes_mid == 0.665 and q.liquidity == 1830000.25
    ga = gms[1]
    assert ga.yes_token_id == "444" and ga.yes_price == 0.55 and ga.best_bid is None  # outcomes listed No, Yes
    assert RM.parse_gamma_markets({"data": fixture("gamma_markets.json")})[0].gamma_id == "501234"
    assert RM.parse_gamma_markets({"question": "no condition id"}) == []


def test_parse_gamma_search_attaches_event_titles():
    gms = RM.parse_gamma_search(fixture("gamma_search.json"))
    assert [g.condition_id for g in gms] == ["0x0sb", "0x0rep", "0x0pri"]
    assert all(g.event_title == "Nebraska Senate Election Winner 2026" for g in gms)
    assert gms[2].closed is True


def test_parse_clob_and_predictit():
    assert RM.parse_clob_midpoint(fixture("clob_midpoint.json")) == 0.555
    assert RM.parse_clob_price(fixture("clob_price_buy.json")) == 0.56
    assert RM.parse_clob_price(None) is None
    quotes = RM.parse_predictit_all(fixture("predictit_all.json"))
    assert set(quotes) == {"31001", "31002"}
    dem = quotes["31001"]
    assert (dem.yes_bid, dem.yes_ask, dem.last) == (0.65, 0.67, 0.66) and dem.yes_mid == pytest.approx(0.66)
    assert dem.title == "TX Senate 2026 / Democratic" and dem.liquidity is None
    rep = quotes["31002"]
    assert rep.yes_bid == 0.33 and rep.yes_ask == pytest.approx(0.36)  # ask from 1 - bestSellNoCost


def test_refquote_normalises_crossed_sides_and_mid():
    q = RM.RefQuote(source="kalshi", key="K", yes_bid=0.7, yes_ask=0.6)
    assert (q.yes_bid, q.yes_ask, q.yes_mid) == (0.6, 0.7, pytest.approx(0.65))
    assert RM.RefQuote(source="kalshi", key="K", last=0.4).yes_mid == 0.4
    assert RM.RefQuote(source="kalshi", key="K").yes_mid is None


def test_pick_and_json_list_helpers():
    assert RM.pick({"a": {"b": 1}}, "x", "a.b") == 1
    assert RM.pick({"a": None, "b": 2}, "a", "b") == 2
    assert RM.pick(None, "a", default=3) == 3
    assert RM.json_list('["0.1", "0.9"]') == ["0.1", "0.9"]
    assert RM.json_list("[0.1, 0.9") == ["0.1", "0.9"]  # tolerant of a broken string
    assert RM.json_list(None) == [] and RM.json_list(("a",)) == ["a"]
    assert RM.to_float("1.5") == 1.5 and RM.to_float(True) is None and RM.to_float("x") is None


# ---------------------------------------------------------------------------------------------
# collection against a mocked exchange + DB
# ---------------------------------------------------------------------------------------------


def _seed(db) -> None:
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="Texas Senate"))
        s.add(Race(race_id="GOV-GA", chamber="governor", state="GA", label="Georgia Governor"))
        s.add(Race(race_id="SEN-WY", chamber="senate", state="WY", label="Wyoming Senate"))
        s.add(
            Market(
                market_id="101", exchange_id="1101", race_id="SEN-TX", party="D", title="TX D", status="open",
                kalshi_ticker="SENATETX-26-D",
                polymarket_condition_id="0xabc0000000000000000000000000000000000000000000000000000000000001",
                polymarket_token_id="111", predictit_id="31001",
            )
        )
        s.add(
            Market(
                market_id="103", exchange_id="1103", race_id="GOV-GA", party="R", title="GA R", status="open",
                kalshi_ticker="GOVPARTYGA-26-R",
                polymarket_condition_id="0xdef0000000000000000000000000000000000000000000000000000000000002",
            )
        )
        s.add(Market(market_id="199", exchange_id="1199", race_id="SEN-WY", party="D", title="no crosswalk", status="open"))
        s.add(Market(market_id="198", exchange_id="1198", race_id="SEN-WY", party="R", title="settled", status="settled", kalshi_ticker="X"))


@pytest.fixture()
def cfg_feeds(cfg):
    cfg.feeds.kalshi_base = KALSHI
    cfg.feeds.polymarket_gamma = GAMMA
    cfg.feeds.polymarket_clob = CLOB
    return cfg


@pytest.fixture()
def router():
    with respx.mock(assert_all_called=False) as r:
        yield r


def _mock_all(router):
    router.get(f"{KALSHI}/markets").mock(return_value=httpx.Response(200, json=fixture("kalshi_markets.json")))
    router.get(f"{GAMMA}/markets").mock(return_value=httpx.Response(200, json=fixture("gamma_markets.json")))
    router.get(f"{CLOB}/midpoint").mock(return_value=httpx.Response(200, json=fixture("clob_midpoint.json")))
    router.get(f"{CLOB}/price", params={"side": "buy"}).mock(return_value=httpx.Response(200, json=fixture("clob_price_buy.json")))
    router.get(f"{CLOB}/price", params={"side": "sell"}).mock(return_value=httpx.Response(200, json=fixture("clob_price_sell.json")))
    router.get(RM.PREDICTIT_ALL_URL).mock(return_value=httpx.Response(200, json=fixture("predictit_all.json")))


def test_legs_from_markets_skips_closed_and_uncrosswalked(db):
    _seed(db)
    legs = RM.load_legs(dbengine.session)
    assert {leg.market_id for leg in legs} == {"101", "103"}


async def test_run_once_writes_ref_prices_for_every_source(db, cfg_feeds, router):
    _seed(db)
    _mock_all(router)
    n = await RM.run_once(dbengine.session, cfg_feeds, None, now=NOW)
    assert n == 5  # kalshi x2, polymarket x2, predictit x1
    with dbengine.session() as s:
        rows = s.execute(select(RefPrice)).scalars().all()
    by = {(r.source, r.key): r for r in rows}
    k = by[("kalshi", "SENATETX-26-D")]
    assert k.market_id == "101" and (k.yes_bid, k.yes_ask, k.yes_mid) == (0.66, 0.68, 0.67) and ensure_utc(k.ts) == NOW
    assert by[("kalshi", "GOVPARTYGA-26-R")].liquidity == 12000.0
    p = by[("polymarket", "0xabc0000000000000000000000000000000000000000000000000000000000001")]
    assert p.market_id == "101" and p.yes_mid == 0.665 and p.yes_bid == 0.66 and p.liquidity == 1830000.25
    ga = by[("polymarket", "0xdef0000000000000000000000000000000000000000000000000000000000002")]
    assert (ga.yes_bid, ga.yes_ask, ga.yes_mid) == (0.55, 0.56, 0.555)  # bid/ask from the CLOB (no bestBid in Gamma)
    assert by[("predictit", "31001")].market_id == "101"
    # the Kalshi batch call carried every ticker; the CLOB was consulted only for the leg lacking bid/ask
    kalshi_call = router.get(f"{KALSHI}/markets").calls[0]
    assert set(kalshi_call.request.url.params["tickers"].split(",")) == {"SENATETX-26-D", "GOVPARTYGA-26-R"}
    assert router.get(f"{CLOB}/midpoint").calls[0].request.url.params["token_id"] == "444"


async def test_run_once_tolerates_source_failures(db, cfg_feeds, router):
    _seed(db)
    router.get(f"{KALSHI}/markets").mock(return_value=httpx.Response(503))
    router.get(f"{KALSHI}/markets/SENATETX-26-D").mock(return_value=httpx.Response(200, json=fixture("kalshi_market_single.json") | {"market": {"ticker": "SENATETX-26-D", "yes_bid": 60, "yes_ask": 62}}))
    router.get(f"{KALSHI}/markets/GOVPARTYGA-26-R").mock(return_value=httpx.Response(404))
    router.get(f"{GAMMA}/markets").mock(side_effect=httpx.ConnectError("blocked"))
    router.get(f"{CLOB}/midpoint").mock(return_value=httpx.Response(200, json={"mid": "0.5"}))
    router.get(f"{CLOB}/price").mock(return_value=httpx.Response(500))
    router.get(RM.PREDICTIT_ALL_URL).mock(side_effect=httpx.ConnectError("gone"))
    n = await RM.run_once(dbengine.session, cfg_feeds, None, now=NOW)
    with dbengine.session() as s:
        rows = s.execute(select(RefPrice)).scalars().all()
    sources = sorted((r.source, r.key) for r in rows)
    # kalshi: per-ticker fallback found one; polymarket: the leg with a token id got a CLOB mid; predictit: nothing
    assert sources == [("kalshi", "SENATETX-26-D"), ("polymarket", "0xabc0000000000000000000000000000000000000000000000000000000000001")]
    assert n == 2


async def test_run_once_without_crosswalk_is_a_noop(db, cfg_feeds, router):
    assert await RM.run_once(dbengine.session, cfg_feeds, None) == 0
    assert not router.calls


async def test_fetch_kalshi_chunks_and_search_helpers(cfg_feeds, router):
    route = router.get(f"{KALSHI}/markets").mock(return_value=httpx.Response(200, json=fixture("kalshi_markets.json")))
    async with httpx.AsyncClient() as http:
        out = await RM.fetch_kalshi(http, KALSHI, ["SENATETX-26-D", "SENATETX-26-R", "GOVPARTYGA-26-R"], chunk=2)
        assert route.call_count == 2 and set(out) >= {"SENATETX-26-D", "SENATETX-26-R", "GOVPARTYGA-26-R"}
        router.get(f"{GAMMA}/public-search").mock(return_value=httpx.Response(200, json=fixture("gamma_search.json")))
        found = await RM.fetch_gamma_search(http, GAMMA, "Nebraska Senate 2026")
        assert [g.condition_id for g in found] == ["0x0sb", "0x0rep", "0x0pri"]
        router.get(f"{GAMMA}/markets", params={"slug": "x"}).mock(return_value=httpx.Response(200, json=[]))
        assert await RM.fetch_gamma_by_slug(http, GAMMA, "x") is None
