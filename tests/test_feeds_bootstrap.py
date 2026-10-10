"""scripts/bootstrap_races.py against a fake exchange adapter, respx-mocked Kalshi / Gamma and the
in-memory DB: title mapping, hand-edit preserving merge, API-only exclusivity, crosswalk verification,
DB upsert and the day-1 probes. Nothing touches the network."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy import select

from sigcup.common.time import ensure_utc
from sigcup.db import engine as dbengine
from sigcup.db.models import Market, Race
from sigcup.feeds import universe as U
from sigcup.settings import Secrets
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import FillInfo, MarketInfo, OrderResult, PriceSnapshot, Relationship, TournamentInfo

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures" / "feeds"
KALSHI = "https://kalshi.test/trade-api/v2"
GAMMA = "https://gamma.test"
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def _load_script():
    spec = importlib.util.spec_from_file_location("bootstrap_races", ROOT / "scripts" / "bootstrap_races.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load_script()


def sig_markets() -> list[MarketInfo]:
    return [MarketInfo.model_validate(m) for m in json.loads((FIX / "sig_markets.json").read_text())]


@dataclass
class FakeAdapter:
    """Just enough of ExchangeAdapter for the bootstrap."""

    name: str = "fake"
    tournament_id: str | None = "t-1"
    markets: list[MarketInfo] = field(default_factory=sig_markets)
    relationships: list[Relationship] = field(default_factory=list)
    fills: list[FillInfo] = field(default_factory=list)
    budget: Budget = field(default_factory=lambda: Budget(90, 25))
    placed: list = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    fail_relationships: bool = False

    async def get_tournament(self) -> TournamentInfo:
        return TournamentInfo(id="t-1", slug="cup", memberCount=1234)

    async def list_markets(self):
        return list(self.markets)

    async def get_relationships(self):
        if self.fail_relationships:
            raise RuntimeError("503")
        return list(self.relationships)

    async def get_fills(self, since=None):
        return list(self.fills)

    async def get_prices(self, ids):
        return {i: PriceSnapshot(exchange_id=i, best_bid=0.5, best_ask=0.9 if i == "1101" else 0.6) for i in ids}

    async def place_batch(self, reqs):
        self.placed.extend(reqs)
        self.budget.writes._stamps.append(self.budget.writes._clock())
        return [OrderResult(orderId=f"o-{i}", status="open") for i, _ in enumerate(reqs)]

    async def cancel_order(self, oid: str) -> bool:
        self.cancelled.append(oid)
        return True


def rel(ids: list[str], exhaustive: bool | None = True, kind: str = "mutually_exclusive", rid: str = "r") -> Relationship:
    return Relationship(id=rid, kind=kind, isExhaustive=exhaustive, exchange_ids=ids)


def fresh_universe(tmp_path) -> Path:
    """A copy of the shipped seed so hand edits can be asserted on."""
    p = tmp_path / "races.yaml"
    p.write_text((ROOT / "config" / "races.yaml").read_text())
    return p


# ---------------------------------------------------------------------------------------------
# mapping and merging
# ---------------------------------------------------------------------------------------------


def test_map_markets_table():
    mapped, unmapped = B.map_markets(sig_markets())
    assert [(m.parsed.race_id, m.parsed.party) for m in mapped] == [
        ("SEN-TX", "D"), ("SEN-TX", "R"), ("GOV-GA", "R"), ("GOV-GA", "D"), ("HOUSE-PA-08", "D"),
        ("CONTROL-SENATE", "D"), ("CONTROL-SENATE", "R"), ("SEN-NE", "I"), ("SEN-FL", "D"),
    ]
    assert [m.id for m in unmapped] == ["110"]


def test_merge_preserves_hand_edits_and_sets_exchange_fields(tmp_path):
    data = U.load_races_yaml(fresh_universe(tmp_path))
    tx = data["races"]["SEN-TX"]
    tx["pvi"] = 5.5
    tx["settlement_note"] = "hand note"
    tx["markets"]["D"]["kalshi_ticker"] = "SENATETX-26-D"
    tx["markets"]["D"]["title"] = "stale title"
    mapped, _ = B.map_markets(sig_markets())
    added, updated, dups = B.merge_markets(data, mapped, now=NOW)
    assert (added, updated, dups) == (9, 0, [])
    leg = tx["markets"]["D"]
    assert leg["sig_market_id"] == "101" and leg["sig_exchange_id"] == "1101"
    assert leg["title"] == "Will the Democratic Party win the Texas Senate race?" and leg["status"] == "open"
    assert leg["settlement_date"] == "2026-11-04T17:00:00+00:00"
    assert leg["kalshi_ticker"] == "SENATETX-26-D" and tx["pvi"] == 5.5 and tx["settlement_note"] == "hand note"
    assert data["races"]["SEN-NE"]["markets"]["I"]["sig_exchange_id"] == "1108"
    assert data["races"]["SEN-NE"]["candidates"]["I"]["name"] == "Dan Osborn"
    assert data["races"]["SEN-FL"]["special"] is True
    assert data["bootstrapped_at"] == NOW.isoformat()
    # second run with the same list: nothing added, nothing updated; a status change counts as an update
    again = B.merge_markets(data, mapped, now=NOW)
    assert again[:2] == (0, 0)
    mapped[0].market.status = "settled"
    assert B.merge_markets(data, mapped, now=NOW)[:2] == (0, 1)


def test_merge_creates_unknown_races_and_reports_duplicates(tmp_path):
    data = U.load_races_yaml(tmp_path / "empty.yaml")
    markets = [
        MarketInfo(id="1", exchangeId="e1", title="Will the Democratic Party win the Utah Senate race?"),
        MarketInfo(id="2", exchangeId="e2", title="Will the Democrats win the Utah Senate race?"),
    ]
    mapped, _ = B.map_markets(markets)
    added, updated, dups = B.merge_markets(data, mapped)
    assert added == 1 and len(dups) == 1 and "kept 1" in dups[0]
    entry = data["races"]["SEN-UT"]
    assert entry["chamber"] == "senate" and entry["pvi"] is None and entry["markets"]["D"]["sig_exchange_id"] == "e1"


# ---------------------------------------------------------------------------------------------
# relationships -> exclusivity (API evidence only)
# ---------------------------------------------------------------------------------------------


def _merged(tmp_path):
    data = U.load_races_yaml(fresh_universe(tmp_path))
    mapped, _ = B.map_markets(sig_markets())
    B.merge_markets(data, mapped)
    return data


def test_apply_relationships_full_superset_partial_and_reset(tmp_path):
    data = _merged(tmp_path)
    data["races"]["GOV-GA"]["exclusive"] = True  # a stale hand edit must not survive without evidence
    rels = [
        rel(["1101", "1102"], exhaustive=True, rid="tx"),  # exactly the TX legs
        rel(["1106", "1107", "9999"], exhaustive=True, rid="ctrl"),  # control legs + another contract
        rel(["1103"], exhaustive=False, rid="ga-partial", kind="MUTUALLY_EXCLUSIVE"),  # only one GA leg
        rel(["1105", "1101"], exhaustive=True, rid="odd", kind="sum_to_one"),  # not an exclusivity kind
    ]
    out = B.apply_relationships(data, rels)
    r = data["races"]
    assert (r["SEN-TX"]["exclusive"], r["SEN-TX"]["exhaustive"]) == (True, True)
    assert (r["CONTROL-SENATE"]["exclusive"], r["CONTROL-SENATE"]["exhaustive"]) == (True, False)  # a larger exhaustive group
    assert (r["GOV-GA"]["exclusive"], r["GOV-GA"]["exhaustive"]) == (None, None)
    assert (r["HOUSE-PA-08"]["exclusive"], r["HOUSE-PA-08"]["exhaustive"]) == (None, None)
    assert out["exclusive_races"] == 2 and out["exhaustive_races"] == 1
    assert len(out["partial"]) == 1 and out["partial"][0].startswith("GOV-GA")
    assert B.apply_relationships(data, [])["relationships"] == 0
    assert r["SEN-TX"]["exclusive"] is None  # no evidence -> null again


def test_apply_relationships_unknown_exhaustive_flag(tmp_path):
    data = _merged(tmp_path)
    B.apply_relationships(data, [rel(["1101", "1102"], exhaustive=None)])
    assert data["races"]["SEN-TX"]["exclusive"] is True and data["races"]["SEN-TX"]["exhaustive"] is None
    B.apply_relationships(data, [rel(["1101", "1102", "7"], exhaustive=None)])
    assert data["races"]["SEN-TX"]["exhaustive"] is None


# ---------------------------------------------------------------------------------------------
# crosswalk
# ---------------------------------------------------------------------------------------------


@pytest.fixture()
def router():
    with respx.mock(assert_all_called=False) as r:
        yield r


KALSHI_LIVE = {
    "SENATETX-26-D": {"ticker": "SENATETX-26-D", "title": "Texas Senate: Democratic", "status": "active", "yes_bid": 66, "yes_ask": 68},
    "SENATETX-26-R": {"ticker": "SENATETX-26-R", "title": "Texas Senate: Republican", "status": "active", "yes_bid": 32, "yes_ask": 34},
    "GOVPARTYGA-26-R": {"ticker": "GOVPARTYGA-26-R", "title": "Georgia Senate: Republican", "status": "active", "yes_bid": 50, "yes_ask": 52},  # contradicts
    "KXGOVPARTYGA-26-R": {"ticker": "KXGOVPARTYGA-26-R", "title": "Georgia Governor: Republican", "status": "active", "yes_bid": 54, "yes_ask": 56},
    "HOUSEPA8-26-D": {"ticker": "HOUSEPA8-26-D", "title": "", "status": "closed", "result": "no"},  # closed: skipped
    "KXHOUSEPA8-26-D": {"ticker": "KXHOUSEPA8-26-D", "title": "PA-8: Democratic", "status": "active", "yes_bid": 50, "yes_ask": 52},
    "CONTROLS-2026-D": {"ticker": "CONTROLS-2026-D", "title": "Senate control: Democrats", "status": "active", "yes_bid": 63, "yes_ask": 64},
}


def _kalshi_side_effect(request: httpx.Request) -> httpx.Response:
    wanted = request.url.params.get("tickers", "").split(",")
    return httpx.Response(200, json={"markets": [KALSHI_LIVE[t] for t in wanted if t in KALSHI_LIVE]})


async def test_crosswalk_kalshi_verifies_tickers(tmp_path, router):
    router.get(f"{KALSHI}/markets").mock(side_effect=_kalshi_side_effect)
    data = _merged(tmp_path)
    data["races"]["SEN-TX"]["markets"]["R"]["kalshi_ticker"] = "HAND-SET"
    legs = {(rid, p): leg for rid, p, _e, leg in U.market_legs(data)}
    async with httpx.AsyncClient() as http:
        found, missing = await B.crosswalk_kalshi(http, KALSHI, data)
    assert legs[("SEN-TX", "D")]["kalshi_ticker"] == "SENATETX-26-D"
    assert legs[("SEN-TX", "R")]["kalshi_ticker"] == "HAND-SET"  # existing ids are kept
    assert legs[("GOV-GA", "R")]["kalshi_ticker"] == "KXGOVPARTYGA-26-R"  # the contradicting title was rejected
    assert legs[("HOUSE-PA-08", "D")]["kalshi_ticker"] == "KXHOUSEPA8-26-D"  # the closed market was skipped
    assert legs[("CONTROL-SENATE", "D")]["kalshi_ticker"] == "CONTROLS-2026-D"
    assert legs[("SEN-WY", "D")]["kalshi_ticker"] == ""  # nothing live -> left empty
    assert found == 4 and missing == 240 - 4
    # every candidate ticker travelled in batched calls of at most KALSHI_CHUNK
    for call in router.get(f"{KALSHI}/markets").calls:
        assert len(call.request.url.params["tickers"].split(",")) <= 50


GAMMA_BY_SLUG = {
    "will-the-democratic-party-win-the-texas-senate-race-in-2026": {
        "id": "1", "question": "Will the Democratic Party win the Texas Senate race in 2026?", "conditionId": "0xtxd",
        "slug": "will-the-democratic-party-win-the-texas-senate-race-in-2026", "clobTokenIds": "[\"t1\", \"t2\"]",
        "outcomePrices": "[\"0.66\", \"0.34\"]", "active": True, "closed": False,
    },
    "will-the-republican-party-win-the-texas-senate-race-in-2026": {
        "id": "2", "question": "Will the Republican Party win the Texas Senate primary in 2026?", "conditionId": "0xbad",
        "slug": "will-the-republican-party-win-the-texas-senate-race-in-2026", "active": True, "closed": False,
    },
}


def _gamma_slug_side_effect(request: httpx.Request) -> httpx.Response:
    slug = request.url.params.get("slug", "")
    return httpx.Response(200, json=[GAMMA_BY_SLUG[slug]] if slug in GAMMA_BY_SLUG else [])


async def test_crosswalk_polymarket_slug_then_search(tmp_path, router):
    router.get(f"{GAMMA}/markets").mock(side_effect=_gamma_slug_side_effect)
    search = router.get(f"{GAMMA}/public-search").mock(return_value=httpx.Response(200, json=json.loads((FIX / "gamma_search.json").read_text())))
    data = U.load_races_yaml(tmp_path / "small.yaml")
    for rid, party in (("SEN-TX", "D"), ("SEN-TX", "R"), ("SEN-NE", "I"), ("SEN-NE", "R"), ("SEN-NE", "D")):
        U.market_entry(U.race_entry(data, rid), party)
    async with httpx.AsyncClient() as http:
        found, missing = await B.crosswalk_polymarket(http, GAMMA, data, pause_s=0)
    legs = {(rid, p): leg for rid, p, _e, leg in U.market_legs(data)}
    txd = legs[("SEN-TX", "D")]
    assert (txd["polymarket_condition_id"], txd["polymarket_token_id"], txd["polymarket_gamma_id"]) == ("0xtxd", "t1", "1")
    assert txd["polymarket_slug"] == "will-the-democratic-party-win-the-texas-senate-race-in-2026"
    assert legs[("SEN-TX", "R")]["polymarket_condition_id"] == ""  # the slug hit was a primary market: rejected
    assert legs[("SEN-NE", "I")]["polymarket_condition_id"] == "0x0sb"  # found through search (Osborn)
    assert legs[("SEN-NE", "R")]["polymarket_condition_id"] == "0x0rep"
    assert legs[("SEN-NE", "D")]["polymarket_condition_id"] == ""  # no Democratic contract in the results
    assert (found, missing) == (3, 2)
    assert search.call_count == 2  # one search per distinct query (Texas, Nebraska); cached across the NE legs


# ---------------------------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------------------------


def test_upsert_db_writes_races_and_markets(db, tmp_path):
    data = _merged(tmp_path)
    data["races"]["SEN-TX"]["markets"]["D"]["kalshi_ticker"] = "SENATETX-26-D"
    data["races"]["SEN-TX"]["markets"]["D"]["polymarket_condition_id"] = "0xtxd"
    data["races"]["SEN-TX"]["markets"]["D"]["polymarket_token_id"] = "t1"
    B.apply_relationships(data, [rel(["1101", "1102"], exhaustive=True)])
    n_races, n_markets = B.upsert_db(dbengine.session, data, now=NOW, resolutions={"1101": "YES follows the party"})
    assert (n_races, n_markets) == (119, 9)
    with dbengine.session() as s:
        tx = s.get(Race, "SEN-TX")
        assert (tx.exclusive, tx.exhaustive, tx.state, tx.chamber) == (True, True, "TX", "senate")
        assert s.get(Race, "GOV-GA").exclusive is None
        m = s.get(Market, "101")
        assert (m.exchange_id, m.race_id, m.party, m.kalshi_ticker, m.polymarket_condition_id, m.polymarket_token_id) == (
            "1101", "SEN-TX", "D", "SENATETX-26-D", "0xtxd", "t1",
        )
        assert ensure_utc(m.settlement_date) == datetime(2026, 11, 4, 17, 0, tzinfo=UTC) and m.resolution_text == "YES follows the party"
        assert ensure_utc(m.updated_at) == NOW
        assert s.get(Market, "108").party == "I" and s.get(Market, "108").race_id == "SEN-NE"
        assert s.execute(select(Market).where(Market.race_id == "SEN-WY")).scalars().all() == []  # no SIG id yet
    # re-run: update in place (no duplicate rows), status change applied
    data["races"]["SEN-TX"]["markets"]["D"]["status"] = "settled"
    B.upsert_db(dbengine.session, data, now=NOW)
    with dbengine.session() as s:
        assert len(s.execute(select(Market)).scalars().all()) == 9
        assert s.get(Market, "101").status == "settled"


# ---------------------------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------------------------


async def test_probe_fees_and_relationship_graph(tmp_path):
    fake = FakeAdapter()
    assert "inconclusive" in (await B.probe_fees(fake))["verdict"]
    fake.fills = [
        FillInfo(fill_id="f1", exchange_id="1101", ts=NOW, side="yes", action="buy", price=0.5, quantity=10, fee=0.0),
        FillInfo(fill_id="f2", exchange_id="1101", ts=NOW, side="yes", action="buy", price=0.5, quantity=10),
    ]
    assert (await B.probe_fees(fake))["verdict"] == "zero fees on all 2 sampled fills"
    fake.fills.append(FillInfo(fill_id="f3", exchange_id="1101", ts=NOW, side="yes", action="buy", price=0.5, quantity=10, fee=0.25))
    out = await B.probe_fees(fake)
    assert out["verdict"].startswith("FEES PRESENT: 1 of 3") and out["total_fee"] == 0.25
    data = _merged(tmp_path)
    assert B.probe_relationship_graph([], data)["verdict"].startswith("EMPTY")
    B.apply_relationships(data, [rel(["1101", "1102"])])
    out = B.probe_relationship_graph([rel(["1101", "1102"])], data)
    assert out["races_exclusive"] == 1 and out["races_exhaustive"] == 1 and "1 groups" in out["verdict"]


async def test_probe_batch_writes_dry_and_executed():
    fake = FakeAdapter()
    dry = await B.probe_batch_writes(fake, fake.markets, execute=False)
    assert dry["verdict"].startswith("skipped") and "procedure" in dry and fake.placed == []
    # executing places real orders: only with LIVE=1 and a trade key (the refusals: test_feeds_fix_probe_gate.py)
    live = Secrets(_env_file=None, live=True, sig_api_key="trade-key")
    out = await B.probe_batch_writes(fake, fake.markets, execute=True, now=NOW, sec=live)
    assert len(fake.placed) == 2 and fake.cancelled == ["o-0", "o-1"]
    req = fake.placed[0]
    assert (req.exchange_id, req.side, req.action, req.price, req.quantity) == ("1101", "yes", "buy", 0.005, 1)  # highest ask first
    assert req.expires_at == NOW + timedelta(seconds=120)
    assert out["orders_in_batch"] == 2 and out["cancelled"] == 2 and out["our_limiter_writes_for_batch"] == 1
    assert "inconclusive" in out["verdict"]  # the fake sends no rate-limit headers


def test_pick_probe_markets_prefers_highest_ask_and_open_status():
    ms = [
        MarketInfo(id="1", exchangeId="a", title="x", status="open"),
        MarketInfo(id="2", exchangeId="b", title="y", status="settled"),
        MarketInfo(id="3", exchangeId="c", title="z", status="open"),
        MarketInfo(id="4", title="no exchange", status="open"),
    ]
    prices = {"a": PriceSnapshot(exchange_id="a", best_ask=0.4), "c": PriceSnapshot(exchange_id="c", best_ask=0.9)}
    assert [m.id for m in B.pick_probe_markets(ms, prices)] == ["3", "1"]


# ---------------------------------------------------------------------------------------------
# the driver end to end (fake adapter, mocked crosswalk, temp yaml, in-memory DB)
# ---------------------------------------------------------------------------------------------


async def test_run_end_to_end(db, tmp_path, cfg, router, monkeypatch):
    cfg.feeds.kalshi_base = KALSHI
    cfg.feeds.polymarket_gamma = GAMMA
    router.get(f"{KALSHI}/markets").mock(side_effect=_kalshi_side_effect)
    router.get(f"{GAMMA}/markets").mock(side_effect=_gamma_slug_side_effect)
    router.get(f"{GAMMA}/public-search").mock(return_value=httpx.Response(200, json={"events": []}))
    yaml_path = fresh_universe(tmp_path)
    fake = FakeAdapter(relationships=[rel(["1101", "1102"], exhaustive=True)])
    args = B.build_parser().parse_args(["--races-yaml", str(yaml_path)])
    monkeypatch.setattr(B, "PAUSE_S", 0)
    async with httpx.AsyncClient() as http:
        report = await B.run(args, cfg=cfg, sec=Secrets(_env_file=None), adapter=fake, http=http, session_factory=dbengine.session)
    assert (report.sig_markets, report.mapped, report.member_count) == (10, 9, 1234)
    assert report.unmapped == ["110: Will turnout exceed 50% in the 2026 midterms?"]
    assert (report.relationships, report.races_exclusive, report.races_exhaustive) == (1, 1, 1)
    assert report.kalshi_found == 5 and report.polymarket_found == 1  # TX D/R, GA R, PA-08 D, Senate control D
    assert (report.db_races, report.db_markets) == (119, 9)
    assert report.probes["batch_writes"]["verdict"].startswith("skipped") and fake.placed == []
    assert report.probes["relationship_graph"]["races_exclusive"] == 1
    assert any("SEN-OH has no SIG market" in n for n in report.notes)
    text = report.text()
    assert "mapped: 9" in text and "probe fees" in text and "unmapped titles" in text
    saved = U.load_races_yaml(yaml_path)
    assert saved["races"]["SEN-TX"]["markets"]["D"]["sig_exchange_id"] == "1101"
    assert saved["races"]["SEN-TX"]["markets"]["D"]["kalshi_ticker"] == "SENATETX-26-D"
    assert saved["races"]["SEN-TX"]["markets"]["D"]["polymarket_condition_id"] == "0xtxd"
    assert saved["races"]["SEN-TX"]["exclusive"] is True and saved["races"]["GOV-GA"]["exclusive"] is None
    with dbengine.session() as s:
        assert s.get(Market, "101").resolution_text.startswith("YES follows")
        assert s.get(Race, "SEN-TX").exhaustive is True


async def test_run_dry_run_and_relationship_failure_keep_evidence(db, tmp_path, cfg):
    yaml_path = fresh_universe(tmp_path)
    data = U.load_races_yaml(yaml_path)
    data["races"]["SEN-TX"]["exclusive"] = True
    U.save_races_yaml(data, yaml_path)
    before = yaml_path.read_text()
    fake = FakeAdapter(fail_relationships=True)
    args = B.build_parser().parse_args(["--races-yaml", str(yaml_path), "--dry-run", "--skip-crosswalk"])
    report = await B.run(args, cfg=cfg, sec=Secrets(_env_file=None), adapter=fake, session_factory=dbengine.session)
    assert yaml_path.read_text() == before  # dry run writes nothing
    assert any("get_relationships failed" in n for n in report.notes) and "relationship_graph" not in report.probes
    assert any("dry run" in n for n in report.notes)
    with dbengine.session() as s:
        assert s.execute(select(Race)).scalars().all() == []


def test_make_adapter_without_key_explains(cfg):
    adapter, reason = B.make_adapter(Secrets(_env_file=None), cfg)
    assert adapter is None and "no SIG key" in reason
