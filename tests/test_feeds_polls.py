"""Ratings / generic-ballot / FEC parsers over synthetic fixtures, the manual ratings loader, the
idempotent writers and a respx-mocked run_once. Nothing touches the network."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy import select

from sigcup.common.time import ensure_utc
from sigcup.db import engine as dbengine
from sigcup.db.models import GenericBallot, Poll, Race, Rating
from sigcup.db.repo import kv_get
from sigcup.feeds import polls as P
from sigcup.feeds.universe import load_races_yaml
from sigcup.settings import Secrets

FIX = Path(__file__).parent / "fixtures" / "feeds"
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
PROB = {"safe": 0.97, "likely": 0.85, "lean": 0.68, "tilt": 0.58, "tossup": 0.50}


def html(name: str) -> str:
    return (FIX / name).read_text()


def as_map(rows: list[P.RatingRow]) -> dict[tuple[str, str], P.RatingRow]:
    return {(r.race_id, r.source): r for r in rows}


# ---------------------------------------------------------------------------------------------
# HTML blocks
# ---------------------------------------------------------------------------------------------


def test_extract_blocks_headings_tables_lists_and_scripts():
    blocks = P.extract_blocks(html("sabato_rating_changes.html"))
    tables = [b for b in blocks if isinstance(b, P.HtmlTable)]
    lists = [b for b in blocks if isinstance(b, P.HtmlList)]
    assert [t.heading for t in tables] == ["Senate", "Governor", "House"]  # the <script> table is ignored
    assert tables[0].header == ["Date", "Race", "Old Rating", "New Rating"]
    assert tables[0].body[0].cells == ["10/6/2026", "Alaska Senate", "Toss-up", "Leans Democratic"]
    assert len(tables[0].body) == 4 and tables[0].ncols == 4
    assert any("Home" in lst.items for lst in lists)
    assert "Georgia Senate" not in P.html_text(html("sabato_rating_changes.html"))


# ---------------------------------------------------------------------------------------------
# ratings parsers
# ---------------------------------------------------------------------------------------------


def test_sabato_rating_changes_keeps_newest_per_race_with_row_dates():
    rows = as_map(P.parse_sabato_rating_changes(html("sabato_rating_changes.html"), ts=NOW, ratings_prob=PROB))
    assert set(rows) == {
        ("SEN-AK", "sabato"), ("SEN-SC", "sabato"), ("SEN-KS", "sabato"), ("GOV-AK", "sabato"), ("GOV-OR", "sabato"),
        ("HOUSE-PA-08", "sabato"), ("HOUSE-NE-02", "sabato"),
    }
    ak = rows[("SEN-AK", "sabato")]
    assert ak.rating == "Lean D" and ak.prob_dem == pytest.approx(0.68) and ak.ts == datetime(2026, 10, 6, tzinfo=UTC)
    assert rows[("SEN-SC", "sabato")].rating == "Likely R" and rows[("SEN-SC", "sabato")].prob_dem == pytest.approx(0.15)
    assert rows[("SEN-KS", "sabato")].ts == datetime(2026, 9, 8, tzinfo=UTC)
    assert rows[("HOUSE-PA-08", "sabato")].rating == "Toss-up"
    assert rows[("HOUSE-NE-02", "sabato")].rating == "Lean D"
    assert all(r.source == "sabato" for r in rows.values())  # the 'Old Rating' column is never used


def test_sabato_ratings_page_lists_under_rating_headings():
    rows = as_map(P.parse_sabato_ratings(html("sabato_senate.html"), "senate", ts=NOW, ratings_prob=PROB))
    expect = {
        "SEN-CO": "Safe D", "SEN-DE": "Safe D", "SEN-GA": "Likely D", "SEN-MN": "Likely D", "SEN-AK": "Lean D",
        "SEN-NH": "Lean D", "SEN-IA": "Toss-up", "SEN-ME": "Toss-up", "SEN-TX": "Toss-up", "SEN-OH": "Toss-up",
        "SEN-KS": "Lean R", "SEN-SC": "Likely R", "SEN-FL": "Likely R", "SEN-WY": "Safe R",
    }
    assert {rid: r.rating for (rid, _s), r in rows.items()} == expect
    assert rows[("SEN-WY", "sabato")].prob_dem == pytest.approx(0.03)
    assert ("GOV-GA", "sabato") not in rows  # the 'More from' nav list carries no rating heading


def test_cook_senate_category_tables_and_house_rating_column():
    sen = as_map(P.parse_cook_ratings(html("cook_senate.html"), "senate", ts=NOW, ratings_prob=PROB))
    tossups = {rid for (rid, _), r in sen.items() if r.rating == "Toss-up"}
    assert tossups == {"SEN-AK", "SEN-IA", "SEN-KS", "SEN-ME", "SEN-MI", "SEN-OH", "SEN-TX"}
    assert sen[("SEN-NH", "cook")].rating == "Lean D"
    assert {rid for (rid, _), r in sen.items() if r.rating == "Likely D"} == {"SEN-GA", "SEN-MN", "SEN-NC"}
    assert sen[("SEN-CO", "cook")].rating == "Safe D" and sen[("SEN-WY", "cook")].rating == "Safe R"
    assert len(sen) == 15
    house = as_map(P.parse_cook_ratings(html("cook_house.html"), "house", ts=NOW, ratings_prob=PROB))
    assert {rid: r.rating for (rid, _), r in house.items()} == {
        "HOUSE-AZ-01": "Toss-up", "HOUSE-AZ-06": "Toss-up", "HOUSE-PA-08": "Toss-up", "HOUSE-CA-22": "Toss-up",
        "HOUSE-IA-01": "Lean D", "HOUSE-WA-03": "Lean D", "HOUSE-NY-13": "Safe D", "HOUSE-TN-05": "Safe R",
    }
    assert house[("HOUSE-TN-05", "cook")].prob_dem == pytest.approx(0.03)


def test_rttwh_table_ignores_probability_and_date_columns():
    rows = as_map(P.parse_rttwh_ratings(html("rttwh_senate.html"), "senate", ts=NOW, ratings_prob=PROB))
    assert {rid: r.rating for (rid, _), r in rows.items()} == {
        "SEN-AK": "Tilt D", "SEN-TX": "Toss-up", "SEN-GA": "Likely D", "SEN-NE": "Lean R", "SEN-FL": "Likely R",
    }
    assert rows[("SEN-AK", "rttwh")].prob_dem == pytest.approx(0.58) and rows[("SEN-AK", "rttwh")].ts == NOW


def test_270towin_consensus_one_row_per_rater_column():
    rows = as_map(P.parse_270towin_table(html("270towin_senate_consensus.html"), "senate", ts=NOW, ratings_prob=PROB))
    assert {s for (_, s) in rows} == {"cook", "inside", "sabato", "split", "270towin"}
    assert rows[("SEN-AK", "cook")].rating == "Toss-up"
    assert rows[("SEN-AK", "sabato")].rating == "Lean D"
    assert rows[("SEN-AK", "split")].rating == "Tilt D"
    assert rows[("SEN-TX", "inside")].rating == "Tilt R" and rows[("SEN-TX", "inside")].prob_dem == pytest.approx(0.42)
    assert rows[("SEN-GA", "270towin")].rating == "Likely D"
    mirror = as_map(P.parse_270towin_table(html("rttwh_senate.html"), "senate", ts=NOW, ratings_prob=PROB, source="split"))
    assert all(s == "split" for (_, s) in mirror)


def test_parse_ratings_html_filters_races_and_tolerates_junk():
    rows = P.parse_ratings_html(html("cook_senate.html"), source="cook", chamber="senate", ts=NOW, ratings_prob=PROB, races={"SEN-TX"})
    assert [(r.race_id, r.rating) for r in rows] == [("SEN-TX", "Toss-up")]
    assert P.parse_ratings_html("<html><body><p>no tables</p></body></html>", source="x", chamber=None, ts=NOW, ratings_prob=PROB) == []
    assert P.parse_ratings_html("<table><tr><td>Lean D</td></tr>", source="x", chamber="senate", ts=NOW, ratings_prob=PROB) == []
    assert P.parse_ratings_html("", source="x", chamber=None, ts=NOW, ratings_prob=PROB) == []


def test_source_from_header_and_dates():
    assert P.source_from_header("Inside Elections") == "inside"
    assert P.source_from_header("Sabato's Crystal Ball") == "sabato"
    assert P.source_from_header("Rating") is None
    assert P.parse_date("10/6/2026") == datetime(2026, 10, 6, tzinfo=UTC)
    assert P.parse_date("October 6, 2026") == datetime(2026, 10, 6, tzinfo=UTC)
    assert P.parse_date("9/16 - 10/5", NOW) == NOW
    assert P.parse_date("garbage", None) is None


# ---------------------------------------------------------------------------------------------
# generic ballot and FEC
# ---------------------------------------------------------------------------------------------


def test_generic_ballot_rcp_average_row_and_fiftyplusone_card():
    rcp = P.parse_generic_ballot(html("rcp_generic_ballot.html"), source="rcp", ts=NOW)
    assert rcp is not None and (rcp.margin, rcp.dem, rcp.rep, rcp.ts) == (7.9, 49.6, 41.7, NOW)
    fpo = P.parse_generic_ballot(html("fiftyplusone_generic.html"), source="fiftyplusone", ts=NOW)
    assert fpo is not None and fpo.margin == 7.8 and fpo.dem == 51.1 and fpo.rep == 43.3
    assert P.parse_generic_ballot("<p>Republicans +2.5 in our average</p>", source="x", ts=NOW).margin == -2.5
    assert P.parse_generic_ballot("<p>nothing here</p>", source="x", ts=NOW) is None


def test_fec_totals_parsing_and_grouping():
    totals = P.parse_fec_totals(json.loads((FIX / "fec_candidate_totals.json").read_text()))
    assert [t.race_id for t in totals] == ["SEN-TX", "SEN-TX", "SEN-TX", "HOUSE-PA-08", "HOUSE-AK-AL", None]
    assert totals[0].party == "D" and totals[1].party == "R" and totals[0].cash_on_hand == 6500000.0
    by_race = P.fec_totals_by_race(totals)
    assert by_race["SEN-TX"]["D"]["name"] == "TALARICO, JAMES"  # best funded Democrat wins over MINOR
    assert by_race["SEN-TX"]["R"]["receipts"] == 15000000.0
    assert by_race["HOUSE-PA-08"]["R"]["candidate_id"] == "H6PA08001"
    assert "HOUSE-AK-AL" in by_race
    assert set(P.fec_totals_by_race(totals, {"SEN-TX"})) == {"SEN-TX"}
    assert P.fec_race_id("H", "pa", 8) == "HOUSE-PA-08" and P.fec_race_id("P", "US", None) is None
    assert P.parse_fec_totals({"results": "nope"}) == []


# ---------------------------------------------------------------------------------------------
# manual loader
# ---------------------------------------------------------------------------------------------


def test_shipped_manual_ratings_cover_the_universe():
    rows = P.load_manual_ratings(ratings_prob=PROB, now=NOW)
    universe = set(load_races_yaml()["races"])
    assert len(rows) >= 100
    assert {r.race_id for r in rows} <= universe
    by = as_map(rows)
    assert by[("SEN-TX", "cook")].rating == "Toss-up" and by[("SEN-TX", "cook")].ts == datetime(2026, 10, 6, tzinfo=UTC)
    assert by[("SEN-TX", "inside")].rating == "Tilt R" and by[("SEN-TX", "inside")].prob_dem == pytest.approx(0.42)
    assert by[("CONTROL-SENATE", "manual")].prob_dem == pytest.approx(0.63)  # explicit override
    assert by[("CONTROL-HOUSE", "manual")].prob_dem == pytest.approx(0.91)
    assert by[("SEN-WY", "manual")].rating == "Safe R" and by[("SEN-WY", "manual")].ts == datetime(2026, 10, 9, tzinfo=UTC)
    assert all(0.0 <= r.prob_dem <= 1.0 for r in rows)
    # every Senate / Governor / House race in the universe has at least one manual fallback rating
    rated = {r.race_id for r in rows}
    missing = {rid for rid in universe if rid not in rated}
    assert not missing, missing


def test_manual_loader_edge_cases(tmp_path):
    p = tmp_path / "ratings_manual.yaml"
    p.write_text(
        "as_of: 2026-10-01\n"
        "ratings:\n"
        "  - {race_id: SEN-TX, rating: 'Lean I'}\n"  # unmappable without prob_dem
        "  - {race_id: NOPE, rating: 'Lean D'}\n"
        "  - {race_id: sen-ga, rating: 'Likely D', as_of: 2026-09-30, source: COOK}\n"
        "  - {race_id: SEN-NE, rating: 'Weird', prob_dem: 0.2}\n"
        "  - {race_id: SEN-AK, rating: 'Toss Up', prob_dem: 1.5}\n"
        "polls:\n"
        "  - {race_id: SEN-TX, pollster: Quinnipiac, end_date: 2026-10-05, dem: 50, rep: 42, sample_size: 1200, population: LV}\n"
        "  - {race_id: SEN-TX, pollster: NoEnd, dem: 50, rep: 42}\n"
        "  - {race_id: SEN-GA, pollster: Emerson, end_date: 2026-10-01, margin: -3}\n"
    )
    rows = P.load_manual_ratings(p, ratings_prob=PROB, now=NOW)
    assert [(r.race_id, r.source, r.rating, r.prob_dem) for r in rows] == [
        ("SEN-GA", "cook", "Likely D", 0.85),
        ("SEN-NE", "manual", "Weird", 0.2),
    ]
    assert rows[0].ts == datetime(2026, 9, 30, tzinfo=UTC)
    polls = P.load_manual_polls(p)
    assert [(q.race_id, q.pollster, q.margin, q.sample_size, q.population) for q in polls] == [
        ("SEN-TX", "Quinnipiac", 8.0, 1200, "LV"),
        ("SEN-GA", "Emerson", -3.0, None, None),
    ]
    assert P.load_manual_ratings(tmp_path / "missing.yaml", ratings_prob=PROB) == []
    p.write_text("ratings: [")
    assert P.load_manual_ratings(p, ratings_prob=PROB) == []


# ---------------------------------------------------------------------------------------------
# writers
# ---------------------------------------------------------------------------------------------


def _ratings(s) -> list[tuple[str, str, str, datetime]]:
    return [(r.race_id, r.source, r.rating, ensure_utc(r.ts)) for r in s.execute(select(Rating).order_by(Rating.id)).scalars().all()]


def test_ensure_race_rows_seeds_from_yaml_once(db):
    assert P.ensure_race_rows(dbengine.session) == 119
    assert P.ensure_race_rows(dbengine.session) == 0
    with dbengine.session() as s:
        tx = s.get(Race, "SEN-TX")
        assert tx.chamber == "senate" and tx.state == "TX" and tx.exclusive is None
        assert s.get(Race, "HOUSE-PA-08").district == "08"


def test_write_ratings_is_idempotent_and_orders_by_time(db):
    P.ensure_race_rows(dbengine.session)
    known = P.known_race_ids(dbengine.session)
    r1 = P.RatingRow("SEN-TX", "cook", "Toss-up", 0.5, NOW - timedelta(days=4))
    assert P.write_ratings(dbengine.session, [r1], known_races=known) == 1
    assert P.write_ratings(dbengine.session, [r1], known_races=known) == 0  # same row again
    older = P.RatingRow("SEN-TX", "cook", "Lean R", 0.32, NOW - timedelta(days=10))
    assert P.write_ratings(dbengine.session, [older], known_races=known) == 0  # older information never overrides
    same_newer = P.RatingRow("SEN-TX", "cook", "Toss-up", 0.5, NOW - timedelta(days=1))
    assert P.write_ratings(dbengine.session, [same_newer], known_races=known) == 0  # unchanged within the refresh window
    changed = P.RatingRow("SEN-TX", "cook", "Lean D", 0.68, NOW)
    assert P.write_ratings(dbengine.session, [changed], known_races=known) == 1
    refresh = P.RatingRow("SEN-TX", "cook", "Lean D", 0.68, NOW + timedelta(days=P.RATING_REFRESH_DAYS))
    assert P.write_ratings(dbengine.session, [refresh], known_races=known) == 1  # stale confirmation re-written
    unknown = P.RatingRow("SEN-ZZ", "cook", "Lean D", 0.68, NOW)
    assert P.write_ratings(dbengine.session, [unknown], known_races=known) == 0
    other_source = P.RatingRow("SEN-TX", "sabato", "Lean D", 0.68, NOW - timedelta(days=30))
    assert P.write_ratings(dbengine.session, [other_source], known_races=known) == 1
    with dbengine.session() as s:
        rows = _ratings(s)
    assert [r[1:3] for r in rows] == [("cook", "Toss-up"), ("cook", "Lean D"), ("cook", "Lean D"), ("sabato", "Lean D")]


def test_write_generic_ballot_and_polls(db):
    P.ensure_race_rows(dbengine.session)
    g1 = P.GenericBallotRow("rcp", NOW - timedelta(hours=6), 7.9)
    assert P.write_generic_ballot(dbengine.session, [g1]) == 1
    assert P.write_generic_ballot(dbengine.session, [P.GenericBallotRow("rcp", NOW, 7.9)]) == 0  # same margin, same day
    assert P.write_generic_ballot(dbengine.session, [P.GenericBallotRow("rcp", NOW, 8.2)]) == 1
    assert P.write_generic_ballot(dbengine.session, [P.GenericBallotRow("rcp", NOW - timedelta(days=2), 1.0)]) == 0  # older
    assert P.write_generic_ballot(dbengine.session, [P.GenericBallotRow("fpo", NOW, 7.8)]) == 1
    poll = P.PollRow("SEN-TX", "Quinnipiac", NOW - timedelta(days=5), 8.0, dem=50, rep=42)
    assert P.write_polls(dbengine.session, [poll, poll], now=NOW) == 1
    assert P.write_polls(dbengine.session, [poll], now=NOW) == 0
    assert P.write_polls(dbengine.session, [P.PollRow("SEN-ZZ", "x", NOW, 1.0)], now=NOW) == 0
    with dbengine.session() as s:
        assert [g.margin for g in s.execute(select(GenericBallot).order_by(GenericBallot.id)).scalars()] == [7.9, 8.2, 7.8]
        p = s.execute(select(Poll)).scalar_one()
        assert ensure_utc(p.fetched_at) == NOW and p.margin == 8.0


# ---------------------------------------------------------------------------------------------
# run_once
# ---------------------------------------------------------------------------------------------


async def test_run_once_offline_writes_manual_fallback(db, cfg):
    n = await P.run_once(dbengine.session, cfg, None, now=NOW, fetch_web=False, fetch_fec=False)
    assert n >= 100
    with dbengine.session() as s:
        assert s.execute(select(Race.race_id)).scalars().all()  # races seeded from yaml
        latest = s.execute(select(Rating).where(Rating.race_id == "SEN-TX", Rating.source == "cook")).scalars().all()
        assert [r.rating for r in latest] == ["Toss-up"]
    assert await P.run_once(dbengine.session, cfg, None, now=NOW, fetch_web=False, fetch_fec=False) == 0


async def test_run_once_with_mocked_web_sources(db, cfg, tmp_path):
    manual = tmp_path / "ratings_manual.yaml"
    manual.write_text("as_of: 2026-10-01\nratings:\n  - {race_id: SEN-TX, rating: 'Lean R', source: cook}\n")
    sources = [
        P.RatingSource("sabato_changes", "https://feeds.test/sabato-changes", "sabato", None),
        P.RatingSource("cook_senate", "https://feeds.test/cook-senate", "cook", "senate"),
        P.RatingSource("dead", "https://feeds.test/dead", "rttwh", "senate"),
    ]
    generic = [("rcp", "https://feeds.test/rcp"), ("fiftyplusone", "https://feeds.test/fpo")]
    with respx.mock(assert_all_called=True) as router:
        router.get("https://feeds.test/sabato-changes").mock(return_value=httpx.Response(200, text=html("sabato_rating_changes.html")))
        router.get("https://feeds.test/cook-senate").mock(return_value=httpx.Response(200, text=html("cook_senate.html")))
        router.get("https://feeds.test/dead").mock(side_effect=httpx.ConnectError("blocked"))
        router.get("https://feeds.test/rcp").mock(return_value=httpx.Response(403))
        router.get("https://feeds.test/fpo").mock(return_value=httpx.Response(200, text=html("fiftyplusone_generic.html")))
        router.get(f"{P.FEC_BASE}/candidates/totals/").mock(return_value=httpx.Response(200, json=json.loads((FIX / "fec_candidate_totals.json").read_text())))
        n = await P.run_once(
            dbengine.session, cfg, Secrets(_env_file=None, fec_api_key="k-SECRET"), now=NOW,
            rating_sources=sources, generic_sources=generic, manual_path=manual,
        )
        fec_calls = router.get(f"{P.FEC_BASE}/candidates/totals/").calls
    assert fec_calls and all(c.request.url.params["api_key"] == "k-SECRET" for c in fec_calls)
    with dbengine.session() as s:
        rows = _ratings(s)
        tx = [(src, rating) for rid, src, rating, _ in rows if rid == "SEN-TX"]
        # manual Lean R (Oct 1) then Cook's live Toss-up (Oct 10): newer and different
        assert tx == [("cook", "Lean R"), ("cook", "Toss-up")]
        assert ("SEN-AK", "sabato", "Lean D", datetime(2026, 10, 6, tzinfo=UTC)) in rows
        gb = s.execute(select(GenericBallot)).scalars().all()
        assert [(g.source, g.margin) for g in gb] == [("fiftyplusone", 7.8)]
        fec = kv_get(s, P.KV_FEC_TOTALS)
        assert fec["races"]["SEN-TX"]["D"]["name"] == "TALARICO, JAMES" and "HOUSE-AK-AL" not in fec["races"]
        assert kv_get(s, P.KV_FEC_LAST_RUN)["ts"] == NOW.isoformat()
    assert n == 1 + 7 + 15 - 1 + 1 + 2  # manual + sabato changes + cook senate (TX already Lean R -> Toss-up counted once) + generic + fec races
    assert not P.fec_due(dbengine.session, NOW + timedelta(hours=1))
    assert P.fec_due(dbengine.session, NOW + P.FEC_MIN_INTERVAL)
