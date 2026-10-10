"""Universe helpers: SIG title parsing table, race ids, crosswalk patterns, rating normalisation,
races.yaml seed invariants and round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sigcup.feeds import universe as U

SEED = Path(__file__).resolve().parent.parent / "config" / "races.yaml"


# ---------------------------------------------------------------------------------------------
# title parsing
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "race_id", "party", "special"),
    [
        ("Will the Democratic Party win the Texas Senate race?", "SEN-TX", "D", False),
        ("Will the Republican Party win the Georgia Governor race?", "GOV-GA", "R", False),
        ("Will the Democratic Party win the PA-08 House race?", "HOUSE-PA-08", "D", False),
        ("Will the Republican Party win the PA-8 House race?", "HOUSE-PA-08", "R", False),
        ("Will the Democratic Party win the AK-AL House race?", "HOUSE-AK-AL", "D", False),
        ("Will the Democratic Party control the Senate?", "CONTROL-SENATE", "D", False),
        ("Will the Republican Party control the House of Representatives?", "CONTROL-HOUSE", "R", False),
        ("Will the Democratic Party win the Florida Senate special election?", "SEN-FL", "D", True),
        ("Will the Republican Party win the Ohio Senate (Special) race?", "SEN-OH", "R", True),
        ("Will Dan Osborn (I) win the Nebraska Senate race?", "SEN-NE", "I", False),
        ("Will an Independent candidate win the Montana Senate race?", "SEN-MT", "I", False),
        ("Will an Independent win the Rhode Island Governor race?", "GOV-RI", "I", False),
        ("Will the Democratic Party win the West Virginia Senate race?", "SEN-WV", "D", False),
        ("Will the Democrats win the North Carolina Senate race?", "SEN-NC", "D", False),
        ("Will the GOP win the New Hampshire Governor race?", "GOV-NH", "R", False),
        ("Will the Democratic Party win Pennsylvania's 8th congressional district?", "HOUSE-PA-08", "D", False),
    ],
)
def test_parse_title_table(title, race_id, party, special):
    p = U.parse_title(title)
    assert p is not None, title
    assert (p.race_id, p.party, p.special) == (race_id, party, special)


@pytest.mark.parametrize(
    "title",
    [
        "Will turnout exceed 50% in the 2026 midterms?",
        "Will the Democratic Party win the Texas Senate primary?",
        "Will the Democratic Party win the Texas Senate race by more than 5 points?",
        "Will the Democratic Party win the House race?",  # no district: ambiguous
        "Will Jane Doe win the Texas Senate race?",  # no party, unknown candidate
        "",
    ],
)
def test_parse_title_rejects(title):
    assert U.parse_title(title) is None


def test_parse_title_candidate_name_kept():
    p = U.parse_title("Will Dan Osborn (I) win the Nebraska Senate race?")
    assert p is not None and p.candidate == "Dan Osborn" and p.label == "Nebraska Senate"


# ---------------------------------------------------------------------------------------------
# race ids and references
# ---------------------------------------------------------------------------------------------


def test_make_and_parse_race_id_roundtrip():
    for rid in ("SEN-TX", "GOV-GA", "HOUSE-PA-08", "HOUSE-AK-AL", "CONTROL-SENATE", "CONTROL-HOUSE"):
        key = U.parse_race_id(rid)
        assert key is not None and key.race_id == rid
    assert U.make_race_id("house", "pa", 8) == "HOUSE-PA-08"
    assert U.make_race_id("house", "AK", "at-large") == "HOUSE-AK-AL"
    assert U.parse_race_id("SEN-TEXAS") is None
    assert U.chamber_of("GOV-NV") == "governor"
    with pytest.raises(ValueError):
        U.make_race_id("house", "PA")


@pytest.mark.parametrize(
    ("text", "chamber", "race_id"),
    [
        ("Alaska", "senate", "SEN-AK"),
        ("AK", "senate", "SEN-AK"),
        ("Alaska Senate", None, "SEN-AK"),
        ("Alaska Governor", None, "GOV-AK"),
        ("Georgia (Special)", "senate", "SEN-GA"),
        ("PA-08", None, "HOUSE-PA-08"),
        ("PA-8", "senate", "HOUSE-PA-08"),  # an explicit district code beats the chamber hint
        ("Pennsylvania 8th", "house", "HOUSE-PA-08"),
        ("AL-02 Figures (D)", "house", "HOUSE-AL-02"),
        ("AK Gov", None, "GOV-AK"),
        ("Sullivan (R)", "senate", None),
        ("", "senate", None),
    ],
)
def test_race_id_from_text(text, chamber, race_id):
    assert U.race_id_from_text(text, chamber) == race_id


# ---------------------------------------------------------------------------------------------
# crosswalk patterns
# ---------------------------------------------------------------------------------------------


def test_kalshi_ticker_candidates_documented_patterns_first():
    assert U.kalshi_ticker_candidates("SEN-TX", "D")[0] == "SENATETX-26-D"
    assert "KXSENATELA-26NOV-D" in U.kalshi_ticker_candidates("SEN-LA", "D")
    assert U.kalshi_ticker_candidates("GOV-GA", "R")[0] == "GOVPARTYGA-26-R"
    assert U.kalshi_ticker_candidates("HOUSE-PA-08", "D")[0] == "HOUSEPA8-26-D"
    assert U.kalshi_ticker_candidates("CONTROL-SENATE", "D")[0] == "CONTROLS-2026-D"
    assert U.kalshi_ticker_candidates("CONTROL-HOUSE", "D")[0] == "CONTROLH-2026-D"
    assert U.kalshi_ticker_candidates("SEN-NE", "I")[0].endswith("-I")
    assert U.kalshi_ticker_candidates("BOGUS", "D") == []
    cands = U.kalshi_ticker_candidates("HOUSE-PA-08", "D")
    assert len(cands) == len(set(cands))


def test_polymarket_slug_candidates_and_query():
    slugs = U.polymarket_slug_candidates("SEN-TX", "D")
    assert slugs[0] == "will-the-democratic-party-win-the-texas-senate-race-in-2026"
    assert any("gubernatorial" in s for s in U.polymarket_slug_candidates("GOV-GA", "R"))
    assert U.polymarket_search_query("HOUSE-PA-08") == "Pennsylvania 8th district House 2026"
    assert U.polymarket_search_query("CONTROL-SENATE") == "Senate control 2026"


def test_crosswalk_match_score_requires_structure():
    good = "Will the Democratic Party win the Texas Senate race in 2026?"
    assert U.crosswalk_match_score("SEN-TX", "D", good) > 0.9
    assert U.crosswalk_match_score("SEN-TX", "R", good) == 0.0  # wrong party
    assert U.crosswalk_match_score("SEN-TX", "D", "Will the Democratic Party win the Texas Governor race in 2026?") == 0.0
    assert U.crosswalk_match_score("SEN-TX", "D", "Will the Democratic Party win the Texas Senate primary in 2026?") == 0.0
    assert U.crosswalk_match_score("SEN-TX", "D", "Texas Senate: Democratic") == 0.0  # no year token
    assert U.crosswalk_match_score("HOUSE-PA-08", "D", "Will the Democratic Party win Pennsylvania's 8th district House race in 2026?") > 0.5
    assert U.crosswalk_match_score("HOUSE-PA-08", "D", "Will the Democratic Party win PA-07 in 2026?") == 0.0
    assert U.crosswalk_match_score("SEN-NE", "I", "Will Dan Osborn win the 2026 Nebraska Senate race?") > 0.5
    assert U.crosswalk_match_score("CONTROL-SENATE", "D", "Will Democrats control the Senate after the 2026 midterms?") > 0.5


@pytest.mark.parametrize(
    ("race_id", "party", "title", "contradicts"),
    [
        ("SEN-TX", "D", "Texas Senate: Democratic", False),
        ("SEN-TX", "D", None, False),
        ("SEN-TX", "D", "Georgia Senate: Democratic", True),
        ("SEN-TX", "D", "Texas Governor: Democratic", True),
        ("SEN-TX", "D", "Texas Senate: Republican", True),
        ("SEN-TX", "D", "Texas Senate Democratic primary winner", True),
        ("HOUSE-PA-08", "D", "PA-7 House: Democratic", True),
        ("HOUSE-PA-08", "D", "Pennsylvania 8th District: Democratic", False),
        ("CONTROL-SENATE", "D", "Democrats control the House", True),
        ("CONTROL-SENATE", "D", "Senate control: Democrats", False),
    ],
)
def test_title_contradicts(race_id, party, title, contradicts):
    assert U.title_contradicts(race_id, party, title) is contradicts


# ---------------------------------------------------------------------------------------------
# ratings
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Leans Democratic", ("lean", "D")),
        ("Lean D", ("lean", "D")),
        ("Likely R", ("likely", "R")),
        ("Likely Republican", ("likely", "R")),
        ("Solid D", ("safe", "D")),
        ("Safe Republican", ("safe", "R")),
        ("Tilt D", ("tilt", "D")),
        ("Toss Up", ("tossup", None)),
        ("Toss-up", ("tossup", None)),
        ("TOSSUP", ("tossup", None)),
        ("Pure Toss-up", ("tossup", None)),
        ("R+8", None),
        ("Sullivan (R)", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_rating(text, expected):
    assert U.normalize_rating(text) == expected


def test_rating_prob_dem_and_canonical(cfg):
    table = cfg.fairvalue.ratings_prob
    assert U.rating_prob_dem("lean", "D", table) == pytest.approx(0.68)
    assert U.rating_prob_dem("lean", "R", table) == pytest.approx(0.32)
    assert U.rating_prob_dem("tossup", None, table) == pytest.approx(0.5)
    assert U.rating_prob_dem("lean", "I", table) is None
    assert U.canonical_rating("lean", "D") == "Lean D"
    assert U.canonical_rating("tossup", None) == "Toss-up"


# ---------------------------------------------------------------------------------------------
# races.yaml seed and I/O
# ---------------------------------------------------------------------------------------------


def test_seed_universe_counts_and_conventions():
    data = U.load_races_yaml(SEED)
    races = data["races"]
    by_chamber: dict[str, int] = {}
    for rid, entry in races.items():
        key = U.parse_race_id(rid)
        assert key is not None, rid
        assert entry["chamber"] == key.chamber
        by_chamber[key.chamber] = by_chamber.get(key.chamber, 0) + 1
        for party, leg in entry["markets"].items():
            assert party in ("D", "R", "I")
            for f in U.MARKET_FIELDS:
                assert f in leg
            assert leg["sig_exchange_id"] == ""  # filled by bootstrap, not by hand
        assert entry["exclusive"] is None and entry["exhaustive"] is None  # API evidence only
        assert entry["incumbent_party"] in (None, "D", "R", "I")
        assert entry["pvi"] is None or isinstance(entry["pvi"], int | float)
    assert by_chamber == {"senate": 35, "governor": 36, "house": 46, "control": 2}
    assert set(races["SEN-NE"]["markets"]) == {"D", "R", "I"}
    assert set(races["SEN-MT"]["markets"]) == {"D", "R", "I"}
    assert set(races["GOV-RI"]["markets"]) == {"D", "R", "I"}
    assert races["SEN-FL"]["special"] and races["SEN-OH"]["special"]
    assert "runoff" in races["SEN-GA"]["settlement_note"].lower()
    assert "fusion" in races["GOV-RI"]["settlement_note"].lower()
    house = {rid for rid in races if rid.startswith("HOUSE-")}
    for d in ("AL-02", "AZ-01", "CA-22", "CO-08", "NE-02", "NY-17", "PA-08", "TX-28", "WA-03", "WI-03"):
        assert f"HOUSE-{d}" in house
    assert U.universe_stats(data).legs == 241


def test_races_yaml_roundtrip_preserves_values(tmp_path):
    data = U.load_races_yaml(SEED)
    data["races"]["SEN-TX"]["pvi"] = 5.5
    leg = U.market_entry(data["races"]["SEN-TX"], "D")
    leg["kalshi_ticker"] = "SENATETX-26-D"
    out = tmp_path / "races.yaml"
    U.save_races_yaml(data, out)
    text = out.read_text()
    assert text.startswith("# config/races.yaml")
    again = U.load_races_yaml(out)
    assert again["races"]["SEN-TX"]["pvi"] == 5.5
    assert again["races"]["SEN-TX"]["markets"]["D"]["kalshi_ticker"] == "SENATETX-26-D"
    assert list(again["races"])[:2] == ["CONTROL-HOUSE", "CONTROL-SENATE"]  # control first, then sorted
    assert yaml.safe_load(text)["races"].keys() == data["races"].keys()


def test_load_missing_yaml_and_race_entry_creation(tmp_path):
    data = U.load_races_yaml(tmp_path / "nope.yaml")
    assert data == {"version": 1, "races": {}}
    entry = U.race_entry(data, "HOUSE-PA-08", create=True)
    assert entry is not None and entry["chamber"] == "house" and entry["state"] == "PA" and entry["district"] == "08"
    assert U.race_entry(data, "NOT-A-RACE") is None
    leg = U.market_entry(entry, "D")
    assert leg == U.empty_market_entry()
    assert U.market_legs(data) == [("HOUSE-PA-08", "D", entry, leg)]


def test_race_row_values():
    vals = U.race_row_values("SEN-TX", {"chamber": "senate", "state": "TX", "pvi": 5, "incumbent_party": "r", "candidates": {}, "exclusive": True})
    assert vals == {
        "race_id": "SEN-TX",
        "chamber": "senate",
        "state": "TX",
        "district": None,
        "label": "Texas Senate",
        "incumbent_party": "R",
        "pvi": 5.0,
        "exclusive": True,
        "exhaustive": None,
        "settlement_note": None,
        "candidates": None,
    }
    assert U.race_row_values("HOUSE-PA-08", None)["district"] == "08"
    assert U.race_row_values("bogus", {}) is None
