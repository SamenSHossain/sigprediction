"""Regression tests for the universe / races.yaml review findings.

F1: every state and district race in config/races.yaml carries a numeric PVI. A null PVI used to be read
    as EVEN by the fundamentals prior, so with a D+8 generic ballot Likely-R races (FL Senate and
    Governor, TX-15) were priced as Democratic favourites by the polls component.
F6 (guard): fairvalue drops the two-party components for a major-party leg whose party fields no
    candidate, reading Race.candidates; the NE Senate seed must say "R and I, no Democrat".
F9: normalize_rating's party fallback ('D Lean', 'Lean (R)', 'R-Likely', 'Solid (D)')."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from sigcup.db import engine as dbengine
from sigcup.db.models import GenericBallot, Race
from sigcup.feeds import polls as P
from sigcup.feeds import universe as U
from sigcup.trader import fairvalue as FV
from sigcup.trader.types import MarketView

SEED = Path(__file__).resolve().parent.parent / "config" / "races.yaml"
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------------------------
# F1: PVI coverage
# ---------------------------------------------------------------------------------------------


def test_every_seat_race_has_a_numeric_pvi():
    races = U.load_races_yaml(SEED)["races"]
    missing = sorted(
        rid for rid, e in races.items() if e["chamber"] != "control" and not isinstance(e.get("pvi"), int | float)
    )
    assert not missing, f"races without a PVI (read as EVEN by the fundamentals prior): {missing}"


def test_florida_and_texas_15_lean_republican():
    races = U.load_races_yaml(SEED)["races"]
    for rid in ("SEN-FL", "GOV-FL", "HOUSE-TX-15", "HOUSE-TX-28"):
        assert races[rid]["pvi"] > 0, rid  # positive = Republican lean (Cook convention)


def _views(*race_ids: str) -> dict[str, MarketView]:
    out = {}
    for rid in race_ids:
        for party in ("D", "R"):
            eid = f"{rid}-{party}"
            out[eid] = MarketView(exchange_id=eid, race_id=rid, party=party, chamber="", state=None)
    return out


async def test_likely_r_races_are_not_priced_as_democratic_favourites(db, cfg):
    """The reviewer's scenario: offline polls pass (manual ratings) plus a D+8 generic ballot."""
    await P.run_once(dbengine.session, cfg, None, now=NOW, fetch_web=False, fetch_fec=False)
    with dbengine.session() as s:
        s.add(GenericBallot(source="rcp", ts=NOW, margin=8.0))
    with dbengine.session() as s:
        fv = FV.compute_all(s, cfg, _views("GOV-FL", "SEN-FL", "HOUSE-TX-15"), NOW)
    for rid in ("GOV-FL", "SEN-FL", "HOUSE-TX-15"):
        d = fv[f"{rid}-D"]
        assert d.components["polls"] < 0.5, (rid, d.components)
        assert d.fair < 0.5, (rid, d.fair)


# ---------------------------------------------------------------------------------------------
# F6 guard: the NE Senate field is R vs I (no Democrat), and it reaches the DB
# ---------------------------------------------------------------------------------------------


def test_nebraska_senate_candidates_mark_the_empty_democratic_leg(db):
    ne = U.load_races_yaml(SEED)["races"]["SEN-NE"]
    assert set(ne["markets"]) == {"D", "R", "I"}
    named = {party for party, c in ne["candidates"].items() if isinstance(c, dict) and c.get("name")}
    assert named == {"R", "I"}
    P.ensure_race_rows(dbengine.session)
    with dbengine.session() as s:
        cands = s.get(Race, "SEN-NE").candidates
    assert set(cands) == {"R", "I"}


# ---------------------------------------------------------------------------------------------
# F9: party fallback in normalize_rating
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Lean (R)", ("lean", "R")),
        ("D Lean", ("lean", "D")),
        ("R-Likely", ("likely", "R")),
        ("Solid (D)", ("safe", "D")),
        ("Likely (I)", ("likely", "I")),
        # unchanged behaviour
        ("Likely R", ("likely", "R")),
        ("Leans Rep.", ("lean", "R")),
        ("Toss-up", ("tossup", None)),
        ("Safe", None),
        ("Sullivan (R)", None),
    ],
)
def test_normalize_rating_party_fallback(text, expected):
    assert U.normalize_rating(text) == expected
