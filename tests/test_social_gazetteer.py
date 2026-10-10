"""Gazetteer linking rules (collisions, hashtags, race codes), parsing helpers and the shared Post ingest."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Post, Race
from sigcup.feeds.nlp.gazetteer import Gazetteer, PostDraft, parse_ts, pick, strip_html, write_posts
from tests.test_social_helpers import NOW, RACES, make_gazetteer


@pytest.fixture()
def g() -> Gazetteer:
    return make_gazetteer()


def links(g: Gazetteer, text: str, **kw) -> dict[tuple[str, str | None], float]:
    return {(r, p): c for r, p, c in g.link(text, **kw)}


# -- name / alias / handle / hashtag ---------------------------------------------------------


def test_full_name_links_with_high_confidence(g):
    assert g.link("Jon Ossoff leads the pack") == [("SEN-GA", "D", 0.95)]


def test_full_name_tolerates_middle_tokens_and_case(g):
    assert links(g, "keisha bottoms is running") == {("GOV-GA", "D"): 0.95}
    assert links(g, "Keisha L. Bottoms is running") == {("GOV-GA", "D"): 0.95}


def test_alias_and_multiword_alias(g):
    assert links(g, "KLB announces a bus tour") == {("GOV-GA", "D"): 0.85}
    assert links(g, "Earl Carter files his paperwork") == {("SEN-GA", "R"): 0.85}


def test_handle_in_text(g):
    assert links(g, "thanks @ossoff for coming") == {("SEN-GA", "D"): 0.95}
    assert links(g, "thanks @OSSOFF for coming") == {("SEN-GA", "D"): 0.95}


def test_author_handle_links_the_candidate(g):
    assert links(g, "What a great night.", author_handle="@BuddyCarter") == {("SEN-GA", "R"): 0.90}


def test_hashtag_case_insensitive(g):
    assert links(g, "big rally tonight #teamossoff") == {("SEN-GA", "D"): 0.90}


def test_partial_word_never_matches(g):
    assert g.link("Jonathan Ossoffsky wrote a book") == []
    assert g.link("email ossoff@example.com") == []


# -- surname-only rules -------------------------------------------------------------------------


def test_surname_alone_needs_a_race_token(g):
    assert g.link("Ossoff is up") == []
    assert links(g, "Ossoff is up in Georgia") == {("SEN-GA", "D"): 0.60}
    assert links(g, "Ossoff is up #GApol") == {("SEN-GA", "D"): 0.60}
    assert links(g, "Ossoff is up #GASen") == {("SEN-GA", "D"): 0.60}


def test_surname_with_a_race_handle_as_evidence(g):
    out = links(g, "Ossoff slams @buddycarter over the vote")
    assert out[("SEN-GA", "R")] == 0.95
    assert out[("SEN-GA", "D")] == 0.60


def test_surname_collision_across_states(g):
    assert links(g, "Jones is up in New York") == {("HOUSE-NY-03", "R"): 0.60}
    assert links(g, "Jones is up in Arkansas") == {}


def test_surname_collision_within_a_state_needs_a_chamber_token(g):
    assert g.link("Jones leads in Georgia") == []
    assert links(g, "Jones leads the Georgia governor race") == {("GOV-GA", "R"): 0.60}
    assert links(g, "Jones gains in #GASen poll") == {("SEN-GA", "I"): 0.60}


def test_common_word_surname_never_links_alone(g):
    assert g.link("Cotton leads in Arkansas") == []
    assert links(g, "Tom Cotton leads in Arkansas") == {("SEN-AR", "R"): 0.95}


def test_hyphenated_surname_pair_with_state(g):
    assert links(g, "Allred-Paxton debate tonight in Texas") == {("SEN-TX", "D"): 0.60, ("SEN-TX", "R"): 0.60}


def test_house_district_phrases(g):
    assert links(g, "Bresnahan wins Pennsylvania's 8th district") == {("HOUSE-PA-08", "R"): 0.60}
    assert links(g, "Cartwright is back in PA-8") == {("HOUSE-PA-08", "D"): 0.60}


# -- race codes ---------------------------------------------------------------------------------


def test_race_code_only_yields_party_none_for_that_chamber_only(g):
    assert g.link("#GASen is tightening") == [("SEN-GA", None, 0.5)]
    assert g.link("GA-Gov is a snoozer") == [("GOV-GA", None, 0.5)]
    assert g.link("PA-08 is close") == [("HOUSE-PA-08", None, 0.5)]
    assert g.link("Texas Senate debate tonight") == [("SEN-TX", None, 0.5)]


def test_bare_state_name_is_not_a_link(g):
    assert g.link("Georgia is lovely in October") == []


def test_link_race_ids_dedupes_and_orders(g):
    assert g.link_race_ids("Jon Ossoff vs Buddy Carter, plus PA-08 chatter") == ["SEN-GA", "HOUSE-PA-08"]
    assert g.link_race_ids("Jon Ossoff vs Buddy Carter, plus PA-08 chatter", min_conf=0.9) == ["SEN-GA"]


# -- construction and queries ---------------------------------------------------------------------


def test_search_terms_and_race_context(g):
    assert g.search_terms("SEN-GA") == ["Jon Ossoff", "Buddy Carter", "Earl Carter", "Chuck Jones"]
    assert g.search_terms("CONTROL-SENATE") == []
    ctx = g.race_context("SEN-TX")
    assert ctx == {"race_id": "SEN-TX", "label": "Texas Senate", "state": "TX", "chamber": "senate",
                   "candidates": {"D": "Colin Allred", "R": "Ken Paxton"}}


def test_from_races_tolerates_junk_and_infers_meta():
    g = Gazetteer.from_races({"SEN-XX": "nope", "GOV-WY": {"candidates": {"D": "Jane Doe", "R": None, "I": {"name": ""}}}})
    assert [c.name for c in g.candidates] == ["Jane Doe"]
    meta = g.races["GOV-WY"]
    assert (meta.state, meta.chamber, meta.label) == ("WY", "governor", "Wyoming governor")
    assert g.link("Doe leads in Wyoming") == [("GOV-WY", "D", 0.6)]


def test_from_db_uses_race_rows_then_yaml_fallback(db, monkeypatch):
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-GA", chamber="senate", state="GA", label="Georgia Senate", candidates=RACES["SEN-GA"]["candidates"]))
    g = Gazetteer.from_db(dbengine.session)
    assert g.link("Jon Ossoff") == [("SEN-GA", "D", 0.95)]

    with dbengine.session() as s:
        s.delete(s.get(Race, "SEN-GA"))
    import sigcup.settings as settings

    monkeypatch.setattr(settings, "load_yaml", lambda name: {"races": RACES} if name == "races" else None)
    g2 = Gazetteer.from_db(dbengine.session)
    assert g2.link("Tom Cotton") == [("SEN-AR", "R", 0.95)]


# -- helpers --------------------------------------------------------------------------------------


def test_parse_ts_shapes():
    assert parse_ts("2026-10-10T12:00:00Z") == datetime(2026, 10, 10, 12, tzinfo=UTC)
    assert parse_ts("20261010T120000Z") == datetime(2026, 10, 10, 12, tzinfo=UTC)
    assert parse_ts("Sat Oct 10 03:00:00 +0000 2026") == datetime(2026, 10, 10, 3, tzinfo=UTC)
    assert parse_ts("Sat, 10 Oct 2026 09:30:00 GMT") == datetime(2026, 10, 10, 9, 30, tzinfo=UTC)
    assert parse_ts(1791900000) == datetime(2026, 10, 13, 14, 0, tzinfo=UTC)
    assert parse_ts(1791900000000) == datetime(2026, 10, 13, 14, 0, tzinfo=UTC)
    assert parse_ts(1791900000000000) == datetime(2026, 10, 13, 14, 0, tzinfo=UTC)
    assert parse_ts(time.gmtime(1791900000)) == datetime(2026, 10, 13, 14, 0, tzinfo=UTC)
    assert parse_ts(datetime(2026, 1, 1)) == datetime(2026, 1, 1, tzinfo=UTC)
    assert parse_ts("garbage") is None
    assert parse_ts(None) is None
    assert parse_ts(True) is None


def test_pick_dotted_paths_and_attributes():
    d = {"a": {"b": 1}, "c": None, "e": ""}
    assert pick(d, "c", "a.b") == 1
    assert pick(d, "e", default="dflt") == "dflt"
    assert pick(SimpleNamespace(x=SimpleNamespace(y="z")), "x.y") == "z"
    assert pick(None, "x", default=3) == 3


def test_strip_html():
    assert strip_html("<p>Hello &amp; <b>bye</b></p>\n\n  x") == "Hello & bye x"
    assert strip_html(None) == ""


def test_write_posts_links_scores_dedupes(db, g):
    drafts = [
        PostDraft("x:1", "x", NOW, "Jon Ossoff is great", author_id="a1", author_handle="gavoter"),
        PostDraft("x:1", "x", NOW, "duplicate id, skipped"),
        PostDraft("x:2", "x", NOW, "the weather is nice"),  # no link -> dropped
        PostDraft("x:3", "x", NOW, "the weather is nice", race_ids=["SEN-TX"]),  # draft-supplied race keeps it
        PostDraft("x:4", "x", NOW, "Jon Ossoff is great", author_id="a2"),
        PostDraft("x:5", "x", NOW, "Jon Ossoff is great", author_id="a3"),
    ]
    assert write_posts(dbengine.session, drafts, g, now=NOW) == 4
    with dbengine.session() as s:
        rows = {p.post_id: p for p in s.execute(select(Post)).scalars()}
    assert set(rows) == {"x:1", "x:3", "x:4", "x:5"}
    assert rows["x:1"].race_ids == ["SEN-GA"]
    assert rows["x:3"].race_ids == ["SEN-TX"]
    assert rows["x:1"].text_hash == rows["x:4"].text_hash
    assert rows["x:1"].bot_score < rows["x:4"].bot_score < rows["x:5"].bot_score  # duplicates raise the score
    assert all(p.classified is False for p in rows.values())
    assert write_posts(dbengine.session, drafts, g, now=NOW) == 0  # idempotent
