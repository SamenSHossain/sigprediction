"""Regression tests for the ratings-parser and race-seeding review findings.

F8: a rating change log whose header row is made of <td> cells (common in WordPress tables) lost the
    'Old Rating' exclusion and the date column, so the OLD rating was kept, stamped ``now``; with no
    chamber hint, bare state names under a 'Senate' / 'Governor' heading were dropped.
F10 (polls side): ``ensure_race_rows`` copied hand-set / stale ``exclusive`` / ``exhaustive`` flags from
    races.yaml into new Race rows; those flags may only come from a relationships fetch (bootstrap).
Nothing touches the network."""

from __future__ import annotations

from datetime import UTC, datetime

from sigcup.db import engine as dbengine
from sigcup.db.models import Race
from sigcup.feeds import polls as P
from sigcup.feeds import universe as U

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
PROB = {"safe": 0.97, "likely": 0.85, "lean": 0.68, "tilt": 0.58, "tossup": 0.50}

TD_HEADER_LOG = """
<html><body>
<h2>Senate</h2>
<table class="wp-block-table">
<tbody>
<tr><td><strong>Date</strong></td><td><strong>Race</strong></td><td><strong>Old Rating</strong></td><td><strong>New Rating</strong></td></tr>
<tr><td>10/6/2026</td><td>Alaska Senate</td><td>Toss-up</td><td>Leans Democratic</td></tr>
<tr><td>9/8/2026</td><td>Kansas Senate</td><td>Likely Republican</td><td>Toss-up</td></tr>
</tbody>
</table>
</body></html>
"""


def as_map(rows):
    return {(r.race_id, r.source): r for r in rows}


def test_td_header_change_log_keeps_the_new_rating_and_row_date():
    rows = as_map(P.parse_sabato_rating_changes(TD_HEADER_LOG, ts=NOW, ratings_prob=PROB))
    ks = rows[("SEN-KS", "sabato")]
    assert (ks.rating, ks.prob_dem, ks.ts) == ("Toss-up", 0.5, datetime(2026, 9, 8, tzinfo=UTC))
    ak = rows[("SEN-AK", "sabato")]
    assert (ak.rating, ak.ts) == ("Lean D", datetime(2026, 10, 6, tzinfo=UTC))


def test_td_header_row_is_promoted_only_when_it_looks_like_a_header():
    tables = P.extract_tables(TD_HEADER_LOG)
    assert tables[0].header == ["Date", "Race", "Old Rating", "New Rating"]
    assert len(tables[0].body) == 2
    # a data-first table (generic-ballot style, <td> only) keeps its first row as data
    gb = P.extract_tables(
        "<table><tr><td>RCP Average</td><td>9/16 - 10/5</td><td>49.6</td><td>41.7</td></tr>"
        "<tr><td>Quinnipiac</td><td>10/1 - 10/5</td><td>50</td><td>42</td></tr></table>"
    )
    assert gb[0].header is None and len(gb[0].body) == 2
    ratings = P.extract_tables("<table><tr><td>Alaska</td><td>Lean D</td></tr><tr><td>Texas</td><td>Toss-up</td></tr></table>")
    assert ratings[0].header is None and len(ratings[0].body) == 2


def test_headerless_old_new_columns_keep_the_rightmost_rating():
    html = (
        "<h2>Senate</h2><table>"
        "<tr><td>Alaska Senate</td><td>Toss-up</td><td>Leans Democratic</td></tr>"
        "<tr><td>Kansas Senate</td><td>Likely Republican</td><td>Toss-up</td></tr>"
        "</table>"
    )
    rows = as_map(P.parse_ratings_html(html, source="sabato", chamber=None, ts=NOW, ratings_prob=PROB))
    assert rows[("SEN-AK", "sabato")].rating == "Lean D"
    assert rows[("SEN-KS", "sabato")].rating == "Toss-up"


def test_change_log_rows_without_a_date_are_not_stamped_now():
    html = (
        "<h2>Senate</h2><table>"
        "<tr><th>Date</th><th>Race</th><th>Old Rating</th><th>New Rating</th></tr>"
        "<tr><td>10/6/2026</td><td>Alaska Senate</td><td>Toss-up</td><td>Leans Democratic</td></tr>"
        "<tr><td>TBD</td><td>Kansas Senate</td><td>Likely Republican</td><td>Toss-up</td></tr>"
        "</table>"
    )
    rows = as_map(P.parse_sabato_rating_changes(html, ts=NOW, ratings_prob=PROB))
    assert ("SEN-KS", "sabato") not in rows
    assert rows[("SEN-AK", "sabato")].ts == datetime(2026, 10, 6, tzinfo=UTC)


def test_block_heading_is_the_chamber_hint_for_bare_state_names():
    html = (
        "<h2>Senate</h2><table>"
        "<tr><th>Date</th><th>Race</th><th>Old Rating</th><th>New Rating</th></tr>"
        "<tr><td>10/6/2026</td><td>Alaska</td><td>Toss-up</td><td>Leans Democratic</td></tr>"
        "<tr><td>10/2/2026</td><td>Maine</td><td>Leans Republican</td><td>Toss-up</td></tr>"
        "</table>"
        "<h2>Governor</h2><table>"
        "<tr><th>Date</th><th>Race</th><th>Old Rating</th><th>New Rating</th></tr>"
        "<tr><td>10/1/2026</td><td>Alaska</td><td>Likely Republican</td><td>Leans Republican</td></tr>"
        "<tr><td>9/30/2026</td><td>Kansas Governor</td><td>Leans Republican</td><td>Toss-up</td></tr>"
        "</table>"
    )
    rows = as_map(P.parse_sabato_rating_changes(html, ts=NOW, ratings_prob=PROB))
    assert {rid: r.rating for (rid, _), r in rows.items()} == {
        "SEN-AK": "Lean D", "SEN-ME": "Toss-up", "GOV-AK": "Lean R", "GOV-KS": "Toss-up",
    }


def test_ensure_race_rows_never_copies_exclusivity_from_yaml(db):
    data = U.load_races_yaml()
    data["races"]["GOV-RI"]["exclusive"] = True  # a stale / hand-set flag
    data["races"]["GOV-RI"]["exhaustive"] = True
    assert P.ensure_race_rows(dbengine.session, data) == len(data["races"])
    with dbengine.session() as s:
        ri = s.get(Race, "GOV-RI")
    assert (ri.exclusive, ri.exhaustive) == (None, None)
