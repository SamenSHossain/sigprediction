"""Regression tests for the gazetteer: race codes and surnames must not match everyday words (F3, F4),
and the shared ingest must not run a per-draft author count or block the event loop (F8)."""

from __future__ import annotations

import threading
from datetime import timedelta
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import event

from sigcup.db import engine as dbengine
from sigcup.db.models import Post
from sigcup.feeds import social_bluesky, social_x
from sigcup.feeds.nlp import botfilter
from sigcup.feeds.nlp.gazetteer import Gazetteer, PostDraft, write_posts
from sigcup.settings import Secrets
from tests.test_social_helpers import NOW, load_text, make_gazetteer, make_post

# A slice of config/races.yaml (names as seeded there) plus a code-only House race.
REAL: dict[str, dict[str, Any]] = {
    "SEN-OR": {"chamber": "senate", "state": "OR", "candidates": {"D": {"name": "Jeff Merkley"}}},
    "GOV-OR": {"chamber": "governor", "state": "OR", "candidates": {"D": {"name": "Tina Kotek"}}},
    "SEN-OK": {"chamber": "senate", "state": "OK", "candidates": {"R": {"name": "Markwayne Mullin"}}},
    "GOV-HI": {"chamber": "governor", "state": "HI", "candidates": {"D": {"name": "Josh Green"}}},
    "SEN-GA": {"chamber": "senate", "state": "GA", "candidates": {"D": {"name": "Jon Ossoff"}}},
    "SEN-TX": {"chamber": "senate", "state": "TX", "candidates": {"R": {"name": "Ken Paxton"}}},
    "SEN-FL": {"chamber": "senate", "state": "FL", "candidates": {"R": {"name": "Ashley Moody"}}},
    "HOUSE-ME-02": {"chamber": "house", "state": "ME", "district": "02", "candidates": {}},
    "HOUSE-CO-08": {"chamber": "house", "state": "CO", "district": "08", "candidates": {"R": {"name": "Gabe Evans"}}},
    "HOUSE-PA-08": {"chamber": "house", "state": "PA", "district": "08", "candidates": {"R": {"name": "Rob Bresnahan"}}},
    "HOUSE-TX-15": {"chamber": "house", "state": "TX", "district": "15", "candidates": {"R": {"name": "Monica De La Cruz"}}},
    "HOUSE-WA-03": {"chamber": "house", "state": "WA", "district": "03", "candidates": {"D": {"name": "Marie Gluesenkamp Perez"}}},
    "HOUSE-WI-03": {"chamber": "house", "state": "WI", "district": "03", "candidates": {"R": {"name": "Derrick Van Orden"}}},
    "HOUSE-AL-02": {"chamber": "house", "state": "AL", "district": "02", "candidates": {"D": {"name": "Shomari Figures"}}},
}


@pytest.fixture()
def real() -> Gazetteer:
    return Gazetteer.from_races(REAL)


# -- F3: race codes ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "does it matter who wins the house or senate?",
        "Will Democrats flip the House or Senate?",
        "WHO WINS THE HOUSE OR SENATE THIS YEAR",
        "the mayor or governor should resign",
        "ok senate, do your job",
        "give me 2 minutes",
        "me 2!!",
        "co 8 is a toss-up",
        "hi gov",
        "Georgia Senate passes new voting bill",
        "the Texas Senate adjourned without a vote on the property tax plan",
        "Michigan's 8th district is in play",
        "the 8th district judge ruled today",
    ],
)
def test_everyday_phrases_are_not_race_codes(real, text):
    assert real.link(text) == []


@pytest.mark.parametrize(
    ("text", "race_id"),
    [
        ("OR-Sen is close", "SEN-OR"),
        ("ORSen is close", "SEN-OR"),
        ("#ORSen is close", "SEN-OR"),
        ("#orsen is close", "SEN-OR"),
        ("HI-Gov is a snoozer", "GOV-HI"),
        ("ME-02 is close", "HOUSE-ME-02"),
        ("ME-2 is close", "HOUSE-ME-02"),
        ("ME02 is close", "HOUSE-ME-02"),
        ("CO-08 is a toss-up", "HOUSE-CO-08"),
        ("Texas Senate debate tonight", "SEN-TX"),
        ("the Georgia Senate race is close", "SEN-GA"),
        ("Georgia's U.S. Senate seat is in play", "SEN-GA"),
        ("Pennsylvania's 8th district is in play", "HOUSE-PA-08"),
    ],
)
def test_race_code_forms_still_link(real, text, race_id):
    assert real.link(text) == [(race_id, None, 0.5)]


def test_ordinal_district_still_backs_a_surname(real):
    assert real.link("Bresnahan wins the 8th district") == [("HOUSE-PA-08", "R", 0.6)]


# -- F4: surnames -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "race_id"),
    [
        ("Florida voters are in a moody place this year", "SEN-FL"),
        ("Moody's downgrades Florida's bond outlook", "SEN-FL"),
        ("Ted Cruz just said Texas will never go blue", "HOUSE-TX-15"),
        ("Senator Ted Cruz (R-Texas) slammed the bill", "HOUSE-TX-15"),
        ("Tom Perez was in Washington today", "HOUSE-WA-03"),
        ("Perez spoke to reporters in Washington today", "HOUSE-WA-03"),
        ("New Figures Show Alabama Jobs Growth", "HOUSE-AL-02"),
        ("ossoff is up in georgia", None),  # lower case: not a capitalised surname
    ],
)
def test_ordinary_words_and_namesakes_do_not_link(real, text, race_id):
    if race_id is None:
        assert [lk for lk in real.link(text) if lk[1] is not None] == []
    else:
        assert race_id not in real.link_race_ids(text)


@pytest.mark.parametrize(
    ("text", "link"),
    [
        ("Ossoff is up in Georgia", ("SEN-GA", "D", 0.6)),
        ("OSSOFF IS UP IN GEORGIA", ("SEN-GA", "D", 0.6)),
        ("De La Cruz leads in Texas", ("HOUSE-TX-15", "R", 0.6)),
        ("de La Cruz leads in Texas", ("HOUSE-TX-15", "R", 0.6)),
        ("Van Orden is up in Wisconsin", ("HOUSE-WI-03", "R", 0.6)),
        ("Gluesenkamp Perez leads in southwest Washington", ("HOUSE-WA-03", "D", 0.6)),
        ("Perez is ahead in Washington state", ("HOUSE-WA-03", "D", 0.6)),
        ("Perez holds on in WA-03", ("HOUSE-WA-03", "D", 0.6)),
    ],
)
def test_capitalised_and_particle_surnames_still_link(real, text, link):
    assert link in real.link(text)


def test_full_names_still_link_regardless_of_case(real):
    assert real.link("monica de la cruz is up") == [("HOUSE-TX-15", "R", 0.95)]
    assert real.link("Ashley Moody leads") == [("SEN-FL", "R", 0.95)]


# -- F8: ingest cost and the event loop -------------------------------------------------------------


def _count_statements(eng: Any) -> list[str]:
    stmts: list[str] = []

    def before(conn, cursor, statement, parameters, context, executemany):  # noqa: ANN001
        stmts.append(statement.lower())

    event.listen(eng, "before_cursor_execute", before)
    return stmts


def test_write_posts_does_not_count_per_draft(db):
    g = make_gazetteer()
    with dbengine.session() as s:
        for i in range(20):
            s.add(make_post(f"x:old{i}", f"old post {i} about Jon Ossoff", ["SEN-GA"], ts=NOW - timedelta(hours=2), author_id="a0"))
    drafts = [PostDraft(f"x:{i}", "x", NOW, f"Jon Ossoff post number {i}", author_id=f"a{i}") for i in range(60)]
    stmts = _count_statements(db)
    assert write_posts(dbengine.session, drafts, g, now=NOW) == 60
    counts = [st for st in stmts if "count(" in st]
    assert len(counts) <= 4, f"{len(counts)} COUNT statements for 60 drafts"


def test_write_posts_rate_and_duplicate_signals_are_unchanged(db):
    g = make_gazetteer()
    with dbengine.session() as s:
        for i in range(30):
            s.add(make_post(f"x:r{i}", f"rate post {i} Jon Ossoff", ["SEN-GA"], ts=NOW - timedelta(hours=3), author_id="busy"))
        s.add(make_post("x:stale", "stale post Jon Ossoff", ["SEN-GA"], ts=NOW - timedelta(days=3), author_id="busy"))
        s.add(make_post("bluesky:other", "other platform Jon Ossoff", ["SEN-GA"], platform="bluesky", ts=NOW, author_id="busy"))
        for i in range(3):
            s.add(make_post(f"x:d{i}", "Vote Jon Ossoff today", ["SEN-GA"], ts=NOW - timedelta(days=5)))
    drafts = [
        PostDraft("x:new1", "x", NOW, "a fresh take on Jon Ossoff", author_id="busy", author_handle="busy"),
        PostDraft("x:new2", "x", NOW, "another fresh take on Jon Ossoff", author_id="busy", author_handle="busy"),
        PostDraft("x:new3", "x", NOW, "Vote Jon Ossoff today", author_id="quiet", author_handle="quiet"),
        PostDraft("x:new4", "x", NOW, "Vote Jon Ossoff today", author_id="quiet2", author_handle="quiet2"),
    ]
    assert write_posts(dbengine.session, drafts, g, now=NOW) == 4
    with dbengine.session() as s:
        got = {pid: s.get(Post, pid).bot_score for pid in ("x:new1", "x:new2", "x:new3", "x:new4")}
    # 30 x posts in the last 24h (the 3-day-old one and the bluesky one do not count); +1 for new1 in-batch
    assert got["x:new1"] == botfilter.score(drafts[0], dup_count=0, author_posts_24h=30, now=NOW)
    assert got["x:new2"] == botfilter.score(drafts[1], dup_count=0, author_posts_24h=31, now=NOW)
    # 3 earlier copies in the DB, then one more from the same batch
    assert got["x:new3"] == botfilter.score(drafts[2], dup_count=3, author_posts_24h=0, now=NOW)
    assert got["x:new4"] == botfilter.score(drafts[3], dup_count=4, author_posts_24h=0, now=NOW)


class _ThreadRecorder:
    def __init__(self) -> None:
        self.threads: list[int] = []
        self._orig = botfilter.score

    def __call__(self, *a: Any, **kw: Any) -> float:
        self.threads.append(threading.get_ident())
        return self._orig(*a, **kw)


@respx.mock
async def test_x_ingest_runs_off_the_event_loop_thread(db, cfg, monkeypatch):
    rec = _ThreadRecorder()
    monkeypatch.setattr(botfilter, "score", rec)
    payload = {"data": [{"id": "1841000000000000009", "text": "Jon Ossoff rally tonight", "author_id": "9"}]}
    respx.get(host="api.x.com").mock(return_value=httpx.Response(200, json=payload))
    n = await social_x.run_once(dbengine.session, cfg, Secrets(_env_file=None, x_vendor="official", x_vendor_key="t"), gazetteer=make_gazetteer(), now=NOW, max_queries=1)
    assert n == 1
    assert rec.threads and threading.get_ident() not in rec.threads


class _FakeWS:
    def __init__(self, msgs: list[str]) -> None:
        self._msgs = list(msgs)

    def __aiter__(self) -> _FakeWS:
        return self

    async def __anext__(self) -> str:
        if not self._msgs:
            raise StopAsyncIteration
        return self._msgs.pop(0)


class _FakeConnect:
    def __init__(self, msgs: list[str]) -> None:
        self.msgs = msgs

    def __call__(self, url: str) -> _FakeConnect:
        return self

    async def __aenter__(self) -> _FakeWS:
        return _FakeWS(self.msgs)

    async def __aexit__(self, *exc: Any) -> bool:
        return False


async def test_bluesky_ingest_runs_off_the_event_loop_thread(db, cfg, monkeypatch):
    rec = _ThreadRecorder()
    monkeypatch.setattr(botfilter, "score", rec)
    lines = load_text("jetstream_messages.jsonl").splitlines()
    n = await social_bluesky.run_once(dbengine.session, cfg, None, duration_s=5, connect=_FakeConnect(lines), gazetteer=make_gazetteer(), now=NOW)
    assert n == 1
    assert rec.threads and threading.get_ident() not in rec.threads
