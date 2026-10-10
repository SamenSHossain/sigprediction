"""Event detection: spike rules, GDELT volume rule, LLM classification plumbing and Event rows."""

from __future__ import annotations

import json
from datetime import timedelta

from sqlalchemy import select

from sigcup.common.time import ensure_utc
from sigcup.db import engine as dbengine
from sigcup.db.models import Event, Feature
from sigcup.feeds import events
from sigcup.settings import StrategyConfig
from tests.test_social_helpers import NOW, FakeAnthropic, make_gazetteer, make_post

DAY = NOW.strftime("%Y-%m-%d")


def feat(platform: str, race_id: str, z: float | None, n: int = 20, day: str = DAY) -> Feature:
    return Feature(platform=platform, race_id=race_id, day=day, n_posts=n, net_stance_dem=0.1, z_score=z, computed_at=NOW)


def test_detect_social_spikes_rule():
    feats = [
        feat("bluesky", "SEN-GA", 3.1), feat("x", "SEN-GA", -2.7), feat("youtube", "SEN-GA", 0.4),
        feat("x", "SEN-TX", 4.0),  # one platform only
        feat("x", "GOV-GA", None), feat("bluesky", "GOV-GA", 2.6),  # None never counts
        feat("x", "SEN-AR", 2.5), feat("rss", "SEN-AR", 2.5),  # threshold is inclusive
    ]
    out = events.detect_social_spikes(feats, spike_z=2.5, min_platforms=2)
    assert set(out) == {"SEN-GA", "SEN-AR"}
    assert {f.platform for f in out["SEN-GA"]} == {"bluesky", "x"}
    assert set(events.detect_social_spikes(feats, spike_z=2.5, min_platforms=1)) == {"SEN-GA", "SEN-TX", "GOV-GA", "SEN-AR"}


def test_detect_gdelt_spikes_rule():
    counts = {}
    for i in range(1, 8):
        counts[("SEN-TX", (NOW - timedelta(days=i)).strftime("%Y-%m-%d"))] = 1
        counts[("SEN-GA", (NOW - timedelta(days=i)).strftime("%Y-%m-%d"))] = 5
    counts[("SEN-TX", DAY)] = 7  # 7 > 3 x 1
    counts[("SEN-GA", DAY)] = 12  # 12 < 3 x 5
    counts[("GOV-GA", DAY)] = 4  # below the minimum article count
    counts[("SEN-AR", DAY)] = 50  # no history at all -> no baseline
    out = events.detect_gdelt_spikes(counts, DAY, baseline_days=7)
    assert out == {"SEN-TX": (7, 1.0)}


def test_gdelt_counts_from_posts():
    posts = [
        make_post("gdelt:1", "a", ["SEN-GA", "GOV-GA"], platform="gdelt", ts=NOW),
        make_post("gdelt:2", "b", ["SEN-GA"], platform="gdelt", ts=NOW - timedelta(days=1)),
        make_post("gdelt:3", "c", None, platform="gdelt", ts=NOW),
    ]
    assert events.gdelt_counts(posts) == {("SEN-GA", DAY): 1, ("GOV-GA", DAY): 1, ("SEN-GA", (NOW - timedelta(days=1)).strftime("%Y-%m-%d")): 1}


def test_parse_classification_lenient():
    assert events.parse_classification('Here: {"kind": "Hard", "description": "Carter withdrew", "confidence": 90}') == ("hard", "Carter withdrew", 0.9)
    assert events.parse_classification('{"kind": "weird"}') == ("unknown", "", 0.5)
    assert events.parse_classification("no json") is None
    assert events.parse_classification("[1, 2]") is None


async def test_llm_classifier_uses_escalation_model_and_payload():
    client = FakeAnthropic(lambda p: '{"kind": "soft", "description": "a new poll", "confidence": 0.6}')
    classify = events.llm_classifier(client, "claude-sonnet-5-5")
    out = await classify({"race_id": "SEN-GA"}, ["[x] post one"], "stance spike")
    assert out == ("soft", "a new poll", 0.6)
    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    payload = json.loads(call["messages"][0]["content"])
    assert payload == {"race": {"race_id": "SEN-GA"}, "trigger": "stance spike", "posts": ["[x] post one"]}

    failing = FakeAnthropic(lambda p: "not json")
    assert await events.llm_classifier(failing, "m")({"race_id": "X"}, [], "trigger") == ("unknown", "trigger", 0.3)


async def test_run_once_social_spike_writes_one_event_per_day(db, cfg):
    seen: list[tuple] = []

    async def classifier(race_context, top_posts, trigger):
        seen.append((race_context["race_id"], list(top_posts), trigger))
        return "hard", "Carter drops out of the race", 0.85

    with dbengine.session() as s:
        s.add(feat("bluesky", "SEN-GA", 3.1, n=40))
        s.add(feat("x", "SEN-GA", -2.7, n=80))
        s.add(feat("x", "SEN-TX", 4.0))
        s.add(feat("x", "SEN-GA", 5.0, day=(NOW - timedelta(days=1)).strftime("%Y-%m-%d")))  # yesterday: ignored
        s.add(make_post("x:1", "Buddy Carter is dropping out!!", ["SEN-GA"], ts=NOW, bot_score=0.1))
        s.add(make_post("x:2", "spam spam", ["SEN-GA"], ts=NOW, bot_score=0.9))
        s.add(make_post("x:3", "unrelated texas post", ["SEN-TX"], ts=NOW, bot_score=0.0))

    n = await events.run_once(dbengine.session, cfg, None, now=NOW, classifier=classifier, gazetteer=make_gazetteer())
    assert n == 1
    assert seen[0][0] == "SEN-GA"
    assert seen[0][1] == ["[x] Buddy Carter is dropping out!!", "[x] spam spam"]
    assert "2 platforms" in seen[0][2] and "bluesky z=+3.1" in seen[0][2]
    with dbengine.session() as s:
        ev = s.execute(select(Event)).scalars().one()
    assert (ev.race_id, ev.kind, ev.source, ev.confidence, ev.acted) == ("SEN-GA", "hard", "social_spike", 0.85, False)
    assert ev.description.startswith("Carter drops out of the race | stance spike")
    assert ensure_utc(ev.ts) == NOW

    # same day again: no duplicate; the classifier is not called again
    assert await events.run_once(dbengine.session, cfg, None, now=NOW + timedelta(hours=2), classifier=classifier, gazetteer=make_gazetteer()) == 0
    assert len(seen) == 1


async def test_run_once_gdelt_volume_without_llm_is_unknown(db):
    cfg = StrategyConfig(sentiment={"baseline_days": 5})
    with dbengine.session() as s:
        for back in range(1, 6):
            s.add(make_post(f"gdelt:h{back}", f"old {back}", ["SEN-TX"], platform="gdelt", ts=NOW - timedelta(days=back), bot_score=None))
        for i in range(7):
            s.add(make_post(f"gdelt:t{i}", f"Texas Senate shock {i}", ["SEN-TX"], platform="gdelt", ts=NOW, bot_score=None))
        for i in range(2):
            s.add(make_post(f"gdelt:g{i}", f"Georgia {i}", ["SEN-GA"], platform="gdelt", ts=NOW, bot_score=None))

    n = await events.run_once(dbengine.session, cfg, None, now=NOW, gazetteer=make_gazetteer())
    assert n == 1
    with dbengine.session() as s:
        ev = s.execute(select(Event)).scalars().one()
    assert (ev.race_id, ev.kind, ev.source, ev.acted) == ("SEN-TX", "unknown", "gdelt_volume", False)
    assert ev.confidence <= 0.5
    assert "7 articles" in ev.description


async def test_run_once_classifier_exception_falls_back_to_unknown(db, cfg):
    async def classifier(race_context, top_posts, trigger):
        raise RuntimeError("boom")

    with dbengine.session() as s:
        s.add(feat("bluesky", "SEN-GA", 3.1))
        s.add(feat("x", "SEN-GA", 3.2))
    assert await events.run_once(dbengine.session, cfg, None, now=NOW, classifier=classifier, gazetteer=make_gazetteer()) == 1
    with dbengine.session() as s:
        ev = s.execute(select(Event)).scalars().one()
    assert ev.kind == "unknown" and ev.confidence <= 0.5
