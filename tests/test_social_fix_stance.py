"""Regression tests for stance classification: escalation must not duplicate rows (F5), the queue is
newest-first with an age cutoff and a batch can carry the whole backlog (F9), deterministic failures
are not retried forever (F11), and the prompt does not invite a sign flip (F12)."""

from __future__ import annotations

import json
from collections import Counter
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Post, Stance
from sigcup.db.repo import kv_get
from sigcup.feeds.nlp import features, stance
from sigcup.settings import StrategyConfig
from tests.test_social_helpers import NOW, FakeAnthropic, keyword_responder, make_gazetteer, make_post


@pytest.fixture()
def small_cfg() -> StrategyConfig:
    return StrategyConfig(sentiment={"max_posts_per_batch": 2})


def rows_for(post_id: str) -> list[Stance]:
    with dbengine.session() as s:
        return s.execute(select(Stance).where(Stance.post_id == post_id).order_by(Stance.id)).scalars().all()


def two_party_responder(cfg: StrategyConfig, first: list[tuple[str, str]], escalated: list[tuple[str, str]]):
    """First pass answers ``first`` [(party, stance)] per post, the escalation model ``escalated``."""

    def responder(params: dict[str, Any]) -> str:
        pairs = escalated if params["model"] == cfg.sentiment.escalation_model else first
        payload = json.loads(params["messages"][0]["content"])
        return json.dumps(
            [
                {"post_id": p["post_id"], "race_id": p["race_ids"][0], "party": party, "stance": st, "confidence": 0.8}
                for p in payload["posts"]
                for party, st in pairs
            ]
        )

    return responder


# -- F5 ---------------------------------------------------------------------------------------------


async def test_escalation_replaces_rows_without_duplicates(db, small_cfg):
    responder = two_party_responder(small_cfg, [("D", "support"), ("R", "unclear")], [("D", "support"), ("R", "oppose")])
    with dbengine.session() as s:
        s.add(make_post("x:1", "Talarico and Paxton ??? both on stage", ["SEN-TX"]))
    client = FakeAnthropic(responder)
    await stance.run_once(dbengine.session, small_cfg, None, client=client, gazetteer=make_gazetteer(), now=NOW)

    rows = rows_for("x:1")
    pairs = Counter((r.race_id, r.party) for r in rows)
    assert all(n == 1 for n in pairs.values()), pairs
    assert sorted((r.party, r.stance, r.model) for r in rows) == [
        ("D", "support", small_cfg.sentiment.escalation_model),
        ("R", "oppose", small_cfg.sentiment.escalation_model),
    ]


async def test_escalation_does_not_change_the_clipped_net(db, small_cfg):
    responder = two_party_responder(small_cfg, [("D", "oppose"), ("R", "unclear")], [("D", "oppose"), ("R", "oppose")])
    with dbengine.session() as s:
        s.add(make_post("x:1", "Talarico and Paxton ??? both awful", ["SEN-TX"]))
    await stance.run_once(dbengine.session, small_cfg, None, client=FakeAnthropic(responder), gazetteer=make_gazetteer(), now=NOW)
    with dbengine.session() as s:
        recs = features.load_records(s, NOW - timedelta(days=1))
    agg = features.aggregate(recs)
    assert agg[("x", "SEN-TX", NOW.strftime("%Y-%m-%d"))] == (1, 0.0)  # D oppose (-1) + R oppose (+1)


async def test_unclear_results_from_a_batch_are_escalated(db):
    cfg = StrategyConfig()
    with dbengine.session() as s:
        for i in range(55):
            s.add(make_post(f"x:{i}", f"I love Jon Ossoff #{i}" if i else "Paxton ??? what now", ["SEN-TX" if i == 0 else "SEN-GA"]))
    g = make_gazetteer()
    client = FakeAnthropic()
    await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=g, now=NOW)
    assert len(client.messages.batches.created) == 1

    def responder(params: dict[str, Any]) -> str:
        if params["model"] == cfg.sentiment.escalation_model:
            payload = json.loads(params["messages"][0]["content"])
            return json.dumps([{"post_id": p["post_id"], "race_id": p["race_ids"][0], "party": "R", "stance": "oppose"} for p in payload["posts"]])
        return keyword_responder(params)

    client2 = FakeAnthropic(responder)
    client2.messages.batches.created = client.messages.batches.created
    client2.messages.batches.status = "ended"
    await stance.run_once(dbengine.session, cfg, None, client=client2, gazetteer=g, now=NOW)
    assert [c["model"] for c in client2.messages.calls] == [cfg.sentiment.escalation_model]
    assert [(r.party, r.stance, r.model) for r in rows_for("x:0")] == [("R", "oppose", cfg.sentiment.escalation_model)]


# -- F9 ---------------------------------------------------------------------------------------------


def batch_post_ids(client: FakeAnthropic) -> list[str]:
    out: list[str] = []
    for req in client.messages.batches.created[-1]:
        out.extend(p["post_id"] for p in json.loads(req["params"]["messages"][0]["content"])["posts"])
    return out


async def test_newest_posts_are_queued_first_and_one_batch_takes_the_backlog(db, cfg):
    with dbengine.session() as s:
        for i in range(700):
            s.add(make_post(f"x:old{i}", f"I love Jon Ossoff old {i}", ["SEN-GA"], ts=NOW - timedelta(days=3, minutes=i)))
        for i in range(30):
            s.add(make_post(f"x:today{i}", f"I love Jon Ossoff today {i}", ["SEN-GA"], ts=NOW - timedelta(minutes=i)))
    client = FakeAnthropic()
    await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=make_gazetteer(), now=NOW)
    ids = batch_post_ids(client)
    assert {f"x:today{i}" for i in range(30)} <= set(ids)
    assert len(ids) == 730  # the whole backlog in one batch, not 500


async def test_posts_older_than_the_feature_window_are_never_sent(db, cfg):
    window = 2 * cfg.sentiment.baseline_days + 1
    with dbengine.session() as s:
        for i in range(5):
            s.add(make_post(f"x:stale{i}", f"I love Jon Ossoff stale {i}", ["SEN-GA"], ts=NOW - timedelta(days=window + 2)))
        s.add(make_post("x:fresh", "I love Jon Ossoff fresh", ["SEN-GA"], ts=NOW))
    client = FakeAnthropic()
    n = await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=make_gazetteer(), now=NOW)
    assert n == 1
    sent = [p["post_id"] for c in client.messages.calls for p in json.loads(c["messages"][0]["content"])["posts"]]
    assert sent == ["x:fresh"]
    with dbengine.session() as s:
        assert all(s.get(Post, f"x:stale{i}").classified for i in range(5))
        assert s.execute(select(Stance).where(Stance.post_id.like("x:stale%"))).scalars().all() == []


async def test_posts_in_a_pending_batch_are_not_classified_twice(db, cfg):
    with dbengine.session() as s:
        for i in range(60):
            s.add(make_post(f"x:{i}", f"I love Jon Ossoff #{i}", ["SEN-GA"]))
    client = FakeAnthropic()
    g = make_gazetteer()
    await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=g, now=NOW)
    with dbengine.session() as s:
        s.add(make_post("x:late", "I love Jon Ossoff late", ["SEN-GA"]))
    # the batch is still processing: only the new post is classified (synchronously)
    n = await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=g, now=NOW)
    assert n == 1
    sent = [p["post_id"] for c in client.messages.calls for p in json.loads(c["messages"][0]["content"])["posts"]]
    assert sent == ["x:late"]
    client.messages.batches.status = "ended"
    assert await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=g, now=NOW) == 60
    with dbengine.session() as s:
        per_post = Counter(r.post_id for r in s.execute(select(Stance)).scalars())
    assert set(per_post.values()) == {1} and len(per_post) == 61
    with dbengine.session() as s:
        assert kv_get(s, stance.KV_BATCH_KEY) is None


# -- F11 --------------------------------------------------------------------------------------------


async def test_a_poison_post_is_retried_alone_then_given_up(db, small_cfg):
    def responder(params: dict[str, Any]) -> str:
        payload = json.loads(params["messages"][0]["content"])
        if any("POISON" in p["text"] for p in payload["posts"]):
            return '[{"post_id": "x:1", "race_id": "SEN-GA", "party": "D", "stan'  # truncated at max_tokens
        return keyword_responder(params)

    with dbengine.session() as s:
        s.add(make_post("x:1", "I love Jon Ossoff", ["SEN-GA"], ts=NOW))
        s.add(make_post("x:2", "POISON Jon Ossoff", ["SEN-GA"], ts=NOW - timedelta(minutes=1)))
    client = FakeAnthropic(responder)
    g = make_gazetteer()
    calls_per_cycle = []
    for _ in range(5):
        before = len(client.messages.calls)
        await stance.run_once(dbengine.session, small_cfg, None, client=client, gazetteer=g, now=NOW)
        calls_per_cycle.append(len(client.messages.calls) - before)
    with dbengine.session() as s:
        assert s.get(Post, "x:1").classified is True
        assert s.get(Post, "x:2").classified is True  # given up after MAX_ATTEMPTS failures
    assert [r.stance for r in rows_for("x:1")] == ["support"]
    assert rows_for("x:2") == []
    assert calls_per_cycle[-1] == 0 and sum(calls_per_cycle) <= 1 + 2 + stance.MAX_ATTEMPTS


async def test_an_api_outage_does_not_count_as_a_failed_attempt(db, small_cfg):
    def boom(params: dict[str, Any]) -> str:
        raise RuntimeError("network down")

    with dbengine.session() as s:
        s.add(make_post("x:1", "I love Jon Ossoff", ["SEN-GA"]))
    for _ in range(stance.MAX_ATTEMPTS + 1):
        await stance.run_once(dbengine.session, small_cfg, None, client=FakeAnthropic(boom), gazetteer=make_gazetteer(), now=NOW)
    with dbengine.session() as s:
        assert s.get(Post, "x:1").classified is False
    assert await stance.run_once(dbengine.session, small_cfg, None, client=FakeAnthropic(), gazetteer=make_gazetteer(), now=NOW) == 1


# -- F12 --------------------------------------------------------------------------------------------


def test_prompt_does_not_relabel_attacks_as_support_for_the_attacked_party():
    p = stance.SYSTEM_PROMPT
    assert "attacking the opponent counts as support" not in p
    assert "stance=oppose" in p and "do not relabel" in p.lower()
