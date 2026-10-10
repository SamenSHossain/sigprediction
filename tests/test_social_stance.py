"""Stance classification: JSON parsing, cache hits, escalation and the Message Batches path with a fake client."""

from __future__ import annotations

import json

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Post, Stance
from sigcup.db.repo import kv_get
from sigcup.feeds.nlp import stance
from sigcup.settings import Secrets, StrategyConfig
from tests.test_social_helpers import NOW, FakeAnthropic, keyword_responder, make_gazetteer, make_post


@pytest.fixture()
def small_cfg() -> StrategyConfig:
    return StrategyConfig(sentiment={"max_posts_per_batch": 2})


def stances(post_id: str) -> list[Stance]:
    with dbengine.session() as s:
        return s.execute(select(Stance).where(Stance.post_id == post_id).order_by(Stance.id)).scalars().all()


# -- parsing --------------------------------------------------------------------------------------


def test_parse_results_lenient():
    text = (
        "Sure, here you go:\n```json\n"
        '[{"post_id":"x:1","race_id":"sen-ga","party":"Dem","stance":"supports","confidence":85,"is_news_or_poll":"false"},'
        ' {"post_id":"x:1","race_id":"SEN-GA","party":"D","stance":"support","confidence":0.9},'
        ' {"post_id":"x:2","race_id":"SEN-TX","party":"R","stance":"against","confidence":0.7,"is_news_or_poll":true},'
        ' {"post_id":"x:9","race_id":"SEN-GA","party":"D","stance":"support"},'
        ' {"post_id":"x:2","race_id":"GOV-GA","party":"R","stance":"oppose"},'
        ' {"post_id":"x:3","race_id":"SEN-GA","party":"Green","stance":"support"},'
        ' "junk"]\n```'
    )
    allowed = {"x:1": ["SEN-GA"], "x:2": ["SEN-TX"], "x:3": ["SEN-GA"]}
    out = stance.parse_results(text, allowed)
    assert [(r.post_id, r.race_id, r.party, r.stance, r.confidence, r.is_news_or_poll) for r in out] == [
        ("x:1", "SEN-GA", "D", "support", 0.85, False),
        ("x:2", "SEN-TX", "R", "oppose", 0.7, True),
    ]


def test_parse_results_wrappers_and_failures():
    assert stance.parse_results('{"results": [{"post_id":"a","race_id":"R1","party":"D","stance":"neutral"}]}') == [
        stance.StanceResult("a", "R1", "D", "neutral", 0.5, False)
    ]
    assert stance.parse_results("I cannot classify these.") is None
    assert stance.parse_results("") is None
    assert stance.parse_results("[]") == []
    assert stance.parse_results('{"post_id":"a","race_id":"R1","party":"R","stance":"bogus"}')[0].stance == "unclear"


def test_response_text_handles_refusal_and_dicts():
    assert stance.response_text({"content": [{"type": "text", "text": "[]"}], "stop_reason": "end_turn"}) == "[]"
    assert stance.response_text({"content": [{"type": "text", "text": "[]"}], "stop_reason": "refusal"}) == ""
    assert stance.response_text(None) == ""


def test_build_params_shape():
    g = make_gazetteer()
    posts = [make_post("x:1", "Jon Ossoff rocks", ["SEN-GA"], author_handle="v")]
    params = stance.build_params(posts, g, "claude-haiku-5-5")
    assert params["model"] == "claude-haiku-5-5" and params["output_config"] == {"effort": "low"}
    payload = json.loads(params["messages"][0]["content"])
    assert payload["races"]["SEN-GA"]["candidates"]["D"] == "Jon Ossoff"
    assert payload["posts"][0] == {"post_id": "x:1", "race_ids": ["SEN-GA"], "author": "v", "platform": "x", "text": "Jon Ossoff rocks"}


# -- run_once -------------------------------------------------------------------------------------


async def test_no_api_key_returns_zero(db, cfg):
    with dbengine.session() as s:
        s.add(make_post("x:1", "I love Jon Ossoff", ["SEN-GA"]))
    assert await stance.run_once(dbengine.session, cfg, Secrets(_env_file=None, anthropic_api_key="")) == 0
    with dbengine.session() as s:
        assert s.get(Post, "x:1").classified is False


async def test_sync_path_writes_stances_and_marks_classified(db, small_cfg):
    with dbengine.session() as s:
        s.add(make_post("x:1", "I love Jon Ossoff", ["SEN-GA"]))
        s.add(make_post("x:2", "I hate Ken Paxton, new poll says", ["SEN-TX"]))
        s.add(make_post("x:3", "Jon Ossoff and Burt Jones both spoke", ["SEN-GA", "GOV-GA"]))
        s.add(make_post("x:4", "no races here", []))
    client = FakeAnthropic()
    n = await stance.run_once(dbengine.session, small_cfg, None, client=client, gazetteer=make_gazetteer(), now=NOW)
    assert n == 4  # x:1 -> 1, x:2 -> 1, x:3 -> 2
    assert len(client.messages.calls) == 2  # 3 posts, 2 per request
    assert client.messages.calls[0]["model"] == small_cfg.sentiment.classifier_model
    r1 = stances("x:1")
    assert [(r.race_id, r.party, r.stance, r.model) for r in r1] == [("SEN-GA", "D", "support", "claude-haiku-5-5")]
    r2 = stances("x:2")
    assert r2[0].stance == "oppose" and r2[0].is_news_or_poll is True
    assert {r.race_id for r in stances("x:3")} == {"SEN-GA", "GOV-GA"}
    with dbengine.session() as s:
        assert all(s.get(Post, pid).classified for pid in ("x:1", "x:2", "x:3", "x:4"))
    assert stances("x:4") == []


async def test_cache_hit_reuses_stance_for_same_text_hash(db, small_cfg):
    with dbengine.session() as s:
        s.add(make_post("x:1", "I love Jon Ossoff", ["SEN-GA"], classified=True))
        s.add(Stance(post_id="x:1", race_id="SEN-GA", party="D", stance="support", confidence=0.9, is_news_or_poll=False, model="claude-haiku-5-5", ts=NOW))
        s.add(make_post("bluesky:2", "  i LOVE jon ossoff ", ["SEN-GA"], platform="bluesky"))
    client = FakeAnthropic()
    n = await stance.run_once(dbengine.session, small_cfg, None, client=client, gazetteer=make_gazetteer(), now=NOW)
    assert n == 1 and client.messages.calls == []
    rows = stances("bluesky:2")
    assert [(r.race_id, r.party, r.stance, r.confidence, r.model) for r in rows] == [("SEN-GA", "D", "support", 0.9, "cache")]
    with dbengine.session() as s:
        assert s.get(Post, "bluesky:2").classified is True


async def test_unclear_is_escalated_to_the_escalation_model(db, small_cfg):
    def responder(params):
        if params["model"] == small_cfg.sentiment.escalation_model:
            payload = json.loads(params["messages"][0]["content"])
            return json.dumps([{"post_id": p["post_id"], "race_id": p["race_ids"][0], "party": "R", "stance": "oppose", "confidence": 0.75} for p in payload["posts"]])
        return keyword_responder(params)

    with dbengine.session() as s:
        s.add(make_post("x:1", "Paxton ??? what is he even doing", ["SEN-TX"]))
        s.add(make_post("x:2", "I love Colin Allred", ["SEN-TX"]))
    client = FakeAnthropic(responder)
    n = await stance.run_once(dbengine.session, small_cfg, None, client=client, gazetteer=make_gazetteer(), now=NOW)
    assert n == 3  # two first-pass rows + one replacement row
    assert [c["model"] for c in client.messages.calls] == [small_cfg.sentiment.classifier_model, small_cfg.sentiment.escalation_model]
    rows = stances("x:1")
    assert [(r.stance, r.party, r.model) for r in rows] == [("oppose", "R", small_cfg.sentiment.escalation_model)]
    assert stances("x:2")[0].stance == "support"


async def test_unparseable_response_leaves_posts_for_retry(db, small_cfg):
    with dbengine.session() as s:
        s.add(make_post("x:1", "I love Jon Ossoff", ["SEN-GA"]))
    client = FakeAnthropic(lambda params: "Sorry, I can't help with that.")
    assert await stance.run_once(dbengine.session, small_cfg, None, client=client, gazetteer=make_gazetteer(), now=NOW) == 0
    with dbengine.session() as s:
        assert s.get(Post, "x:1").classified is False


async def test_api_error_is_swallowed(db, small_cfg):
    def boom(params):
        raise RuntimeError("network down")

    with dbengine.session() as s:
        s.add(make_post("x:1", "I love Jon Ossoff", ["SEN-GA"]))
    assert await stance.run_once(dbengine.session, small_cfg, None, client=FakeAnthropic(boom), gazetteer=make_gazetteer(), now=NOW) == 0


async def test_batch_path_submits_persists_and_resumes(db, cfg):
    with dbengine.session() as s:
        for i in range(60):
            s.add(make_post(f"x:{i}", f"I love Jon Ossoff #{i}", ["SEN-GA"]))
    client = FakeAnthropic()
    g = make_gazetteer()

    n = await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=g, now=NOW)
    assert n == 0 and client.messages.calls == []
    assert len(client.messages.batches.created) == 1
    requests = client.messages.batches.created[0]
    assert len(requests) == 3  # 60 posts / 25 per request
    assert requests[0]["params"]["model"] == cfg.sentiment.classifier_model
    with dbengine.session() as s:
        state = kv_get(s, stance.KV_BATCH_KEY)
        assert state["batch_id"] == "msgbatch_1"
        assert sum(len(v) for v in state["chunks"].values()) == 60
        assert s.get(Post, "x:0").classified is False

    # still processing: nothing new is submitted, nothing classified
    n = await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=g, now=NOW)
    assert n == 0 and len(client.messages.batches.created) == 1

    # a restart: a fresh client object, the batch id comes from KV
    client2 = FakeAnthropic()
    client2.messages.batches.created = client.messages.batches.created
    client2.messages.batches.status = "ended"
    n = await stance.run_once(dbengine.session, cfg, None, client=client2, gazetteer=g, now=NOW)
    assert n == 60
    with dbengine.session() as s:
        assert kv_get(s, stance.KV_BATCH_KEY) is None
        assert s.execute(select(Post).where(Post.classified.is_(False))).scalars().all() == []
        assert s.execute(select(Stance)).scalars().all()[0].model == cfg.sentiment.classifier_model


async def test_batches_can_be_disabled(db, cfg):
    with dbengine.session() as s:
        for i in range(60):
            s.add(make_post(f"x:{i}", f"I love Jon Ossoff #{i}", ["SEN-GA"]))
    client = FakeAnthropic()
    n = await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=make_gazetteer(), now=NOW, use_batches=False)
    assert n == 60 and len(client.messages.calls) == 3 and client.messages.batches.created == []
