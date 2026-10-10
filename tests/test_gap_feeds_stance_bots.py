"""C1: posts the bot filter flags (``bot_score > BOT_THRESHOLD``, the cut features.py applies) are never
sent to the classifier; they are marked classified without a Stance row so they never re-enter the queue."""

from __future__ import annotations

import json
from datetime import timedelta

from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Post, Stance
from sigcup.feeds.nlp import features, stance
from sigcup.feeds.nlp.botfilter import BOT_THRESHOLD
from sigcup.settings import StrategyConfig
from tests.test_social_helpers import NOW, FakeAnthropic, make_gazetteer, make_post


def sent_ids(client: FakeAnthropic) -> list[str]:
    return [p["post_id"] for c in client.messages.calls for p in json.loads(c["messages"][0]["content"])["posts"]]


def batch_ids(client: FakeAnthropic) -> list[str]:
    return [
        p["post_id"]
        for batch in client.messages.batches.created
        for req in batch
        for p in json.loads(req["params"]["messages"][0]["content"])["posts"]
    ]


def stance_rows(prefix: str) -> list[Stance]:
    with dbengine.session() as s:
        return s.execute(select(Stance).where(Stance.post_id.like(f"{prefix}%"))).scalars().all()


def test_the_threshold_is_the_one_features_uses():
    assert stance.BOT_THRESHOLD is BOT_THRESHOLD is features.BOT_THRESHOLD


async def test_bot_flagged_posts_are_not_sent_and_not_requeued(db):
    cfg = StrategyConfig(sentiment={"max_posts_per_batch": 10})
    with dbengine.session() as s:
        s.add(make_post("x:bot", "I love Jon Ossoff bot", ["SEN-GA"], bot_score=0.95))
        s.add(make_post("x:edge", "I love Jon Ossoff edge", ["SEN-GA"], bot_score=BOT_THRESHOLD))  # not above: kept
        s.add(make_post("x:human", "I love Jon Ossoff human", ["SEN-GA"], bot_score=0.10))
        s.add(make_post("x:unscored", "I love Jon Ossoff unscored", ["SEN-GA"], bot_score=None))
    client = FakeAnthropic()
    g = make_gazetteer()
    n = await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=g, now=NOW)
    assert n == 3
    assert sorted(sent_ids(client)) == ["x:edge", "x:human", "x:unscored"]
    with dbengine.session() as s:
        assert s.get(Post, "x:bot").classified is True
    assert stance_rows("x:bot") == []
    # nothing is re-queued on the next cycle
    calls = len(client.messages.calls)
    assert await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=g, now=NOW) == 0
    assert len(client.messages.calls) == calls


async def test_bot_flagged_posts_never_reach_a_message_batch(db):
    cfg = StrategyConfig()
    with dbengine.session() as s:
        for i in range(60):
            s.add(make_post(f"x:h{i}", f"I love Jon Ossoff human {i}", ["SEN-GA"], bot_score=0.2))
        for i in range(40):
            s.add(make_post(f"x:b{i}", f"I love Jon Ossoff bot {i}", ["SEN-GA"], bot_score=0.9))
    client = FakeAnthropic()
    await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=make_gazetteer(), now=NOW)
    ids = batch_ids(client)
    assert len(ids) == 60 and all(pid.startswith("x:h") for pid in ids)
    with dbengine.session() as s:
        assert all(s.get(Post, f"x:b{i}").classified for i in range(40))


async def test_bot_posts_do_not_crowd_the_queue_limit(db):
    """Many unclassified bot posts newer than a human post must not push it out of this cycle's queue."""
    cfg = StrategyConfig(sentiment={"max_posts_per_batch": 5})
    with dbengine.session() as s:
        for i in range(30):
            s.add(make_post(f"x:b{i}", f"I love Jon Ossoff bot {i}", ["SEN-GA"], bot_score=0.99, ts=NOW - timedelta(minutes=i)))
        s.add(make_post("x:human", "I love Jon Ossoff human", ["SEN-GA"], bot_score=0.0, ts=NOW - timedelta(hours=2)))
    client = FakeAnthropic()
    await stance.run_once(dbengine.session, cfg, None, client=client, gazetteer=make_gazetteer(), now=NOW, use_batches=False, max_posts=5)
    assert sent_ids(client) == ["x:human"]
