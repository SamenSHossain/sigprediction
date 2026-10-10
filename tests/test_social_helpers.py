"""Shared helpers for the news/social/NLP tests (no tests in this module).

An inline race dict (the gazetteer is never read from config in tests), fixture loaders, a fake
Anthropic client with Messages + Message Batches surfaces, and a Post factory.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from sigcup.common.ids import text_hash
from sigcup.db.models import Post
from sigcup.feeds.nlp.gazetteer import Gazetteer

FIXTURES = Path(__file__).parent / "fixtures" / "social"
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)

RACES: dict[str, dict[str, Any]] = {
    "SEN-GA": {
        "chamber": "senate", "state": "GA", "district": None, "label": "Georgia Senate",
        "candidates": {
            "D": {"name": "Jon Ossoff", "aliases": [], "handles": ["ossoff"], "hashtags": ["TeamOssoff"]},
            "R": {"name": "Buddy Carter", "aliases": ["Earl Carter"], "handles": ["buddycarter"], "hashtags": []},
            "I": {"name": "Chuck Jones"},
        },
    },
    "GOV-GA": {
        "chamber": "governor", "state": "GA", "district": None, "label": "Georgia Governor",
        "candidates": {
            "R": {"name": "Burt Jones", "handles": ["burtjonesga"]},
            "D": {"name": "Keisha Lance Bottoms", "aliases": ["KLB"]},
        },
    },
    "SEN-TX": {
        "chamber": "senate", "state": "TX", "district": None, "label": "Texas Senate",
        "candidates": {"D": {"name": "Colin Allred"}, "R": {"name": "Ken Paxton"}},
    },
    "HOUSE-PA-08": {
        "chamber": "house", "state": "PA", "district": "08", "label": "PA-08 House",
        "candidates": {"R": {"name": "Rob Bresnahan"}, "D": {"name": "Matt Cartwright"}},
    },
    "HOUSE-NY-03": {
        "chamber": "house", "state": "NY", "district": "03", "label": "NY-03 House",
        "candidates": {"R": {"name": "Mike Jones"}},
    },
    "SEN-AR": {
        "chamber": "senate", "state": "AR", "district": None, "label": "Arkansas Senate",
        "candidates": {"R": {"name": "Tom Cotton"}},
    },
    "CONTROL-SENATE": {"chamber": "control", "state": None, "district": None, "label": "Senate control", "candidates": {}},
}


def make_gazetteer() -> Gazetteer:
    return Gazetteer.from_races(RACES)


def load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text())


def load_text(name: str) -> str:
    return (FIXTURES / name).read_text()


def make_post(
    post_id: str,
    text: str,
    race_ids: list[str] | None,
    *,
    platform: str = "x",
    ts: datetime = NOW,
    bot_score: float | None = 0.0,
    classified: bool = False,
    author_id: str | None = None,
    author_handle: str | None = None,
) -> Post:
    return Post(
        post_id=post_id,
        platform=platform,
        ts=ts,
        author_id=author_id,
        author_handle=author_handle,
        author_created_at=None,
        text=text,
        text_hash=text_hash(text),
        url=None,
        race_ids=race_ids,
        bot_score=bot_score,
        classified=classified,
        fetched_at=ts,
    )


# ---------------------------------------------------------------------------------------------
# Fake Anthropic client
# ---------------------------------------------------------------------------------------------


def text_message(text: str) -> SimpleNamespace:
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")


def keyword_responder(params: dict[str, Any]) -> str:
    """Deterministic stance answers from keywords: love -> support, hate -> oppose, ??? -> unclear."""
    payload = json.loads(params["messages"][0]["content"])
    races = payload["races"]
    out = []
    for p in payload["posts"]:
        text = p["text"].lower()
        for rid in p["race_ids"]:
            cands = races.get(rid, {}).get("candidates") or {"D": "?"}
            party = sorted(cands)[0]
            if "love" in text:
                stance = "support"
            elif "hate" in text:
                stance = "oppose"
            elif "???" in text:
                stance = "unclear"
            else:
                stance = "neutral"
            out.append(
                {"post_id": p["post_id"], "race_id": rid, "party": party, "stance": stance, "confidence": 0.8,
                 "is_news_or_poll": "poll" in text}
            )
    return json.dumps(out)


class FakeBatches:
    def __init__(self, responder: Callable[[dict[str, Any]], str]) -> None:
        self.responder = responder
        self.created: list[list[dict[str, Any]]] = []
        self.status = "in_progress"

    def create(self, requests: Any) -> SimpleNamespace:
        self.created.append(list(requests))
        return SimpleNamespace(id=f"msgbatch_{len(self.created)}", processing_status="in_progress")

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        return SimpleNamespace(id=batch_id, processing_status=self.status)

    def results(self, batch_id: str):  # noqa: ANN201 - generator like the SDK's JSONLDecoder
        reqs = self.created[int(batch_id.rsplit("_", 1)[-1]) - 1]
        for r in reqs:
            yield SimpleNamespace(
                custom_id=r["custom_id"],
                result=SimpleNamespace(type="succeeded", message=text_message(self.responder(r["params"]))),
            )


class FakeMessages:
    def __init__(self, responder: Callable[[dict[str, Any]], str]) -> None:
        self.responder = responder
        self.calls: list[dict[str, Any]] = []
        self.batches = FakeBatches(responder)

    def create(self, **params: Any) -> SimpleNamespace:
        self.calls.append(params)
        return text_message(self.responder(params))


class FakeAnthropic:
    """Enough of ``anthropic.Anthropic`` for stance.py and events.py: ``messages.create`` and ``messages.batches``."""

    def __init__(self, responder: Callable[[dict[str, Any]], str] | None = None) -> None:
        self.messages = FakeMessages(responder or keyword_responder)
