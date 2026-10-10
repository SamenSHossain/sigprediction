"""Regression tests for the platform collectors: the shared test ``secrets()`` helper is hermetic (F14),
a Jetstream slice is bounded even on an endless or stalled stream, and YouTube rotates through the
configured channels instead of always reading the first ~33 (F15)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import respx

from sigcup.db import engine as dbengine
from sigcup.db.repo import kv_get
from sigcup.feeds import social_bluesky, social_youtube
from tests import test_social_platforms
from tests.test_social_helpers import NOW, make_gazetteer

# -- F14 --------------------------------------------------------------------------------------------


def test_platform_tests_secrets_ignore_exported_credentials(monkeypatch):
    for var, val in {
        "BLUESKY_HANDLE": "someone.bsky.social",
        "BLUESKY_APP_PASSWORD": "abcd-efgh-ijkl-mnop",
        "X_VENDOR": "official",
        "X_VENDOR_KEY": "live-key",
        "YOUTUBE_API_KEY": "live-yt",
        "REDDIT_CLIENT_ID": "id",
        "REDDIT_CLIENT_SECRET": "secret",
        "ANTHROPIC_API_KEY": "sk-live",
    }.items():
        monkeypatch.setenv(var, val)
    s = test_social_platforms.secrets()
    assert (s.bluesky_handle, s.bluesky_app_password) == ("", "")
    assert (s.x_vendor, s.x_vendor_key, s.youtube_api_key) == ("none", "", "")
    assert (s.reddit_client_id, s.reddit_client_secret, s.anthropic_api_key) == ("", "", "")
    assert social_bluesky._make_search_client(s) is None
    # explicit keyword arguments still win
    assert test_social_platforms.secrets(bluesky_handle="h").bluesky_handle == "h"


class _EndlessWS:
    """A firehose that never ends: every message is an unrelated English post."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s, self.n = delay_s, 0

    def __aiter__(self) -> _EndlessWS:
        return self

    async def __anext__(self) -> str:
        self.n += 1
        await asyncio.sleep(self.delay_s)
        return json.dumps({"did": "did:plc:z", "time_us": 1791900000000000, "kind": "commit",
                           "commit": {"operation": "create", "collection": "app.bsky.feed.post", "rkey": f"r{self.n}",
                                      "record": {"text": f"lunch was great {self.n}", "langs": ["en"]}}})


class _Connect:
    def __init__(self, ws: Any) -> None:
        self.ws = ws

    def __call__(self, url: str) -> _Connect:
        return self

    async def __aenter__(self) -> Any:
        return self.ws

    async def __aexit__(self, *exc: Any) -> bool:
        return False


async def test_jetstream_slice_is_bounded_on_an_endless_stream():
    ws = _EndlessWS()
    t0 = time.monotonic()
    kept = await social_bluesky.consume_jetstream(make_gazetteer(), 0.3, connect=_Connect(ws))
    assert kept == [] and ws.n > 10
    assert time.monotonic() - t0 < 2.0


async def test_jetstream_slice_is_bounded_on_a_stalled_stream():
    ws = _EndlessWS(delay_s=30.0)
    t0 = time.monotonic()
    assert await social_bluesky.consume_jetstream(make_gazetteer(), 0.3, connect=_Connect(ws)) == []
    assert time.monotonic() - t0 < 2.0


# -- F15 --------------------------------------------------------------------------------------------


@respx.mock
async def test_youtube_rotates_through_channels_across_runs(db, cfg):
    playlist = respx.get(host="www.googleapis.com", path="/youtube/v3/playlistItems").mock(return_value=httpx.Response(200, json={"items": []}))
    sources = {"global": {"youtube_channels": ["UCone", "UCtwo", "UCthree"]}}
    secrets = test_social_platforms.secrets(youtube_api_key="key")
    seen: list[str] = []
    for _ in range(4):
        before = playlist.call_count
        await social_youtube.run_once(dbengine.session, cfg, secrets, sources=sources, gazetteer=make_gazetteer(), now=NOW, max_units=1)
        seen.extend(c.request.url.params["playlistId"] for c in playlist.calls[before:])
    assert seen == ["UUone", "UUtwo", "UUthree", "UUone"]
    with dbengine.session() as s:
        assert kv_get(s, social_youtube.KV_CURSOR) == 1
