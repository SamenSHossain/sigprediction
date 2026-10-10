"""Regression tests for review findings on RealtimeFeed (sigcup/sig/realtime.py): token reuse and backoff on a
flapping socket (F2), the hourly token cap, refresh after an auth-type join failure, and a token refresh that
must not stall the heartbeat loop (F9). Everything runs against websockets servers on 127.0.0.1."""

from __future__ import annotations

import asyncio
import json
import socket
from datetime import UTC, datetime, timedelta

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from sigcup.sig.client import RealtimeToken
from sigcup.sig.realtime import BookUpdate, RealtimeFeed
from tests.test_sig_realtime import FakeSupabase, TokenSource, collect

TID = "tour-2026-midterms"


@pytest.fixture()
async def start_server():
    servers: list = []

    async def start(handler) -> int:
        server = await serve(handler, "127.0.0.1", 0)
        servers.append(server)
        return server.sockets[0].getsockname()[1]

    yield start
    for s in servers:
        s.close()
        await s.wait_closed()


async def run_for(feed: RealtimeFeed, seconds: float) -> list:
    out: list = []
    gen = feed.events()

    async def consume() -> None:
        async for ev in gen:
            out.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(seconds)
    await feed.stop()
    try:
        await asyncio.wait_for(task, 2)
    except (TimeoutError, asyncio.CancelledError):
        task.cancel()
    await gen.aclose()
    return out


def closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------------------------
# F2: a socket that accepts the handshake and then drops must not burn a write per second
# ---------------------------------------------------------------------------------------------


async def test_flapping_socket_reuses_the_token_and_backs_off(start_server):
    connections = 0

    async def accept_then_close(ws) -> None:
        nonlocal connections
        connections += 1  # returning closes the socket right after the handshake

    port = await start_server(accept_then_close)
    tokens = TokenSource(port)
    feed = RealtimeFeed(
        tokens,
        TID,
        markets={"387": "1076"},
        user_channel="user:prof-42",
        max_channels=94,
        tick_s=0.02,
        reconnect_min_s=0.05,
        reconnect_max_s=5.0,
    )
    await run_for(feed, 1.0)
    assert connections >= 3  # it does keep trying
    assert connections <= 6  # 0.05 -> 0.1 -> 0.2 -> 0.4 -> 0.8: no reset without a successful join
    assert tokens.calls == 1  # one token for every attempt while it is valid


async def test_expiring_token_is_not_reused(start_server):
    async def accept_then_close(ws) -> None:
        return None

    port = await start_server(accept_then_close)
    calls = 0

    async def short_lived() -> RealtimeToken:
        nonlocal calls
        calls += 1
        return RealtimeToken(
            supabaseUrl=f"http://127.0.0.1:{port}",
            anonKey="anon",
            token=f"jwt-{calls}",
            expiresAt=datetime.now(UTC) + timedelta(minutes=5),
        )

    feed = RealtimeFeed(
        short_lived, TID, markets={"387": "1076"}, max_channels=94, tick_s=0.02, reconnect_min_s=0.05
    )
    await run_for(feed, 0.4)
    assert calls >= 2  # 5 minutes left is too little to reuse


async def test_token_requests_are_capped_per_hour():
    port = closed_port()
    calls = 0

    async def provider() -> RealtimeToken:
        nonlocal calls
        calls += 1
        return RealtimeToken(supabaseUrl=f"http://127.0.0.1:{port}", anonKey="a", token=f"t{calls}")

    feed = RealtimeFeed(
        provider,
        TID,
        markets={"387": "1076"},
        max_channels=94,
        reconnect_min_s=0.01,
        reconnect_max_s=0.02,
        max_tokens_per_hour=3,
    )
    await run_for(feed, 0.6)
    assert calls == 3  # then REST only until the hour rolls over
    assert not feed.connected


async def test_auth_join_failure_refreshes_the_token_and_rejoins(start_server):
    fake = FakeSupabase(expected_joins=2)
    joins: list[tuple[str, str]] = []

    async def handler(ws) -> None:
        # reject any join made with the first token as an auth failure; delegate the rest to FakeSupabase
        async def guarded():
            async for raw in ws:
                msg = json.loads(raw)
                if msg["event"] == "phx_join":
                    joins.append((msg["topic"], msg["payload"]["access_token"]))
                    if msg["payload"]["access_token"] == "jwt-1":
                        await ws.send(
                            json.dumps(
                                {
                                    "topic": msg["topic"],
                                    "event": "phx_reply",
                                    "ref": msg["ref"],
                                    "payload": {
                                        "status": "error",
                                        "response": {"reason": "Unauthorized: invalid or expired token"},
                                    },
                                }
                            )
                        )
                        continue
                yield raw

        class Proxy:
            def __init__(self, inner):
                self.inner = inner

            def __aiter__(self):
                return guarded()

            async def send(self, data):
                await self.inner.send(data)

        try:
            await fake.handler(Proxy(ws))
        except ConnectionClosed:
            pass

    fake.port = await start_server(handler)
    tokens = TokenSource(fake.port)
    feed = RealtimeFeed(
        tokens,
        TID,
        markets={"387": "1076"},
        user_channel="user:prof-42",
        max_channels=94,
        tick_s=0.02,
        reconnect_min_s=0.05,
        join_retry_s=0.1,
    )
    evs = await collect(feed, lambda out: any(isinstance(e, BookUpdate) for e in out), timeout=4.0)
    assert any(isinstance(e, BookUpdate) for e in evs)
    assert tokens.calls == 2
    assert {tok for _, tok in joins} == {"jwt-1", "jwt-2"}
    assert fake.connections == 1  # recovered on the same socket


# ---------------------------------------------------------------------------------------------
# F9: a slow token refresh must not block heartbeats or reads
# ---------------------------------------------------------------------------------------------


async def test_slow_token_refresh_does_not_stall_the_session(start_server):
    fake = FakeSupabase(expected_joins=2)
    fake.port = await start_server(fake.handler)
    release = asyncio.Event()
    calls = 0
    hb_at_block: list[int] = []

    async def provider() -> RealtimeToken:
        nonlocal calls
        calls += 1
        if calls == 2:  # the refresh: stuck behind a frozen write budget
            hb_at_block.append(fake.heartbeats)
            await release.wait()
        return RealtimeToken(
            supabaseUrl=f"http://127.0.0.1:{fake.port}",
            anonKey="anon",
            token=f"jwt-{calls}",
            expiresAt=datetime.now(UTC) + timedelta(hours=3),
            user_channel="user:prof-42",
        )

    feed = RealtimeFeed(
        provider,
        TID,
        markets={"387": "1076"},
        user_channel="user:prof-42",
        max_channels=94,
        heartbeat_s=0.1,
        tick_s=0.02,
        token_refresh_s=0.3,
        reconnect_min_s=0.05,
    )

    def heartbeats_continue(_out) -> bool:
        if hb_at_block and fake.heartbeats >= hb_at_block[0] + 3:
            release.set()  # the refresh completes only after the session proved it kept running
        return release.is_set() and len(fake.access_tokens) >= 2

    await collect(feed, heartbeats_continue, timeout=4.0)
    assert {tok for _, tok in fake.access_tokens} == {"jwt-2"}
    assert fake.connections == 1
