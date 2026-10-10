"""RealtimeFeed: fixture-driven payload parsing, revision-gap detection and a fake Supabase/Phoenix server
on localhost (join, broadcast, heartbeat, token refresh, reconnect with full rejoin, channel cap)."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from sigcup.sig.client import RealtimeToken
from sigcup.sig.realtime import (
    BookUpdate,
    FillEvent,
    RealtimeFeed,
    ResyncNeeded,
    RevisionTracker,
    SettledEvent,
    decode_frame,
    join_payload,
    market_id_from_topic,
    market_topic,
    parse_broadcast,
    unwrap_broadcast,
    websocket_url,
)

FIX = Path(__file__).parent / "fixtures" / "sig"
TID = "tour-2026-midterms"
MARKETS = {"387": "1076", "388": "1077", "999": None}


def frame(name: str) -> str:
    return (FIX / name).read_text()


def events_from(name: str, tracker: RevisionTracker | None = None, markets=MARKETS):
    topic, event, payload, _ref = decode_frame(frame(name))
    inner_event, inner_payload = unwrap_broadcast(event, payload)
    return parse_broadcast(inner_event, inner_payload, topic=topic, tracker=tracker, markets=markets)


# ---------------------------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------------------------


def test_topic_helpers_and_url():
    assert market_topic("T", "M") == "tournament:T:market:M"
    assert market_id_from_topic("realtime:tournament:T:market:387") == "387"
    assert market_id_from_topic("tournament:T:market:387") == "387"
    assert market_id_from_topic("realtime:user:prof-42") is None
    assert market_id_from_topic(None) is None
    url = websocket_url("https://abcd.supabase.co", "anon key")
    assert url == "wss://abcd.supabase.co/realtime/v1/websocket?apikey=anon%20key&vsn=1.0.0"
    assert websocket_url("http://127.0.0.1:1234", "a").startswith(
        "ws://127.0.0.1:1234/realtime/v1/websocket?"
    )
    assert websocket_url("wss://x.supabase.co/realtime/v1", "a").startswith(
        "wss://x.supabase.co/realtime/v1/websocket"
    )


def test_decode_frame_object_and_array_encodings():
    topic, event, payload, ref = decode_frame(frame("rt_phx_reply.json"))
    assert (
        topic == "realtime:tournament:tour-2026-midterms:market:387" and event == "phx_reply" and ref == "1"
    )
    assert payload["status"] == "ok"
    topic, event, payload, ref = decode_frame(frame("rt_market_batch_v2.json"))
    assert event == "broadcast" and ref is None and payload["event"] == "market_batch"
    assert decode_frame("not json") == (None, None, None, None)
    assert decode_frame("[1,2]") == (None, None, None, None)


def test_join_payload_is_private_and_carries_the_token():
    p = join_payload("jwt")
    assert p["access_token"] == "jwt" and p["config"]["private"] is True
    assert p["config"]["broadcast"] == {"ack": False, "self": False}


# ---------------------------------------------------------------------------------------------
# Fixture-driven parser
# ---------------------------------------------------------------------------------------------


def test_market_batch_fixture_yields_book_updates():
    evs = events_from("rt_market_batch.json")
    assert [type(e) for e in evs] == [BookUpdate, BookUpdate]
    b1076, b1077 = evs
    assert b1076.exchange_id == "1076" and b1076.market_id == "387"
    assert [lv.price for lv in b1076.bids] == [0.845, 0.84]  # best first
    assert [lv.price for lv in b1076.asks] == [0.88, 0.9]
    assert b1076.best_bid == 0.845 and b1076.best_ask == 0.88
    assert b1076.revision == 12 and b1076.sequence == 993
    assert b1076.ts == datetime(2026, 10, 10, 12, 0, 2, tzinfo=UTC)
    assert b1076.next_expiry_at == datetime(2026, 10, 10, 12, 10, tzinfo=UTC)
    assert len(b1076.trades) == 1 and b1076.trades[0]["price"] == 0.88
    assert b1077.exchange_id == "1077" and b1077.bids[0].quantity == 20 and b1077.asks[0].price == 0.155
    assert b1077.trades == []  # the trade belonged to 1076


def test_market_batch_array_frame_parses_the_same_way():
    evs = events_from("rt_market_batch_v2.json")
    assert len(evs) == 1 and isinstance(evs[0], BookUpdate)
    assert evs[0].revision == 13 and evs[0].asks == [] and evs[0].bids[0].price == 0.85


def test_book_dirty_and_market_settled_fixtures():
    (dirty,) = events_from("rt_book_dirty.json")
    assert isinstance(dirty, ResyncNeeded)
    assert dirty.exchange_id == "1076" and dirty.market_id == "387" and dirty.reason == "book_dirty"
    (settled,) = events_from("rt_market_settled.json")
    assert isinstance(settled, SettledEvent)
    assert settled.exchange_id == "1077" and settled.market_id == "388"


def test_account_batch_fixture_yields_fills_and_resync():
    evs = events_from("rt_account_batch.json")
    fills = [e for e in evs if isinstance(e, FillEvent)]
    resyncs = [e for e in evs if isinstance(e, ResyncNeeded)]
    assert len(fills) == 2 and len(resyncs) == 1
    assert resyncs[0].exchange_id is None and resyncs[0].reason == "account_resync_required"
    f1, f2 = fills
    assert f1.fill.order_id == "o-1" and f1.fill.exchange_id == "1076" and f1.fill.quantity == 20
    assert f1.fill.fill_id.startswith("synth:") and f1.revision == 3
    assert f1.fill.ts == datetime(2026, 10, 10, 12, 0, 5, tzinfo=UTC)
    assert f2.fill.quantity == 50 and f2.fill.side == "no"  # signed quantity -> NO


def test_sparse_market_batch_with_missing_fields():
    # no exchangeId in the book and no mapping for the market: ask for a REST reload by market id
    evs = events_from("rt_market_batch_sparse.json")
    assert len(evs) == 1 and isinstance(evs[0], ResyncNeeded)
    assert evs[0].exchange_id is None and evs[0].market_id == "999" and evs[0].reason == "unknown_exchange"
    # with the market -> exchange mapping known, the book is attributed through the topic
    evs = events_from("rt_market_batch_sparse.json", markets={"999": "5000"})
    assert len(evs) == 1 and isinstance(evs[0], BookUpdate)
    ev = evs[0]
    assert ev.exchange_id == "5000" and ev.market_id == "999"
    assert ev.bids[0].quantity == 7 and ev.asks[0].price == 0.35 and ev.asks[0].quantity == 9
    assert ev.revision is None and ev.sequence is None and ev.ts is None


def test_non_broadcast_and_unknown_events_are_ignored():
    assert events_from("rt_phx_reply.json") == []
    assert parse_broadcast("something_else", {"books": []}) == []
    assert parse_broadcast("market_batch", None) == []
    assert parse_broadcast("market_batch", {}) == []
    assert parse_broadcast("market_batch", {"books": "nonsense", "trades": 5}) == []
    assert parse_broadcast("account_batch", {"fills": [{"price": 0.5}]}) == []


def test_market_batch_settled_flag_and_resync_required():
    payload = {
        "delivery": {"revision": 1},
        "books": {"1076": {"bids": [], "asks": [], "asOf": {"sequence": 1}}},
        "marketSettled": True,
        "resyncRequired": True,
    }
    evs = parse_broadcast("market_batch", payload, topic="realtime:tournament:T:market:387", markets=MARKETS)
    kinds = [type(e) for e in evs]
    assert kinds == [ResyncNeeded, BookUpdate, SettledEvent]
    assert evs[0].reason == "resync_required" and evs[0].exchange_id == "1076"
    assert evs[2].exchange_id == "1076" and evs[2].market_id == "387"


def test_trade_without_sequence_requests_resync():
    payload = {
        "delivery": {"revision": 1},
        "books": [],
        "trades": [{"exchangeId": "1076", "price": 0.5, "sequence": None}],
    }
    evs = parse_broadcast("market_batch", payload, topic="realtime:tournament:T:market:387", markets=MARKETS)
    assert len(evs) == 1 and isinstance(evs[0], ResyncNeeded) and evs[0].reason == "trade_without_sequence"


# ---------------------------------------------------------------------------------------------
# Revision gaps, duplicates, book versions
# ---------------------------------------------------------------------------------------------


def batch(
    revision: int, previous: int | None, seq: int, price: float = 0.5, exchange_id: str = "1076"
) -> dict:
    return {
        "delivery": {"revision": revision, "previousRevision": previous},
        "books": [
            {
                "exchangeId": exchange_id,
                "bids": [{"price": price, "quantity": 1}],
                "asks": [],
                "asOf": {"sequence": seq},
            }
        ],
        "trades": [],
    }


def test_revision_gap_emits_resync_and_duplicates_are_dropped():
    tracker = RevisionTracker()
    topic = "realtime:tournament:T:market:387"

    def run(rev, prev, seq, price=0.5):
        return parse_broadcast(
            "market_batch", batch(rev, prev, seq, price), topic=topic, tracker=tracker, markets=MARKETS
        )

    assert [type(e) for e in run(1, 0, 10)] == [BookUpdate]
    assert [type(e) for e in run(2, 1, 11)] == [BookUpdate]
    assert run(2, 1, 11) == []  # the same batch can arrive up to three times
    assert run(2, 1, 11) == []
    evs = run(4, 3, 13, 0.6)  # revision 3 was never delivered
    assert [type(e) for e in evs] == [ResyncNeeded, BookUpdate]
    assert evs[0].reason == "revision_gap" and evs[0].exchange_id == "1076" and evs[0].market_id == "387"
    assert evs[1].revision == 4 and evs[1].bids[0].price == 0.6  # the newest book is still applied
    assert run(3, 2, 12) == []  # late delivery of the missing batch is stale now
    assert [type(e) for e in run(5, 4, 14)] == [BookUpdate]
    assert tracker.revisions[topic] == 5


def test_revisions_are_tracked_per_topic():
    tracker = RevisionTracker()
    a = "realtime:tournament:T:market:387"
    b = "realtime:tournament:T:market:388"
    assert len(parse_broadcast("market_batch", batch(10, 9, 1), topic=a, tracker=tracker)) == 1
    assert (
        len(parse_broadcast("market_batch", batch(2, 1, 1, exchange_id="1077"), topic=b, tracker=tracker))
        == 1
    )
    assert (
        len(parse_broadcast("market_batch", batch(3, 2, 2, exchange_id="1077"), topic=b, tracker=tracker))
        == 1
    )
    assert parse_broadcast("market_batch", batch(10, 9, 1), topic=a, tracker=tracker) == []
    assert tracker.revisions == {a: 10, b: 3}


def test_book_versions_never_go_backwards():
    tracker = RevisionTracker()
    topic = "realtime:tournament:T:market:387"
    newer = {
        "books": [
            {
                "exchangeId": "1076",
                "bids": [],
                "asks": [],
                "asOf": {"sequence": 20, "at": "2026-10-10T00:00:02Z"},
            }
        ]
    }
    older = {
        "books": [
            {
                "exchangeId": "1076",
                "bids": [],
                "asks": [],
                "asOf": {"sequence": 19, "at": "2026-10-10T00:00:01Z"},
            }
        ]
    }
    same = {
        "books": [
            {
                "exchangeId": "1076",
                "bids": [],
                "asks": [],
                "asOf": {"sequence": 20, "at": "2026-10-10T00:00:02Z"},
            }
        ]
    }
    assert len(parse_broadcast("market_batch", newer, topic=topic, tracker=tracker)) == 1
    assert parse_broadcast("market_batch", older, topic=topic, tracker=tracker) == []
    assert parse_broadcast("market_batch", same, topic=topic, tracker=tracker) == []
    unversioned = {"books": [{"exchangeId": "1076", "bids": [], "asks": []}]}
    assert len(parse_broadcast("market_batch", unversioned, topic=topic, tracker=tracker)) == 1
    assert len(parse_broadcast("market_batch", unversioned, topic=topic, tracker=tracker)) == 1
    tracker.reset()
    assert len(parse_broadcast("market_batch", older, topic=topic, tracker=tracker)) == 1


def test_account_batch_revision_gap():
    tracker = RevisionTracker()
    fill = {"exchangeId": "1076", "price": 0.5, "quantity": 1, "at": "2026-10-10T00:00:00Z"}
    first = parse_broadcast(
        "account_batch",
        {"delivery": {"revision": 1, "previousRevision": 0}, "fills": [fill]},
        topic="realtime:user:p",
        tracker=tracker,
    )
    assert [type(e) for e in first] == [FillEvent]
    gap = parse_broadcast(
        "account_batch",
        {"delivery": {"revision": 3, "previousRevision": 2}, "fills": [fill]},
        topic="realtime:user:p",
        tracker=tracker,
    )
    assert [type(e) for e in gap] == [ResyncNeeded, FillEvent]
    assert gap[0].reason == "account_revision_gap"


# ---------------------------------------------------------------------------------------------
# Fake Supabase Realtime server (Phoenix protocol over websockets on localhost)
# ---------------------------------------------------------------------------------------------


class FakeSupabase:
    """Answers phx_join / heartbeat / access_token, records what it saw, and pushes one market_batch per
    connection once every expected channel has joined. Optionally drops the first connection."""

    def __init__(
        self, expected_joins: int, *, close_first: bool = False, reject_topics: set[str] | None = None
    ):
        self.expected_joins = expected_joins
        self.close_first = close_first
        self.reject_topics = reject_topics or set()
        self.joins: list[list[str]] = []
        self.heartbeats = 0
        self.access_tokens: list[tuple[str, str]] = []
        self.join_tokens: list[str] = []
        self.leaves: list[str] = []
        self.connections = 0
        self.port = 0

    async def handler(self, ws):
        idx = self.connections
        self.connections += 1
        self.joins.append([])
        joined: list[str] = []
        batch_sent = False
        try:
            async for raw in ws:
                msg = json.loads(raw)
                topic, event, payload, ref = msg["topic"], msg["event"], msg["payload"], msg["ref"]
                if event == "phx_join":
                    self.joins[idx].append(topic)
                    self.join_tokens.append(payload.get("access_token", ""))
                    if topic in self.reject_topics:
                        await ws.send(
                            json.dumps(
                                {
                                    "topic": topic,
                                    "event": "phx_reply",
                                    "ref": ref,
                                    "payload": {
                                        "status": "error",
                                        "response": {"reason": "ChannelRateLimitReached: Too many channels"},
                                    },
                                }
                            )
                        )
                        continue
                    joined.append(topic)
                    await ws.send(
                        json.dumps(
                            {
                                "topic": topic,
                                "event": "phx_reply",
                                "ref": ref,
                                "payload": {"status": "ok", "response": {}},
                            }
                        )
                    )
                    if len(joined) >= self.expected_joins and not batch_sent:
                        batch_sent = True
                        market = next(t for t in joined if ":market:" in t)
                        await ws.send(
                            json.dumps(
                                {
                                    "topic": market,
                                    "event": "broadcast",
                                    "ref": None,
                                    "payload": {
                                        "type": "broadcast",
                                        "event": "market_batch",
                                        "payload": {
                                            "delivery": {"revision": idx + 1, "previousRevision": idx},
                                            "books": [
                                                {
                                                    "exchangeId": "1076",
                                                    "bids": [{"price": 0.5, "quantity": 1}],
                                                    "asks": [{"price": 0.55, "quantity": 2}],
                                                    "asOf": {"sequence": idx + 1},
                                                }
                                            ],
                                            "trades": [],
                                        },
                                    },
                                }
                            )
                        )
                        if idx == 0 and self.close_first:
                            asyncio.get_running_loop().call_later(
                                0.2, lambda: asyncio.ensure_future(ws.close())
                            )
                elif event == "heartbeat":
                    self.heartbeats += 1
                    await ws.send(
                        json.dumps(
                            {
                                "topic": "phoenix",
                                "event": "phx_reply",
                                "ref": ref,
                                "payload": {"status": "ok", "response": {}},
                            }
                        )
                    )
                elif event == "access_token":
                    self.access_tokens.append((topic, payload.get("access_token", "")))
                elif event == "phx_leave":
                    self.leaves.append(topic)
                    await ws.send(
                        json.dumps(
                            {
                                "topic": topic,
                                "event": "phx_reply",
                                "ref": ref,
                                "payload": {"status": "ok", "response": {}},
                            }
                        )
                    )
        except ConnectionClosed:
            pass


class TokenSource:
    def __init__(self, port: int):
        self.port = port
        self.calls = 0

    async def __call__(self) -> RealtimeToken:
        self.calls += 1
        return RealtimeToken(
            supabaseUrl=f"http://127.0.0.1:{self.port}",
            anonKey="anon",
            token=f"jwt-{self.calls}",
            expiresAt=datetime.now(UTC) + timedelta(hours=3),
            user_channel="user:prof-42",
        )


async def collect(feed: RealtimeFeed, done, timeout: float = 8.0) -> list:
    """Consume feed events until `done(events)` holds (polled, since some conditions are server-side)."""
    out: list = []
    gen = feed.events()

    async def consume() -> None:
        async for ev in gen:
            out.append(ev)

    task = asyncio.create_task(consume())
    try:
        async with asyncio.timeout(timeout):
            while not done(out):
                if task.done():
                    task.result()
                    raise AssertionError("feed ended early")
                await asyncio.sleep(0.02)
    finally:
        await feed.stop()
        try:
            await asyncio.wait_for(task, 2)
        except (TimeoutError, asyncio.CancelledError):
            task.cancel()
        await gen.aclose()
    return out


@pytest.fixture()
async def fake_server():
    servers: list = []

    async def start(fake: FakeSupabase) -> FakeSupabase:
        server = await serve(fake.handler, "127.0.0.1", 0)
        servers.append(server)
        fake.port = server.sockets[0].getsockname()[1]
        return fake

    yield start
    for s in servers:
        s.close()
        await s.wait_closed()


async def test_feed_joins_channels_heartbeats_and_yields_events(fake_server):
    fake = await fake_server(FakeSupabase(expected_joins=3))
    tokens = TokenSource(fake.port)
    feed = RealtimeFeed(
        tokens,
        TID,
        markets={"387": "1076", "388": "1077"},
        user_channel="user:prof-42",
        max_channels=94,
        heartbeat_s=0.15,
        tick_s=0.02,
        reconnect_min_s=0.05,
    )
    evs = await collect(
        feed, lambda out: fake.heartbeats >= 2 and any(isinstance(e, BookUpdate) for e in out)
    )
    assert isinstance(evs[0], ResyncNeeded) and evs[0].exchange_id is None and evs[0].reason == "connected"
    books = [e for e in evs if isinstance(e, BookUpdate)]
    assert books and books[0].exchange_id == "1076" and books[0].revision == 1 and books[0].best_ask == 0.55
    assert books[0].market_id == "387"
    assert sorted(fake.joins[0]) == sorted(
        [
            "realtime:tournament:tour-2026-midterms:market:387",
            "realtime:tournament:tour-2026-midterms:market:388",
            "realtime:user:prof-42",
        ]
    )
    assert all(t == "jwt-1" for t in fake.join_tokens)
    assert fake.heartbeats >= 2
    assert feed.joined == set() and not feed.connected  # stopped cleanly


async def test_feed_reconnects_and_rejoins_everything(fake_server):
    fake = await fake_server(FakeSupabase(expected_joins=2, close_first=True))
    tokens = TokenSource(fake.port)
    feed = RealtimeFeed(
        tokens,
        TID,
        markets={"387": "1076"},
        user_channel="user:prof-42",
        max_channels=94,
        tick_s=0.02,
        reconnect_min_s=0.05,
        reconnect_max_s=0.1,
    )
    evs = await collect(feed, lambda out: sum(isinstance(e, BookUpdate) for e in out) >= 2)
    reasons = [e.reason for e in evs if isinstance(e, ResyncNeeded)]
    assert reasons[:1] == ["connected"] and "reconnected" in reasons
    books = [e for e in evs if isinstance(e, BookUpdate)]
    assert [b.revision for b in books] == [1, 2]  # the tracker was reset on reconnect
    assert fake.connections == 2 and len(fake.joins) == 2
    assert sorted(fake.joins[0]) == sorted(fake.joins[1])  # full rejoin, including the user channel
    assert "realtime:user:prof-42" in fake.joins[1]
    assert feed.connections == 2 and tokens.calls == 2  # a fresh token per connection


async def test_feed_refreshes_token_on_joined_channels(fake_server):
    fake = await fake_server(FakeSupabase(expected_joins=2))
    tokens = TokenSource(fake.port)
    feed = RealtimeFeed(
        tokens,
        TID,
        markets={"387": "1076"},
        user_channel="user:prof-42",
        max_channels=94,
        tick_s=0.02,
        token_refresh_s=0.2,
        reconnect_min_s=0.05,
    )
    await collect(feed, lambda out: len(fake.access_tokens) >= 2)
    topics = {t for t, _ in fake.access_tokens}
    assert topics == {"realtime:tournament:tour-2026-midterms:market:387", "realtime:user:prof-42"}
    assert all(tok == "jwt-2" for _, tok in fake.access_tokens)
    assert tokens.calls >= 2 and feed.token is not None and feed.token.token == "jwt-2"


async def test_feed_caps_channels_and_tolerates_rejected_joins(fake_server):
    rejected = f"realtime:{market_topic(TID, '388')}"
    fake = await fake_server(FakeSupabase(expected_joins=2, reject_topics={rejected}))
    tokens = TokenSource(fake.port)
    feed = RealtimeFeed(
        tokens,
        TID,
        markets={"387": "1076", "388": "1077", "389": "1078", "390": "1079"},
        user_channel="user:prof-42",
        max_channels=3,
        tick_s=0.02,
        reconnect_min_s=0.05,
        join_retry_s=0.1,
    )
    assert len(feed.markets) == 2  # 3 channels minus the user channel
    assert set(feed.markets) == {"387", "388"}
    await collect(
        feed, lambda out: any(isinstance(e, BookUpdate) for e in out) and fake.joins[0].count(rejected) >= 2
    )
    assert rejected in fake.joins[0]  # tried, rejected, retried after join_retry_s
    assert f"realtime:{market_topic(TID, '389')}" not in fake.joins[0]


async def test_feed_keeps_retrying_when_token_fetch_fails():
    calls = 0

    async def failing() -> RealtimeToken:
        nonlocal calls
        calls += 1
        raise RuntimeError("exchange down")

    feed = RealtimeFeed(
        failing, TID, markets={"387": "1076"}, max_channels=94, reconnect_min_s=0.02, reconnect_max_s=0.05
    )
    gen = feed.events()

    async def first() -> None:
        async for _ in gen:
            raise AssertionError("no events expected")

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(first(), 0.3)
    await feed.stop()
    await gen.aclose()
    assert calls >= 3 and not feed.connected


async def test_feed_recovers_from_unreachable_server(fake_server):
    fake = await fake_server(FakeSupabase(expected_joins=1))
    tokens = TokenSource(fake.port)
    bad_then_good = [RealtimeToken(supabaseUrl="http://127.0.0.1:1", anonKey="a", token="t")]

    async def provider() -> RealtimeToken:
        if bad_then_good:
            return bad_then_good.pop()
        return await tokens()

    feed = RealtimeFeed(
        provider, TID, markets={"387": "1076"}, max_channels=94, tick_s=0.02, reconnect_min_s=0.05
    )
    evs = await collect(feed, lambda out: any(isinstance(e, BookUpdate) for e in out))
    assert any(isinstance(e, BookUpdate) for e in evs)
    assert fake.connections == 1
