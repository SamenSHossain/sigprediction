"""BookState: quotes persisted only on change, duck-typed realtime updates, views joined with metadata."""

from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import IntentState, Market, OrderIntent, Quote, Race
from sigcup.sig.schemas import Level, OrderBook, PositionInfo, PriceSnapshot
from sigcup.trader.book_state import BookState
from tests.fakes_exec import FakeClock


def quotes() -> list[Quote]:
    with dbengine.session() as s:
        return list(s.execute(select(Quote).order_by(Quote.id)).scalars().all())


def seed_meta() -> None:
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="Texas Senate", exclusive=True, exhaustive=True))
        s.add(Market(market_id="1", exchange_id="tx-d", race_id="SEN-TX", party="D", title="D wins TX", status="open", kalshi_ticker="SENATETX-26-D"))
        s.add(Market(market_id="2", exchange_id="tx-r", race_id="SEN-TX", party="R", title="R wins TX", status="open"))


def test_quotes_persisted_only_on_change(db, cfg):
    book = BookState(cfg)
    clock = FakeClock()
    snap = {"m1": PriceSnapshot(exchange_id="m1", best_bid=0.4, best_ask=0.45, bid_size=10, ask_size=20, ts=clock.now())}
    assert book.update_prices(snap) == ["m1"]
    assert book.update_prices(snap, ts=clock.now() + timedelta(seconds=30)) == []
    assert len(quotes()) == 1
    snap["m1"] = PriceSnapshot(exchange_id="m1", best_bid=0.4, best_ask=0.45, bid_size=10, ask_size=25)
    assert book.update_prices(snap) == ["m1"]
    rows = quotes()
    assert len(rows) == 2 and rows[-1].ask_size == 25 and rows[-1].source == "rest"
    assert book.top("m1").last is None and book.mid("m1") == pytest.approx(0.425)


def test_lenient_dict_snapshots_and_renamed_fields(db, cfg):
    book = BookState(cfg, persist=False)
    changed = book.update_prices([{"exchangeId": "x", "bestBid": "0.30", "bestAsk": 0.36, "latestPrice": 0.33, "ts": "2026-10-10T12:00:00Z"}])
    assert changed == ["x"]
    t = book.top("x")
    assert (t.bid, t.ask, t.last) == (0.30, 0.36, 0.33) and t.ts.tzinfo is not None
    assert book.pending_quotes == 1 and quotes() == []
    assert book.flush_quotes() == 1 and len(quotes()) == 1


def test_realtime_duck_typed_update_and_depth(db, cfg):
    book = BookState(cfg)
    ev = SimpleNamespace(exchange_id="m1", bids=[{"price": 0.5, "quantity": 30}, (0.49, 5)], asks=[Level(price=0.55, quantity=7)], ts=None, revision=3)
    assert book.update_realtime(ev) is True
    t = book.top("m1")
    assert (t.bid, t.ask, t.bid_size, t.ask_size, t.source) == (0.5, 0.55, 30, 7, "realtime")
    assert book.book("m1").bids[1].price == 0.49
    assert book.update_realtime(ev) is False
    assert book.pending_quotes == 1  # realtime quotes are batched until flush
    book.flush_quotes()
    assert quotes()[-1].source == "realtime"
    # a REST depth read with the same top does not add a row
    assert book.update_book(OrderBook(exchange_id="m1", bids=[Level(price=0.5, quantity=30)], asks=[Level(price=0.55, quantity=7)])) is False
    assert len(quotes()) == 1
    assert book.stale_books(0) == ["m1"] and book.stale_books(3600) == []


def test_views_join_metadata_positions_and_open_orders(db, cfg):
    seed_meta()
    book = BookState(cfg)
    assert book.load_meta() == 2
    races = book.races()
    assert races["SEN-TX"].markets == {"D": "tx-d", "R": "tx-r"} and races["SEN-TX"].exclusive is True
    book.update_prices({"tx-d": PriceSnapshot(exchange_id="tx-d", best_bid=0.6, best_ask=0.64), "zz": PriceSnapshot(exchange_id="zz", best_bid=0.1, best_ask=0.2)})
    clock = FakeClock()
    intents = [
        OrderIntent(id="a" * 32, exchange_id="tx-d", side="yes", action="buy", price=0.6, quantity=10, strategy="theta", state=IntentState.ACKED.value, created_at=clock.now(), updated_at=clock.now()),
        OrderIntent(id="b" * 32, exchange_id="tx-d", side="yes", action="buy", price=0.6, quantity=10, strategy="theta", state=IntentState.UNCERTAIN.value, created_at=clock.now(), updated_at=clock.now()),
    ]
    views = book.views([PositionInfo(exchange_id="tx-r", quantity=-5, avg_price=0.4)], intents)
    assert set(views) == {"tx-d", "tx-r", "zz"}
    d = views["tx-d"]
    assert (d.race_id, d.party, d.chamber, d.state, d.mid) == ("SEN-TX", "D", "senate", "TX", 0.62)
    assert [i.id for i in d.open_orders] == ["a" * 32]
    assert views["tx-r"].position == -5 and views["tx-r"].avg_price == 0.4 and views["tx-r"].bid is None
    assert views["zz"].chamber == "unknown" and views["zz"].race_id == "zz"
    assert book.hot_markets(1) == ["tx-d"]
    book.set_account(cash=1000.0, reserved=10.0)
    assert book.equity == 1000.0 and book.reserved == 10.0


def test_bulk_prices_fixture_parses_leniently(db, cfg):
    """The documented /exchanges/prices shape (camelCase, strings, nulls) goes straight into the book."""
    import json
    from pathlib import Path

    payload = json.loads((Path(__file__).parent / "fixtures" / "exec" / "bulk_prices.json").read_text())
    book = BookState(cfg)
    changed = book.update_prices(payload["data"])
    assert len(changed) == 4
    tx = book.top("ex-sen-tx-d")
    assert (tx.bid, tx.ask, tx.last, tx.bid_size) == (0.615, 0.635, 0.62, 20) and tx.ts.isoformat().startswith("2026-10-10T11:59:30")
    ri = book.top("ex-gov-ri-d")
    assert ri.bid is None and ri.ask == 0.995 and book.mid("ex-gov-ri-d") == 0.93
    pa = book.top("ex-house-pa-08-d")
    assert (pa.bid, pa.ask, pa.bid_size, pa.ask_size, pa.last) == (0.485, 0.515, 4500, 4200, None)
    assert len(quotes()) == 4
    assert book.update_prices(payload["data"]) == []
