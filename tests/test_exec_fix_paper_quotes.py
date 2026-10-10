"""Paper trader in DB-quote mode (no SIG key): the PaperAdapter serves the latest Quote rows back to the
trader, and BookState used to persist those echoes as source='rest', indistinguishable from real
exchange quotes. They are labelled source='paper'; quotes from a real read source stay 'rest'. The CLV /
reporter readers do not filter on source and the echo carries the served quote's price and timestamp,
so CLV is unchanged."""

from __future__ import annotations

from datetime import timedelta

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Quote
from sigcup.paper import PaperAdapter
from sigcup.reporter import clv
from sigcup.trader import loop as trader_loop
from tests.fakes_exec import FakeAdapter, FakeClock
from tests.test_exec_loop import make_sec, seed


@pytest.fixture(autouse=True)
def _reset_module_cache():
    trader_loop._modules.clear()
    yield
    trader_loop._modules.clear()


def quote_rows() -> list[Quote]:
    with dbengine.session() as s:
        return list(s.execute(select(Quote).order_by(Quote.id)).scalars().all())


def seed_quotes(clock: FakeClock) -> None:
    """What an earlier run / scripts/seed_paper.py left in the quotes table."""
    ts = clock.now() - timedelta(minutes=5)
    with dbengine.session() as s:
        s.add(Quote(ts=ts, exchange_id="tx-d", bid=0.60, ask=0.64, bid_size=50, ask_size=50, source="seed"))
        s.add(Quote(ts=ts, exchange_id="tx-r", bid=0.34, ask=0.38, bid_size=50, ask_size=50, source="seed"))


async def test_db_quote_paper_echoes_are_labelled_paper_not_rest(db, cfg, tmp_path):
    clock = FakeClock()
    seed(clock)
    seed_quotes(clock)
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), clock=clock.now)  # no key: DB quotes
    assert isinstance(t.adapter, PaperAdapter) and t.adapter.use_db_quotes
    await t.refresh_books()
    assert t.book.top("tx-d").ask == 0.64  # the served quote reached the book
    rows = quote_rows()
    echoes = rows[2:]
    assert echoes, "the served quotes are persisted as the book's history"
    assert {q.source for q in echoes} == {"paper"}, [(q.exchange_id, q.source) for q in echoes]
    assert all(q.source != "rest" for q in rows), "no echo may pass for a real exchange quote"
    # CLV / reporter readers see the same price and time as the quote that was served
    with dbengine.session() as s:
        q = clv.quote_as_of(s, "tx-d", clock.now())
        assert clv.quote_mid(q) == pytest.approx(0.62)
        assert q.ts.replace(tzinfo=None) == (clock.now() - timedelta(minutes=5)).replace(tzinfo=None)


async def test_paper_with_a_real_read_source_keeps_rest_quotes(db, cfg, tmp_path):
    clock = FakeClock()
    seed(clock)
    source = FakeAdapter(clock=clock.now)
    source.set_top("tx-d", 0.58, 0.62)
    paper = PaperAdapter(cfg, source=source, clock=clock.now)
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=paper, clock=clock.now)
    await t.refresh_books()
    assert {q.source for q in quote_rows()} == {"rest"}  # real exchange quotes (via the read client)


async def test_live_style_adapter_keeps_rest_quotes(db, cfg, tmp_path):
    clock = FakeClock()
    seed(clock)
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("tx-d", 0.58, 0.62)
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    await t.refresh_books()
    assert {q.source for q in quote_rows()} == {"rest"}
