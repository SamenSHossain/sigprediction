"""Integration: scripts/seed_paper.py seeds an offline paper database from config/races.yaml and
config/ratings_manual.yaml, and one `sigcup trade --paper --once` cycle runs against it with no network,
writing the trader heartbeat and only paper intents/fills."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

from sigcup.db import engine as dbengine
from sigcup.db.models import Fill, Heartbeat, Market, OrderIntent, Quote, Race, Rating
from sigcup.settings import ROOT, Secrets
from sigcup.trader import loop as trader_loop

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def _load_script():
    path = ROOT / "scripts" / "seed_paper.py"
    spec = importlib.util.spec_from_file_location("seed_paper", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def seed_paper():
    return _load_script()


@pytest.fixture()
def sec(tmp_path: Path) -> Secrets:
    return Secrets(_env_file=None, kill_file=str(tmp_path / "KILL"), data_dir=str(tmp_path))


async def test_seed_is_idempotent_and_offline(db, cfg, sec, seed_paper):
    rep = await seed_paper.seed(cfg=cfg, sec=sec, session_factory=dbengine.session, now=NOW)
    assert rep.races_inserted > 100 and rep.markets_inserted == 2 * rep.races_inserted + 3  # 3 independent legs
    assert rep.ratings_written > 0 and rep.quotes_written == rep.markets_inserted
    with dbengine.session() as s:
        assert s.execute(select(func.count()).select_from(Race)).scalar() == rep.races_inserted
        markets = s.execute(select(Market)).scalars().all()
        assert all(m.exchange_id.startswith(seed_paper.SYNTH_EXCHANGE_PREFIX) for m in markets)
        quotes = s.execute(select(Quote)).scalars().all()
        assert all(q.source == seed_paper.QUOTE_SOURCE and q.bid is not None and q.ask is not None and q.bid < q.ask for q in quotes)
        # a Likely D rating (0.85) puts the D leg near 0.85 and the R leg near 0.15
        ga = {m.party: m.exchange_id for m in markets if m.race_id == "SEN-GA"}
        by_eid = {q.exchange_id: q for q in quotes}
        assert by_eid[ga["D"]].last > 0.7 and by_eid[ga["R"]].last < 0.3
        assert abs(by_eid[ga["D"]].last + by_eid[ga["R"]].last - 1.0) < 0.011
        n_ratings = s.execute(select(func.count()).select_from(Rating)).scalar()
    # second run: nothing new except another quote snapshot
    rep2 = await seed_paper.seed(cfg=cfg, sec=sec, session_factory=dbengine.session, now=NOW)
    assert rep2.races_inserted == 0 and rep2.markets_inserted == 0 and rep2.markets_existing == rep.markets_inserted
    assert rep2.ratings_written == 0
    with dbengine.session() as s:
        assert s.execute(select(func.count()).select_from(Rating)).scalar() == n_ratings
        assert s.execute(select(func.count()).select_from(Quote)).scalar() == 2 * rep.quotes_written
    # purge removes exactly the synthetic rows
    nm, nq = seed_paper.purge(dbengine.session)
    assert nm == rep.markets_inserted and nq == 2 * rep.quotes_written


async def test_quotes_fixture_keys_by_leg(db, cfg, sec, seed_paper, tmp_path: Path):
    fixture = tmp_path / "quotes.yaml"
    fixture.write_text("SEN-GA:D: {bid: 0.80, ask: 0.84, bid_size: 50, ask_size: 60}\nnot-a-market: {bid: 0.1, ask: 0.2}\n")
    rep = await seed_paper.seed(cfg=cfg, sec=sec, session_factory=dbengine.session, quotes_path=fixture, now=NOW)
    assert rep.quotes_written == 1
    with dbengine.session() as s:
        q = s.execute(select(Quote)).scalars().one()
        m = s.get(Market, seed_paper.synthetic_ids("SEN-GA", "D")[0])
        assert q.exchange_id == m.exchange_id and q.bid == 0.80 and q.ask == 0.84 and q.bid_size == 50 and q.ask_size == 60


async def test_paper_cycle_runs_offline_on_seeded_db(db, cfg, sec, seed_paper, monkeypatch):
    trader_loop._modules.clear()
    await seed_paper.seed(cfg=cfg, sec=sec, session_factory=dbengine.session, now=NOW)
    # no SIG keys -> PaperAdapter on DB quotes; building a live client would be a seam bug
    monkeypatch.setattr(trader_loop, "build_live_adapter", lambda *a, **k: pytest.fail("live client built in paper mode"))
    res = await trader_loop.run(paper=True, once=True, cfg=cfg, sec=sec, clock=lambda: NOW)
    assert res is not None and res.skipped is None and res.killed is None
    assert res.fair > 0 and res.desired > 0 and res.sync is not None
    with dbengine.session() as s:
        hb = s.get(Heartbeat, "trader")
        assert hb is not None and hb.status == "paper"
        intents = s.execute(select(OrderIntent)).scalars().all()
        fills = s.execute(select(Fill)).scalars().all()
    assert intents and all(i.paper for i in intents)
    assert all(f.paper for f in fills)
    trader_loop._modules.clear()
