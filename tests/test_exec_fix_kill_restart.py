"""A kill file that already exists when the trader (re)starts is honoured by the startup reconcile itself,
not only by the first model cycle / kill-watch: the OMS starts killed, persists this mode's KV kill flag
(as `sigcup kill` and `sigcup trade` do), sends cancel-all, and sends nothing new, not even the re-POST of
an UNCERTAIN order left behind by a crash. This covers a file touched by hand while no flag was set, and
a trader started without the CLI (python -m sigcup.trader.loop) that never sets the flag itself."""

from __future__ import annotations

import pytest

from sigcup.db import engine as dbengine
from sigcup.db.models import IntentState, OrderIntent, Strategy
from sigcup.db.repo import kv_get
from sigcup.sig.ratelimit import Budget
from sigcup.trader import loop as trader_loop
from sigcup.trader.oms import KV_KILLED, KV_KILLED_PAPER, OMS
from sigcup.trader.types import DesiredOrder
from tests.fakes_exec import FakeAdapter, FakeClock
from tests.test_exec_loop import fake_models, make_sec, seed


@pytest.fixture(autouse=True)
def _reset_module_cache():
    trader_loop._modules.clear()
    yield
    trader_loop._modules.clear()


def order(eid: str) -> DesiredOrder:
    return DesiredOrder(exchange_id=eid, side="yes", action="buy", price=0.6, quantity=10, strategy=Strategy.VALUE)


def kinds(fake: FakeAdapter) -> list[str]:
    return [k for _, k in fake.writes]


async def _crash_with_lost_order(fake: FakeAdapter, cfg, budget, clock, tmp_path, *, paper: bool) -> None:
    """A previous run: one resting order (m-rest) and one POST whose answer was lost (UNCERTAIN, m-lost)."""
    oms0 = OMS(fake, cfg, budget, paper=paper, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, max_retries=0, backoff_s=0.0)
    await oms0.sync([order("tx-r")])
    fake.default_script = ["lost"]
    await oms0.sync([order("tx-r"), order("tx-d")])
    states = {}
    with dbengine.session() as s:
        for i in s.query(OrderIntent).all():
            states[i.exchange_id] = i.state
    assert states == {"tx-r": IntentState.ACKED.value, "tx-d": IntentState.UNCERTAIN.value}
    assert len(fake.open_orders()) == 1


async def test_startup_reconcile_with_kill_file_cancels_all_and_sends_nothing(db, cfg, tmp_path):
    clock = FakeClock()
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
    fake = FakeAdapter(budget=budget, clock=clock.now)
    await _crash_with_lost_order(fake, cfg, budget, clock, tmp_path, paper=False)
    (tmp_path / "KILL").write_text("")  # touched by hand: no KV flag was set
    with dbengine.session() as s:
        assert kv_get(s, KV_KILLED) is None
    before = len(fake.writes)

    oms = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, backoff_s=0.0)
    await oms.startup_reconcile()

    new = kinds(fake)[before:]
    assert "place" not in new, "the UNCERTAIN order must not be re-POSTed while the kill file exists"
    assert "cancel_all" in new
    assert oms.killed is not None and oms.killed.startswith("kill file present")
    assert oms.kill_confirmed is True and fake.open_orders() == []
    with dbengine.session() as s:
        flag = kv_get(s, KV_KILLED)
    assert flag and flag["reason"].startswith("kill file present") and flag["paper"] is False
    # and it stays dead
    rep = await oms.sync([order("tx-d")])
    assert rep.blocked and "place" not in kinds(fake)[before:]


async def test_trader_started_with_kill_file_places_nothing_and_cancels_all(db, cfg, tmp_path, monkeypatch):
    """The whole trader (startup + one cycle): zero placements, a cancel-all, the paper kill flag set."""
    clock = FakeClock()
    seed(clock)
    fake_models(monkeypatch, lambda ctx, arbs: [order("tx-d"), order("tx-r")])
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
    fake = FakeAdapter(budget=budget, clock=clock.now)
    fake.set_top("tx-d", 0.58, 0.62)
    fake.set_top("tx-r", 0.36, 0.40)
    await _crash_with_lost_order(fake, cfg, budget, clock, tmp_path, paper=True)
    (tmp_path / "KILL").write_text("")
    before = len(fake.writes)

    res = await trader_loop.run(paper=True, once=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, budget=budget, clock=clock.now)

    new = kinds(fake)[before:]
    assert new.count("place") == 0 and "multi" not in new, new
    assert "cancel_all" in new
    assert res.killed and "kill file" in res.killed and res.sync is None
    assert fake.open_orders() == []
    with dbengine.session() as s:
        flag = kv_get(s, KV_KILLED_PAPER)
        assert kv_get(s, KV_KILLED) is None, "a paper trader never sets the live flag"
    assert flag and flag["paper"] is True and "kill file" in flag["reason"]
