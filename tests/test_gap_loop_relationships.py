"""B2: the trader refreshes race exclusivity from the relationships API (at startup and hourly).

Only a ``mutually_exclusive`` group with ``isExhaustive: true`` covering exactly a race's legs sets
``Race.exclusive`` / ``Race.exhaustive`` True; nothing is ever inferred from titles; a True is never
flipped to False unless the API positively lists the race's legs as non-exclusive (or non-exhaustive).
The read goes through the trader's adapter (the single shared budget) at housekeeping priority."""

from __future__ import annotations

import asyncio

import pytest

from sigcup.db import engine as dbengine
from sigcup.db.models import Market, Race
from sigcup.settings import Secrets
from sigcup.sig.schemas import Relationship
from sigcup.trader import loop as trader_loop
from sigcup.trader.types import PRIORITY_HOUSEKEEPING
from tests.fakes_exec import FakeAdapter, FakeClock


@pytest.fixture(autouse=True)
def _reset_module_cache():
    trader_loop._modules.clear()
    yield
    trader_loop._modules.clear()


def make_sec(tmp_path) -> Secrets:
    return Secrets(_env_file=None, live=False, kill_file=str(tmp_path / "KILL"), sig_api_key="", sig_read_key="")


def race(rid: str, *, exclusive=None, exhaustive=None, label: str | None = None) -> Race:
    return Race(race_id=rid, chamber="senate", state=rid[-2:], label=label or f"{rid} race", exclusive=exclusive, exhaustive=exhaustive)


def seed() -> None:
    with dbengine.session() as s:
        s.add(race("SEN-TX"))  # unknown -> confirmed by an exact exhaustive group
        s.add(race("SEN-GA", exclusive=True, exhaustive=True))  # True, absent from the API: kept
        s.add(race("SEN-RI", exclusive=True, exhaustive=True))  # True, API says non-exclusive: flipped
        s.add(race("SEN-NE"))  # a group covering only part of its legs: untouched
        s.add(race("SEN-MT", label="mutually exclusive, exactly one winner"))  # a title never decides
        s.add(race("SEN-AZ", exclusive=True, exhaustive=True))  # exact group, isExhaustive false
        for i, (rid, party) in enumerate(
            [("SEN-TX", "D"), ("SEN-TX", "R"), ("SEN-GA", "D"), ("SEN-GA", "R"), ("SEN-RI", "D"), ("SEN-RI", "R"),
             ("SEN-NE", "R"), ("SEN-NE", "I"), ("SEN-MT", "D"), ("SEN-MT", "R"), ("SEN-AZ", "D"), ("SEN-AZ", "R")]
        ):
            s.add(Market(market_id=str(i), exchange_id=f"{rid.lower()}-{party.lower()}", race_id=rid, party=party, title=f"{rid} {party}", status="open"))


def rel(rid: str, kind: str, ids: list[str], exhaustive: bool | None) -> Relationship:
    return Relationship(id=rid, kind=kind, isExhaustive=exhaustive, exchange_ids=ids)


def flags() -> dict[str, tuple[bool | None, bool | None]]:
    with dbengine.session() as s:
        return {r.race_id: (r.exclusive, r.exhaustive) for r in s.query(Race).all()}


class PriorityFake(FakeAdapter):
    """A client whose read accepts the limiter priority (as SigClient reads do)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.rel_priorities: list[int | None] = []

    async def get_relationships(self, *, priority: int | None = None):
        self.rel_priorities.append(priority)
        return await super().get_relationships()


async def test_refresh_sets_flags_only_from_exact_api_groups(db, cfg, tmp_path):
    seed()
    clock = FakeClock()
    fake = PriorityFake(clock=clock.now)
    fake.relationships = [
        rel("g1", "mutually_exclusive", ["sen-tx-d", "sen-tx-r"], True),
        rel("g2", "not_mutually_exclusive", ["sen-ri-d", "sen-ri-r"], None),
        rel("g3", "mutually_exclusive", ["sen-ne-r"], True),  # partial coverage
        rel("g4", "mutually_exclusive", ["sen-az-d", "sen-az-r"], False),
    ]
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    rep = await t.refresh_relationships()
    assert rep.error is None and rep.relationships == 4
    f = flags()
    assert f["SEN-TX"] == (True, True)
    assert f["SEN-GA"] == (True, True), "no API evidence: a True is kept"
    assert f["SEN-RI"][0] is False, "the API positively lists the legs as non-exclusive"
    assert f["SEN-NE"] == (None, None), "a group covering part of a race decides nothing"
    assert f["SEN-MT"] == (None, None), "titles never set exclusivity"
    assert f["SEN-AZ"] == (True, False), "isExhaustive false on exactly the race's legs"
    assert fake.rel_priorities == [PRIORITY_HOUSEKEEPING]
    assert "SEN-TX" in rep.set_true and "SEN-RI" in rep.set_false
    # the trader's in-memory race metadata follows the DB
    assert t.book.races()["SEN-TX"].exclusive is True


async def test_failed_or_empty_fetch_changes_nothing(db, cfg, tmp_path):
    seed()
    clock = FakeClock()

    class Broken(FakeAdapter):
        async def get_relationships(self):
            raise RuntimeError("503")

    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=Broken(clock=clock.now), clock=clock.now)
    before = flags()
    rep = await t.refresh_relationships()
    assert rep.error and flags() == before
    # an empty graph (the Oct-7 audit) is no evidence either way: plain-signature adapters still work
    t2 = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=FakeAdapter(clock=clock.now), clock=clock.now)
    rep2 = await t2.refresh_relationships()
    assert rep2.error is None and rep2.relationships == 0 and flags() == before


async def test_relationships_refresh_runs_at_startup_and_hourly(db, cfg, tmp_path, monkeypatch):
    seed()
    clock = FakeClock()
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("sen-tx-d", 0.4, 0.5)
    t = trader_loop.build_trader(paper=True, cfg=cfg, sec=make_sec(tmp_path), adapter=fake, clock=clock.now)
    scheduled: dict[str, tuple[float, bool]] = {}

    async def fake_every(seconds, fn, *, name, stop=None, run_immediately=True, **kw):
        scheduled[name] = (seconds, run_immediately)
        if stop is not None:
            await stop.wait()

    monkeypatch.setattr(trader_loop, "every", fake_every)
    task = asyncio.create_task(t.run_forever())
    for _ in range(200):
        if "relationships" in scheduled:
            break
        await asyncio.sleep(0.01)
    t.stop.set()
    await asyncio.wait_for(task, 5)
    assert scheduled["relationships"] == (3600, True)
