"""Regression tests for the feeds-loop observability finding (F11).

* The heartbeat reflects the LATEST run of each collector: one transient failure no longer leaves the
  process 'degraded' until restart.
* A collector whose import failed is retried after a backoff instead of being disabled for good.
Nothing touches the network."""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime, timedelta

from sigcup.db import engine as dbengine
from sigcup.db.models import Heartbeat
from sigcup.feeds import loop as L
from sigcup.settings import Secrets

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self) -> None:
        self.t = NOW

    def __call__(self) -> datetime:
        return self.t


def _runner(cfg, specs, clock) -> L.FeedsRunner:
    return L.FeedsRunner(cfg, Secrets(_env_file=None), dbengine.session, specs=specs, now=clock, stagger_s=0.0)


def _status() -> str:
    with dbengine.session() as s:
        return s.get(Heartbeat, L.PROCESS_NAME).status


async def test_heartbeat_recovers_after_a_transient_failure(cfg, db, monkeypatch):
    calls = {"n": 0}
    flaky = types.ModuleType("tests._feed_flaky")

    async def run_once(session_factory, cfg, secrets):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("GDELT 429")
        return 2

    flaky.run_once = run_once
    monkeypatch.setitem(sys.modules, "tests._feed_flaky", flaky)
    r = _runner(cfg, [L.CollectorSpec("flaky", "tests._feed_flaky", 60)], Clock())
    await r.run_all_once()
    await r.beat()
    assert _status() == "degraded"  # the latest run failed
    await r.run_all_once()
    await r.beat()
    assert _status() == "ok"  # it recovered: old noise must not mask a later real outage
    assert r.stats["flaky"].errors == 1  # the cumulative count is still reported in the detail


async def test_failed_import_is_retried_after_a_backoff(cfg, db, monkeypatch):
    clock = Clock()
    spec = L.CollectorSpec("late", "tests._feed_late_module", 60)
    monkeypatch.delitem(sys.modules, "tests._feed_late_module", raising=False)
    r = _runner(cfg, [spec], clock)
    assert await r.run_all_once() == {"late": None}  # not importable yet (sibling module being redeployed)
    late = types.ModuleType("tests._feed_late_module")

    async def run_once(session_factory, cfg, secrets):
        return 5

    late.run_once = run_once
    monkeypatch.setitem(sys.modules, "tests._feed_late_module", late)
    assert await r.run_all_once() == {"late": None}  # within the backoff: no retry yet
    clock.t = NOW + timedelta(hours=1)
    assert await r.run_all_once() == {"late": 5}  # retried and running without a process restart
    assert r.stats["late"].skipped == 2 and r.stats["late"].runs == 1
