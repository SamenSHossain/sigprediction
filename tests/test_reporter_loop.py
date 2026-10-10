"""Reporter process: the hourly / daily schedule driven by an injected clock, the KV day marker,
the daily JSON file, the heartbeat, and that a failing job never kills the tick."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from sigcup.db import engine as dbengine
from sigcup.db.models import CLV, Fill, Heartbeat, Quote
from sigcup.db.repo import kv_get
from sigcup.reporter import loop as rloop
from sigcup.settings import Secrets

T0 = datetime(2026, 10, 10, 11, 0, tzinfo=UTC)


class Clock:
    def __init__(self, t: datetime = T0) -> None:
        self.t = t

    def now(self) -> datetime:
        return self.t

    def set(self, t: datetime) -> None:
        self.t = t


def make(tmp_path, cfg, clock: Clock) -> rloop.Reporter:
    sec = Secrets(_env_file=None, kill_file=str(tmp_path / "KILL"), data_dir=str(tmp_path / "data"))
    return rloop.Reporter(cfg, sec, clock=clock.now, reports_dir=tmp_path / "reports")


def test_schedule_hourly_and_daily(db, cfg, tmp_path):
    clock = Clock(T0)
    r = make(tmp_path, cfg, clock)
    assert r.tick() == {"hourly": True, "daily": False}  # first tick: hourly runs immediately
    assert r.last_status is not None and r.last_hourly == T0
    clock.set(T0 + timedelta(minutes=30))
    assert r.tick() == {"hourly": False, "daily": False}
    clock.set(T0 + timedelta(hours=1))  # 12:00 UTC: hourly due again and the daily fires
    assert r.tick() == {"hourly": True, "daily": True}
    assert (tmp_path / "reports" / "2026-10-10.json").exists()
    saved = json.loads((tmp_path / "reports" / "2026-10-10.json").read_text())
    assert saved["day"] == "2026-10-10"
    with dbengine.session() as s:
        assert kv_get(s, rloop.KV_LAST_DAILY_DAY)["day"] == "2026-10-10"
        assert kv_get(s, rloop.KV_LAST_DAILY)["day"] == "2026-10-10"
        assert kv_get(s, rloop.KV_LAST_STATUS)["ts"].startswith("2026-10-10T12:00")
        hb = s.get(Heartbeat, "reporter")
        assert hb is not None and hb.status == "ok" and "ticks=3" in hb.detail and "daily=10-10 12:00" in hb.detail
    clock.set(T0 + timedelta(hours=1, minutes=5))
    assert r.tick() == {"hourly": False, "daily": False}  # not twice in one day
    clock.set(T0 + timedelta(days=1, hours=1, minutes=1))
    assert r.tick()["daily"] is True  # next day after noon


def test_daily_runs_once_after_restart_past_noon(db, cfg, tmp_path):
    clock = Clock(T0 + timedelta(hours=7))  # 18:00 UTC, process (re)started after the noon slot
    r = make(tmp_path, cfg, clock)
    assert r.tick() == {"hourly": True, "daily": True}
    r2 = make(tmp_path, cfg, clock)  # a second restart the same day: the KV marker prevents a repeat
    clock.set(T0 + timedelta(hours=8))
    assert r2.tick() == {"hourly": True, "daily": False}
    assert r.daily_due(T0 - timedelta(hours=2)) is False  # before noon: never due


def test_hourly_computes_clv(db, cfg, tmp_path):
    fill_ts = T0 - timedelta(hours=25)
    with dbengine.session() as s:
        s.add(Fill(fill_id="f1", exchange_id="m1", ts=fill_ts, side="yes", action="buy", price=0.6, quantity=10, fee=0.0, strategy="theta", paper=True))
        s.add(Quote(ts=fill_ts + timedelta(hours=24), exchange_id="m1", bid=0.69, ask=0.71))
    make(tmp_path, cfg, Clock(T0)).tick()
    with dbengine.session() as s:
        row = s.get(CLV, "f1")
        assert row is not None and abs(row.clv_24h - 0.10) < 1e-9


def test_failing_job_is_logged_not_raised(db, cfg, tmp_path, monkeypatch):
    clock = Clock(T0 + timedelta(hours=1))
    r = make(tmp_path, cfg, clock)

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(rloop.report, "status", boom)
    monkeypatch.setattr(rloop.report, "daily", boom)
    assert r.tick() == {"hourly": False, "daily": False}
    assert r.last_hourly == clock.now()  # marked so it is not retried every tick
    with dbengine.session() as s:
        assert s.get(Heartbeat, "reporter") is not None


def test_next_daily_at_and_render_text():
    assert rloop.next_daily_at(T0) == T0.replace(hour=12)
    assert rloop.next_daily_at(T0.replace(hour=13)) == (T0 + timedelta(days=1)).replace(hour=12)
    assert "hello" in rloop.render_text("[bold]hello[/bold]")


async def test_run_once(db, cfg, tmp_path, monkeypatch):
    sec = Secrets(_env_file=None, kill_file=str(tmp_path / "KILL"), data_dir=str(tmp_path / "data"))
    r = await rloop.run(once=True, cfg=cfg, sec=sec, clock=lambda: T0)
    assert r.ticks == 1 and r.last_status is not None
