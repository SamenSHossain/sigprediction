"""Feeds scheduler: collector planning / skip rules, election-night ref cadence, lazy module
resolution and the guarded runner (missing, failing and slow collectors), heartbeats."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import UTC, datetime

import pytest

from sigcup.db import engine as dbengine
from sigcup.db.models import Heartbeat
from sigcup.feeds import loop as L
from sigcup.settings import Secrets

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


def secrets(**kw) -> Secrets:
    return Secrets(_env_file=None, **kw)


def spec_map(cfg, sec) -> dict[str, L.CollectorSpec]:
    return {s.name: s for s in L.plan_collectors(cfg, sec)}


# ---------------------------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------------------------


def test_plan_skips_collectors_without_credentials(cfg):
    specs = spec_map(cfg, secrets())
    assert specs["social_reddit"].enabled is False and "reddit" in specs["social_reddit"].reason
    assert specs["social_x"].enabled is False and "x_vendor" in specs["social_x"].reason
    assert specs["social_youtube"].enabled is False and "youtube" in specs["social_youtube"].reason
    assert specs["nlp_stance"].enabled is False and "anthropic" in specs["nlp_stance"].reason
    for name in ("ref_markets", "polls", "news_gdelt", "news_rss", "social_bluesky", "nlp_features", "events"):
        assert specs[name].enabled is True and specs[name].reason == ""
    assert specs["ref_markets"].module == "sigcup.feeds.ref_markets" and specs["ref_markets"].dynamic_cadence
    assert specs["polls"].cadence_s == cfg.feeds.polls_cadence_s
    assert specs["news_gdelt"].cadence_s == cfg.feeds.news_cadence_s
    assert specs["social_bluesky"].cadence_s == cfg.feeds.social_cadence_s
    assert specs["social_youtube"].cadence_s == cfg.feeds.youtube_cadence_s


def test_plan_enables_collectors_with_credentials(cfg):
    sec = secrets(
        x_vendor="socialdata", x_vendor_key="k", youtube_api_key="y", reddit_client_id="id", reddit_client_secret="s",
        anthropic_api_key="a",
    )
    specs = spec_map(cfg, sec)
    assert all(specs[n].enabled for n in ("social_x", "social_youtube", "social_reddit", "nlp_stance"))
    half = spec_map(cfg, secrets(reddit_client_id="id", x_vendor="NONE"))
    assert half["social_reddit"].enabled is False and half["social_x"].enabled is False


def test_ref_cadence_switches_on_election_night(cfg):
    assert L.ref_cadence_s(cfg, NOW) == cfg.feeds.ref_cadence_s
    assert L.ref_cadence_s(cfg, datetime(2026, 11, 4, 2, 0, tzinfo=UTC)) == cfg.election_night.ref_cadence_s
    assert L.ref_cadence_s(cfg, datetime(2026, 11, 4, 18, 0, tzinfo=UTC)) == cfg.feeds.ref_cadence_s  # after the deadline


def test_ref_cadence_falls_back_when_election_night_unimportable(cfg, monkeypatch):
    import importlib

    real = importlib.import_module

    def broken(name, *a, **k):
        if name == "sigcup.trader.election_night":
            raise ImportError("not ready")
        return real(name, *a, **k)

    monkeypatch.setattr(L.importlib, "import_module", broken)
    assert L.ref_cadence_s(cfg, datetime(2026, 11, 4, 2, 0, tzinfo=UTC)) == cfg.feeds.ref_cadence_s


def test_run_timeout_bounds():
    assert L.run_timeout_s(10) == 60.0
    assert L.run_timeout_s(120) == 240.0
    assert L.run_timeout_s(21600) == L.MAX_RUN_TIMEOUT_S


# ---------------------------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------------------------


@pytest.fixture()
def fake_modules(monkeypatch):
    """Register throwaway collector modules under tests.* so no other workstream's code is touched."""
    calls: dict[str, list] = {"ok": [], "boom": [], "slow": []}

    ok = types.ModuleType("tests._feed_ok")

    async def ok_run(session_factory, cfg, secrets):
        calls["ok"].append((session_factory, cfg, secrets))
        return 3

    ok.run_once = ok_run

    boom = types.ModuleType("tests._feed_boom")

    async def boom_run(session_factory, cfg, secrets):
        calls["boom"].append(1)
        raise RuntimeError("vendor exploded")

    boom.run_once = boom_run

    slow = types.ModuleType("tests._feed_slow")

    async def slow_run(session_factory, cfg, secrets):
        calls["slow"].append(1)
        await asyncio.sleep(10)
        return 1

    slow.run_once = slow_run

    norun = types.ModuleType("tests._feed_norun")

    for name, mod in (("tests._feed_ok", ok), ("tests._feed_boom", boom), ("tests._feed_slow", slow), ("tests._feed_norun", norun)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def _runner(cfg, specs, **kw) -> L.FeedsRunner:
    return L.FeedsRunner(cfg, secrets(), dbengine.session, specs=specs, now=lambda: NOW, stagger_s=0.0, **kw)


async def test_run_collector_guards_every_failure_mode(cfg, db, fake_modules, monkeypatch):
    monkeypatch.setattr(L, "MAX_RUN_TIMEOUT_S", 0.05)
    specs = [
        L.CollectorSpec("ok", "tests._feed_ok", 60),
        L.CollectorSpec("boom", "tests._feed_boom", 60),
        L.CollectorSpec("slow", "tests._feed_slow", 60),
        L.CollectorSpec("norun", "tests._feed_norun", 60),
        L.CollectorSpec("missing", "tests._feed_does_not_exist", 60),
        L.CollectorSpec("stub", "tests._feed_stub_raises", 60),
    ]
    stub = types.ModuleType("tests._feed_stub_raises")
    monkeypatch.setitem(sys.modules, "tests._feed_stub_raises", stub)  # module without run_once (unfinished stub)
    r = _runner(cfg, specs)
    results = await r.run_all_once()
    assert results == {"ok": 3, "boom": None, "slow": None, "norun": None, "missing": None, "stub": None}
    assert fake_modules["ok"][0][0] is dbengine.session and fake_modules["ok"][0][1] is cfg
    st = r.stats
    assert (st["ok"].runs, st["ok"].rows, st["ok"].last_rows, st["ok"].last_run) == (1, 3, 3, NOW.isoformat())
    assert st["boom"].errors == 1 and "vendor exploded" in st["boom"].last_error
    assert st["slow"].errors == 1 and "timeout" in st["slow"].last_error
    assert st["norun"].skipped == 1 and st["missing"].skipped == 1 and st["stub"].skipped == 1
    # resolution is cached: a second pass does not retry the import machinery but still skips cleanly
    await r.run_all_once(only=["missing", "ok"])
    assert st["missing"].skipped == 2 and st["ok"].runs == 2


async def test_run_all_once_respects_only_and_disabled(cfg, db, fake_modules):
    specs = [L.CollectorSpec("ok", "tests._feed_ok", 60), L.CollectorSpec("off", "tests._feed_boom", 60, enabled=False, reason="no creds")]
    r = _runner(cfg, specs)
    assert await r.run_all_once() == {"ok": 3}
    assert await r.run_all_once(only=["off"]) == {}
    assert fake_modules["boom"] == []


async def test_run_schedules_and_heartbeats_until_stopped(cfg, db, fake_modules):
    specs = [L.CollectorSpec("ok", "tests._feed_ok", 60), L.CollectorSpec("ref", "tests._feed_ok", 60, dynamic_cadence=True)]
    r = _runner(cfg, specs, heartbeat_s=60)
    task = asyncio.create_task(r.run())
    for _ in range(50):
        await asyncio.sleep(0.01)
        if r.stats["ok"].runs and r.stats["ref"].runs:
            break
    r.stop.set()
    await asyncio.wait_for(task, timeout=2)
    assert r.stats["ok"].runs == 1 and r.stats["ref"].runs == 1
    with dbengine.session() as s:
        hb = s.get(Heartbeat, L.PROCESS_NAME)
    assert hb is not None and hb.status == "ok"
    detail = json.loads(hb.detail)
    assert detail["collectors"]["ok"]["rows"] == 3 and detail["ts"] == NOW.isoformat()


async def test_beat_marks_degraded_after_errors(cfg, db, fake_modules):
    r = _runner(cfg, [L.CollectorSpec("boom", "tests._feed_boom", 60)])
    await r.run_all_once()
    await r.beat()
    with dbengine.session() as s:
        hb = s.get(Heartbeat, L.PROCESS_NAME)
    assert hb.status == "degraded" and json.loads(hb.detail)["collectors"]["boom"]["errors"] == 1


def test_main_once_prints_counts(cfg, db, fake_modules, monkeypatch, capsys):
    monkeypatch.setattr(L, "plan_collectors", lambda c, s: [L.CollectorSpec("ok", "tests._feed_ok", 60)])
    monkeypatch.setattr(L, "config", lambda: cfg)
    monkeypatch.setattr(L, "load_secrets", lambda: secrets())
    assert L.main(["--once"]) == 0
    assert "ok" in capsys.readouterr().out
