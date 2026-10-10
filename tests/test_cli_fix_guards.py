"""Regression tests for the CLI safety findings: `trade` follows LIVE (F1), the `--probe-writes`
passthrough is guarded (F2), `sigcup kill` sets the database kill flag so a restarted trader starts
killed (F3), the canary's read usage (F4), `kill` survives an invalid LIVE value (F11), and the
`kill --cancel-now` success path."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from typer.testing import CliRunner

from sigcup import cli, settings
from sigcup.db import engine as dbengine
from sigcup.db.models import Strategy
from sigcup.db.repo import kv_get
from sigcup.settings import Secrets
from sigcup.sig.ratelimit import Budget
from sigcup.sig.schemas import OrderRequest
from sigcup.trader.oms import KV_KILLED, KV_KILLED_PAPER, OMS
from sigcup.trader.types import DesiredOrder
from tests.fakes_exec import FakeAdapter, FakeClock

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(cli.app, list(args))


def make_sec(tmp_path, **kw) -> Secrets:
    kw.setdefault("live", False)
    return Secrets(_env_file=None, kill_file=str(tmp_path / "KILL"), data_dir=str(tmp_path / "data"), **kw)


def use(monkeypatch, s: Secrets) -> Secrets:
    monkeypatch.setattr(settings, "secrets", lambda: s)
    return s


def fake_trader(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    mod = ModuleType("sigcup.trader.loop")

    async def run(paper: bool = False, once: bool = False, **kw):
        calls.append({"paper": paper, "once": once})
        return None

    mod.run = run
    monkeypatch.setitem(sys.modules, "sigcup.trader.loop", mod)
    return calls


# -- F1: `trade` without a flag follows LIVE ----------------------------------------------------
def test_trade_default_follows_live(db, tmp_path, monkeypatch):
    calls = fake_trader(monkeypatch)
    use(monkeypatch, make_sec(tmp_path, live=True, sig_api_key="k"))
    r = invoke("trade", "--once")  # what pm2 runs (no flag) once .env has LIVE=1 + a key
    assert r.exit_code == 0, r.output
    assert "mode=LIVE" in r.output
    assert calls == [{"paper": False, "once": True}]

    # explicit --paper still overrides LIVE=1
    r = invoke("trade", "--paper", "--once")
    assert r.exit_code == 0, r.output
    assert "mode=paper" in r.output and calls[-1] == {"paper": True, "once": True}

    # LIVE=1 without a key: refused, nothing started
    use(monkeypatch, make_sec(tmp_path, live=True))
    r = invoke("trade")
    assert r.exit_code == 2 and "SIG_API_KEY is empty" in r.output
    assert len(calls) == 2

    # LIVE unset: paper
    use(monkeypatch, make_sec(tmp_path, live=False, sig_api_key="k"))
    r = invoke("trade", "--once")
    assert r.exit_code == 0, r.output
    assert calls[-1] == {"paper": True, "once": True}


# -- F2: --probe-writes passed through `--` is still guarded ------------------------------------
def test_bootstrap_probe_writes_passthrough_is_guarded(db, tmp_path, monkeypatch):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "bootstrap_races.py").write_text("def main(argv):\n    print('SCRIPT ARGV', argv)\n    return 0\n")
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    use(monkeypatch, make_sec(tmp_path, live=False, sig_api_key="k"))
    for args in (("--", "--probe-writes"), ("--", "--dry-run", "--probe-writes"), ("--", "--probe")):
        r = invoke("bootstrap", *args)
        assert r.exit_code == 2, (args, r.output)
        assert "refusing --probe-writes" in r.output and "SCRIPT ARGV" not in r.output
    r = invoke("bootstrap", "--", "--dry-run")  # unrelated passthrough still works
    assert r.exit_code == 0 and "SCRIPT ARGV ['--dry-run']" in r.output

    use(monkeypatch, make_sec(tmp_path, live=True, sig_api_key="k"))
    r = invoke("bootstrap", "--", "--dry-run", "--probe-writes")
    assert r.exit_code == 0, r.output
    assert "SCRIPT ARGV ['--dry-run', '--probe-writes']" in r.output  # passed once, not duplicated


# -- F3: `sigcup kill` sets the database flag ---------------------------------------------------
def test_kill_sets_and_clear_removes_both_kill_flags(db, tmp_path, monkeypatch):
    use(monkeypatch, make_sec(tmp_path))
    r = invoke("kill")
    assert r.exit_code == 0, r.output
    with dbengine.session() as s:
        live_flag, paper_flag = kv_get(s, KV_KILLED), kv_get(s, KV_KILLED_PAPER)
    assert live_flag and live_flag["paper"] is False and "sigcup kill" in live_flag["reason"]
    assert paper_flag and paper_flag["paper"] is True
    r = invoke("kill", "--clear")
    assert r.exit_code == 0, r.output
    with dbengine.session() as s:
        assert kv_get(s, KV_KILLED) is None and kv_get(s, KV_KILLED_PAPER) is None


async def test_trader_restarted_after_kill_does_not_repost(db, cfg, tmp_path, monkeypatch):
    """A dead trader left a lost (UNCERTAIN) live intent; the operator runs `sigcup kill`; pm2 restarts
    the trader: its startup reconcile must not re-POST the order."""
    use(monkeypatch, make_sec(tmp_path))
    clock = FakeClock()
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min, clock=clock.mono)
    fake = FakeAdapter(budget=budget, clock=clock.now)
    oms1 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, max_retries=0, backoff_s=0.0)
    fake.default_script = ["lost"]  # never reached the exchange
    await oms1.sync([DesiredOrder(exchange_id="m1", side="yes", action="buy", price=0.5, quantity=10, strategy=Strategy.VALUE)])
    assert fake.orders == {}
    places_before = sum(1 for _, k in fake.writes if k == "place")

    r = invoke("kill")
    assert r.exit_code == 0, r.output

    oms2 = OMS(fake, cfg, budget, kill_path=tmp_path / "KILL", clock=clock.now, sleep=clock.sleep, backoff_s=0.0)
    await oms2.startup_reconcile()
    assert oms2.killed is not None
    assert sum(1 for _, k in fake.writes if k == "place") == places_before, "no order may be sent after the kill"
    assert fake.open_orders() == []


def test_trade_with_kill_file_present_starts_killed(db, tmp_path, monkeypatch):
    calls = fake_trader(monkeypatch)
    s = use(monkeypatch, make_sec(tmp_path))
    s.kill_path.parent.mkdir(parents=True, exist_ok=True)
    s.kill_path.touch()  # touched by hand, not through `sigcup kill`
    r = invoke("trade", "--once")
    assert r.exit_code == 0, r.output
    assert calls == [{"paper": True, "once": True}]
    with dbengine.session() as sess:
        flag = kv_get(sess, KV_KILLED_PAPER)
    assert flag and flag["paper"] is True and "kill file" in flag["reason"]


# -- F4: the canary's read usage ------------------------------------------------------------------
async def test_canary_default_poll_keeps_reads_small(db, cfg):
    clock = FakeClock()
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("m1", 0.60, 0.64)
    res = await cli.run_canary(fake, cfg, exchange_id="m1", quantity=10, wait_s=30, clock=clock.now, sleep=clock.sleep)
    assert res.ok and res.status == "cancelled"
    # the account shares 100 reads/min across every process: one canary run must stay a small slice
    assert len(fake.reads) <= 20, fake.reads


# -- F11: `kill` works when LIVE has an invalid value -------------------------------------------
def test_kill_survives_invalid_live_value(db, tmp_path, monkeypatch):
    monkeypatch.setenv("LIVE", "live")
    monkeypatch.setenv("KILL_FILE", str(tmp_path / "KILL"))

    def broken() -> Secrets:
        return Secrets(_env_file=None)  # LIVE=live -> pydantic ValidationError

    with pytest.raises(Exception):  # noqa: B017 - the exact pydantic class is irrelevant here
        broken()
    monkeypatch.setattr(settings, "secrets", broken)
    r = invoke("kill")
    assert r.exit_code == 0, r.output
    assert (tmp_path / "KILL").exists()


# -- kill --cancel-now success path -------------------------------------------------------------
async def _place(fake: FakeAdapter, key: str) -> None:
    await fake.place_order(OrderRequest(exchange_id="m1", side="yes", action="buy", price=0.40, quantity=5, idempotency_key=key, tournament_id=fake.tournament_id))


def test_kill_cancel_now_cancels_through_a_direct_client(db, tmp_path, monkeypatch):
    import asyncio

    use(monkeypatch, make_sec(tmp_path, live=True, sig_api_key="k"))
    fake = FakeAdapter()
    fake.set_top("m1", 0.40, 0.45)
    asyncio.run(_place(fake, "k1"))
    assert len(fake.open_orders()) == 1
    seen: list[bool] = []

    def build(sec, cfg, *, live, budget=None):
        seen.append(live)
        return fake

    monkeypatch.setattr(cli, "build_adapter", build)
    r = invoke("kill", "--cancel-now")
    assert r.exit_code == 0, r.output
    assert seen == [True] and fake.open_orders() == []
    assert "exchange cancelled 1 orders" in r.output
