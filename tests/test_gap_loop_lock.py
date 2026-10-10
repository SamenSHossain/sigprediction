"""B4: the trader holds data/trader.lock (its pid) so the one-shot commands that build their own limiter
do not run beside a live trader: `sigcup canary` and `bootstrap --probe-writes` refuse (clear message,
non-zero exit) while a live trader holds the lock; `kill --cancel-now` (the emergency path) still runs
but warns. A lock whose pid is not alive (or was reused by another process) is stale."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sigcup import cli, settings
from sigcup.settings import Secrets
from sigcup.trader import loop as trader_loop
from tests.fakes_exec import FakeAdapter, FakeClock

runner = CliRunner()


@pytest.fixture(autouse=True)
def _reset_module_cache():
    trader_loop._modules.clear()
    yield
    trader_loop._modules.clear()


def make_sec(tmp_path, **kw) -> Secrets:
    kw.setdefault("live", False)
    return Secrets(_env_file=None, kill_file=str(tmp_path / "data" / "KILL"), data_dir=str(tmp_path / "data"), **kw)


def use(monkeypatch, s: Secrets) -> Secrets:
    monkeypatch.setattr(settings, "secrets", lambda: s)
    return s


@pytest.fixture()
def other_process():
    """A live process that is not this one (stands in for a running trader)."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    yield p
    p.kill()
    p.wait()


def dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def hold(path: Path, pid: int, mode: str | None = "live") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n" + (f"mode={mode}\n" if mode else ""))


# -- the lock itself --------------------------------------------------------------------------------
def test_lock_path_defaults_to_data_trader_lock(tmp_path):
    assert trader_loop.trader_lock_path(Secrets(_env_file=None)) == settings.ROOT / "data" / "trader.lock"
    assert trader_loop.trader_lock_path(make_sec(tmp_path)) == tmp_path / "data" / "trader.lock"


def test_acquire_writes_pid_and_release_removes_it(tmp_path):
    path = tmp_path / "data" / "trader.lock"
    lock = trader_loop.acquire_trader_lock(path, mode="live")
    assert lock.pid == os.getpid() and path.read_text().splitlines()[0] == str(os.getpid())
    held = trader_loop.read_trader_lock(path)
    assert held is not None and held.pid == os.getpid() and held.mode == "live" and held.alive()
    assert trader_loop.running_trader(path) is not None
    assert trader_loop.release_trader_lock(path) is True
    assert not path.exists() and trader_loop.running_trader(path) is None


def test_stale_lock_is_replaced(tmp_path):
    path = tmp_path / "trader.lock"
    hold(path, dead_pid())
    assert trader_loop.running_trader(path) is None, "a dead pid is stale"
    lock = trader_loop.acquire_trader_lock(path, mode="paper")
    assert lock.pid == os.getpid() and trader_loop.read_trader_lock(path).mode == "paper"
    # garbage is stale too
    path.write_text("not a pid\n")
    assert trader_loop.running_trader(path) is None
    trader_loop.acquire_trader_lock(path, mode="paper")
    trader_loop.release_trader_lock(path)


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="pid start time needs /proc")
def test_reused_pid_is_stale(tmp_path, other_process):
    path = tmp_path / "trader.lock"
    path.write_text(f"{other_process.pid}\nmode=live\nstart=1\n")  # alive pid, different start time
    assert trader_loop.running_trader(path) is None


def test_live_lock_of_another_process_refuses_a_second_trader(tmp_path, other_process):
    path = tmp_path / "trader.lock"
    hold(path, other_process.pid)
    with pytest.raises(trader_loop.TraderLockHeld, match=str(other_process.pid)):
        trader_loop.acquire_trader_lock(path, mode="live")
    assert trader_loop.release_trader_lock(path) is False, "never remove another process's lock"
    assert path.exists()


async def test_trader_run_holds_the_lock_and_removes_it_at_exit(db, cfg, tmp_path, monkeypatch):
    sec = make_sec(tmp_path)
    path = trader_loop.trader_lock_path(sec)
    clock = FakeClock()
    seen: list[bool] = []

    async def run_once(self):
        seen.append(trader_loop.running_trader(path) is not None)
        return trader_loop.CycleResult(ts=clock.now())

    monkeypatch.setattr(trader_loop.Trader, "run_once", run_once)
    await trader_loop.run(paper=True, once=True, cfg=cfg, sec=sec, adapter=FakeAdapter(clock=clock.now), clock=clock.now)
    assert seen == [True] and not path.exists()


async def test_trader_run_refuses_while_another_trader_holds_the_lock(db, cfg, tmp_path, other_process):
    sec = make_sec(tmp_path)
    path = trader_loop.trader_lock_path(sec)
    hold(path, other_process.pid, "paper")
    with pytest.raises(trader_loop.TraderLockHeld):
        await trader_loop.run(paper=True, once=True, cfg=cfg, sec=sec, adapter=FakeAdapter())
    assert trader_loop.read_trader_lock(path).pid == other_process.pid


# -- the one-shot commands ----------------------------------------------------------------------------
def test_canary_refuses_while_a_live_trader_holds_the_lock(db, tmp_path, monkeypatch, other_process):
    s = use(monkeypatch, make_sec(tmp_path, live=True, sig_api_key="k"))
    hold(trader_loop.trader_lock_path(s), other_process.pid, "live")
    monkeypatch.setattr(cli, "build_adapter", lambda *a, **k: pytest.fail("no client may be built beside a live trader"))
    for args in (("canary", "--live", "--market", "m1"), ("canary", "--market", "m1", "--wait", "0")):
        r = runner.invoke(cli.app, list(args))
        assert r.exit_code != 0, r.output
        assert "live trader" in r.output and str(other_process.pid) in r.output


def test_canary_runs_beside_a_paper_trader_or_a_stale_lock(db, tmp_path, monkeypatch, other_process):
    s = use(monkeypatch, make_sec(tmp_path))
    fake = FakeAdapter()
    fake.set_top("m1", 0.40, 0.45)
    monkeypatch.setattr(cli, "build_adapter", lambda *a, **k: fake)
    hold(trader_loop.trader_lock_path(s), other_process.pid, "paper")  # the runbook's paper phase
    r = runner.invoke(cli.app, ["canary", "--market", "m1", "--wait", "0"])
    assert r.exit_code == 0, r.output
    hold(trader_loop.trader_lock_path(s), dead_pid(), "live")
    r = runner.invoke(cli.app, ["canary", "--market", "m1", "--wait", "0"])
    assert r.exit_code == 0, r.output


def test_bootstrap_probe_writes_refuses_while_a_live_trader_holds_the_lock(db, tmp_path, monkeypatch, other_process):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "bootstrap_races.py").write_text("def main(argv):\n    print('SCRIPT ARGV', argv)\n    return 0\n")
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    s = use(monkeypatch, make_sec(tmp_path, live=True, sig_api_key="k"))
    hold(trader_loop.trader_lock_path(s), other_process.pid, "live")
    for args in (("bootstrap", "--probe-writes"), ("bootstrap", "--", "--probe-writes")):
        r = runner.invoke(cli.app, list(args))
        assert r.exit_code != 0, (args, r.output)
        assert "live trader" in r.output and "SCRIPT ARGV" not in r.output
    r = runner.invoke(cli.app, ["bootstrap", "--", "--dry-run"])  # the read-only bootstrap is unaffected
    assert r.exit_code == 0 and "SCRIPT ARGV" in r.output


def test_bootstrap_script_refuses_probe_writes_beside_a_live_trader(tmp_path, monkeypatch, capsys, other_process):
    from tests.test_feeds_bootstrap import B

    s = make_sec(tmp_path, live=True, sig_api_key="trade-key")
    monkeypatch.setattr(B, "load_secrets", lambda: s)
    monkeypatch.setattr(B, "make_adapter", lambda *a, **k: pytest.fail("no client for a refused probe"))
    hold(trader_loop.trader_lock_path(s), other_process.pid, None)  # mode unknown counts as live
    code = B.main(["--probe-writes", "--skip-crosswalk", "--dry-run"])
    err = capsys.readouterr().err
    assert code != 0 and "refusing --probe-writes" in err and "live trader" in err


def test_kill_cancel_now_runs_beside_a_live_trader_with_a_warning(db, tmp_path, monkeypatch, other_process):
    s = use(monkeypatch, make_sec(tmp_path, live=True, sig_api_key="k"))
    hold(trader_loop.trader_lock_path(s), other_process.pid, "live")
    fake = FakeAdapter()
    monkeypatch.setattr(cli, "build_adapter", lambda *a, **k: fake)
    r = runner.invoke(cli.app, ["kill", "--cancel-now"])
    assert r.exit_code == 0, r.output
    assert "warning" in r.output and str(other_process.pid) in r.output
    assert [w[1] for w in fake.writes] == ["cancel_all"]


def test_trade_command_reports_a_held_lock_clearly(db, tmp_path, monkeypatch, other_process):
    s = use(monkeypatch, make_sec(tmp_path))
    monkeypatch.setattr(trader_loop, "secrets", lambda: s)  # the trader resolves its own settings
    monkeypatch.setattr(trader_loop, "build_trader", lambda **k: pytest.fail("no trader may be built beside another"))
    hold(trader_loop.trader_lock_path(s), other_process.pid, "paper")
    r = runner.invoke(cli.app, ["trade", "--paper", "--once"])
    assert r.exit_code != 0
    assert "another trader" in r.output and str(other_process.pid) in r.output
