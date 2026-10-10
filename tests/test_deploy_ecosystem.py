"""pm2 deployment: ecosystem.config.js must launch. pm2 resolves ``script`` as a FILE, so each app has to
point at the venv's ``sigcup`` console script (created by ``uv sync``) with ``interpreter: "none"``; a
``script: "-m"`` entry fails with ``Script not found``. The trader runs plain ``trade``, which is paper
unless ``LIVE=1`` (+ a trade key) is set in ``.env`` (docs/RUNBOOK.md section 4/5).

The config is evaluated by node itself (``node -e``); skipped when node is not installed."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from sigcup import cli, settings
from sigcup.settings import Secrets

ROOT = Path(__file__).resolve().parent.parent
ECOSYSTEM = ROOT / "ecosystem.config.js"
VENV = ROOT / ".venv"
SIGCUP = VENV / "bin" / "sigcup"


def _node() -> str | None:
    found = shutil.which("node")
    if found:
        return found
    fallback = Path("/opt/node22/bin/node")
    return str(fallback) if fallback.exists() else None


NODE = _node()
pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")

EXPECTED_ARGS = {"trader": "trade", "feeds": "feeds", "reporter": "report loop"}


@pytest.fixture(scope="module")
def apps() -> dict[str, dict]:
    """The apps exactly as pm2 sees them (the config is plain node: require() it and dump JSON)."""
    code = "process.stdout.write(JSON.stringify(require(process.argv[1])))"
    out = subprocess.run([NODE, "-e", code, str(ECOSYSTEM)], capture_output=True, text=True, timeout=60, check=True)
    data = json.loads(out.stdout)
    return {a["name"]: a for a in data["apps"]}


def test_every_app_runs_the_venv_console_script_without_an_interpreter(apps):
    assert set(apps) == set(EXPECTED_ARGS)
    for name, app in apps.items():
        assert app["interpreter"] == "none", name
        assert Path(app["script"]) == SIGCUP, (name, app["script"])
        assert app["args"] == EXPECTED_ARGS[name], name
        assert app["cwd"] == str(ROOT), name


@pytest.mark.skipif(not VENV.is_dir(), reason="no .venv (not installed with `uv sync`)")
def test_every_app_script_exists_and_is_executable_after_uv_sync(apps):
    for name, app in apps.items():
        script = Path(app["script"])
        assert script.is_file(), f"{name}: {script} does not exist (run `uv sync`)"
        assert os.access(script, os.X_OK), f"{name}: {script} is not executable"


@pytest.mark.skipif(not SIGCUP.is_file(), reason="no .venv/bin/sigcup (run `uv sync`)")
def test_every_app_command_line_is_a_valid_sigcup_command(apps):
    """What pm2 executes (``script args``) parses: ``--help`` on the same command line exits 0."""
    for name, app in apps.items():
        cmd = [app["script"], *app["args"].split(), "--help"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=app["cwd"])
        assert r.returncode == 0, (name, cmd, r.stdout[-500:], r.stderr[-500:])


def test_operational_options_are_kept(apps):
    for name, app in apps.items():
        assert app["autorestart"] is True, name
        assert app["restart_delay"] == 5000 and app["max_restarts"] == 20 and app["min_uptime"] == "30s", name
        assert app["kill_timeout"] == 15000, name
        assert app["env"]["PYTHONUNBUFFERED"] == "1" and app["time"] is True, name
        assert "LIVE" not in app["env"], f"{name}: LIVE must come from .env, not be pinned by pm2"
        assert app["out_file"] == f"logs/{name}.out.log" and app["error_file"] == f"logs/{name}.err.log", name


def test_trader_args_start_paper_unless_live_is_set(apps, tmp_path, monkeypatch):
    """The trader's pm2 command line never forces live: with LIVE unset it runs paper (even with a trade
    key), and it only trades live when .env has LIVE=1 and a key, the RUNBOOK's explicit opt-in."""
    args = apps["trader"]["args"].split()
    assert "--live" not in args
    calls: list[dict] = []
    mod = ModuleType("sigcup.trader.loop")

    async def run(paper: bool = False, once: bool = False, **kw):
        calls.append({"paper": paper})
        return None

    mod.run = run
    monkeypatch.setitem(sys.modules, "sigcup.trader.loop", mod)

    def use(**kw) -> None:
        sec = Secrets(_env_file=None, kill_file=str(tmp_path / "KILL"), data_dir=str(tmp_path / "data"), **kw)
        monkeypatch.setattr(settings, "secrets", lambda: sec)

    runner = CliRunner()
    use(live=False, sig_api_key="trade-key")
    r = runner.invoke(cli.app, args)
    assert r.exit_code == 0, r.output
    assert "mode=paper" in r.output and calls == [{"paper": True}]

    use(live=True, sig_api_key="trade-key")
    r = runner.invoke(cli.app, args)
    assert r.exit_code == 0, r.output
    assert "mode=LIVE" in r.output and calls[-1] == {"paper": False}
