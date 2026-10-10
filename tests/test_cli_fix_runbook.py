"""Regression tests for the operator RUNBOOK: the go-live steps match what the CLI does (F1: LIVE=1 only
in the shell for the canary, plain `trade` follows LIVE), no step runs canaries against a live trader
or claims cross-process queuing (F4), the canary worst case covers --take (F4), the kill section says
`sigcup kill` sets the database flag (F3), LIVE's accepted values (F11) and the --json exit code (F10)."""

from __future__ import annotations

import re
from pathlib import Path

RUNBOOK = (Path(__file__).resolve().parent.parent / "docs" / "RUNBOOK.md").read_text()


def section(number: int) -> str:
    m = re.search(rf"^## {number}\. .*?(?=^## {number + 1}\. |\Z)", RUNBOOK, flags=re.S | re.M)
    assert m, f"section {number} missing"
    return m.group(0)


def test_go_live_sets_live_only_in_the_shell_for_the_canary():
    go_live = section(5)
    assert "LIVE=1 .venv/bin/sigcup canary --live" in go_live
    assert "in .env: LIVE=1" not in go_live
    assert "`sigcup trade` without a flag follows `LIVE`" in RUNBOOK
    assert "ignores `LIVE`" not in RUNBOOK


def test_no_canary_burst_against_the_live_trader():
    go_live = section(5)
    assert "while the trader is live" not in go_live
    assert "the limiter must queue" not in go_live
    assert re.search(r"per\s+account", go_live)


def test_canary_worst_case_covers_take():
    assert re.search(r"worst case[^.]*--take", section(5).replace("\n", " "))


def test_kill_section_mentions_the_database_flag():
    kill = section(7).replace("\n", " ")
    assert "sets the database kill flags" in kill
    assert "UNCERTAIN" in kill


def test_live_values_and_json_exit_code_documented():
    env = section(2)
    assert "`true`" in env and "LIVE=live" in env
    assert "`--json` always exits 0" in RUNBOOK
