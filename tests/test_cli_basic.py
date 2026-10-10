"""CLI: --help, db init/stats, lazy entry points (missing module -> clear error, fake module -> wired
with the right arguments), the LIVE guard, kill / kill --clear, status / report commands on an empty
DB, universe (DB then yaml fallback), and the canary runner against the in-memory FakeAdapter."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from types import ModuleType

import pytest
from sqlalchemy import select
from typer.testing import CliRunner

from sigcup import cli, settings
from sigcup.db import engine as dbengine
from sigcup.db.models import Fill, IntentState, Market, OrderIntent, Race
from sigcup.db.repo import kv_get, kv_set
from sigcup.settings import Secrets
from tests.fakes_exec import FakeAdapter, FakeClock

runner = CliRunner()


@pytest.fixture()
def sec(tmp_path, monkeypatch) -> Secrets:
    s = Secrets(_env_file=None, kill_file=str(tmp_path / "KILL"), data_dir=str(tmp_path / "data"), dsn="sqlite:///" + str(tmp_path / "t.db"))
    monkeypatch.setattr(settings, "secrets", lambda: s)
    import sigcup.reporter.loop as rloop

    monkeypatch.setattr(rloop, "secrets", lambda: s)
    return s


def invoke(*args: str):
    return runner.invoke(cli.app, list(args))


def compact(text: str) -> str:
    """Rich wraps table cells; compare without whitespace and box-drawing characters."""
    return "".join(ch for ch in text if not ch.isspace() and ch not in "│┃")


# -- help / db ----------------------------------------------------------------------------------
def test_help_lists_every_command():
    r = invoke("--help")
    assert r.exit_code == 0, r.output
    for cmd in ("db", "bootstrap", "trade", "feeds", "report", "status", "kill", "canary", "universe"):
        assert cmd in r.output
    r = invoke("report", "--help")
    assert r.exit_code == 0
    for sub in ("daily", "status", "leadlag", "loop"):
        assert sub in r.output
    assert invoke().exit_code != 0  # no args -> help, non-zero
    assert "sigcup" in invoke("--version").output


def test_db_init_and_stats(db, sec):
    r = invoke("db", "init")
    assert r.exit_code == 0, r.output
    assert "database ready" in r.output and "t.db" in r.output
    r = invoke("db", "stats")
    assert r.exit_code == 0, r.output
    assert "fills" in r.output and "heartbeats" in r.output


def test_describe_dsn_strips_credentials():
    assert cli.describe_dsn("postgresql://user:pw@host:5432/db") == "postgresql://***@host:5432/db"
    assert cli.describe_dsn("sqlite:///data/x.db").endswith("data/x.db")


# -- lazy entry points -----------------------------------------------------------------------
def test_missing_trader_module_is_a_clear_error(db, sec, monkeypatch):
    monkeypatch.setitem(sys.modules, "sigcup.trader.loop", None)  # import -> ModuleNotFoundError
    r = invoke("trade", "--once")
    assert r.exit_code == 2
    assert "the trader" in r.output and "cannot be imported" in r.output


def test_stub_module_raising_not_implemented(db, sec, monkeypatch, tmp_path):
    # a stub whose import raises NotImplementedError (the shape of every unfinished module here)
    mod = ModuleType("sigcup.feeds.loop")
    monkeypatch.setitem(sys.modules, "sigcup.feeds.loop", mod)  # exists but has neither main nor run
    r = invoke("feeds", "--once")
    assert r.exit_code == 2 and "neither main" in r.output

    def fake_import(name):
        raise NotImplementedError("implemented in the feeds workstream")

    monkeypatch.setattr(cli.importlib, "import_module", fake_import)
    r = invoke("feeds")
    assert r.exit_code == 2 and "not implemented yet" in r.output


def test_trade_wires_paper_and_once(db, sec, monkeypatch):
    calls: list[dict] = []
    mod = ModuleType("sigcup.trader.loop")

    async def run(paper: bool = False, once: bool = False, **kw):
        calls.append({"paper": paper, "once": once})
        return None

    mod.run = run
    monkeypatch.setitem(sys.modules, "sigcup.trader.loop", mod)
    r = invoke("trade", "--once")
    assert r.exit_code == 0, r.output
    assert calls == [{"paper": True, "once": True}]
    r = invoke("trade", "--paper")
    assert r.exit_code == 0 and calls[-1] == {"paper": True, "once": False}


def test_trade_live_refused_without_live_env_or_key(db, sec, monkeypatch, tmp_path):
    called = []
    mod = ModuleType("sigcup.trader.loop")

    async def run(**kw):
        called.append(kw)

    mod.run = run
    monkeypatch.setitem(sys.modules, "sigcup.trader.loop", mod)
    r = invoke("trade", "--live")
    assert r.exit_code == 2 and "LIVE is not 1" in r.output and called == []
    # LIVE=1 but no key
    s2 = Secrets(_env_file=None, live=True, kill_file=str(tmp_path / "KILL"))
    monkeypatch.setattr(settings, "secrets", lambda: s2)
    r = invoke("trade", "--live")
    assert r.exit_code == 2 and "SIG_API_KEY is empty" in r.output and called == []
    # both present -> runs with paper=False
    s3 = Secrets(_env_file=None, live=True, sig_api_key="k", kill_file=str(tmp_path / "KILL"))
    monkeypatch.setattr(settings, "secrets", lambda: s3)
    r = invoke("trade", "--live", "--once")
    assert r.exit_code == 0, r.output
    assert called == [{"paper": False, "once": True}]
    assert cli.live_allowed(s3) == (True, "")


def test_feeds_passes_args_to_main(db, sec, monkeypatch):
    seen: list[list[str]] = []
    mod = ModuleType("sigcup.feeds.loop")
    mod.main = lambda argv: seen.append(list(argv)) or 0
    monkeypatch.setitem(sys.modules, "sigcup.feeds.loop", mod)
    r = invoke("feeds", "--once", "--only", "polls", "--only", "ref_markets", "--log-level", "DEBUG")
    assert r.exit_code == 0, r.output
    assert seen == [["--log-level", "DEBUG", "--once", "--only=polls", "--only=ref_markets"]]


def test_bootstrap_loads_script_via_importlib(db, sec, monkeypatch, tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    # a dataclass with postponed annotations resolves them through sys.modules: the loader must register the module
    (scripts / "bootstrap_races.py").write_text(
        "from __future__ import annotations\nfrom dataclasses import dataclass\n"
        "@dataclass\nclass R:\n    n: int = 0\n"
        "def main(argv):\n    print('ARGV', argv, R().n)\n    return 0\n"
    )
    monkeypatch.setattr(settings, "ROOT", tmp_path)
    r = invoke("bootstrap", "--dry-run", "--skip-crosswalk")
    assert r.exit_code == 0, r.output
    assert "ARGV ['--dry-run', '--skip-crosswalk']" in r.output
    r = invoke("bootstrap", "--probe-writes")
    assert r.exit_code == 2 and "refusing --probe-writes" in r.output
    monkeypatch.setattr(settings, "ROOT", tmp_path / "nowhere")
    r = invoke("bootstrap")
    assert r.exit_code == 2 and "does not exist" in r.output


# -- kill ---------------------------------------------------------------------------------------
def test_kill_touches_file_and_clear_removes_it(db, sec):
    r = invoke("kill")
    assert r.exit_code == 0, r.output
    assert sec.kill_path.exists()
    assert "What happens next" in r.output and "cancel" in r.output
    assert "trader heartbeat is stale" in r.output  # nothing running in the test DB
    with dbengine.session() as s:
        kv_set(s, "killed", {"reason": "test"})
    r = invoke("kill", "--clear")
    assert r.exit_code == 0, r.output
    assert not sec.kill_path.exists()
    assert "cleared" in r.output and "pm2 restart trader" in r.output
    with dbengine.session() as s:
        assert kv_get(s, "killed") is None
    r = invoke("kill", "--cancel-now")
    assert r.exit_code == 2 and "refusing --cancel-now" in r.output


# -- status / report ----------------------------------------------------------------------------
def test_status_and_reports_on_empty_db(db, sec):
    r = invoke("status")
    assert r.exit_code == 1, r.output  # alerts (no heartbeats) -> exit 1
    assert "Heartbeats" in r.output and "missing" in r.output
    r = invoke("report", "status", "--json")
    assert r.exit_code == 0, r.output
    d = json.loads(r.output)
    assert d["exposure"]["n_positions"] == 0 and len(d["heartbeats"]) == 3
    r = invoke("report", "daily")
    assert r.exit_code == 0, r.output
    assert "Daily report" in r.output
    r = invoke("report", "daily", "--json", "--save")
    assert r.exit_code == 0, r.output
    assert list((sec.data_path / "reports").glob("*.json"))
    r = invoke("report", "leadlag", "--days", "3")
    assert r.exit_code == 0, r.output
    assert "nofeaturerows" in compact(r.output)
    r = invoke("report", "loop", "--once")
    assert r.exit_code == 0, r.output
    r = invoke("status", "--json")
    assert r.exit_code == 0 and json.loads(r.output)["heartbeats"][2]["process"] == "reporter"


# -- universe -----------------------------------------------------------------------------------
def test_universe_db_then_yaml_fallback(db, sec):
    r = invoke("universe", "--source", "db")
    assert r.exit_code == 1 and "empty universe" in r.output
    r = invoke("universe", "--chamber", "control")
    assert r.exit_code == 0, r.output
    assert "(yaml)" in r.output and "CONTROL-HOUSE" in compact(r.output)
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX Senate", exclusive=True, exhaustive=False, settlement_note="runoff"))
        s.add(Market(market_id="1", exchange_id="tx-d", race_id="SEN-TX", party="D", title="D", status="open", kalshi_ticker="K"))
        s.add(Market(market_id="2", exchange_id="tx-r", race_id="SEN-TX", party="R", title="R", status="open"))
    r = invoke("universe", "--state", "tx")
    assert r.exit_code == 0, r.output
    out = compact(r.output)
    assert "(db)" in r.output and "SEN-TX" in out and "D:tx-dopenK-" in out and "R:tx-ropen--" in out
    assert invoke("universe", "--state", "ZZ", "--source", "db").exit_code == 1


# -- canary -------------------------------------------------------------------------------------
async def test_canary_joins_bid_then_cancels(db, cfg, sec):
    clock = FakeClock()
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("m1", 0.60, 0.64)
    res = await cli.run_canary(fake, cfg, exchange_id="m1", title="T", quantity=10, wait_s=4, clock=clock.now, sleep=clock.sleep, poll_s=2)
    assert res.ok and res.status == "cancelled" and res.cancelled and res.filled == 0
    assert res.price == 0.60 and res.bid == 0.60 and res.ask == 0.64 and res.tournament == "fake"
    (o,) = fake.orders.values()
    assert o.status == "cancelled" and o.req.quantity == 10 and o.req.tournament_id == "t-fake" and o.req.expires_at is not None
    assert [k for _, k in fake.writes] == ["place", "cancel"] and res.writes_used == 2
    with dbengine.session() as s:
        row = s.execute(select(OrderIntent)).scalars().one()
        assert row.state == IntentState.CANCELLED.value and row.strategy == "manual" and row.paper is True and row.sig_order_id == o.order_id


async def test_canary_fill_is_recorded(db, cfg):
    clock = FakeClock()
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("m1", 0.60, 0.64)

    async def sleep(seconds: float) -> None:  # a counterparty hits our resting order during the wait
        clock.advance(seconds)
        for o in fake.open_orders():
            fake.fill(o.order_id)

    res = await cli.run_canary(fake, cfg, exchange_id="m1", quantity=10, wait_s=10, take=True, live=False, clock=clock.now, sleep=sleep)
    assert res.ok and res.status == "filled" and res.filled == 10 and res.avg_fill == 0.64 and not res.cancelled
    with dbengine.session() as s:
        assert s.get(OrderIntent, res.order_id and s.execute(select(OrderIntent.id)).scalar_one()).state == IntentState.FILLED.value
        f = s.execute(select(Fill)).scalars().one()
        assert f.quantity == 10 and f.strategy == "manual" and f.intent_id is not None


async def test_canary_uncertain_paths(db, cfg):
    clock = FakeClock()
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("m1", 0.60, 0.64)
    fake.default_script = ["lost"]  # never reached the exchange
    res = await cli.run_canary(fake, cfg, exchange_id="m1", quantity=10, wait_s=0, clock=clock.now, sleep=clock.sleep)
    assert not res.ok and res.status == "uncertain" and "ORDER_STATUS_UNKNOWN" in res.error
    with dbengine.session() as s:
        assert s.execute(select(OrderIntent)).scalars().one().state == IntentState.UNCERTAIN.value
    fake.default_script = ["uncertain"]  # recorded, response lost -> found by idempotency key, then cancelled
    res = await cli.run_canary(fake, cfg, exchange_id="m1", quantity=10, wait_s=0, clock=clock.now, sleep=clock.sleep)
    assert res.ok and res.status == "cancelled" and res.uncertain and res.order_id in fake.orders


async def test_canary_refuses_after_deadline_and_reports_errors(db, cfg):
    clock = FakeClock(start=cfg.deadline_utc + timedelta(minutes=1))
    fake = FakeAdapter(clock=clock.now)
    res = await cli.run_canary(fake, cfg, exchange_id="m1", clock=clock.now, sleep=clock.sleep)
    assert not res.ok and "deadline" in res.error and fake.writes == []
    clock = FakeClock()
    fake = FakeAdapter(clock=clock.now)
    fake.set_top("m1", 0.60, 0.64)
    from sigcup.sig.errors import ValidationError

    fake.default_script = [ValidationError("insufficient funds", 400, "INSUFFICIENT_FUNDS")]
    res = await cli.run_canary(fake, cfg, exchange_id="m1", clock=clock.now, sleep=clock.sleep)
    assert res.status == "rejected" and "place_order failed" in res.error
    assert "CHECK" in str(cli.render_canary(res).title) or cli.render_canary(res).row_count > 5


def test_canary_command_paper(db, sec, monkeypatch):
    with dbengine.session() as s:
        s.add(Race(race_id="SEN-TX", chamber="senate", state="TX", label="TX Senate"))
        s.add(Market(market_id="1", exchange_id="tx-d", race_id="SEN-TX", party="D", title="TX D", status="open"))
    fake = FakeAdapter()
    fake.set_top("tx-d", 0.60, 0.64)
    monkeypatch.setattr(cli, "build_adapter", lambda sec, cfg, live, budget=None: fake)
    r = invoke("canary", "--wait", "0")
    assert r.exit_code == 0, r.output
    assert "Canary (paper) on tx-d" in r.output and "cancelled" in r.output
    r = invoke("canary", "--live")
    assert r.exit_code == 2 and "refusing a live canary" in r.output
    r = invoke("canary", "--market", "nope", "--wait", "0")
    assert r.exit_code == 1, r.output
    assert "no price data" in compact(r.output).replace("nopricedata", "no price data") and "CHECK" in r.output
    assert len(fake.orders) == 1  # the unknown market placed nothing
