"""Typer CLI: db init|stats | bootstrap | trade [--paper/--live] [--once] | feeds | report daily|status|leadlag|loop |
status | kill | canary | universe.

The heavy entry points (trader, feeds, bootstrap script) are imported LAZILY inside each command so
``sigcup --help`` works even while those modules are unfinished; a command whose target is missing
prints a clear error and exits with code 2.

Live-trading guard (one rule everywhere): anything that can send a real order (``trade`` in live mode,
``canary --live``, ``bootstrap --probe-writes`` including when passed through ``--``, ``kill --cancel-now``)
requires ``LIVE=1`` (environment or .env) AND a non-empty ``SIG_API_KEY``; otherwise the command refuses
and exits non-zero. ``trade`` without a flag follows ``LIVE`` (that is what pm2 runs; ``--paper`` /
``--live`` override it); every other command is paper unless asked.

Kill switch: ``sigcup kill`` touches the kill file AND sets the database kill flags (``killed`` for the
live trader, ``killed:paper`` for the paper one), so a trader that is (re)started afterwards begins
killed: its startup reconcile cancels what is open and re-sends nothing. ``trade`` sets the same flag
when it finds the kill file. ``kill --clear`` removes both.

One limiter: the trader holds ``trader.lock`` (next to the kill file, ``data/trader.lock`` by default) while
it runs. ``canary`` and ``bootstrap --probe-writes`` build their own limiter, so they refuse (exit 2) while a
LIVE trader holds the lock (a paper trader, as in the runbook's canary step, or a stale lock does not
block them); ``kill --cancel-now`` is the emergency path and only warns. A second ``trade`` refuses while
another trader holds the lock.
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from sigcup import __version__, settings
from sigcup.common.ids import new_idempotency_key
from sigcup.common.log import get_logger, setup_logging
from sigcup.common.time import before_deadline, ensure_utc, utc_now
from sigcup.settings import Secrets, StrategyConfig
from sigcup.sig.schemas import OrderRequest, round_tick

log = get_logger(__name__)
# a pipe (pm2 logs, tests) gets a fixed wide console so tables are not wrapped at 80 columns
console = Console(width=None if sys.stdout.isatty() else 140)
err_console = Console(stderr=True)

app = typer.Typer(help="sigcup: SIG Predictions Cup trading system", no_args_is_help=True, rich_markup_mode="rich")
db_app = typer.Typer(help="Database maintenance.", no_args_is_help=True)
report_app = typer.Typer(help="Status, daily report, lead-lag analysis, and the reporter process.", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(report_app, name="report")

EXIT_USAGE = 2
# the account shares 100 reads/min across every process: a 30 s canary polls ~7 times (2 reads each)
CANARY_POLL_S = 5.0
KV_KILLED = "killed"  # == sigcup.trader.oms.KV_KILLED (not imported: `kill` must work if the trader is broken)
KV_KILLED_PAPER = "killed:paper"  # == sigcup.trader.oms.KV_KILLED_PAPER
PROBE_WRITES = "--probe-writes"
ONLY_OPTION = typer.Option(None, "--only", help="Collector name(s) to run (with --once).")


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def fail(msg: str, code: int = EXIT_USAGE) -> NoReturn:
    """Print an error to stderr and exit non-zero."""
    err_console.print(f"[bold red]error:[/bold red] {msg}")
    raise typer.Exit(code)


def load_module(module: str, *, what: str) -> ModuleType:
    """Import a module lazily; an unfinished/missing module becomes a clear CLI error."""
    try:
        return importlib.import_module(module)
    except NotImplementedError as e:
        fail(f"{what} is not implemented yet ({module}: {e})")
    except ImportError as e:
        fail(f"{what} cannot be imported ({module}): {e}")
    except Exception as e:  # noqa: BLE001 - a stub may raise anything at import
        fail(f"{what} failed to import ({module}): {type(e).__name__}: {e}")


def load_entry(module: str, *attrs: str, what: str) -> Any:
    """The first of ``attrs`` found on the lazily imported module, else a clear error."""
    mod = load_module(module, what=what)
    for a in attrs:
        fn = getattr(mod, a, None)
        if fn is not None:
            return fn
    fail(f"{what}: none of {', '.join(attrs)} is defined in {module}")


def load_script(name: str, *, what: str) -> ModuleType:
    """Load ``scripts/<name>.py`` (scripts/ is not a package) from the repo root via importlib."""
    path = settings.ROOT / "scripts" / f"{name}.py"
    if not path.exists():
        fail(f"{what}: {path} does not exist")
    spec = importlib.util.spec_from_file_location(f"sigcup_scripts.{name}", path)
    if spec is None or spec.loader is None:
        fail(f"{what}: cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses with postponed annotations resolve them via sys.modules
    try:
        spec.loader.exec_module(mod)
    except NotImplementedError as e:
        sys.modules.pop(spec.name, None)
        fail(f"{what} is not implemented yet ({path.name}: {e})")
    except Exception as e:  # noqa: BLE001
        sys.modules.pop(spec.name, None)
        fail(f"{what} failed to load ({path.name}): {type(e).__name__}: {e}")
    return mod


def live_allowed(sec: Secrets) -> tuple[bool, str]:
    """Whether real orders may be sent: LIVE=1 and a trade key."""
    if not sec.live:
        return False, "LIVE is not 1 (set LIVE=1 in .env or the environment)"
    if not sec.sig_api_key:
        return False, "SIG_API_KEY is empty (create a read+trade key under My Profile -> API Keys)"
    return True, ""


def require_live(sec: Secrets, action: str) -> None:
    ok, why = live_allowed(sec)
    if not ok:
        fail(f"refusing {action}: {why}. Paper mode needs no key (LIVE unset or 0, or pass --paper).")


def trader_lock_holder(sec: Secrets) -> Any | None:
    """The lock of a running trader (``sigcup.trader.loop.running_trader``), None when there is none or the
    check itself is unavailable (a trader module that cannot be imported cannot be running either)."""
    try:
        mod = importlib.import_module("sigcup.trader.loop")
        return mod.running_trader(sec)
    except Exception as e:  # noqa: BLE001
        log.warning("trader lock check unavailable: %s: %s", type(e).__name__, e)
        return None


def require_no_live_trader(sec: Secrets, action: str) -> None:
    """Refuse a command that builds its own exchange limiter while a LIVE trader runs (a lock without a
    mode counts as live). A paper trader does not block it (the runbook runs canaries beside one)."""
    holder = trader_lock_holder(sec)
    if holder is None:
        return
    if getattr(holder, "mode", None) == "paper":
        console.print(f"note: a paper trader (pid {holder.pid}) is running; it sends no orders")
        return
    fail(
        f"refusing {action}: a live trader (pid {holder.pid}) holds {holder.path}. It owns the account's "
        "exchange budget (30 writes + 100 reads per minute) and this command would add its own limiter on top. "
        "Stop it first (`pm2 stop trader`), or remove the lock file if no trader is running."
    )


def _kill_secrets() -> Secrets:
    """Settings for the kill command, which must work even when the environment is broken: a value such
    as ``LIVE=live`` fails pydantic's bool parsing, so retry with ``LIVE`` forced to 0 (the kill file and
    DSN still come from the environment / .env), then fall back to the defaults."""
    try:
        return settings.secrets()
    except Exception as e:  # noqa: BLE001
        err_console.print(f"[yellow]warning:[/yellow] the settings do not validate ({type(e).__name__}); treating LIVE as 0 to find the kill file")
        try:
            return Secrets(live=False)
        except Exception:  # noqa: BLE001
            return Secrets.model_construct()


def _set_kill_flags(reason: str, *, live: bool = True, paper: bool = True) -> list[str]:
    """Persist the kill flag(s) the OMS reads at startup (an existing flag and its reason are kept).
    Returns the keys written."""
    from sigcup.db import engine as dbengine
    from sigcup.db.repo import kv_get, kv_set

    _init_db()
    written: list[str] = []
    with dbengine.session() as s:
        for key, is_paper, wanted in ((KV_KILLED, False, live), (KV_KILLED_PAPER, True, paper)):
            if wanted and not kv_get(s, key):
                kv_set(s, key, {"reason": reason, "ts": utc_now().isoformat(), "source": "cli", "paper": is_paper})
                written.append(key)
    return written


def _is_probe_writes(arg: str) -> bool:
    """``--probe-writes`` or anything argparse would expand to it (``--probe``, ``--probe-writes=...``)."""
    name = arg.split("=", 1)[0]
    return len(name) >= 3 and name.startswith("--") and PROBE_WRITES.startswith(name)


def describe_dsn(dsn: str) -> str:
    """A loggable form of the DSN (credentials stripped)."""
    if dsn.startswith("sqlite:///"):
        p = Path(dsn[len("sqlite:///") :])
        return str(p if p.is_absolute() else settings.ROOT / p)
    if "@" in dsn:
        scheme, rest = dsn.split("://", 1) if "://" in dsn else ("", dsn)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return dsn


def run_async(coro: Awaitable[Any]) -> Any:
    try:
        return asyncio.run(coro)  # type: ignore[arg-type]
    except KeyboardInterrupt:
        err_console.print("interrupted")
        raise typer.Exit(130) from None


def _init_db() -> None:
    from sigcup.db import engine as dbengine

    dbengine.init_db()


# ---------------------------------------------------------------------------------------------
# root
# ---------------------------------------------------------------------------------------------
def _version(value: bool) -> None:
    if value:
        console.print(f"sigcup {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    log_level: str = typer.Option("INFO", "--log-level", envvar="SIGCUP_LOG_LEVEL", help="Python log level for this process."),
    version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Print the version and exit."),
) -> None:
    setup_logging(log_level.upper())


# ---------------------------------------------------------------------------------------------
# db
# ---------------------------------------------------------------------------------------------
@db_app.command("init")
def db_init() -> None:
    """Create every table (idempotent) in the database named by DSN."""
    _init_db()
    console.print(f"database ready: {describe_dsn(settings.secrets().dsn)}")


@db_app.command("stats")
def db_stats() -> None:
    """Row counts per table."""
    from sqlalchemy import func, select

    from sigcup.db import engine as dbengine
    from sigcup.db.models import Base

    _init_db()
    t = Table(title=f"Tables in {describe_dsn(settings.secrets().dsn)}")
    t.add_column("table")
    t.add_column("rows", justify="right")
    with dbengine.session() as s:
        for table in Base.metadata.sorted_tables:
            n = s.execute(select(func.count()).select_from(table)).scalar()
            t.add_row(table.name, str(n))
    console.print(t)


# ---------------------------------------------------------------------------------------------
# bootstrap / trade / feeds
# ---------------------------------------------------------------------------------------------
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def bootstrap(
    ctx: typer.Context,
    probe_writes: bool = typer.Option(False, "--probe-writes", help="Run the batch-write-count probe (places and cancels 2 tiny REAL orders; needs LIVE=1 and the trade key)."),
) -> None:
    """Pull the SIG market list + relationships, crosswalk to Kalshi/Polymarket, write config/races.yaml and the Race/Market rows, run the day-1 probes.

    Extra arguments are passed through to scripts/bootstrap_races.py (see `sigcup bootstrap -- --help`).
    """
    sec = settings.secrets()
    extra = list(ctx.args)
    passthrough_probe = any(_is_probe_writes(a) for a in extra)  # `sigcup bootstrap -- --probe-writes`
    if probe_writes or passthrough_probe:
        require_live(sec, "--probe-writes (it places real orders)")
        require_no_live_trader(sec, "--probe-writes")
    mod = load_script("bootstrap_races", what="the bootstrap script")
    main = getattr(mod, "main", None)
    if main is None:
        fail("the bootstrap script has no main(argv)")
    argv = extra + ([PROBE_WRITES] if probe_writes and not passthrough_probe else [])
    _init_db()
    code = main(argv)
    raise typer.Exit(int(code or 0))


@app.command()
def trade(
    paper: bool | None = typer.Option(None, "--paper/--live", help="Force paper mode, or LIVE orders (needs LIVE=1 and SIG_API_KEY). Default: LIVE=1 -> live, otherwise paper."),
    once: bool = typer.Option(False, "--once", help="One refresh / model / reconcile pass, then exit."),
) -> None:
    """Run the trader process (the only process that holds the trade key). Without a flag it follows LIVE."""
    sec = settings.secrets()
    explicit = paper is not None
    if paper is None:
        paper = not sec.live
    if not paper:
        require_live(sec, "--live" if explicit else "live trading (LIVE=1 and no --paper)")
    if sec.kill_path.exists():
        err_console.print(f"[yellow]warning:[/yellow] kill file present at {sec.kill_path}: the trader starts killed (cancels, sends nothing) until `sigcup kill --clear`")
        try:
            _set_kill_flags(f"kill file present: {sec.kill_path}", live=not paper, paper=paper)
        except Exception as e:  # noqa: BLE001 - the trader still checks the file on every cycle
            log.warning("could not set the kill flag in the database: %s", e)
    run = load_entry("sigcup.trader.loop", "run", what="the trader")
    console.print(f"trader: mode={'LIVE' if not paper else 'paper'} once={once}")
    try:
        result = run_async(run(paper=paper, once=once))
    except Exception as e:
        if type(e).__name__ == "TraderLockHeld":  # another trader runs: one process owns the exchange budget
            fail(str(e))
        raise
    if once and result is not None:
        console.print(_summarise_cycle(result))


def _summarise_cycle(res: Any) -> str:
    parts = [f"{k}={getattr(res, k)}" for k in ("skipped", "fair", "arbs", "desired", "accepted", "killed") if hasattr(res, k)]
    return "cycle: " + " ".join(parts)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def feeds(
    ctx: typer.Context,
    once: bool = typer.Option(False, "--once", help="Run every enabled collector once and exit."),
    only: list[str] | None = ONLY_OPTION,
) -> None:
    """Run the feeds process (reference prices, polls, news, social, NLP, events)."""
    mod = load_module("sigcup.feeds.loop", what="the feeds process")
    argv = list(ctx.args) + (["--once"] if once else []) + [f"--only={o}" for o in (only or [])]
    main = getattr(mod, "main", None)
    if main is not None:
        _init_db()
        raise typer.Exit(int(main(argv) or 0))
    run = getattr(mod, "run", None)
    if run is None:
        fail("the feeds process defines neither main(argv) nor run()")
    _init_db()
    run_async(run())


# ---------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------
def _status_dict() -> dict[str, Any]:
    from sigcup.db import engine as dbengine
    from sigcup.reporter import report

    _init_db()
    with dbengine.session() as s:
        return report.status(s, now=utc_now(), cfg=settings.config(), sec=settings.secrets())


@report_app.command("status")
def report_status(json_out: bool = typer.Option(False, "--json", help="Print the plain dict as JSON.")) -> None:
    """Heartbeats, cash/equity/reserved, open orders by strategy, exposure, rank, alerts."""
    from sigcup.reporter import report

    d = _status_dict()
    if json_out:
        console.print_json(json.dumps(d, default=str))
        return
    console.print(report.render_status(d))
    if d["alerts"]:
        raise typer.Exit(1)


@report_app.command("daily")
def report_daily(
    json_out: bool = typer.Option(False, "--json", help="Print the plain dict as JSON."),
    save: bool = typer.Option(False, "--save", help="Also write <data_dir>/reports/<day>.json."),
) -> None:
    """P&L by strategy, CLV, calibration, SIG-vs-reference gaps, rank history."""
    from sigcup.db import engine as dbengine
    from sigcup.reporter import clv, report

    _init_db()
    now = utc_now()
    cfg = settings.config()
    with dbengine.session() as s:
        clv.compute(s, now, deadline=cfg.deadline_utc)
        d = report.daily(s, now=now, cfg=cfg)
    if json_out:
        console.print_json(json.dumps(d, default=str))
    else:
        console.print(report.render_daily(d))
    if save:
        out = settings.secrets().data_path / "reports"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{d['day']}.json"
        path.write_text(json.dumps(d, indent=1, default=str))
        console.print(f"saved {path}")


@report_app.command("leadlag")
def report_leadlag(
    days: int = typer.Option(14, "--days", help="Trailing window in days."),
    json_out: bool = typer.Option(False, "--json", help="Print the plain dict as JSON."),
) -> None:
    """Lead-lag cross-correlation (1-24 h) and Spearman IC of stance z-scores vs price changes, per platform."""
    from sigcup.db import engine as dbengine
    from sigcup.reporter import leadlag

    _init_db()
    with dbengine.session() as s:
        d = leadlag.compute(s, utc_now(), days=days)
    if json_out:
        console.print_json(json.dumps(d, default=str))
    else:
        console.print(leadlag.render(d))


@report_app.command("loop")
def report_loop(once: bool = typer.Option(False, "--once", help="One tick (CLV + status; daily if due) and exit.")) -> None:
    """Run the reporter process: hourly status + CLV, daily report at 12:00 UTC, heartbeat."""
    from sigcup.reporter import loop as reporter_loop

    raise typer.Exit(int(reporter_loop.main(once=once) or 0))


@app.command()
def status(json_out: bool = typer.Option(False, "--json", help="Print the plain dict as JSON.")) -> None:
    """Alias for `report status`."""
    report_status(json_out=json_out)


# ---------------------------------------------------------------------------------------------
# kill
# ---------------------------------------------------------------------------------------------
KILL_NEXT = """[bold]What happens next[/bold]
 1. The trader checks for the file every 5 s (and on every model cycle): it cancels every resting
    order (cancel-all), logs KILL SWITCH and exits once nothing is left open.
 2. The 'killed' flag is already set in the database, so pm2's restart (or any later start) begins
    killed: it cancels what is open and sends nothing while the file exists or the flag is set.
 3. Check: `sigcup status` (open orders 0, trader status 'killed') and `pm2 logs trader`.
 4. To resume after investigating: `sigcup kill --clear`, then `pm2 restart trader`."""


@app.command()
def kill(
    clear: bool = typer.Option(False, "--clear", help="Remove the kill file and clear the database flag (resume trading)."),
    cancel_now: bool = typer.Option(False, "--cancel-now", help="Also cancel every resting order right now through a direct client (needs LIVE=1 and the trade key; use when the trader is dead)."),
) -> None:
    """Touch the kill file: the trader cancels all orders and stops. `--clear` undoes it."""
    from sigcup.db import engine as dbengine
    from sigcup.db.repo import kv_get, kv_set

    sec = _kill_secrets()
    path = sec.kill_path
    if clear:
        if path.exists():
            path.unlink()
            console.print(f"removed {path}")
        else:
            console.print(f"no kill file at {path}")
        try:
            _init_db()
            with dbengine.session() as s:
                flag = kv_get(s, KV_KILLED)
                kv_set(s, KV_KILLED, None)
                kv_set(s, KV_KILLED_PAPER, None)
        except Exception as e:  # noqa: BLE001 - e.g. settings that do not validate
            fail(f"could not clear the database kill flags ({type(e).__name__}: {str(e).splitlines()[0]}): fix the settings and rerun `sigcup kill --clear`")
        console.print(f"database kill flags cleared{f' (was: {flag})' if flag else ''}. Run `pm2 restart trader` to resume trading.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    console.print(f"[bold red]kill file created:[/bold red] {path}")
    try:
        _set_kill_flags(f"sigcup kill (kill file {path})")
        console.print("database kill flags set: a trader that (re)starts begins killed")
    except Exception as e:  # noqa: BLE001 - the file alone still stops a running trader
        err_console.print(f"[yellow]warning:[/yellow] could not set the database kill flag ({type(e).__name__}: {e}); a running trader still stops on the file")
    console.print(KILL_NEXT)
    try:
        d = _status_dict()
        trader = next((h for h in d["heartbeats"] if h["process"] == "trader"), None)
        if trader and trader["stale"]:
            err_console.print("[yellow]warning:[/yellow] the trader heartbeat is stale or missing: nothing is running to cancel resting orders. Use `sigcup kill --cancel-now` (LIVE=1 + trade key) or cancel in the web app.")
    except Exception as e:  # noqa: BLE001
        log.warning("status check after kill failed: %s", e)
    if cancel_now:
        require_live(sec, "--cancel-now")
        holder = trader_lock_holder(sec)
        if holder is not None:  # the emergency path is never refused, but it adds a second limiter
            err_console.print(
                f"[yellow]warning:[/yellow] a {holder.mode or 'live'} trader (pid {holder.pid}) is running: --cancel-now uses its own "
                "limiter beside it (one cancel-all write); the account's shared exchange limits may be briefly exceeded"
            )
        n = run_async(_cancel_all_direct(sec, settings.config()))
        console.print(f"cancel-all: exchange cancelled {n} orders")


async def _cancel_all_direct(sec: Secrets, cfg: StrategyConfig) -> int:
    adapter = build_adapter(sec, cfg, live=True)
    try:
        return int(await adapter.cancel_all())
    finally:
        await _close(adapter)


# ---------------------------------------------------------------------------------------------
# canary
# ---------------------------------------------------------------------------------------------
@dataclass
class CanaryResult:
    """What the canary did; ``ok`` means the order was acknowledged and ended in a known state."""

    mode: str
    exchange_id: str
    title: str
    side: str
    action: str
    price: float
    quantity: int
    bid: float | None = None
    ask: float | None = None
    order_id: str | None = None
    status: str = "not sent"
    filled: int = 0
    avg_fill: float | None = None
    cancelled: bool = False
    uncertain: bool = False
    elapsed_s: float = 0.0
    error: str | None = None
    tournament: str | None = None
    writes_used: int = 0

    @property
    def ok(self) -> bool:
        return self.error is None and self.status in ("filled", "cancelled", "partial-cancelled", "expired")


def build_adapter(sec: Secrets, cfg: StrategyConfig, *, live: bool, budget: Any | None = None) -> Any:
    """The live SigClient (trade key) or a PaperAdapter wrapping a read-only client when a key exists."""
    from sigcup.sig.ratelimit import Budget

    budget = budget or Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min)
    client_mod = load_module("sigcup.sig.client", what="the SIG client") if (live or sec.sig_read_key or sec.sig_api_key) else None
    if live:
        return client_mod.SigClient(sec.sig_api_key, budget, cfg=cfg, base_url=sec.sig_base_url, tournament_slug=sec.sig_tournament_slug)  # type: ignore[union-attr]
    paper_mod = load_module("sigcup.paper", what="the paper adapter")
    source = None
    if client_mod is not None:
        key = sec.sig_read_key or sec.sig_api_key
        try:
            source = client_mod.SigClient(key, budget, cfg=cfg, base_url=sec.sig_base_url, tournament_slug=sec.sig_tournament_slug)
        except Exception as e:  # noqa: BLE001
            log.warning("paper: no live market-data source (%s); using DB quotes", type(e).__name__)
    return paper_mod.PaperAdapter(cfg, source=source, budget=budget)


async def _close(adapter: Any) -> None:
    for obj in (adapter, getattr(adapter, "source", None)):
        aclose = getattr(obj, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:  # noqa: BLE001
                pass


def _pick_market(explicit: str | None) -> tuple[str, str]:
    """(exchange_id, title) from the DB: the given id, else the first open market."""
    from sqlalchemy import select

    from sigcup.db import engine as dbengine
    from sigcup.db.models import Market

    with dbengine.session() as s:
        if explicit:
            m = s.execute(select(Market).where((Market.exchange_id == explicit) | (Market.market_id == explicit))).scalars().first()
            return (m.exchange_id, m.title) if m else (explicit, explicit)
        m = s.execute(select(Market).where(Market.status == "open").order_by(Market.race_id, Market.party)).scalars().first()
        if m is None:
            m = s.execute(select(Market).order_by(Market.race_id)).scalars().first()
        if m is None:
            fail("no markets in the database: run `sigcup bootstrap` first or pass --market EXCHANGE_ID")
        return m.exchange_id, m.title


async def run_canary(
    adapter: Any,
    cfg: StrategyConfig,
    *,
    exchange_id: str,
    title: str = "",
    quantity: int = 10,
    wait_s: float = 30.0,
    take: bool = False,
    live: bool = False,
    clock: Callable[[], datetime] = utc_now,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    poll_s: float = CANARY_POLL_S,
) -> CanaryResult:
    """Place ONE small buy-YES order (joining the bid, or taking the ask with ``take``), wait for a
    fill, cancel what remains, and record the intent/fills in the database (strategy 'manual')."""
    from sigcup.db import engine as dbengine
    from sigcup.db.models import Fill, IntentState, OrderIntent, Strategy

    res = CanaryResult(mode="LIVE" if live else "paper", exchange_id=exchange_id, title=title, side="yes", action="buy", price=0.0, quantity=int(quantity))
    start = clock()
    if not before_deadline(cfg.deadline_utc, start):
        res.error = f"past the trading deadline {cfg.deadline_utc.isoformat()}"
        return res
    try:
        get_t = getattr(adapter, "get_tournament", None)
        if callable(get_t):
            t = await get_t()
            res.tournament = getattr(t, "slug", None) or getattr(t, "id", None)
        snaps = await adapter.get_prices([exchange_id])
        snap = snaps.get(exchange_id)
        res.bid = getattr(snap, "best_bid", None) if snap else None
        res.ask = getattr(snap, "best_ask", None) if snap else None
        last = getattr(snap, "last", None) if snap else None
        if take and res.ask is not None:
            price = res.ask
        elif res.bid is not None:
            price = res.bid
        elif last is not None:
            price = last
        else:
            res.error = f"no price data for {exchange_id}: is the exchange id right and the market open?"
            return res
        res.price = round_tick(float(price))
        if live and res.price > 0.5 and not take:
            # a canary must stay cheap: never bid more than 0.50 unless the user asked to take
            res.price = round_tick(0.5)
    except Exception as e:  # noqa: BLE001
        res.error = f"market data failed: {type(e).__name__}: {e}"
        return res

    key = new_idempotency_key()
    now = clock()
    expires = now + timedelta(seconds=max(60.0, wait_s + 60.0))
    req = OrderRequest(
        exchange_id=exchange_id, side="yes", action="buy", price=res.price, quantity=res.quantity, idempotency_key=key,
        expires_at=expires, strategy="manual", tournament_id=getattr(adapter, "tournament_id", None),
    )
    intent = OrderIntent(
        id=key, exchange_id=exchange_id, side="yes", action="buy", price=res.price, quantity=res.quantity, strategy=Strategy.MANUAL.value,
        state=IntentState.SENT.value, created_at=now, updated_at=now, expires_at=expires, paper=not live, error="canary",
    )
    with dbengine.session() as s:
        s.merge(intent)

    def persist(state: str, **fields: Any) -> None:
        with dbengine.session() as s:
            row = s.get(OrderIntent, key)
            if row is not None:
                row.state = state
                row.updated_at = clock()
                for k, v in fields.items():
                    setattr(row, k, v)

    try:
        r = await adapter.place_order(req)
        res.writes_used += 1
    except Exception as e:  # noqa: BLE001
        res.error = f"place_order failed: {type(e).__name__}: {e}"
        persist(IntentState.REJECTED.value, error=res.error)
        res.status = "rejected"
        return res
    res.order_id = getattr(r, "order_id", None)
    res.uncertain = bool(getattr(r, "uncertain", False))
    if res.order_id is None and res.uncertain:
        # the exchange may have the order: look it up by idempotency key before giving up
        try:
            for o in await adapter.get_open_orders():
                if getattr(o, "idempotency_key", None) == key:
                    res.order_id = o.order_id
                    break
        except Exception as e:  # noqa: BLE001
            log.warning("canary: open-orders lookup failed: %s", type(e).__name__)
    if res.order_id is None:
        res.status = "uncertain" if res.uncertain else "rejected"
        res.error = "no order id returned" + (" (ORDER_STATUS_UNKNOWN: reconcile before retrying)" if res.uncertain else "")
        persist(IntentState.UNCERTAIN.value if res.uncertain else IntentState.REJECTED.value, error=res.error)
        return res
    persist(IntentState.ACKED.value, sig_order_id=res.order_id)
    res.status = "acked"
    traded = int(getattr(r, "quantity_traded", 0) or 0)
    res.filled = traded

    # -- wait for fills -----------------------------------------------------------------------
    deadline = start + timedelta(seconds=wait_s)
    seen: dict[str, Any] = {}
    while True:
        try:
            fills = await adapter.get_fills(since=start - timedelta(minutes=1))
            for f in fills:
                if getattr(f, "order_id", None) == res.order_id and f.fill_id not in seen:
                    seen[f.fill_id] = f
            if seen:
                qty = sum(int(f.quantity) for f in seen.values())
                res.filled = max(res.filled, qty)
                res.avg_fill = sum(float(f.price) * int(f.quantity) for f in seen.values()) / qty if qty else None
            if res.filled >= res.quantity:
                res.status = "filled"
                break
            open_ids = {o.order_id for o in await adapter.get_open_orders()}
            if res.order_id not in open_ids and res.filled < res.quantity:
                res.status = "gone"  # expired / cancelled elsewhere / filled but fills not yet visible
                break
        except Exception as e:  # noqa: BLE001
            log.warning("canary: poll failed: %s", type(e).__name__)
        if clock() >= deadline:
            break
        await sleep(poll_s)

    # -- cancel the remainder -----------------------------------------------------------------
    if res.status in ("acked",):
        try:
            res.cancelled = bool(await adapter.cancel_order(res.order_id))
            res.writes_used += 1
            res.status = "partial-cancelled" if res.filled else "cancelled"
        except Exception as e:  # noqa: BLE001
            res.error = f"cancel failed: {type(e).__name__}: {e}"
    elif res.status == "gone":
        res.status = "expired"

    # -- persist -------------------------------------------------------------------------------
    final = {"filled": IntentState.FILLED.value, "cancelled": IntentState.CANCELLED.value, "partial-cancelled": IntentState.CANCELLED.value, "expired": IntentState.EXPIRED.value}.get(res.status, IntentState.ACKED.value)
    persist(final, filled_qty=res.filled, avg_fill=res.avg_fill, error=res.error)
    if seen:
        with dbengine.session() as s:
            for f in seen.values():
                if s.get(Fill, f.fill_id) is None:
                    s.add(
                        Fill(
                            fill_id=f.fill_id, sig_order_id=res.order_id, intent_id=key, exchange_id=exchange_id, ts=ensure_utc(f.ts),
                            side=str(f.side), action=str(f.action), price=float(f.price), quantity=int(f.quantity), fee=float(getattr(f, "fee", 0.0) or 0.0),
                            strategy=Strategy.MANUAL.value, paper=not live,
                        )
                    )
    res.elapsed_s = (clock() - start).total_seconds()
    return res


def render_canary(res: CanaryResult) -> Table:
    t = Table(title=f"Canary ({res.mode}) on {res.exchange_id} {res.title[:60]}", show_header=False)
    t.add_column("k")
    t.add_column("v")
    for k, v in asdict(res).items():
        if k in ("mode", "exchange_id", "title"):
            continue
        t.add_row(k, "-" if v is None else str(v))
    t.add_row("verdict", "[green]OK[/green]" if res.ok else "[red]CHECK[/red]")
    return t


@app.command()
def canary(
    live: bool = typer.Option(False, "--live/--paper", help="Send a REAL 10-share order (needs LIVE=1 and SIG_API_KEY) or a paper one (default)."),
    market: str | None = typer.Option(None, "--market", help="Exchange id (default: the first open market in the DB)."),
    quantity: int = typer.Option(10, "--qty", min=1, max=100, help="Shares (keep it tiny)."),
    wait: float = typer.Option(30.0, "--wait", help="Seconds to wait for a fill before cancelling."),
    take: bool = typer.Option(False, "--take", help="Cross the spread (buy at the best ask) so it fills; default joins the bid and is cancelled."),
) -> None:
    """One small buy-YES order on one market: place, wait, show fill or cancel. The live go/no-go check."""
    sec = settings.secrets()
    cfg = settings.config()
    if live:
        require_live(sec, "a live canary")
        if sec.kill_path.exists():
            fail(f"kill file present at {sec.kill_path}: clear it first (`sigcup kill --clear`)")
    require_no_live_trader(sec, "the canary")
    _init_db()
    exchange_id, title = _pick_market(market)
    adapter = build_adapter(sec, cfg, live=live)

    async def go() -> CanaryResult:
        try:
            return await run_canary(adapter, cfg, exchange_id=exchange_id, title=title, quantity=quantity, wait_s=wait, take=take, live=live)
        finally:
            await _close(adapter)

    res = run_async(go())
    console.print(render_canary(res))
    if not res.ok:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------------------------
def _universe_rows_db() -> list[dict[str, Any]]:
    from sqlalchemy import select

    from sigcup.db import engine as dbengine
    from sigcup.db.models import Market, Race

    _init_db()
    rows: list[dict[str, Any]] = []
    with dbengine.session() as s:
        markets: dict[str, list[Market]] = {}
        for m in s.execute(select(Market)).scalars().all():
            markets.setdefault(m.race_id, []).append(m)
        for r in s.execute(select(Race).order_by(Race.chamber, Race.state, Race.district)).scalars().all():
            legs = sorted(markets.get(r.race_id, []), key=lambda m: m.party)
            rows.append(
                {
                    "race_id": r.race_id,
                    "chamber": r.chamber,
                    "state": r.state,
                    "label": r.label,
                    "exclusive": r.exclusive,
                    "exhaustive": r.exhaustive,
                    "legs": [
                        {"party": m.party, "exchange_id": m.exchange_id, "status": m.status, "kalshi": bool(m.kalshi_ticker), "polymarket": bool(m.polymarket_condition_id)}
                        for m in legs
                    ],
                    "note": r.settlement_note,
                }
            )
    return rows


def _universe_rows_yaml() -> list[dict[str, Any]]:
    mod = load_module("sigcup.feeds.universe", what="the universe helpers")
    data = mod.load_races_yaml()
    rows: list[dict[str, Any]] = []
    for race_id, e in sorted((data or {}).get("races", {}).items()):
        legs = []
        for party, m in sorted((e.get("markets") or {}).items()):
            legs.append({"party": party, "exchange_id": m.get("sig_exchange_id") or "", "status": m.get("status") or "", "kalshi": bool(m.get("kalshi_ticker")), "polymarket": bool(m.get("polymarket_condition_id"))})
        rows.append({"race_id": race_id, "chamber": e.get("chamber"), "state": e.get("state"), "label": e.get("label"), "exclusive": e.get("exclusive"), "exhaustive": e.get("exhaustive"), "legs": legs, "note": e.get("settlement_note")})
    return rows


@app.command()
def universe(
    chamber: str | None = typer.Option(None, "--chamber", help="senate | governor | house | control"),
    state: str | None = typer.Option(None, "--state", help="Two-letter state code."),
    source: str = typer.Option("auto", "--source", help="db | yaml | auto (DB, falling back to config/races.yaml)."),
) -> None:
    """Print the race / market universe (exchange ids, exclusivity flags, crosswalk coverage)."""
    rows: list[dict[str, Any]] = []
    used = source
    if source in ("db", "auto"):
        rows = _universe_rows_db()
        used = "db"
    if not rows and source in ("yaml", "auto"):
        rows = _universe_rows_yaml()
        used = "yaml"
    if chamber:
        rows = [r for r in rows if (r["chamber"] or "").lower() == chamber.lower()]
    if state:
        rows = [r for r in rows if (r["state"] or "").upper() == state.upper()]
    t = Table(title=f"Universe ({used}): {len(rows)} races, {sum(len(r['legs']) for r in rows)} markets")
    for col in ("race", "chamber", "st", "label", "excl", "exh", "legs (party:exchange_id status K/P)", "note"):
        t.add_column(col)
    by_chamber: dict[str, int] = {}
    mapped = 0
    for r in rows:
        by_chamber[r["chamber"] or "?"] = by_chamber.get(r["chamber"] or "?", 0) + 1
        legs = []
        for leg in r["legs"]:
            if leg["exchange_id"]:
                mapped += 1
            flags = ("K" if leg["kalshi"] else "-") + ("P" if leg["polymarket"] else "-")
            legs.append(f"{leg['party']}:{leg['exchange_id'] or '?'} {leg['status'] or ''} {flags}")
        t.add_row(r["race_id"], str(r["chamber"]), str(r["state"] or ""), str(r["label"] or "")[:36], _flag(r["exclusive"]), _flag(r["exhaustive"]), "\n".join(legs), (r["note"] or "")[:40])
    console.print(t)
    console.print(f"by chamber: {by_chamber}; markets with exchange ids: {mapped}")
    if not rows:
        err_console.print("[yellow]empty universe:[/yellow] run `sigcup bootstrap` (or check config/races.yaml)")
        raise typer.Exit(1)


def _flag(v: bool | None) -> str:
    return "?" if v is None else ("yes" if v else "no")


def main() -> None:
    app()


if __name__ == "__main__":
    main()


__all__ = ["app", "main", "run_canary", "CanaryResult", "build_adapter", "live_allowed", "load_entry", "load_script"]
