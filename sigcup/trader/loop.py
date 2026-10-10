"""The `trader` process entry point (asyncio). Owns the ONLY exchange client in the system.
Cycle every cadence.model_s: refresh books (Realtime + bulk REST), load fair values, run consistency +
signals, apply risk, hand desired orders to the OMS; reconcile every cadence.reconcile_s; snapshot the
leaderboard every cadence.leaderboard_s; heartbeat. `sigcup trade [--paper]` calls run().

Model modules are imported lazily by the exact contract names in ``sigcup.trader.types`` and every call
is wrapped: a missing or failing *required* step (fair value, signals) logs and skips the cycle, an
optional step (consistency, endgame, election night) falls back to a safe default. ``run(once=True)``
performs a single refresh / model cycle / reconcile / leaderboard pass and returns its result (canary,
tests). ``run_forever`` retries a failing startup (503 / 429 / timeouts) with backoff while the heartbeat and
kill-watch run, and after a kill it stops only once the OMS confirmed that no order is left open.

Besides the model cycle the process runs:

* the election-night fast task (``election_tick``, every ``election_night.ref_cadence_s``): while
  ``election_night.is_active`` it reads the newest reference prices from the DB (rows younger than
  ``ELECTION_REF_FRESH_CADENCES`` collector cadences only), refreshes SIG top-of-book for the affected
  markets only (reference beyond the take threshold and the cached book not already showing no lag), runs
  ``election_night.taker_orders`` (whose fair-value confirmation re-checks the reference age) and sends the
  takes through the same OMS / Budget at priority 0, holding every working order so the resting book is
  never cancelled by a fast tick. Outside the window it returns after one cheap check.
* ``refresh_relationships`` (at startup and every ``RELATIONSHIPS_S``): ``Race.exclusive`` /
  ``Race.exhaustive`` from the relationships API only (see the method).
* Signal rows: after each cycle one row per accepted order (at most once per market and strategy per
  ``SIGNAL_DEDUPE_S``) for the reporter's calibration section.

Trader lock: ``run`` holds ``trader.lock`` next to the kill file (``data/trader.lock`` by default; first
line the pid, then ``mode=`` / ``start=`` / ``ts=``) and removes it on exit. A lock whose pid is not alive,
or whose pid now belongs to a process started at another time, is stale. A second trader refuses to start
while another one holds it; the one-shot commands that build their own limiter (`sigcup canary`,
`bootstrap --probe-writes`) refuse while a live trader holds it (``running_trader``).
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import signal
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

from sqlalchemy import select

from sigcup.common.log import get_logger, setup_logging
from sigcup.common.scheduler import every
from sigcup.common.time import ensure_utc, hours_to, utc_now
from sigcup.db import engine as dbengine
from sigcup.db.models import Event, LeaderboardSnap, Market, Race, Signal, Strategy
from sigcup.db.repo import heartbeat, kv_get, kv_set, latest_ref_prices
from sigcup.paper import PaperAdapter
from sigcup.settings import ROOT, Secrets, StrategyConfig, config, secrets
from sigcup.sig.adapter import ExchangeAdapter
from sigcup.sig.ratelimit import Budget
from sigcup.trader import risk
from sigcup.trader.book_state import BookState
from sigcup.trader.oms import OMS, ReconcileReport, SyncReport, desired_key, intent_key, priority_for
from sigcup.trader.types import (
    PRIORITY_ARB,
    PRIORITY_HOUSEKEEPING,
    ActiveEvent,
    DesiredOrder,
    FairValue,
    RaceInfo,
    TraderContext,
)

log = get_logger(__name__)

PROCESS = "trader"
KV_DAY_START = "trader.day_start"  # live; paper keeps its own baseline under KV_DAY_START_PAPER
KV_DAY_START_PAPER = "trader.day_start:paper"
STARTUP_RETRY_S = 5.0
STARTUP_RETRY_MAX_S = 60.0
BULK_CHUNK = 100  # /exchanges/prices accepts <= 100 ids per read
DEPTH_PER_PASS = 4  # depth reads per bulk pass (bounded: reads are shared with everything else)
PAPER_QUOTE_SOURCE = "paper"  # Quote.source of the rows a DB-quote paper trader echoes back
# -- tunables not (yet) in settings.yaml (read through getattr so a settings field can override them) --
ELECTION_REF_FRESH_CADENCES = 3  # election_night.ref_fresh_cadences: older references never take
RELATIONSHIPS_S = 3600  # cadence.relationships_s: relationships API refresh
SIGNAL_DEDUPE_S = 600  # cadence.signal_dedupe_s: one Signal row per (market, strategy) per window
HOLD_EXPIRY_MARGIN_S = 2.0  # a working order this close to expiry is not held by a fast tick (it expires)
TRADER_LOCK_NAME = "trader.lock"
EXCLUSIVE_KINDS = frozenset({"mutually_exclusive"})
NON_EXCLUSIVE_KINDS = frozenset(
    {"not_mutually_exclusive", "non_mutually_exclusive", "non_exclusive", "nonexclusive", "not_exclusive", "independent"}
)
_EPS = 1e-9

_modules: dict[str, ModuleType | None] = {}


def _module(name: str) -> ModuleType | None:
    """Import a model module lazily; a stub that raises at import is treated as missing (once logged)."""
    if name in _modules:
        return _modules[name]
    try:
        mod: ModuleType | None = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001 - stubs raise NotImplementedError at import
        log.warning("model module %s unavailable: %s", name, e)
        mod = None
    _modules[name] = mod
    return mod


def _step(name: str, default: Any, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, bool]:
    """Call a model step; return (result, ok). Failures are logged, never raised."""
    try:
        return fn(*args, **kwargs), True
    except Exception as e:  # noqa: BLE001
        log.exception("model step %s failed: %s", name, e)
        return default, False


def _accepts_priority(fn: Callable[..., Any]) -> bool:
    """True when an adapter method takes the limiter ``priority`` keyword (SigClient reads do)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "priority" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


async def _call_at(priority: int, fn: Callable[..., Any], *args: Any) -> Any:
    """Call an adapter read at ``priority`` when it accepts one (the limiter serves lower numbers first)."""
    if _accepts_priority(fn):
        return await fn(*args, priority=priority)
    return await fn(*args)


def _setting(section: Any, name: str, default: float) -> float:
    try:
        return float(getattr(section, name, default))
    except (TypeError, ValueError):
        return float(default)


def _strategy_tag(value: Any) -> Strategy | str:
    s = str(getattr(value, "value", value))
    try:
        return Strategy(s)
    except ValueError:
        return s  # an unknown tag must keep its exact spelling (it is part of the OMS diff key)


def _norm_kind(kind: str | None) -> str:
    return (kind or "").strip().lower().replace("-", "_").replace(" ", "_")


def _rel_active(rel: Any) -> bool:
    """False only when the API marks the relationship inactive (``active: false``); absent = active."""
    active = getattr(rel, "active", None)
    if active is None:
        return True
    if isinstance(active, str):
        return active.strip().lower() not in ("false", "0", "no", "inactive", "")
    return bool(active)


@dataclass
class CycleResult:
    """What one model cycle did (returned by ``run_once`` and kept as ``Trader.last_cycle``)."""

    ts: datetime
    skipped: str | None = None
    fair: int = 0
    arbs: int = 0
    desired: int = 0
    accepted: int = 0
    sync: SyncReport | None = None
    reconcile: ReconcileReport | None = None
    killed: str | None = None
    signals: int = 0  # Signal rows written for this cycle (after the per-market dedupe window)
    context: TraderContext | None = field(default=None, repr=False)


@dataclass
class ElectionTick:
    """What one fast election-night tick did (``Trader.last_election_tick``)."""

    ts: datetime
    active: bool = False
    skipped: str | None = None
    refs: dict[str, float] = field(default_factory=dict)  # affected markets -> fresh reference
    refreshed: list[str] = field(default_factory=list)  # markets whose SIG top-of-book was re-read
    takes: list[DesiredOrder] = field(default_factory=list)
    accepted: list[DesiredOrder] = field(default_factory=list)
    sync: SyncReport | None = None


@dataclass
class RelationshipsReport:
    """What one relationships refresh changed in the Race table."""

    ts: datetime
    relationships: int = 0
    set_true: list[str] = field(default_factory=list)  # races whose exclusive and/or exhaustive became True
    set_false: list[str] = field(default_factory=list)  # races the API positively lists as non-exclusive / non-exhaustive
    kept_true: list[str] = field(default_factory=list)  # True flags kept without API evidence this run
    error: str | None = None


# -- trader lock -----------------------------------------------------------------------------------
class TraderLockHeld(RuntimeError):
    """Another running trader holds the lock."""

    def __init__(self, holder: TraderLock) -> None:
        self.holder = holder
        super().__init__(
            f"another trader is running (pid {holder.pid}, mode {holder.mode or 'unknown'}) and holds {holder.path}: "
            "stop it first (`pm2 stop trader`); if no trader is running, delete the file"
        )


def _proc_start(pid: int) -> str | None:
    """The process start time (clock ticks since boot, /proc/<pid>/stat field 22), None where unavailable."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    rest = raw.rsplit(")", 1)[-1].split()  # fields after the (comm): rest[0] is field 3
    return rest[19] if len(rest) > 19 else None


def pid_alive(pid: int, start: str | None = None) -> bool:
    """True when ``pid`` is a running process (and, when ``start`` is given and /proc knows the pid, the
    same process that wrote the lock: a reused pid is not alive for this purpose)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass  # exists, owned by another user
    except OSError:
        return False
    if start:
        current = _proc_start(pid)
        if current is not None and current != start:
            return False
    return True


@dataclass(frozen=True)
class TraderLock:
    path: Path
    pid: int
    mode: str | None = None  # live | paper (None: unknown, treated as live)
    start: str | None = None
    ts: str | None = None

    @property
    def live(self) -> bool:
        return self.mode != "paper"

    def alive(self) -> bool:
        return pid_alive(self.pid, self.start)


def trader_lock_path(sec: Secrets) -> Path:
    """``Secrets.trader_lock_file`` when such a setting exists, else ``trader.lock`` next to the kill file
    (both are trader control files: ``data/trader.lock`` by default)."""
    explicit = getattr(sec, "trader_lock_file", None)
    if explicit:
        p = Path(str(explicit))
        return p if p.is_absolute() else ROOT / p
    return sec.kill_path.parent / TRADER_LOCK_NAME


def read_trader_lock(path: Path | str) -> TraderLock | None:
    """The parsed lock file (alive or not), None when absent or unreadable."""
    p = Path(path)
    try:
        lines = p.read_text().splitlines()
    except OSError:
        return None
    if not lines:
        return None
    try:
        pid = int(lines[0].strip())
    except ValueError:
        return None
    kv: dict[str, str] = {}
    for line in lines[1:]:
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    return TraderLock(path=p, pid=pid, mode=kv.get("mode") or None, start=kv.get("start") or None, ts=kv.get("ts") or None)


def running_trader(target: Secrets | Path | str) -> TraderLock | None:
    """The lock of a running trader (``target`` is the Secrets or the lock path), else None (absent / stale)."""
    path = trader_lock_path(target) if isinstance(target, Secrets) else Path(target)
    held = read_trader_lock(path)
    if held is None or not held.alive():
        return None
    return held


def acquire_trader_lock(path: Path | str, *, mode: str) -> TraderLock:
    """Create the lock for this process (``O_EXCL``); a stale lock is replaced, a running other trader's
    lock raises ``TraderLockHeld``. Re-acquiring a lock this process already holds is allowed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    me = os.getpid()
    lock = TraderLock(path=p, pid=me, mode=mode, start=_proc_start(me), ts=utc_now().isoformat())
    text = f"{me}\nmode={mode}\n" + (f"start={lock.start}\n" if lock.start else "") + f"ts={lock.ts}\n"
    for _ in range(3):
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            held = read_trader_lock(p)
            if held is not None and held.pid != me and held.alive():
                raise TraderLockHeld(held) from None
            if held is not None and held.pid != me:
                log.warning("trader lock %s is stale (pid %d is not running): replacing it", p, held.pid)
            elif held is None:
                log.warning("trader lock %s is unreadable: replacing it", p)
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w") as f:
            f.write(text)
        log.info("trader lock %s acquired (pid %d, %s)", p, me, mode)
        return lock
    raise RuntimeError(f"could not create the trader lock {p} (another process keeps recreating it)")


def release_trader_lock(path: Path | str) -> bool:
    """Remove the lock if this process holds it. Never removes another process's lock."""
    p = Path(path)
    held = read_trader_lock(p)
    if held is None or held.pid != os.getpid():
        if held is not None:
            log.warning("trader lock %s belongs to pid %d: left in place", p, held.pid)
        return False
    try:
        p.unlink()
    except FileNotFoundError:
        return False
    return True


def build_live_adapter(sec: Secrets, cfg: StrategyConfig, budget: Budget, *, read_only: bool = False) -> ExchangeAdapter:
    """Construct the live SigClient tolerant of its constructor signature (written concurrently)."""
    mod = _module("sigcup.sig.client")
    if mod is None or not hasattr(mod, "SigClient"):
        raise RuntimeError("sigcup.sig.client.SigClient is not available")
    cls = mod.SigClient
    key = (sec.sig_read_key or sec.sig_api_key) if read_only else sec.sig_api_key
    attempts: list[dict[str, Any]] = [
        {"api_key": key, "cfg": cfg, "budget": budget, "base_url": sec.sig_base_url, "tournament_slug": sec.sig_tournament_slug},
        {"api_key": key, "cfg": cfg, "budget": budget},
        {"api_key": key, "budget": budget},
        {"api_key": key},
    ]
    last: Exception | None = None
    for kwargs in attempts:
        try:
            return cls(**kwargs)
        except TypeError as e:
            last = e
    raise RuntimeError(f"cannot construct SigClient: {last}")


class Trader:
    """All periodic tasks of the trader process share this object."""

    def __init__(
        self,
        cfg: StrategyConfig,
        sec: Secrets,
        adapter: ExchangeAdapter,
        budget: Budget,
        book: BookState,
        oms: OMS,
        *,
        live: bool,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.cfg = cfg
        self.sec = sec
        self.adapter = adapter
        self.budget = budget
        self.book = book
        self.oms = oms
        self.live = live
        self.clock = clock or utc_now
        self.stop = asyncio.Event()
        self.my_rank: int | None = None
        self.last_cycle: CycleResult | None = None
        self.last_leaderboard: LeaderboardSnap | None = None
        self._depth_cursor = 0
        self._realtime_task: asyncio.Task | None = None
        self._realtime_feed: Any = None
        self._resync_requested = False
        self.startup_retry_s = STARTUP_RETRY_S
        self._stop_when_flat = False  # this process fired the kill: stop once it is confirmed
        self.last_election_tick: ElectionTick | None = None
        self.last_relationships: RelationshipsReport | None = None

    # -- books ---------------------------------------------------------------------------------
    async def refresh_books(self) -> list[str]:
        """Bulk prices for every known market (<= 100 ids per read) plus a few stale depth reads."""
        ids = self.book.exchange_ids()
        if not ids:
            markets = await self.adapter.list_markets()
            ids = [m.exchange_id or m.id for m in markets]
        changed: list[str] = []
        for i in range(0, len(ids), BULK_CHUNK):
            chunk = ids[i : i + BULK_CHUNK]
            snaps = await self.adapter.get_prices(chunk)
            changed.extend(self.book.update_prices(snaps))
        stale = self.book.stale_books(self.cfg.cadence.depth_refresh_s, self.clock())
        if stale:
            start = self._depth_cursor % len(stale)
            picks = (stale[start:] + stale[:start])[:DEPTH_PER_PASS]
            self._depth_cursor += len(picks)
            for eid in picks:
                try:
                    self.book.update_book(await self.adapter.get_orderbook(eid))
                except Exception as e:  # noqa: BLE001
                    log.warning("depth %s failed: %s", eid, e)
        self.book.flush_quotes()
        self._resync_requested = False
        return changed

    async def consume_realtime(self, events: AsyncIterator[Any]) -> None:
        """Apply realtime events (duck-typed) to the book; fills/resyncs just flag a REST refresh."""
        async for ev in events:
            if self.stop.is_set():
                break
            try:
                if hasattr(ev, "bids") or hasattr(ev, "asks"):
                    self.book.update_realtime(ev)
                elif hasattr(ev, "fill"):
                    self._resync_requested = True
                else:
                    self._resync_requested = True
            except Exception:  # noqa: BLE001
                log.exception("realtime event failed")
        self.book.flush_quotes()

    def start_realtime(self) -> bool:
        """Start the Supabase Realtime feed (``sigcup.sig.realtime.RealtimeFeed``) for the hottest markets
        when the adapter can mint a realtime token (live client, or paper wrapping one). REST polling
        continues regardless; the feed only makes the book fresher between bulk reads."""
        if not getattr(self.adapter, "has_realtime", hasattr(self.adapter, "realtime_token")):
            return False
        mod = _module("sigcup.sig.realtime")
        if mod is None or not hasattr(mod, "RealtimeFeed"):
            return False
        tid = getattr(self.adapter, "tournament_id", None)
        if not tid:
            log.warning("realtime: tournament id unknown; REST only")
            return False
        ids = self.book.hot_markets(self.cfg.limits.realtime_channels)
        # RealtimeFeed takes MARKET id -> exchange id (topics are per market id); unmapped books have no topic
        markets = {self.book.meta[eid].market_id: eid for eid in ids if eid in self.book.meta and self.book.meta[eid].market_id}
        try:
            feed = mod.RealtimeFeed(self.adapter.realtime_token, str(tid), markets=markets, max_channels=self.cfg.limits.realtime_channels)
            self._realtime_feed = feed
            self._realtime_task = asyncio.create_task(self.consume_realtime(feed.events()), name="realtime")
            log.info("realtime: subscribing %d markets", len(markets))
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("realtime feed unavailable (%s); REST only", e)
            return False

    # -- context -------------------------------------------------------------------------------
    def _ref_quotes(self, s: Any, since: datetime | None = None) -> dict[str, tuple[float, datetime | None]]:
        """Best real-money YES mid per exchange id via Market.kalshi_ticker / polymarket_condition_id (the
        more liquid source wins) with the row's timestamp; ``since`` drops rows older than it."""
        refs = latest_ref_prices(s, since=since)
        out: dict[str, tuple[float, datetime | None]] = {}
        for eid, meta in self.book.meta.items():
            best: tuple[float, float, datetime | None] | None = None
            for source, key in (("kalshi", meta.kalshi_ticker), ("polymarket", meta.polymarket_condition_id)):
                if not key:
                    continue
                r = refs.get((source, key))
                if r is None:
                    continue
                ts = ensure_utc(r.ts) if r.ts is not None else None
                if since is not None and (ts is None or ts < since):
                    continue
                mid = r.yes_mid
                if mid is None and r.yes_bid is not None and r.yes_ask is not None:
                    mid = (r.yes_bid + r.yes_ask) / 2
                if mid is None:
                    continue
                liq = float(r.liquidity or 0.0)
                if best is None or liq > best[0]:
                    best = (liq, float(mid), ts)
            if best is not None:
                out[eid] = (best[1], best[2])
        return out

    def _ref_prices(self, s: Any) -> dict[str, float]:
        """Best real-money YES mid per exchange id (any age: takes re-check freshness via the fair value)."""
        return {eid: mid for eid, (mid, _ts) in self._ref_quotes(s).items()}

    def _sync_book_from_oms(self) -> None:
        self.book.set_open_intents(self.oms.acked())
        self.book.set_positions(list(self.oms.positions.values()))
        self.book.set_account(cash=self.oms.cash, reserved=self.oms.reserved)
        self.book.positions_value = self.oms.positions_value

    def _events(self, s: Any, now: datetime) -> list[ActiveEvent]:
        rows = s.execute(select(Event).where(Event.ts >= now - timedelta(hours=24))).scalars().all()
        return [
            ActiveEvent(race_id=e.race_id, kind=e.kind, ts=ensure_utc(e.ts), description=e.description, confidence=e.confidence)
            for e in rows
        ]

    def _daily_pnl(self, s: Any, now: datetime, equity: float) -> float:
        day = now.strftime("%Y-%m-%d")
        key = KV_DAY_START if self.live else KV_DAY_START_PAPER  # a paper book never sets the live baseline
        start = kv_get(s, key) or {}
        if not isinstance(start, dict) or start.get("day") != day:
            start = {"day": day, "equity": equity}
            kv_set(s, key, start)
        return equity - float(start.get("equity", equity))

    def build_context(self) -> TraderContext:
        """Assemble the TraderContext for one model cycle."""
        cfg, now = self.cfg, self.clock()
        self._sync_book_from_oms()
        views = self.book.views()
        fv = _module("sigcup.trader.fairvalue")
        with dbengine.session() as s:
            races: dict[str, RaceInfo] = self.book.races()
            if fv is not None and hasattr(fv, "load_races"):
                loaded, ok = _step("fairvalue.load_races", None, fv.load_races, s)
                if ok and loaded:
                    races = loaded
            fair: dict[str, FairValue] = {}
            if fv is not None and hasattr(fv, "compute_all"):
                fair, _ = _step("fairvalue.compute_all", {}, fv.compute_all, s, cfg, views, now)
            ref_prices = self._ref_prices(s)
            events = self._events(s, now)
            equity = self.oms.equity
            daily_pnl = self._daily_pnl(s, now, equity)
        eg = _module("sigcup.trader.endgame")
        sleeve = cfg.convex_sleeve.budget_frac
        if eg is not None and hasattr(eg, "sleeve_fraction"):
            sleeve, _ = _step("endgame.sleeve_fraction", sleeve, eg.sleeve_fraction, cfg, now, self.last_leaderboard, equity)
        en = _module("sigcup.trader.election_night")
        active = False
        if en is not None and hasattr(en, "is_active"):
            active, _ = _step("election_night.is_active", False, en.is_active, cfg, now)
        return TraderContext(
            cfg=cfg,
            now=now,
            markets=views,
            races=races,
            fair=fair or {},
            cash=self.oms.cash,
            reserved=self.oms.reserved,
            equity=equity,
            bankroll_start=cfg.bankroll_start,
            events=events,
            sleeve_frac=float(sleeve),
            election_night=bool(active),
            hours_to_deadline=hours_to(cfg.deadline_utc, now),
            ref_prices=ref_prices,
            daily_pnl=daily_pnl,
            my_rank=self.my_rank,
        )

    # -- model cycle ---------------------------------------------------------------------------
    async def model_cycle(self) -> CycleResult:
        res = CycleResult(ts=self.clock())
        self.last_cycle = res
        if self.oms.killed is not None:
            res.skipped = res.killed = self.oms.killed
            return res
        if self._resync_requested:
            await self.refresh_books()
        ctx = self.build_context()
        res.context = ctx
        res.fair = len(ctx.fair)
        reason = risk.kill_switch_triggered(ctx, self.sec.kill_path)
        if reason:
            await self.oms.kill(reason)
            res.skipped = res.killed = reason
            self._stop_after_kill()
            return res
        cons = _module("sigcup.trader.consistency")
        arbs: list = []
        if cons is not None and hasattr(cons, "scan"):
            arbs, _ = _step("consistency.scan", [], cons.scan, ctx)
        res.arbs = len(arbs or [])
        sig = _module("sigcup.trader.signals")
        if sig is None or not hasattr(sig, "generate"):
            res.skipped = "signals module unavailable"
            log.warning("model cycle skipped: %s", res.skipped)
            return res
        desired, ok = _step("signals.generate", None, sig.generate, ctx, arbs or [])
        if not ok or desired is None:
            res.skipped = "signals.generate failed"
            return res
        desired = list(desired)
        if ctx.election_night:
            en = _module("sigcup.trader.election_night")
            if en is not None and hasattr(en, "taker_orders"):
                takes, _ = _step("election_night.taker_orders", [], en.taker_orders, ctx, ctx.ref_prices)
                desired.extend(takes or [])
        res.desired = len(desired)
        accepted: list[DesiredOrder] = risk.apply(ctx, desired)
        res.accepted = len(accepted)
        res.sync = await self.oms.sync(accepted)
        if res.sync.killed:
            res.killed = res.sync.killed
            self._stop_after_kill()
        try:
            res.signals = self.record_signals(ctx, accepted)
        except Exception as e:  # noqa: BLE001 - bookkeeping never breaks trading
            log.exception("signal rows not written: %s", e)
        log.info(
            "cycle: fair=%d arbs=%d desired=%d accepted=%d placed=%d cancelled=%d kept=%d deferred=%d skipped=%d%s",
            res.fair, res.arbs, res.desired, res.accepted, len(res.sync.placed), len(res.sync.cancelled),
            res.sync.kept, res.sync.deferred, len(res.sync.skipped), f" blocked={res.sync.blocked}" if res.sync.blocked else "",
        )
        return res

    # -- signals (calibration record) ----------------------------------------------------------
    def record_signals(self, ctx: TraderContext, orders: list[DesiredOrder]) -> int:
        """Persist one Signal row per order (those that survived risk), in one transaction, skipping any
        (market, strategy) that already has a row within ``SIGNAL_DEDUPE_S``. ``fair`` is P(YES) and
        ``price`` is in the same YES units (a NO at p is a YES at 1 - p; the order's own price is in
        ``meta.order_price``); ``edge`` is the order's own edge. Returns the rows written."""
        if not orders:
            return 0
        now = ensure_utc(ctx.now)
        window = timedelta(seconds=_setting(self.cfg.cadence, "signal_dedupe_s", SIGNAL_DEDUPE_S))
        with dbengine.session() as s:
            recent = {
                (str(eid), str(strat))
                for eid, strat in s.execute(select(Signal.exchange_id, Signal.strategy).where(Signal.ts > now - window)).all()
            }
            rows: list[Signal] = []
            for d in orders:
                strat = str(getattr(d.strategy, "value", d.strategy))[:16]
                key = (d.exchange_id, strat)
                if key in recent:
                    continue
                recent.add(key)
                rows.append(self._signal_row(ctx, d, strat, now))
            s.add_all(rows)
        return len(rows)

    def _signal_row(self, ctx: TraderContext, d: DesiredOrder, strategy: str, now: datetime) -> Signal:
        fv = ctx.fair.get(d.exchange_id)
        view = ctx.markets.get(d.exchange_id)
        fair = d.fair if d.fair is not None else (fv.fair if fv is not None else None)
        price_yes = float(d.price) if d.side == "yes" else 1.0 - float(d.price)
        edge = d.edge
        if edge is None and fair is not None:
            edge = (float(fair) - price_yes) if d.is_long_yes else (price_yes - float(fair))

        def nums(m: Any) -> dict[str, float]:
            out: dict[str, float] = {}
            for k, v in (m or {}).items():
                try:
                    out[str(k)] = float(v)
                except (TypeError, ValueError):
                    continue
            return out

        meta: dict[str, Any] = {
            "side": d.side,
            "action": d.action,
            "order_price": float(d.price),
            "priority": int(d.priority),
            "take": bool(d.take),
            "group_id": d.group_id,
            "reason": (d.reason or "")[:200],
            "ref": ctx.ref_prices.get(d.exchange_id),
        }
        if fv is not None:
            meta.update(
                fair_yes=float(fv.fair), sd=float(fv.sd), components=nums(fv.components), weights=nums(fv.weights),
                fair_ts=ensure_utc(fv.ts).isoformat() if fv.ts is not None else None,
                ref_liquidity=float(fv.ref_liquidity) if fv.ref_liquidity is not None else None,
            )
        if view is not None:
            meta.update(bid=view.bid, ask=view.ask, mid=view.mid)
        return Signal(
            ts=now, exchange_id=d.exchange_id, strategy=strategy, fair=float(fair) if fair is not None else None,
            price=round(price_yes, 6), edge=float(edge) if edge is not None else None, size=int(d.quantity), meta=meta,
        )

    # -- election night: the fast taker task ---------------------------------------------------
    def _hold_working(self, wanted: list[DesiredOrder], now: datetime) -> list[DesiredOrder]:
        """Desired orders that re-state every working order at its own price, so an ``OMS.sync`` of
        ``wanted + hold`` only adds ``wanted`` (the sync otherwise cancels whatever is not desired). Orders
        about to expire are left out: the sync expires them locally instead of re-placing them."""
        keys = {desired_key(d) for d in wanted}
        out: list[DesiredOrder] = []
        for i in self.oms.working():
            k = intent_key(i)
            if k in keys:
                continue
            if i.expires_at is not None and (ensure_utc(i.expires_at) - now).total_seconds() <= HOLD_EXPIRY_MARGIN_S:
                continue
            keys.add(k)
            out.append(
                DesiredOrder(
                    exchange_id=i.exchange_id, side=i.side, action=i.action, price=float(i.price),  # type: ignore[arg-type]
                    quantity=max(1, int(i.quantity) - int(i.filled_qty or 0)), strategy=_strategy_tag(i.strategy),  # type: ignore[arg-type]
                    priority=priority_for(i.strategy), reason="held by the election fast tick",
                )
            )
        return out

    def _affected(self, refs: dict[str, tuple[float, datetime | None]]) -> list[str]:
        """Markets whose fresh reference is beyond the take threshold and whose cached SIG top does not
        already rule a lag out (unknown top -> affected). Largest cached lag first, at most one bulk read."""
        en = self.cfg.election_night
        thr, lag = float(en.take_when_ref_ge), float(en.min_lag)
        scored: list[tuple[float, str]] = []
        for eid, (ref, _ts) in refs.items():
            top = self.book.top(eid)
            if ref >= thr - _EPS:
                if top is not None and top.ask is not None and float(top.ask) > ref - lag + _EPS:
                    continue
                gap = (ref - float(top.ask)) if top is not None and top.ask is not None else 0.0
            elif ref <= 1.0 - thr + _EPS:
                if top is not None and top.bid is not None and float(top.bid) < ref + lag - _EPS:
                    continue
                gap = (float(top.bid) - ref) if top is not None and top.bid is not None else 0.0
            else:
                continue
            scored.append((gap, eid))
        scored.sort(key=lambda t: (-t[0], t[1]))
        return [eid for _, eid in scored[:BULK_CHUNK]]

    def _fast_context(self, now: datetime, refs: dict[str, tuple[float, datetime | None]]) -> TraderContext:
        """A lean context for the fast tick: live book views, the last cycle's races / fair values, and a
        fair value per affected market carrying the fresh reference and ITS timestamp, so
        ``election_night.confirmed_ref`` re-checks the reference age (``fairvalue.ref_max_age``)."""
        cfg = self.cfg
        self._sync_book_from_oms()
        views = self.book.views()
        last = self.last_cycle.context if self.last_cycle is not None else None
        races = dict(last.races) if last is not None and last.races else self.book.races()
        fair = dict(last.fair) if last is not None else {}
        for eid, (ref, ts) in refs.items():
            prev = fair.get(eid)
            fair[eid] = FairValue(
                exchange_id=eid, fair=ref, sd=prev.sd if prev is not None else 0.0,
                components={**(prev.components if prev is not None else {}), "ref": ref},
                weights=dict(prev.weights) if prev is not None else {}, ts=ts,
                ref_liquidity=prev.ref_liquidity if prev is not None else None,
            )
        equity = self.oms.equity
        with dbengine.session() as s:
            daily_pnl = self._daily_pnl(s, now, equity)
        return TraderContext(
            cfg=cfg,
            now=now,
            markets=views,
            races=races,
            fair=fair,
            cash=self.oms.cash,
            reserved=self.oms.reserved,
            equity=equity,
            bankroll_start=cfg.bankroll_start,
            events=list(last.events) if last is not None else [],
            sleeve_frac=last.sleeve_frac if last is not None else cfg.convex_sleeve.budget_frac,
            election_night=True,
            hours_to_deadline=hours_to(cfg.deadline_utc, now),
            ref_prices={**(last.ref_prices if last is not None else {}), **{eid: r for eid, (r, _ts) in refs.items()}},
            daily_pnl=daily_pnl,
            my_rank=self.my_rank,
        )

    async def election_tick(self) -> ElectionTick:
        """One fast election-night pass (see the module docstring). Outside the window: one cheap check."""
        now = self.clock()
        res = ElectionTick(ts=now)
        self.last_election_tick = res
        en = _module("sigcup.trader.election_night")
        if en is None or not hasattr(en, "is_active") or not hasattr(en, "taker_orders"):
            res.skipped = "election_night module unavailable"
            return res
        active, _ = _step("election_night.is_active", False, en.is_active, self.cfg, now)
        res.active = bool(active)
        if not res.active:
            return res
        if self.oms.killed is not None:
            res.skipped = f"killed: {self.oms.killed}"
            return res
        cadence = max(1.0, float(self.cfg.election_night.ref_cadence_s))
        fresh_cadences = _setting(self.cfg.election_night, "ref_fresh_cadences", ELECTION_REF_FRESH_CADENCES)
        since = now - timedelta(seconds=fresh_cadences * cadence)
        with dbengine.session() as s:
            fresh = self._ref_quotes(s, since=since)
        affected = self._affected(fresh)
        if not affected:
            return res
        res.refs = {eid: fresh[eid][0] for eid in affected}
        snaps = await _call_at(PRIORITY_ARB, self.adapter.get_prices, affected)  # one read, ahead of bulk polling
        self.book.update_prices(snaps)
        res.refreshed = list(affected)
        ctx = self._fast_context(now, {eid: fresh[eid] for eid in affected})
        reason = risk.kill_switch_triggered(ctx, self.sec.kill_path)
        if reason:
            await self.oms.kill(reason)
            res.skipped = reason
            self._stop_after_kill()
            return res
        takes, _ = _step("election_night.taker_orders", [], en.taker_orders, ctx, res.refs)
        res.takes = [replace(d, priority=PRIORITY_ARB, take=True) for d in (takes or [])]
        if not res.takes:
            return res
        res.accepted = risk.apply(ctx, res.takes)
        if not res.accepted:
            return res
        res.sync = await self.oms.sync(res.accepted + self._hold_working(res.accepted, self.clock()))
        if res.sync.killed:
            self._stop_after_kill()
        try:
            self.record_signals(ctx, res.accepted)
        except Exception as e:  # noqa: BLE001
            log.exception("election signal rows not written: %s", e)
        log.info(
            "election tick: refs=%d refreshed=%d takes=%d accepted=%d placed=%d deferred=%d%s",
            len(res.refs), len(res.refreshed), len(res.takes), len(res.accepted), len(res.sync.placed),
            res.sync.deferred, f" blocked={res.sync.blocked}" if res.sync.blocked else "",
        )
        return res

    # -- relationships -------------------------------------------------------------------------
    async def refresh_relationships(self) -> RelationshipsReport:
        """``Race.exclusive`` / ``Race.exhaustive`` from the relationships API (read at housekeeping
        priority through the shared adapter / budget). A ``mutually_exclusive`` group covering exactly a
        race's legs sets ``exclusive`` True, and ``exhaustive`` True when ``isExhaustive`` is true. A flag is
        set False only when the API positively says so for exactly the race's legs (a non-exclusive kind, or
        ``isExhaustive: false``); no evidence (an empty graph, partial groups, a failed read) changes nothing,
        and titles are never consulted."""
        rep = RelationshipsReport(ts=self.clock())
        self.last_relationships = rep
        try:
            rels = list(await _call_at(PRIORITY_HOUSEKEEPING, self.adapter.get_relationships))
        except Exception as e:  # noqa: BLE001
            rep.error = f"{type(e).__name__}: {e}"
            log.warning("relationships refresh failed (flags unchanged): %s", rep.error)
            return rep
        rep.relationships = len(rels)
        with dbengine.session() as s:
            legs: dict[str, set[str]] = {}
            for m in s.execute(select(Market)).scalars().all():
                if m.exchange_id:
                    legs.setdefault(m.race_id, set()).add(str(m.exchange_id))
            race_of_legs = {frozenset(v): rid for rid, v in legs.items() if v}
            races = {r.race_id: r for r in s.execute(select(Race)).scalars().all()}
            evidence: set[str] = set()
            for rel in rels:
                if not _rel_active(rel):
                    continue
                ids = frozenset(str(x) for x in (rel.exchange_ids or []))
                rid = race_of_legs.get(ids)
                race = races.get(rid) if rid is not None else None
                if race is None:
                    continue
                kind = _norm_kind(rel.kind)
                if kind in EXCLUSIVE_KINDS:
                    evidence.add(rid)
                    changed = race.exclusive is not True
                    race.exclusive = True
                    if rel.is_exhaustive is True and race.exhaustive is not True:
                        race.exhaustive = True
                        changed = True
                    elif rel.is_exhaustive is False and race.exhaustive is not False:
                        log.warning("relationships: %s group %s is exclusive but NOT exhaustive: exhaustive %s -> False", rid, rel.id, race.exhaustive)
                        race.exhaustive = False
                        rep.set_false.append(rid)
                    if changed:
                        log.info("relationships: %s exclusive=%s exhaustive=%s (group %s)", rid, race.exclusive, race.exhaustive, rel.id)
                        rep.set_true.append(rid)
                elif kind in NON_EXCLUSIVE_KINDS:
                    evidence.add(rid)
                    if race.exclusive is not False:
                        log.warning("relationships: %s listed as %s by group %s: exclusive %s -> False", rid, kind, rel.id, race.exclusive)
                        race.exclusive = False
                        rep.set_false.append(rid)
            rep.kept_true = sorted(
                rid for rid, r in races.items() if rid not in evidence and (r.exclusive is True or r.exhaustive is True)
            )
        log.info(
            "relationships: %d groups; set True %s; set False %s; %d races keep True flags without API evidence this run",
            rep.relationships, sorted(set(rep.set_true)) or "-", sorted(set(rep.set_false)) or "-", len(rep.kept_true),
        )
        if rep.set_true or rep.set_false:
            try:
                self.book.load_meta()
            except Exception as e:  # noqa: BLE001
                log.warning("race metadata reload failed: %s", e)
        return rep

    # -- housekeeping --------------------------------------------------------------------------
    async def reconcile(self) -> ReconcileReport:
        rep = await self.oms.reconcile()
        self.book.set_positions(list(self.oms.positions.values()))
        self.book.set_account(cash=self.oms.cash, reserved=self.oms.reserved)
        return rep

    async def leaderboard(self) -> LeaderboardSnap | None:
        try:
            lb = await self.adapter.get_leaderboard()
        except Exception as e:  # noqa: BLE001
            log.warning("leaderboard failed: %s", e)
            return None
        my_rank = lb.my_rank
        my_total = lb.my_total
        for e in lb.entries:
            if e.is_me:
                my_rank, my_total = e.rank, e.total
        snap = LeaderboardSnap(
            ts=self.clock(),
            rank=my_rank,
            our_total=my_total,
            top_totals=[float(e.total) for e in sorted(lb.entries, key=lambda e: e.rank)[:10]],
            member_count=lb.member_count,
        )
        with dbengine.session() as s:
            s.add(snap)
        self.my_rank = my_rank
        self.last_leaderboard = snap
        return snap

    async def heartbeat(self) -> None:
        status = "killed" if self.oms.killed else ("live" if self.live else "paper")
        detail = f"cash={self.oms.cash:.0f} reserved={self.oms.reserved:.0f} working={len(self.oms.working())}"
        with dbengine.session() as s:
            heartbeat(s, PROCESS, status, detail)

    def _stop_after_kill(self) -> None:
        """Stop once the kill is confirmed (no open orders); until then keep running so watch_kill retries."""
        self._stop_when_flat = True
        if self.oms.kill_confirmed:
            self.stop.set()
        else:
            log.critical("kill not confirmed yet: staying up to retry the cancel")

    async def watch_kill(self) -> None:
        if self.oms.killed is None and self.sec.kill_path.exists():
            await self.oms.kill(f"kill file present: {self.sec.kill_path}")
            self._stop_when_flat = True
        elif self.oms.killed is not None and not self.oms.kill_confirmed:
            await self.oms.confirm_kill()
        if self._stop_when_flat and self.oms.kill_confirmed:
            self.stop.set()

    # -- entry points --------------------------------------------------------------------------
    async def run_once(self) -> CycleResult:
        """Startup reconcile + one refresh / cycle / reconcile / leaderboard / heartbeat pass."""
        await self.oms.startup_reconcile()
        await self.refresh_books()
        res = await self.model_cycle()
        res.reconcile = await self.reconcile()
        await self.leaderboard()
        await self.heartbeat()
        return res

    async def _startup(self) -> bool:
        """Startup reconcile + first book refresh, retried with backoff: a 503 / 429 / timeout at startup must
        not crash the process (pm2 gives up after a few fast restarts). False if stopped meanwhile."""
        delay = self.startup_retry_s
        reconciled = False
        while not self.stop.is_set():
            try:
                if not reconciled:
                    await self.oms.startup_reconcile()
                    reconciled = True
                await self.refresh_books()
                return True
            except Exception as e:  # noqa: BLE001
                log.warning("startup failed (%s: %s); retrying in %.0f s", type(e).__name__, e, delay)
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=delay)
            except TimeoutError:
                pass
            delay = min(delay * 2, STARTUP_RETRY_MAX_S)
        return False

    async def run_forever(self) -> None:
        loop = asyncio.get_running_loop()
        for sig_ in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig_, self.stop.set)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        c = self.cfg.cadence
        tasks = [
            asyncio.create_task(every(15, self.heartbeat, name="heartbeat", jitter=0.0, stop=self.stop)),
            asyncio.create_task(every(5, self.watch_kill, name="kill-watch", jitter=0.0, stop=self.stop)),
        ]
        try:
            if await self._startup():
                self.start_realtime()
                tasks += [
                    asyncio.create_task(every(c.bulk_prices_s, self.refresh_books, name="books", run_immediately=False, stop=self.stop)),
                    asyncio.create_task(every(c.model_s, self.model_cycle, name="model", stop=self.stop)),
                    asyncio.create_task(every(c.reconcile_s, self.reconcile, name="reconcile", stop=self.stop)),
                    asyncio.create_task(every(c.leaderboard_s, self.leaderboard, name="leaderboard", stop=self.stop)),
                    # election night: takes on the reference cadence (idles outside the window)
                    asyncio.create_task(
                        every(self.cfg.election_night.ref_cadence_s, self.election_tick, name="election", jitter=0.0, stop=self.stop)
                    ),
                    asyncio.create_task(
                        every(_setting(c, "relationships_s", RELATIONSHIPS_S), self.refresh_relationships, name="relationships", stop=self.stop)
                    ),
                ]
                await self.stop.wait()
        finally:
            log.info("trader stopping")
            for t in tasks:
                t.cancel()
            if self._realtime_task is not None:
                stop_feed = getattr(self._realtime_feed, "stop", None)
                if callable(stop_feed):
                    try:
                        res = stop_feed()
                        if asyncio.iscoroutine(res):
                            await res
                    except Exception:  # noqa: BLE001
                        log.exception("realtime stop failed")
                self._realtime_task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.live:
                try:
                    if self.oms.killed is None:
                        await self.oms.cancel_all()
                    elif not self.oms.kill_confirmed:
                        await self.oms.confirm_kill()
                except Exception:  # noqa: BLE001
                    log.exception("cancel_all on shutdown failed")
            try:
                await self.heartbeat()
            except Exception:  # noqa: BLE001
                log.exception("final heartbeat failed")


def build_trader(
    *,
    paper: bool,
    cfg: StrategyConfig | None = None,
    sec: Secrets | None = None,
    adapter: ExchangeAdapter | None = None,
    budget: Budget | None = None,
    clock: Callable[[], datetime] | None = None,
) -> Trader:
    """Wire the single Budget, the adapter, BookState and OMS. LIVE without a trade key is refused."""
    cfg = cfg or config()
    sec = sec or secrets()
    budget = budget or Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min)
    live = bool(sec.live) and not paper
    if adapter is None:
        if live:
            if not sec.sig_api_key:
                raise RuntimeError("LIVE=1 but SIG_API_KEY is empty: refusing to start the live trader")
            adapter = build_live_adapter(sec, cfg, budget)
        else:
            source: ExchangeAdapter | None = None
            if sec.sig_read_key or sec.sig_api_key:
                try:
                    source = build_live_adapter(sec, cfg, budget, read_only=True)
                except Exception as e:  # noqa: BLE001
                    log.warning("paper: no live market-data source (%s); using DB quotes", e)
            paper_adapter = PaperAdapter(cfg, source=source, budget=budget, clock=clock)
            try:
                paper_adapter.resume_from_ledger()  # a restart continues the same paper book
            except Exception as e:  # noqa: BLE001
                log.warning("paper: cannot resume from the fills ledger (%s); starting a fresh book", e)
            adapter = paper_adapter
    elif live and not sec.sig_api_key:
        raise RuntimeError("LIVE=1 but SIG_API_KEY is empty: refusing to start the live trader")
    book = BookState(cfg)
    if getattr(adapter, "use_db_quotes", False) is True:
        book.persist_source = PAPER_QUOTE_SOURCE  # the paper adapter serves the quotes table back: label the echoes
    try:
        book.load_meta()
    except Exception as e:  # noqa: BLE001
        log.warning("race/market metadata unavailable (%s); views will be unmapped", e)
    oms = OMS(adapter, cfg, budget, paper=not live, kill_path=sec.kill_path, clock=clock)
    return Trader(cfg, sec, adapter, budget, book, oms, live=live, clock=clock)


async def run(
    paper: bool = False,
    once: bool = False,
    *,
    cfg: StrategyConfig | None = None,
    sec: Secrets | None = None,
    adapter: ExchangeAdapter | None = None,
    budget: Budget | None = None,
    clock: Callable[[], datetime] | None = None,
) -> CycleResult | None:
    """Run the trader. ``once`` performs a single pass and returns its CycleResult (canary / tests).

    Holds the trader lock for the whole run (``TraderLockHeld`` when another trader is running)."""
    setup_logging()
    dbengine.init_db()
    sec = sec or secrets()
    lock_path = trader_lock_path(sec)
    acquire_trader_lock(lock_path, mode="live" if (bool(sec.live) and not paper) else "paper")
    try:
        trader = build_trader(paper=paper, cfg=cfg, sec=sec, adapter=adapter, budget=budget, clock=clock)
        log.info("trader starting: mode=%s adapter=%s", "LIVE" if trader.live else "paper", getattr(trader.adapter, "name", "?"))
        if once:
            return await trader.run_once()
        await trader.run_forever()
        return trader.last_cycle
    finally:
        release_trader_lock(lock_path)


def main(paper: bool = False, once: bool = False) -> None:
    """Synchronous entry for the CLI / pm2 (`sigcup trade`, paper unless LIVE=1)."""
    asyncio.run(run(paper=paper, once=once))


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(paper="--paper" in sys.argv, once="--once" in sys.argv)
