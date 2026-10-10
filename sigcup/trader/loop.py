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
"""

from __future__ import annotations

import asyncio
import importlib
import signal
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import ModuleType
from typing import Any

from sqlalchemy import select

from sigcup.common.log import get_logger, setup_logging
from sigcup.common.scheduler import every
from sigcup.common.time import ensure_utc, hours_to, utc_now
from sigcup.db import engine as dbengine
from sigcup.db.models import Event, LeaderboardSnap
from sigcup.db.repo import heartbeat, kv_get, kv_set, latest_ref_prices
from sigcup.paper import PaperAdapter
from sigcup.settings import Secrets, StrategyConfig, config, secrets
from sigcup.sig.adapter import ExchangeAdapter
from sigcup.sig.ratelimit import Budget
from sigcup.trader import risk
from sigcup.trader.book_state import BookState
from sigcup.trader.oms import OMS, ReconcileReport, SyncReport
from sigcup.trader.types import ActiveEvent, DesiredOrder, FairValue, RaceInfo, TraderContext

log = get_logger(__name__)

PROCESS = "trader"
KV_DAY_START = "trader.day_start"  # live; paper keeps its own baseline under KV_DAY_START_PAPER
KV_DAY_START_PAPER = "trader.day_start:paper"
STARTUP_RETRY_S = 5.0
STARTUP_RETRY_MAX_S = 60.0
BULK_CHUNK = 100  # /exchanges/prices accepts <= 100 ids per read
DEPTH_PER_PASS = 4  # depth reads per bulk pass (bounded: reads are shared with everything else)
PAPER_QUOTE_SOURCE = "paper"  # Quote.source of the rows a DB-quote paper trader echoes back

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
    context: TraderContext | None = field(default=None, repr=False)


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
    def _ref_prices(self, s: Any) -> dict[str, float]:
        """Best real-money YES mid per exchange id via Market.kalshi_ticker / polymarket_condition_id."""
        refs = latest_ref_prices(s)
        out: dict[str, float] = {}
        for eid, meta in self.book.meta.items():
            best: tuple[float, float] | None = None
            for source, key in (("kalshi", meta.kalshi_ticker), ("polymarket", meta.polymarket_condition_id)):
                if not key:
                    continue
                r = refs.get((source, key))
                if r is None:
                    continue
                mid = r.yes_mid
                if mid is None and r.yes_bid is not None and r.yes_ask is not None:
                    mid = (r.yes_bid + r.yes_ask) / 2
                if mid is None:
                    continue
                liq = float(r.liquidity or 0.0)
                if best is None or liq > best[0]:
                    best = (liq, float(mid))
            if best is not None:
                out[eid] = best[1]
        return out

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
        self.book.set_open_intents(self.oms.acked())
        self.book.set_positions(list(self.oms.positions.values()))
        self.book.set_account(cash=self.oms.cash, reserved=self.oms.reserved)
        self.book.positions_value = self.oms.positions_value
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
        log.info(
            "cycle: fair=%d arbs=%d desired=%d accepted=%d placed=%d cancelled=%d kept=%d deferred=%d skipped=%d%s",
            res.fair, res.arbs, res.desired, res.accepted, len(res.sync.placed), len(res.sync.cancelled),
            res.sync.kept, res.sync.deferred, len(res.sync.skipped), f" blocked={res.sync.blocked}" if res.sync.blocked else "",
        )
        return res

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
    """Run the trader. ``once`` performs a single pass and returns its CycleResult (canary / tests)."""
    setup_logging()
    dbengine.init_db()
    trader = build_trader(paper=paper, cfg=cfg, sec=sec, adapter=adapter, budget=budget, clock=clock)
    log.info("trader starting: mode=%s adapter=%s", "LIVE" if trader.live else "paper", getattr(trader.adapter, "name", "?"))
    if once:
        return await trader.run_once()
    await trader.run_forever()
    return trader.last_cycle


def main(paper: bool = False, once: bool = False) -> None:
    """Synchronous entry for the CLI / pm2 (`sigcup trade`, paper unless LIVE=1)."""
    asyncio.run(run(paper=paper, once=once))


if __name__ == "__main__":  # pragma: no cover
    import sys

    main(paper="--paper" in sys.argv, once="--once" in sys.argv)
