"""The `feeds` process entry point: runs every enabled collector on its cadence with
sigcup.common.scheduler.every, then NLP and features, with heartbeats.

Collector convention (every module under sigcup.feeds, including the news/social/NLP ones written by
another workstream)::

    async def run_once(session_factory, cfg, secrets) -> int      # rows written this pass

Modules are imported *lazily by dotted name* the first time they are due, and every call is wrapped in
try/except with a timeout, so an unfinished or crashing collector never kills the process (it is logged
and retried on its next tick). Cadences come from ``cfg.feeds``; ``ref_markets`` re-evaluates its
cadence every cycle and switches to ``cfg.election_night.ref_cadence_s`` while
``sigcup.trader.election_night.is_active`` says election night is on (if that import fails, the normal
cadence is kept).

Skip rules (``plan_collectors``): social_reddit needs reddit_client_id + reddit_client_secret;
social_x needs ``x_vendor != 'none'``; social_youtube needs youtube_api_key; nlp_stance needs
anthropic_api_key. Everything else is always scheduled (Bluesky's Jetstream and GDELT need no key).

Usage::

    python -m sigcup.feeds.loop              # serve (pm2 `feeds` process); SIGINT/SIGTERM stop it
    python -m sigcup.feeds.loop --once       # run every enabled collector once, print counts, exit
    python -m sigcup.feeds.loop --once --only polls --only ref_markets
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import random
import signal
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from sigcup.common.log import get_logger, setup_logging
from sigcup.common.scheduler import every
from sigcup.common.time import utc_now
from sigcup.db import engine as dbengine
from sigcup.db.repo import heartbeat
from sigcup.settings import Secrets, StrategyConfig, config
from sigcup.settings import secrets as load_secrets

log = get_logger(__name__)

PROCESS_NAME = "feeds"
HEARTBEAT_S = 30
MIN_CADENCE_S = 5
STAGGER_S = 2.0  # start collectors a few seconds apart so the first pass does not burst every host at once
MAX_RUN_TIMEOUT_S = 900.0

SessionFactory = Callable[[], Any]
RunOnce = Callable[[SessionFactory, StrategyConfig, Secrets], Awaitable[int | None]]


@dataclass(frozen=True)
class CollectorSpec:
    """One scheduled collector."""

    name: str
    module: str  # dotted module path exposing `async def run_once(session_factory, cfg, secrets) -> int`
    cadence_s: int
    enabled: bool = True
    reason: str = ""  # why it is disabled (empty when enabled)
    dynamic_cadence: bool = False  # ref_markets: cadence re-evaluated each cycle (election night)


@dataclass
class CollectorStats:
    runs: int = 0
    rows: int = 0
    errors: int = 0
    skipped: int = 0  # module/function missing (unfinished workstream)
    last_run: str | None = None
    last_rows: int | None = None
    last_error: str | None = None


def _present(*values: str | None) -> bool:
    return all(bool(v and str(v).strip()) for v in values)


def plan_collectors(cfg: StrategyConfig, secrets: Secrets) -> list[CollectorSpec]:
    """The collector schedule for this config and these credentials (pure; see the skip rules above)."""
    f = cfg.feeds
    x_on = (secrets.x_vendor or "none").strip().lower() != "none"
    yt_on = _present(secrets.youtube_api_key)
    reddit_on = _present(secrets.reddit_client_id, secrets.reddit_client_secret)
    llm_on = _present(secrets.anthropic_api_key)
    return [
        CollectorSpec("ref_markets", "sigcup.feeds.ref_markets", f.ref_cadence_s, dynamic_cadence=True),
        CollectorSpec("polls", "sigcup.feeds.polls", f.polls_cadence_s),
        CollectorSpec("news_gdelt", "sigcup.feeds.news_gdelt", f.news_cadence_s),
        CollectorSpec("news_rss", "sigcup.feeds.news_rss", f.news_cadence_s),
        CollectorSpec("social_bluesky", "sigcup.feeds.social_bluesky", f.social_cadence_s),
        CollectorSpec("social_x", "sigcup.feeds.social_x", f.social_cadence_s, x_on, "" if x_on else "x_vendor is 'none'"),
        CollectorSpec(
            "social_youtube", "sigcup.feeds.social_youtube", f.youtube_cadence_s, yt_on, "" if yt_on else "no youtube_api_key"
        ),
        CollectorSpec(
            "social_reddit", "sigcup.feeds.social_reddit", f.social_cadence_s, reddit_on, "" if reddit_on else "no reddit credentials"
        ),
        CollectorSpec("nlp_stance", "sigcup.feeds.nlp.stance", f.news_cadence_s, llm_on, "" if llm_on else "no anthropic_api_key"),
        CollectorSpec("nlp_features", "sigcup.feeds.nlp.features", f.news_cadence_s),
        CollectorSpec("events", "sigcup.feeds.events", f.news_cadence_s),
    ]


def ref_cadence_s(cfg: StrategyConfig, now: datetime | None = None) -> int:
    """Reference-price cadence: election_night.ref_cadence_s while election night is active, else
    feeds.ref_cadence_s. The election-night module is imported lazily; any failure means 'not active'."""
    now = now or utc_now()
    try:
        en = importlib.import_module("sigcup.trader.election_night")
        active = bool(en.is_active(cfg, now))
    except Exception as e:  # noqa: BLE001
        log.debug("election_night unavailable (%s); normal ref cadence", type(e).__name__)
        active = False
    return int(cfg.election_night.ref_cadence_s if active else cfg.feeds.ref_cadence_s)


def run_timeout_s(cadence_s: float) -> float:
    return float(min(MAX_RUN_TIMEOUT_S, max(60.0, 2.0 * cadence_s)))


class FeedsRunner:
    """Schedules the planned collectors; keeps per-collector stats for the heartbeat."""

    def __init__(
        self,
        cfg: StrategyConfig,
        secrets: Secrets,
        session_factory: SessionFactory | None = None,
        *,
        specs: Sequence[CollectorSpec] | None = None,
        stop: asyncio.Event | None = None,
        now: Callable[[], datetime] = utc_now,
        heartbeat_s: float = HEARTBEAT_S,
        stagger_s: float = STAGGER_S,
    ) -> None:
        self.cfg = cfg
        self.secrets = secrets
        self.session_factory = session_factory or dbengine.session
        self.specs: list[CollectorSpec] = list(specs if specs is not None else plan_collectors(cfg, secrets))
        self.stop = stop or asyncio.Event()
        self.now = now
        self.heartbeat_s = heartbeat_s
        self.stagger_s = stagger_s
        self.stats: dict[str, CollectorStats] = {s.name: CollectorStats() for s in self.specs}
        self._fns: dict[str, RunOnce | None] = {}

    # -- lazy module resolution --------------------------------------------------------------
    def resolve(self, spec: CollectorSpec) -> RunOnce | None:
        """Import spec.module and return its run_once (cached); None (logged once) when unavailable."""
        if spec.name in self._fns:
            return self._fns[spec.name]
        fn: RunOnce | None = None
        try:
            mod = importlib.import_module(spec.module)
            cand = getattr(mod, "run_once", None)
            if callable(cand):
                fn = cand
            else:
                log.warning("collector %s: %s has no run_once; skipping until it does", spec.name, spec.module)
        except NotImplementedError as e:
            log.warning("collector %s not implemented yet (%s); skipping", spec.name, e)
        except Exception as e:  # noqa: BLE001
            log.warning("collector %s: import of %s failed: %s: %s", spec.name, spec.module, type(e).__name__, e)
        self._fns[spec.name] = fn
        return fn

    # -- one guarded run ---------------------------------------------------------------------
    async def run_collector(self, spec: CollectorSpec) -> int | None:
        st = self.stats.setdefault(spec.name, CollectorStats())
        fn = self.resolve(spec)
        if fn is None:
            st.skipped += 1
            return None
        timeout = run_timeout_s(spec.cadence_s)
        try:
            n = await asyncio.wait_for(fn(self.session_factory, self.cfg, self.secrets), timeout=timeout)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            st.errors += 1
            st.last_error = f"timeout after {timeout:.0f}s"
            log.error("collector %s timed out after %.0fs", spec.name, timeout)
            return None
        except Exception as e:  # noqa: BLE001
            st.errors += 1
            st.last_error = f"{type(e).__name__}: {e}"[:200]
            log.exception("collector %s failed", spec.name)
            return None
        rows = int(n) if isinstance(n, int | float) else 0
        st.runs += 1
        st.rows += rows
        st.last_rows = rows
        st.last_run = self.now().isoformat()
        st.last_error = None
        log.info("collector %s: %d rows", spec.name, rows)
        return rows

    async def run_all_once(self, only: Sequence[str] | None = None) -> dict[str, int | None]:
        """Run every enabled collector once, sequentially (``--once``)."""
        out: dict[str, int | None] = {}
        for spec in self.specs:
            if only and spec.name not in only:
                continue
            if not spec.enabled:
                log.info("collector %s disabled: %s", spec.name, spec.reason)
                continue
            out[spec.name] = await self.run_collector(spec)
        return out

    # -- loops -------------------------------------------------------------------------------
    async def _ref_loop(self, spec: CollectorSpec) -> None:
        """Like scheduler.every, but the delay is re-evaluated every cycle (election-night cadence)."""
        while not self.stop.is_set():
            await self.run_collector(spec)
            delay = max(MIN_CADENCE_S, ref_cadence_s(self.cfg, self.now())) * (1 + random.uniform(-0.1, 0.1))
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def _every(self, spec: CollectorSpec, delay: float) -> None:
        if delay > 0:
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=delay)
                return
            except TimeoutError:
                pass
        if spec.dynamic_cadence:
            await self._ref_loop(spec)
            return

        async def tick() -> None:
            await self.run_collector(spec)

        await every(max(MIN_CADENCE_S, spec.cadence_s), tick, name=spec.name, stop=self.stop)

    def summary(self) -> dict[str, Any]:
        return {
            "ts": self.now().isoformat(),
            "collectors": {name: asdict(st) for name, st in self.stats.items()},
            "disabled": {s.name: s.reason for s in self.specs if not s.enabled},
        }

    async def beat(self) -> None:
        errors = sum(st.errors for st in self.stats.values())
        status = "ok" if errors == 0 else "degraded"
        detail = json.dumps(self.summary(), default=str)[:4000]
        with self.session_factory() as s:
            heartbeat(s, PROCESS_NAME, status, detail)

    async def run(self) -> None:
        """Serve until ``stop`` is set."""
        for spec in self.specs:
            if not spec.enabled:
                log.info("collector %s disabled: %s", spec.name, spec.reason)
        enabled = [s for s in self.specs if s.enabled]
        tasks: list[asyncio.Task] = []
        for i, spec in enumerate(enabled):
            tasks.append(asyncio.create_task(self._every(spec, i * self.stagger_s), name=f"feeds:{spec.name}"))
        tasks.append(asyncio.create_task(every(self.heartbeat_s, self.beat, name="heartbeat", stop=self.stop), name="feeds:heartbeat"))
        log.info("feeds: %d collectors scheduled (%s)", len(enabled), ", ".join(s.name for s in enabled))
        try:
            await self.stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await self.beat()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------------------------


async def _once(runner: FeedsRunner, only: Sequence[str] | None) -> dict[str, int | None]:
    """``--once``: run the selected collectors, then leave a heartbeat so `sigcup status` sees the pass."""
    results = await runner.run_all_once(only)
    try:
        await runner.beat()
    except Exception:  # noqa: BLE001
        log.exception("feeds: heartbeat failed")
    return results


async def serve(runner: FeedsRunner) -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, runner.stop.set)
        except (NotImplementedError, RuntimeError):  # Windows / non-main thread
            pass
    await runner.run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sigcup feeds", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="run every enabled collector once and exit")
    parser.add_argument("--only", action="append", default=[], help="collector name(s) to run (with --once)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    cfg = config()
    sec = load_secrets()
    dbengine.init_db()
    runner = FeedsRunner(cfg, sec)
    if args.once:
        results = asyncio.run(_once(runner, args.only or None))
        for name, n in results.items():
            print(f"{name:16s} {'skipped' if n is None else n}")
        return 0
    asyncio.run(serve(runner))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
