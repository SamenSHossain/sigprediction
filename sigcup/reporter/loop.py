"""The `reporter` process entry point.

Every minute ``Reporter.tick(now)`` runs and decides what is due:

* **hourly** (first tick, then every ``hourly_s``): ``clv.compute`` for every fill still missing a
  24 h / closing price, then one ``status`` log line (heartbeat staleness, cash/equity/reserved, open
  orders, exposure, rank, alerts).
* **daily at 12:00 UTC**: the full ``report.daily`` tables are logged, the plain dict is written to
  ``<data_dir>/reports/<day>.json`` and to KV ``reporter.last_daily``. The day is remembered in KV
  (``reporter.last_daily_day``) so a restart after noon still produces the report once, and a restart
  before noon does not produce it twice.
* a ``reporter`` heartbeat on every tick (``detail`` says when the last hourly/daily ran).

The reporter only reads the database (it holds no exchange key), so it can never spend the write
budget; ``sigcup report loop`` and pm2's ``reporter`` app call ``main()``.
"""

from __future__ import annotations

import asyncio
import io
import json
import signal
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from rich.console import Console

from sigcup.common.log import get_logger, setup_logging
from sigcup.common.time import ensure_utc, utc_now
from sigcup.db import engine as dbengine
from sigcup.db.repo import heartbeat, kv_get, kv_set
from sigcup.reporter import clv, report
from sigcup.settings import Secrets, StrategyConfig, config, secrets

log = get_logger(__name__)

PROCESS = "reporter"
TICK_S = 60
HOURLY_S = 3600
DAILY_HOUR_UTC = 12
KV_LAST_DAILY_DAY = "reporter.last_daily_day"
KV_LAST_DAILY = "reporter.last_daily"
KV_LAST_STATUS = "reporter.last_status"


def render_text(renderable: Any, width: int = 140) -> str:
    """Render a rich object to plain text (for log files)."""
    buf = io.StringIO()
    Console(file=buf, width=width, force_terminal=False, color_system=None, highlight=False).print(renderable)
    return buf.getvalue()


class Reporter:
    """Holds the schedule state; ``tick`` is pure with respect to the clock so tests can drive it."""

    def __init__(
        self,
        cfg: StrategyConfig | None = None,
        sec: Secrets | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
        hourly_s: float = HOURLY_S,
        daily_hour_utc: int = DAILY_HOUR_UTC,
        reports_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg or config()
        self.sec = sec or secrets()
        self.clock = clock or utc_now
        self.hourly_s = hourly_s
        self.daily_hour_utc = daily_hour_utc
        self.reports_dir = reports_dir if reports_dir is not None else self.sec.data_path / "reports"
        self.stop = asyncio.Event()
        self.last_hourly: datetime | None = None
        self.last_daily: datetime | None = None
        self.last_status: dict[str, Any] | None = None
        self.last_daily_report: dict[str, Any] | None = None
        self.ticks = 0

    # -- schedule ------------------------------------------------------------------------------
    def hourly_due(self, now: datetime) -> bool:
        return self.last_hourly is None or (now - self.last_hourly).total_seconds() >= self.hourly_s

    def daily_due(self, now: datetime) -> bool:
        if now.hour < self.daily_hour_utc:
            return False
        day = now.strftime("%Y-%m-%d")
        with dbengine.session() as s:
            done = kv_get(s, KV_LAST_DAILY_DAY)
        return not (isinstance(done, dict) and done.get("day") == day)

    # -- work ----------------------------------------------------------------------------------
    def run_hourly(self, now: datetime) -> dict[str, Any]:
        with dbengine.session() as s:
            stats = clv.compute(s, now, deadline=self.cfg.deadline_utc)
            d = report.status(s, now=now, cfg=self.cfg, sec=self.sec)
            kv_set(s, KV_LAST_STATUS, {"ts": d["ts"], "alerts": d["alerts"], "rank": d["leaderboard"]["rank"], "total": d["account"]["total"]})
        log.info("clv: examined=%d created=%d 24h=%d close=%d pending=%d", stats.examined, stats.created, stats.updated_24h, stats.updated_close, stats.pending)
        log.info(report.status_line(d))
        self.last_hourly = now
        self.last_status = d
        return d

    def run_daily(self, now: datetime) -> dict[str, Any]:
        with dbengine.session() as s:
            d = report.daily(s, now=now, cfg=self.cfg)
            kv_set(s, KV_LAST_DAILY, d)
            kv_set(s, KV_LAST_DAILY_DAY, {"day": d["day"], "ts": now.isoformat()})
        for line in render_text(report.render_daily(d)).splitlines():
            if line.strip():
                log.info("daily | %s", line)
        try:
            self.reports_dir.mkdir(parents=True, exist_ok=True)
            (self.reports_dir / f"{d['day']}.json").write_text(json.dumps(d, indent=1, default=str))
        except OSError as e:
            log.warning("could not write the daily report file: %s", e)
        self.last_daily = now
        self.last_daily_report = d
        return d

    def tick(self, now: datetime | None = None) -> dict[str, bool]:
        """One scheduler step. Returns which jobs ran. Exceptions are logged, never raised."""
        now = ensure_utc(now or self.clock())
        self.ticks += 1
        ran = {"hourly": False, "daily": False}
        if self.hourly_due(now):
            try:
                self.run_hourly(now)
                ran["hourly"] = True
            except Exception:  # noqa: BLE001
                log.exception("hourly report failed")
                self.last_hourly = now  # do not retry in a tight loop
        if self.daily_due(now):
            try:
                self.run_daily(now)
                ran["daily"] = True
            except Exception:  # noqa: BLE001
                log.exception("daily report failed")
        try:
            with dbengine.session() as s:
                heartbeat(s, PROCESS, "ok", self._detail())
        except Exception:  # noqa: BLE001
            log.exception("reporter heartbeat failed")
        return ran

    def _detail(self) -> str:
        h = self.last_hourly.strftime("%H:%M") if self.last_hourly else "-"
        d = self.last_daily.strftime("%m-%d %H:%M") if self.last_daily else "-"
        alerts = len(self.last_status["alerts"]) if self.last_status else 0
        return f"ticks={self.ticks} hourly={h} daily={d} alerts={alerts}"

    # -- serving -------------------------------------------------------------------------------
    async def serve(self, tick_s: float = TICK_S) -> None:
        loop = asyncio.get_running_loop()
        for sig_ in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig_, self.stop.set)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        log.info("reporter starting: hourly status, daily report at %02d:00 UTC, reports in %s", self.daily_hour_utc, self.reports_dir)
        while not self.stop.is_set():
            await asyncio.to_thread(self.tick)
            now = self.clock()
            delay = tick_s - (now.second + now.microsecond / 1e6) % tick_s  # align to the minute
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=max(1.0, delay))
            except TimeoutError:
                pass
        log.info("reporter stopping")


async def run(*, once: bool = False, cfg: StrategyConfig | None = None, sec: Secrets | None = None, clock: Callable[[], datetime] | None = None) -> Reporter:
    """Run the reporter; ``once`` performs a single tick (hourly status + CLV; daily if due) and returns."""
    setup_logging()
    dbengine.init_db()
    r = Reporter(cfg, sec, clock=clock)
    if once:
        r.tick()
        return r
    await r.serve()
    return r


def main(once: bool = False) -> int:
    """Synchronous entry for the CLI / pm2 (`python -m sigcup.cli report loop`)."""
    asyncio.run(run(once=once))
    return 0


def next_daily_at(now: datetime, hour: int = DAILY_HOUR_UTC) -> datetime:
    """When the next daily report is scheduled (informational, for `sigcup status`)."""
    now = ensure_utc(now)
    at = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return at if at > now else at + timedelta(days=1)


if __name__ == "__main__":  # pragma: no cover
    import sys

    raise SystemExit(main(once="--once" in sys.argv))
