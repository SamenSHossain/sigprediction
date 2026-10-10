"""Hourly status (heartbeats, cash, exposure by party/state, open orders) and the daily report
(P&L by strategy tag, CLV table and per-strategy CLV gate, calibration of fair values vs reference,
the correlated settlement P&L distribution of the current positions, rank history).

Both entry points return plain dicts (JSON-serialisable, used by tests, the KV store and the daily
JSON file) and have a rich renderer::

    d = status(session, now=utc_now(), cfg=config(), sec=secrets())
    console.print(render_status(d))

    d = daily(session, now=utc_now(), cfg=config())
    console.print(render_daily(d))

P&L convention: every fill is reduced to a signed YES-equivalent trade (buy NO at p == sell YES at
1 - p) and run through an average-cost ledger per (strategy, market); realised P&L is booked when a
fill reduces the position, the remainder is marked at ``Position.mark`` (the exchange's mark; the
latest quote mid, then the average cost, are the fallbacks). Fees are subtracted. Paper and live fills
are never mixed in one row (P&L and CLV both carry a ``mode`` column).

Position convention (the producers: PaperAdapter and the OMS that copies its PositionInfo): ``quantity``
is signed (+YES / -NO) and ``avg_price`` is the price paid per share of the side held, so a NO position
carries the NO price; ``mark`` is the YES mid. ``yes_avg_price`` converts the cost to YES terms.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from rich.console import Group
from rich.table import Table
from sqlalchemy import select
from sqlalchemy.orm import Session

from sigcup.common.log import get_logger
from sigcup.common.time import ensure_utc, hours_to, utc_now
from sigcup.db.models import (
    AccountSnap,
    Fill,
    Heartbeat,
    IntentState,
    LeaderboardSnap,
    Market,
    OrderIntent,
    Position,
    Race,
    Signal,
)
from sigcup.db.repo import kv_get, latest_quotes, latest_ref_prices
from sigcup.reporter import clv as clvmod
from sigcup.reporter.clv import fill_direction, fill_yes_price, quote_mid
from sigcup.settings import Secrets, StrategyConfig, config

log = get_logger(__name__)

EXPECTED_PROCESSES = ("trader", "feeds", "reporter")
STALE_AFTER_S: dict[str, float] = {"trader": 90, "feeds": 240, "reporter": 4500}
DEFAULT_STALE_S = 600.0
WORKING_STATES = {IntentState.SENT.value, IntentState.UNCERTAIN.value, IntentState.ACKED.value}
KV_KILLED = "killed"
PORTFOLIO_LOSS_FRAC = 0.10  # the daily report's P(settlement loss > this fraction of bankroll_start)


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _age_s(ts: datetime | None, now: datetime) -> float | None:
    return None if ts is None else max(0.0, (now - ensure_utc(ts)).total_seconds())


def _iso(ts: datetime | None) -> str | None:
    return None if ts is None else ensure_utc(ts).isoformat()


def _market_index(session: Session) -> tuple[dict[str, Market], dict[str, Race]]:
    markets = {m.exchange_id: m for m in session.execute(select(Market)).scalars().all()}
    races = {r.race_id: r for r in session.execute(select(Race)).scalars().all()}
    return markets, races


def _mark_for(exchange_id: str, positions: dict[str, Position], quotes: dict[str, Any], fallback: float | None) -> float | None:
    p = positions.get(exchange_id)
    if p is not None and p.mark is not None:
        return float(p.mark)
    mid = quote_mid(quotes.get(exchange_id))
    if mid is not None:
        return mid
    return fallback


def yes_avg_price(quantity: int, avg_price: float | None) -> float | None:
    """``Position.avg_price`` in YES terms: the YES price paid for a long position, ``1 - NO price paid``
    for a NO (negative) position."""
    if avg_price is None:
        return None
    return float(avg_price) if quantity >= 0 else 1.0 - float(avg_price)


def at_risk(quantity: int, mark: float | None) -> float:
    """What a position is worth at the YES ``mark``: YES shares x mark, NO shares x (1 - mark)."""
    if mark is None or not quantity:
        return 0.0
    return quantity * mark if quantity > 0 else -quantity * (1.0 - mark)


def _latest(session: Session, model: Any) -> Any:
    return session.execute(select(model).order_by(model.ts.desc(), model.id.desc()).limit(1)).scalars().first()


# ---------------------------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------------------------
def heartbeats(session: Session, now: datetime) -> list[dict[str, Any]]:
    """Every expected process (plus any extra heartbeat row) with its age and staleness."""
    rows = {h.process: h for h in session.execute(select(Heartbeat)).scalars().all()}
    out: list[dict[str, Any]] = []
    for name in list(EXPECTED_PROCESSES) + sorted(k for k in rows if k not in EXPECTED_PROCESSES):
        h = rows.get(name)
        age = _age_s(h.ts, now) if h else None
        limit = STALE_AFTER_S.get(name, DEFAULT_STALE_S)
        out.append(
            {
                "process": name,
                "ts": _iso(h.ts) if h else None,
                "age_s": age,
                "status": h.status if h else "missing",
                "detail": h.detail if h else None,
                "stale": age is None or age > limit,
            }
        )
    return out


def account(session: Session, now: datetime, cfg: StrategyConfig) -> dict[str, Any]:
    snap = _latest(session, AccountSnap)
    if snap is None:
        return {"ts": None, "age_s": None, "cash": None, "positions_value": None, "total": None, "reserved": None, "free_cash": None, "pnl": None, "pnl_frac": None}
    total = float(snap.total)
    return {
        "ts": _iso(snap.ts),
        "age_s": _age_s(snap.ts, now),
        "cash": float(snap.cash),
        "positions_value": float(snap.positions_value),
        "total": total,
        "reserved": float(snap.reserved or 0.0),
        "free_cash": float(snap.cash) - float(snap.reserved or 0.0),
        "pnl": total - cfg.bankroll_start,
        "pnl_frac": (total - cfg.bankroll_start) / cfg.bankroll_start if cfg.bankroll_start else None,
    }


def open_intents(session: Session) -> dict[str, Any]:
    """Working intents by strategy: ACKED (resting) counts/notional, plus SENT/UNCERTAIN counts."""
    rows = session.execute(select(OrderIntent).where(OrderIntent.state.in_(sorted(WORKING_STATES)))).scalars().all()
    by: dict[str, dict[str, Any]] = {}
    uncertain = 0
    sent = 0
    for i in rows:
        st = str(i.state)
        if st == IntentState.UNCERTAIN.value:
            uncertain += 1
            continue
        if st == IntentState.SENT.value:
            sent += 1
            continue
        d = by.setdefault(str(i.strategy or clvmod.UNKNOWN_STRATEGY), {"n": 0, "shares": 0, "notional": 0.0, "paper": 0, "live": 0})
        remaining = max(0, int(i.quantity) - int(i.filled_qty or 0))
        price = float(i.price) if str(i.action) == "buy" else 1.0 - float(i.price)
        d["n"] += 1
        d["shares"] += remaining
        d["notional"] += price * remaining
        d["paper" if i.paper else "live"] += 1
    return {
        "by_strategy": dict(sorted(by.items())),
        "n_acked": sum(d["n"] for d in by.values()),
        "notional": sum(d["notional"] for d in by.values()),
        "n_sent": sent,
        "n_uncertain": uncertain,
    }


def exposure(session: Session) -> dict[str, Any]:
    """Net YES-equivalent exposure (signed shares x YES mark: ``value``, ``gross`` = |value|) and the
    capital at risk (YES shares x mark + NO shares x (1 - mark): ``at_risk``) by party, state and
    chamber, plus the partisan tilt (long D / short R count as pro-Dem)."""
    positions = {p.exchange_id: p for p in session.execute(select(Position)).scalars().all() if p.quantity}
    markets, races = _market_index(session)
    quotes = latest_quotes(session)
    def new_bucket() -> dict[str, float]:
        return {"shares": 0.0, "value": 0.0, "gross": 0.0, "at_risk": 0.0, "n": 0}

    by_party: dict[str, dict[str, float]] = defaultdict(new_bucket)
    by_state: dict[str, dict[str, float]] = defaultdict(new_bucket)
    by_chamber: dict[str, dict[str, float]] = defaultdict(new_bucket)
    rows: list[dict[str, Any]] = []
    net_dem = 0.0
    gross = 0.0
    risk_total = 0.0
    for eid, p in positions.items():
        m = markets.get(eid)
        r = races.get(m.race_id) if m else None
        qty = int(p.quantity)
        avg_yes = yes_avg_price(qty, p.avg_price)
        mark = _mark_for(eid, positions, quotes, avg_yes)
        value = qty * mark if mark is not None else 0.0
        risk = at_risk(qty, mark)
        party = (m.party if m else "?") or "?"
        state = (r.state if r and r.state else (r.chamber if r else "?")) or "?"
        chamber = (r.chamber if r else "?") or "?"
        for key, bucket in ((party, by_party), (state, by_state), (chamber, by_chamber)):
            bucket[key]["shares"] += qty
            bucket[key]["value"] += value
            bucket[key]["gross"] += abs(value)
            bucket[key]["at_risk"] += risk
            bucket[key]["n"] += 1
        if party.upper() == "D":
            net_dem += value
        elif party.upper() == "R":
            net_dem -= value
        gross += abs(value)
        risk_total += risk
        rows.append(
            {
                "exchange_id": eid,
                "race_id": m.race_id if m else None,
                "party": party,
                "state": r.state if r else None,
                "quantity": qty,
                "avg_price": float(p.avg_price) if p.avg_price is not None else None,
                "avg_yes": avg_yes,
                "mark": mark,
                "value": value,
                "at_risk": risk,
                "unrealised": (qty * (mark - avg_yes)) if (mark is not None and avg_yes is not None) else None,
            }
        )
    rows.sort(key=lambda d: -abs(d["value"]))
    return {
        "n_positions": len(rows),
        "gross": gross,
        "at_risk": risk_total,
        "net_dem_equivalent": net_dem,
        "by_party": {k: dict(v) for k, v in sorted(by_party.items())},
        "by_state": {k: dict(v) for k, v in sorted(by_state.items(), key=lambda kv: -kv[1]["gross"])},
        "by_chamber": {k: dict(v) for k, v in sorted(by_chamber.items())},
        "positions": rows,
    }


def leaderboard(session: Session, now: datetime) -> dict[str, Any]:
    snap = _latest(session, LeaderboardSnap)
    if snap is None:
        return {"ts": None, "age_s": None, "rank": None, "our_total": None, "member_count": None, "top_totals": []}
    return {
        "ts": _iso(snap.ts),
        "age_s": _age_s(snap.ts, now),
        "rank": snap.rank,
        "our_total": snap.our_total,
        "member_count": snap.member_count,
        "top_totals": list(snap.top_totals or [])[:5],
    }


def kill_state(session: Session, sec: Secrets | None) -> dict[str, Any]:
    flag = kv_get(session, KV_KILLED)
    return {
        "kill_file": str(sec.kill_path) if sec else None,
        "kill_file_present": bool(sec and sec.kill_path.exists()),
        "kv_flag": flag if flag else None,
    }


def status(session: Session, now: datetime | None = None, cfg: StrategyConfig | None = None, sec: Secrets | None = None) -> dict[str, Any]:
    """The hourly status snapshot as a plain dict."""
    now = ensure_utc(now or utc_now())
    cfg = cfg or config()
    d = {
        "ts": now.isoformat(),
        "hours_to_deadline": hours_to(cfg.deadline_utc, now),
        "heartbeats": heartbeats(session, now),
        "account": account(session, now, cfg),
        "open_intents": open_intents(session),
        "exposure": exposure(session),
        "leaderboard": leaderboard(session, now),
        "kill": kill_state(session, sec),
        "live_env": bool(sec.live) if sec is not None else None,
    }
    d["alerts"] = alerts(d, cfg)
    return d


def alerts(d: dict[str, Any], cfg: StrategyConfig) -> list[str]:
    """Human-readable warnings derived from a status dict."""
    out: list[str] = []
    for h in d["heartbeats"]:
        if h["stale"]:
            how = "missing" if h["age_s"] is None else f"stale ({h['age_s']:.0f} s)"
            out.append(f"{h['process']} heartbeat {how}")
        elif h["status"] not in ("ok", "paper", "live"):
            out.append(f"{h['process']} status {h['status']}")
        elif h["process"] == "trader" and h["status"] == "paper" and d.get("live_env"):
            out.append("LIVE=1 but the trader is running in paper mode (started with --paper?): nothing reaches the exchange")
    if d["kill"]["kill_file_present"]:
        out.append("KILL file present: trader will not trade")
    if d["kill"]["kv_flag"]:
        out.append(f"kill flag set in KV: {d['kill']['kv_flag']}")
    if d["open_intents"]["n_uncertain"]:
        out.append(f"{d['open_intents']['n_uncertain']} UNCERTAIN intents awaiting reconciliation")
    acct = d["account"]
    if acct["total"] is not None:
        reserve = cfg.risk.liquid_reserve_frac_final72h if d["hours_to_deadline"] <= cfg.endgame.final_hours else cfg.risk.liquid_reserve_frac
        if acct["free_cash"] is not None and acct["free_cash"] < reserve * cfg.bankroll_start:
            out.append(f"free cash {acct['free_cash']:.0f} below the {reserve:.0%} liquid reserve")
        if acct["pnl_frac"] is not None and acct["pnl_frac"] <= -cfg.risk.daily_loss_limit_frac:
            out.append(f"equity {acct['pnl_frac']:+.1%} vs start")
    return out


def status_line(d: dict[str, Any]) -> str:
    """One log line summarising a status dict (used by the hourly reporter log)."""
    hb = " ".join(f"{h['process']}={'STALE' if h['stale'] else h['status']}" for h in d["heartbeats"])
    a = d["account"]
    lb = d["leaderboard"]
    cash = "-" if a["cash"] is None else f"{a['cash']:.0f}"
    total = "-" if a["total"] is None else f"{a['total']:.0f}"
    reserved = "-" if a["reserved"] is None else f"{a['reserved']:.0f}"
    rank = "-" if lb["rank"] is None else f"{lb['rank']}/{lb['member_count'] or '?'}"
    return (
        f"status: {hb} | cash={cash} total={total} reserved={reserved} | open={d['open_intents']['n_acked']} "
        f"uncertain={d['open_intents']['n_uncertain']} | positions={d['exposure']['n_positions']} gross={d['exposure']['gross']:.0f} "
        f"netD={d['exposure']['net_dem_equivalent']:+.0f} | rank={rank} | T-{d['hours_to_deadline']:.0f}h"
        + (f" | ALERTS: {'; '.join(d['alerts'])}" if d.get("alerts") else "")
    )


# ---------------------------------------------------------------------------------------------
# daily
# ---------------------------------------------------------------------------------------------
@dataclass
class _Ledger:
    """Average-cost ledger of one (strategy, market) in YES-equivalent shares."""

    pos: float = 0.0
    avg: float = 0.0
    realised: float = 0.0
    fees: float = 0.0
    shares: int = 0
    n: int = 0

    def apply(self, dq: float, price: float, fee: float) -> None:
        self.n += 1
        self.shares += int(abs(dq))
        self.fees += fee
        if self.pos == 0 or (dq > 0) == (self.pos > 0):
            total = self.pos + dq
            self.avg = (self.pos * self.avg + dq * price) / total if total else 0.0
            self.pos = total
            return
        closing = min(abs(dq), abs(self.pos))
        self.realised += closing * (price - self.avg) * (1 if self.pos > 0 else -1)
        remainder = abs(dq) - closing
        if remainder > 0:
            self.pos = remainder * (1 if dq > 0 else -1)
            self.avg = price
        else:
            self.pos += dq
            if self.pos == 0:
                self.avg = 0.0

    def unrealised(self, mark: float | None) -> float:
        if self.pos == 0:
            return 0.0
        m = self.avg if mark is None else mark
        return self.pos * (m - self.avg)


def pnl_by_strategy(session: Session, fills: Iterable[Fill] | None = None) -> list[dict[str, Any]]:
    """Realised + mark-to-market P&L per (strategy, paper/live) with a total row per mode."""
    if fills is None:
        fills = session.execute(select(Fill).order_by(Fill.ts, Fill.fill_id)).scalars().all()
    fills = sorted(fills, key=lambda f: (ensure_utc(f.ts), f.fill_id))
    positions = {p.exchange_id: p for p in session.execute(select(Position)).scalars().all()}
    quotes = latest_quotes(session)
    ledgers: dict[tuple[bool, str, str], _Ledger] = defaultdict(_Ledger)
    for f in fills:
        key = (bool(f.paper), str(f.strategy or clvmod.UNKNOWN_STRATEGY), f.exchange_id)
        dq = fill_direction(f.side, f.action) * int(f.quantity)
        ledgers[key].apply(float(dq), fill_yes_price(f.side, float(f.price)), float(f.fee or 0.0))
    agg: dict[tuple[bool, str], dict[str, Any]] = {}
    for (paper, strategy, eid), led in ledgers.items():
        mark = _mark_for(eid, positions, quotes, None)
        row = agg.setdefault(
            (paper, strategy),
            {"mode": "paper" if paper else "live", "strategy": strategy, "n_fills": 0, "shares": 0, "markets": 0, "open_shares": 0.0, "realised": 0.0, "unrealised": 0.0, "fees": 0.0, "total": 0.0},
        )
        row["n_fills"] += led.n
        row["shares"] += led.shares
        row["markets"] += 1
        row["open_shares"] += abs(led.pos)
        row["realised"] += led.realised
        row["unrealised"] += led.unrealised(mark)
        row["fees"] += led.fees
    rows: list[dict[str, Any]] = []
    for mode_paper in (False, True):
        mode_rows = [v for (p, _), v in sorted(agg.items(), key=lambda kv: kv[0]) if p == mode_paper]
        if not mode_rows:
            continue
        for r in mode_rows:
            r["total"] = r["realised"] + r["unrealised"] - r["fees"]
            rows.append(r)
        tot = {"mode": "paper" if mode_paper else "live", "strategy": "all", "n_fills": 0, "shares": 0, "markets": 0, "open_shares": 0.0, "realised": 0.0, "unrealised": 0.0, "fees": 0.0, "total": 0.0}
        for r in mode_rows:
            for k in ("n_fills", "shares", "markets", "open_shares", "realised", "unrealised", "fees", "total"):
                tot[k] += r[k]
        rows.append(tot)
    return rows


def equity_curve(session: Session, now: datetime, cfg: StrategyConfig) -> dict[str, Any]:
    """Today's and overall equity change from AccountSnap rows."""
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    first_today = session.execute(select(AccountSnap).where(AccountSnap.ts >= day_start).order_by(AccountSnap.ts, AccountSnap.id).limit(1)).scalars().first()
    latest = _latest(session, AccountSnap)
    if latest is None:
        return {"start_of_day": None, "now": None, "change_today": None, "change_total": None}
    base = float(first_today.total) if first_today is not None else None
    return {
        "start_of_day": base,
        "now": float(latest.total),
        "change_today": (float(latest.total) - base) if base is not None else None,
        "change_total": float(latest.total) - cfg.bankroll_start,
    }


# (strategy, fair, price, edge, size, reference mid)
_CalSample = tuple[str, float, float | None, float | None, int, float | None]


def calibration(
    session: Session,
    now: datetime,
    days: int = 1,
    cfg: StrategyConfig | None = None,
    fair_values: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fair-vs-price and fair-vs-reference summary.

    From the Signal rows of the last ``days`` when a producer wrote any (``source: "signals"``);
    otherwise from the model's current fair values (``fairvalue.compute_all`` over every market, priced
    against the latest SIG mid and the fair value's own ``ref`` component: ``source: "fair_values"``).
    ``fair_values`` supplies them (the daily report shares one computation); by default they are computed.
    """
    since = now - timedelta(days=days)
    rows = session.execute(select(Signal).where(Signal.ts >= since)).scalars().all()
    samples: list[_CalSample] = []
    source = "signals"
    error: str | None = None
    for s in rows:
        if s.fair is None:
            continue
        meta = s.meta if isinstance(s.meta, dict) else {}
        samples.append(
            (
                str(s.strategy or "?"),
                float(s.fair),
                float(s.price) if s.price is not None else None,
                float(s.edge) if s.edge is not None else None,
                int(s.size or 0),
                _signal_ref(meta),
            )
        )
    n_rows = len(rows)
    if not rows:
        source = "fair_values"
        try:
            samples = _fair_value_samples(session, now, cfg or config(), fair_values)
        except Exception as e:  # noqa: BLE001 - the report must render even if the model cannot run
            log.warning("calibration: fair values unavailable: %s: %s", type(e).__name__, e)
            error = f"{type(e).__name__}: {e}"
            samples = []
        n_rows = len(samples)
    by_strategy: dict[str, dict[str, Any]] = {}
    buckets: dict[str, dict[str, Any]] = {}
    ref_gaps: list[float] = []
    for strategy, fair, price, edge_raw, size, ref in samples:
        edge = edge_raw if edge_raw is not None else ((fair - price) if price is not None else None)
        d = by_strategy.setdefault(strategy, {"n": 0, "edge_sum": 0.0, "abs_edge_sum": 0.0, "size": 0})
        d["n"] += 1
        if edge is not None:
            d["edge_sum"] += edge
            d["abs_edge_sum"] += abs(edge)
        d["size"] += size
        if price is not None:
            lo = int(fair * 10) / 10 if fair < 1 else 0.9
            b = buckets.setdefault(f"{lo:.1f}-{lo + 0.1:.1f}", {"n": 0, "fair_sum": 0.0, "price_sum": 0.0})
            b["n"] += 1
            b["fair_sum"] += fair
            b["price_sum"] += price
        if ref is not None:
            ref_gaps.append(fair - ref)
    return {
        "source": source,
        "error": error,
        "n_signals": n_rows,
        "window_days": days,
        "by_strategy": {
            k: {"n": v["n"], "mean_edge": v["edge_sum"] / v["n"] if v["n"] else None, "mean_abs_edge": v["abs_edge_sum"] / v["n"] if v["n"] else None, "size": v["size"]}
            for k, v in sorted(by_strategy.items())
        },
        "buckets": {k: {"n": v["n"], "mean_fair": v["fair_sum"] / v["n"], "mean_price": v["price_sum"] / v["n"]} for k, v in sorted(buckets.items())},
        "fair_vs_ref": {"n": len(ref_gaps), "mean_gap": (sum(ref_gaps) / len(ref_gaps)) if ref_gaps else None, "mean_abs_gap": (sum(abs(g) for g in ref_gaps) / len(ref_gaps)) if ref_gaps else None},
    }


def _signal_ref(meta: dict[str, Any]) -> float | None:
    """The reference mid recorded with a signal: ``components.ref`` first, then any reference key (never
    a FairValue ``weights`` entry, whose ``ref`` is a blending weight, not a price)."""
    comps = meta.get("components")
    if isinstance(comps, dict):
        n = _first_number(comps, *REF_KEYS)
        if n is not None:
            return n
    return _first_number(meta, *REF_KEYS)


def current_fair_values(session: Session, now: datetime, cfg: StrategyConfig) -> dict[str, Any]:
    """``fairvalue.compute_all`` over every market with a known race, quoted at the latest SIG quote.

    The trader persists no per-market fair values (Signal rows exist only for orders that survived risk,
    deduplicated), so the reporter recomputes them from the same DB inputs. The trader package is
    imported lazily: the reporter never depends on it being importable for its other sections."""
    from sigcup.trader import fairvalue
    from sigcup.trader.types import MarketView

    markets, races = _market_index(session)
    quotes = latest_quotes(session)
    views: list[MarketView] = []
    for eid, m in markets.items():
        r = races.get(m.race_id)
        if r is None:
            continue
        q = quotes.get(eid)
        views.append(
            MarketView(
                exchange_id=eid, race_id=m.race_id, party=m.party, chamber=r.chamber, state=r.state,
                bid=float(q.bid) if q is not None and q.bid is not None else None,
                ask=float(q.ask) if q is not None and q.ask is not None else None,
                last=float(q.last) if q is not None and q.last is not None else None,
                ts=q.ts if q is not None else None, status=m.status or "open",
            )
        )
    return fairvalue.compute_all(session, cfg, views, now)


def _shared_fair_values(session: Session, now: datetime, cfg: StrategyConfig) -> Callable[[], dict[str, Any]]:
    """A thunk that computes :func:`current_fair_values` on first use and then returns the same result
    (or re-raises the same error), so one daily report runs the model at most once."""
    memo: dict[str, Any] = {}

    def get() -> dict[str, Any]:
        if "error" in memo:
            raise memo["error"]
        if "fair" not in memo:
            try:
                memo["fair"] = current_fair_values(session, now, cfg)
            except Exception as e:
                memo["error"] = e
                raise
        return memo["fair"]

    return get


def _fair_value_samples(
    session: Session, now: datetime, cfg: StrategyConfig, fair_values: Callable[[], dict[str, Any]] | None = None
) -> list[_CalSample]:
    """One calibration sample per market the model prices now."""
    fair = fair_values() if fair_values is not None else current_fair_values(session, now, cfg)
    quotes = latest_quotes(session)
    out: list[_CalSample] = []
    for eid, fv in sorted(fair.items()):
        comps = fv.components if isinstance(fv.components, dict) else {}
        ref = comps.get("ref")
        out.append(("fair_value", float(fv.fair), quote_mid(quotes.get(eid)), None, 0, float(ref) if ref is not None else None))
    return out


REF_KEYS = ("ref", "ref_mid", "reference", "ref_price", "kalshi", "polymarket")
VALUE_KEYS = ("mid", "yes_mid", "price", "fair", "value")
SKIP_KEYS = ("weights", "weight")  # blending weights share key names with the components


def _as_number(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, dict):
        for k in VALUE_KEYS:
            n = _as_number(v.get(k))
            if n is not None:
                return n
    return None


def _first_number(meta: dict[str, Any], *keys: str, depth: int = 3) -> float | None:
    """The first numeric value under any of ``keys`` (a dict value yields its mid/price), searching
    nested dicts up to ``depth`` levels so ``{"components": {"ref": {...}}}`` is found too (never
    inside a ``weights`` dict)."""
    for k in keys:
        if k in meta:
            n = _as_number(meta[k])
            if n is not None:
                return n
    if depth > 0:
        for k, v in meta.items():
            if isinstance(v, dict) and k not in SKIP_KEYS:
                n = _first_number(v, *keys, depth=depth - 1)
                if n is not None:
                    return n
    return None


def sig_vs_ref(session: Session, top: int = 8) -> dict[str, Any]:
    """Gap between the latest SIG mid and the best real-money reference mid per crosswalked market."""
    markets, _ = _market_index(session)
    quotes = latest_quotes(session)
    refs = latest_ref_prices(session)
    gaps: list[dict[str, Any]] = []
    for eid, m in markets.items():
        mid = quote_mid(quotes.get(eid))
        if mid is None:
            continue
        best: tuple[float, float, str] | None = None
        for source, key in (("kalshi", m.kalshi_ticker), ("polymarket", m.polymarket_condition_id)):
            if not key:
                continue
            r = refs.get((source, key))
            if r is None:
                continue
            ref_mid = float(r.yes_mid) if r.yes_mid is not None else ((float(r.yes_bid) + float(r.yes_ask)) / 2 if r.yes_bid is not None and r.yes_ask is not None else None)
            if ref_mid is None:
                continue
            liq = float(r.liquidity or 0.0)
            if best is None or liq > best[0]:
                best = (liq, ref_mid, source)
        if best is None:
            continue
        gaps.append({"exchange_id": eid, "race_id": m.race_id, "party": m.party, "sig_mid": mid, "ref_mid": best[1], "source": best[2], "gap": mid - best[1]})
    gaps.sort(key=lambda g: -abs(g["gap"]))
    n = len(gaps)
    return {
        "n": n,
        "mean_gap": (sum(g["gap"] for g in gaps) / n) if n else None,
        "mean_abs_gap": (sum(abs(g["gap"]) for g in gaps) / n) if n else None,
        "largest": gaps[:top],
    }


def rank_history(session: Session, now: datetime, days: int = 14) -> dict[str, Any]:
    """One leaderboard point per UTC day (the last snapshot of each day) plus the latest and the 24 h change."""
    since = now - timedelta(days=days)
    rows = session.execute(select(LeaderboardSnap).where(LeaderboardSnap.ts >= since).order_by(LeaderboardSnap.ts, LeaderboardSnap.id)).scalars().all()
    per_day: dict[str, LeaderboardSnap] = {}
    for r in rows:
        per_day[ensure_utc(r.ts).strftime("%Y-%m-%d")] = r
    latest = rows[-1] if rows else None
    day_ago = None
    for r in rows:
        if ensure_utc(r.ts) <= now - timedelta(hours=24):
            day_ago = r
    return {
        "points": [
            {"day": day, "ts": _iso(r.ts), "rank": r.rank, "our_total": r.our_total, "member_count": r.member_count, "top3": list(r.top_totals or [])[:3]}
            for day, r in sorted(per_day.items())
        ],
        "rank": latest.rank if latest else None,
        "our_total": latest.our_total if latest else None,
        "member_count": latest.member_count if latest else None,
        "rank_change_24h": (int(latest.rank) - int(day_ago.rank)) if (latest and day_ago and latest.rank is not None and day_ago.rank is not None) else None,
    }


def _loss_frac(cfg: StrategyConfig) -> float:
    v = getattr(getattr(cfg, "reporter", None), "portfolio_loss_frac", None)
    try:
        return float(v) if v is not None else PORTFOLIO_LOSS_FRAC
    except (TypeError, ValueError):
        return PORTFOLIO_LOSS_FRAC


def portfolio_risk(
    session: Session,
    now: datetime,
    cfg: StrategyConfig,
    fair_values: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Settlement P&L distribution of the current positions under the correlated simulation.

    Inputs: every seat race's legs at the model's current fair values (``fair_values``, by default
    :func:`current_fair_values`; the daily report shares one computation with the calibration
    fallback, and a flat book computes nothing). A positioned leg without a fair value is priced at its
    mark (``Position.mark``, then the latest SIG mid; listed in ``mark_fallback``). As in
    ``fairvalue.compute_all``, a major-party leg that fields no candidate is left out of the race and
    settles NO. ``simulate.simulate`` (``simulation.n_sims``, ``simulation.seed``) draws the outcomes; a
    control market settles on the simulated seat count reaching the majority. ``simulate.portfolio_pnl``
    then values each position against its cost in YES terms (``yes_avg_price``, else the mark), so the
    figures are what the book makes at settlement relative to what was paid; the same distribution
    relative to today's marks is shifted by ``-unrealised_at_mark``. Positions that cannot be simulated
    (unknown race, no price, a control market whose chamber has no simulated seats) are listed in
    ``unsimulated`` and left out of the figures. Never raises: a failure is reported in ``error``.
    """
    loss_frac = _loss_frac(cfg)
    threshold = loss_frac * float(cfg.bankroll_start)
    held = {p.exchange_id: p for p in session.execute(select(Position)).scalars().all() if p.quantity}
    out: dict[str, Any] = {
        "n_positions": len(held), "n_simulated": 0, "n_sims": 0, "basis": "cost",
        "mean": None, "p05": None, "p50": None, "p95": None, "p_loss": None,
        "loss_frac": loss_frac, "loss_threshold": threshold, "unrealised_at_mark": None,
        "mark_fallback": [], "unsimulated": [], "error": None,
    }
    if not held:
        return out
    try:
        import numpy as np

        from sigcup.trader import fairvalue
        from sigcup.trader import simulate as sim_mod

        fair = fair_values() if fair_values is not None else current_fair_values(session, now, cfg)
        quotes = latest_quotes(session)
        meta = fairvalue.load_races(session)
        candidates = dict(session.execute(select(Race.race_id, Race.candidates)).all())
        race_fairs: dict[str, dict[str, float]] = {}
        empty_legs: list[str] = []
        fallback: list[str] = []
        for race_id, info in sorted(meta.items()):
            if info.chamber == "control":
                continue
            roles = {eid: fairvalue._role(party) for party, eid in info.markets.items()}
            empty_roles = fairvalue._empty_major_roles(candidates.get(race_id), roles)
            for party, eid in info.markets.items():
                if roles[eid] in empty_roles:
                    empty_legs.append(eid)  # fields no candidate: cannot win the seat
                    continue
                fv = fair.get(eid)
                p = float(fv.fair) if fv is not None and fv.fair is not None else None
                if p is None and eid in held:
                    p = _mark_for(eid, held, quotes, None)
                    if p is not None:
                        fallback.append(eid)
                if p is not None:
                    race_fairs.setdefault(race_id, {})[party] = p
        n = int(cfg.simulation.n_sims)
        sim = sim_mod.simulate(race_fairs, meta, cfg, n=n, seed=cfg.simulation.seed)
        samples = dict(sim.samples)
        for eid in empty_legs:
            samples[eid] = np.zeros(sim.n, dtype=np.uint8)
        majority = {"senate": sim_mod.SENATE_MAJORITY, "house": sim_mod.HOUSE_MAJORITY}
        for info in meta.values():
            chamber = fairvalue.control_chamber(info) if info.chamber == "control" else None
            if chamber not in sim.seats:
                continue
            d_control = sim.seats[chamber] >= majority[chamber]
            for party, eid in info.markets.items():
                role = fairvalue._role(party)
                if role in ("D", "R"):
                    samples[eid] = (d_control if role == "D" else ~d_control).astype(np.uint8)
        book: dict[str, tuple[float, float | None]] = {}
        unrealised = 0.0
        for eid, p in sorted(held.items()):
            qty = int(p.quantity)
            mark = _mark_for(eid, held, quotes, None)
            basis = yes_avg_price(qty, p.avg_price)
            basis = basis if basis is not None else mark
            if eid not in samples or basis is None:
                out["unsimulated"].append({"exchange_id": eid, "quantity": qty, "at_risk": at_risk(qty, mark if mark is not None else basis)})
                continue
            book[eid] = (float(qty), float(basis))
            if mark is not None:
                unrealised += qty * (mark - basis)
        out["mark_fallback"] = sorted(e for e in fallback if e in book)
        out["n_simulated"] = len(book)
        out["n_sims"] = sim.n
        if book:
            pnl = sim_mod.portfolio_pnl(book, replace(sim, samples=samples))
            p05, p50, p95 = (float(x) for x in np.percentile(pnl, [5, 50, 95]))
            out.update(
                mean=float(pnl.mean()), p05=p05, p50=p50, p95=p95,
                p_loss=float(np.mean(pnl < -threshold)), unrealised_at_mark=float(unrealised),
            )
    except Exception as e:  # noqa: BLE001 - the report must render even if the model cannot run
        log.warning("portfolio P&L simulation unavailable: %s: %s", type(e).__name__, e)
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def clv_by_mode(session: Session) -> list[dict[str, Any]]:
    """CLV rows per strategy with a ``mode`` key: the live rows (and their ``all``) first, then paper.
    Paper fills (synthetic crosses at our resting price) never dilute the live scorecard."""
    return [{"mode": mode, **r} for mode, paper in (("live", False), ("paper", True)) for r in clvmod.aggregate_by_strategy(session, paper=paper)]


def daily(session: Session, now: datetime | None = None, cfg: StrategyConfig | None = None) -> dict[str, Any]:
    """The daily report as a plain dict (P&L by strategy, CLV and its per-strategy gate, calibration,
    rank history)."""
    now = ensure_utc(now or utc_now())
    cfg = cfg or config()
    clv_rows = clv_by_mode(session)
    gate_n = clvmod.gate_min_fills(cfg)
    fair_values = _shared_fair_values(session, now, cfg)  # computed at most once, only when needed
    return {
        "ts": now.isoformat(),
        "day": now.strftime("%Y-%m-%d"),
        "hours_to_deadline": hours_to(cfg.deadline_utc, now),
        "equity": equity_curve(session, now, cfg),
        "pnl": pnl_by_strategy(session),
        "clv": clv_rows,
        "clv_gate": clvmod.strategy_gate(clv_rows, gate_n),
        "clv_gate_min_fills": gate_n,
        "calibration": calibration(session, now, cfg=cfg, fair_values=fair_values),
        "portfolio_pnl": portfolio_risk(session, now, cfg, fair_values=fair_values),
        "sig_vs_ref": sig_vs_ref(session),
        "rank_history": rank_history(session, now),
        "exposure": exposure(session),
    }


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------
def _n(v: float | None, digits: int = 0, sign: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:{'+' if sign else ''}.{digits}f}"


def render_status(d: dict[str, Any]) -> Group:
    hb = Table(title=f"Heartbeats  ({d['ts'][:19]}Z, T-{d['hours_to_deadline']:.1f} h)")
    for col in ("process", "status", "age", "detail"):
        hb.add_column(col)
    for h in d["heartbeats"]:
        age = "-" if h["age_s"] is None else f"{h['age_s']:.0f} s"
        style = "red" if h["stale"] else "green"
        hb.add_row(h["process"], f"[{style}]{h['status']}[/{style}]", age, (h["detail"] or "")[:80])

    a = d["account"]
    lb = d["leaderboard"]
    acct = Table(title="Account")
    for col in ("cash", "reserved", "free", "positions", "total", "P&L", "rank", "members", "snap age"):
        acct.add_column(col, justify="right")
    acct.add_row(
        _n(a["cash"]), _n(a["reserved"]), _n(a["free_cash"]), _n(a["positions_value"]), _n(a["total"]),
        "-" if a["pnl"] is None else f"{a['pnl']:+.0f} ({a['pnl_frac']:+.1%})",
        _n(lb["rank"]), _n(lb["member_count"]), "-" if a["age_s"] is None else f"{a['age_s']:.0f} s",
    )

    oi = Table(title=f"Open orders  (acked={d['open_intents']['n_acked']} sent={d['open_intents']['n_sent']} uncertain={d['open_intents']['n_uncertain']})")
    for col in ("strategy", "orders", "shares", "notional", "paper", "live"):
        oi.add_column(col, justify="right" if col != "strategy" else "left")
    for k, v in d["open_intents"]["by_strategy"].items():
        oi.add_row(k, str(v["n"]), str(v["shares"]), _n(v["notional"]), str(v["paper"]), str(v["live"]))
    if not d["open_intents"]["by_strategy"]:
        oi.add_row("(none)", "", "", "", "", "")

    ex = d["exposure"]
    exp = Table(title=f"Exposure  (positions={ex['n_positions']} YES-equiv gross={ex['gross']:.0f} at risk={ex.get('at_risk', 0.0):.0f} net D-equivalent={ex['net_dem_equivalent']:+.0f})")
    for col in ("group", "key", "positions", "shares", "value", "gross", "at risk"):
        exp.add_column(col, justify="right" if col not in ("group", "key") else "left")
    for group in ("by_party", "by_chamber", "by_state"):
        for k, v in list(ex[group].items())[:12]:
            exp.add_row(group[3:], k, str(int(v["n"])), _n(v["shares"]), _n(v["value"], sign=True), _n(v["gross"]), _n(v.get("at_risk")))
    if not ex["positions"]:
        exp.add_row("(flat)", "", "", "", "", "", "")

    parts: list[Any] = [hb, acct, oi, exp]
    if d.get("alerts"):
        al = Table(title="Alerts", show_header=False)
        al.add_column("alert")
        for a_ in d["alerts"]:
            al.add_row(f"[yellow]{a_}[/yellow]")
        parts.append(al)
    if d["kill"]["kill_file_present"] or d["kill"]["kv_flag"]:
        parts.append(f"[bold red]KILL: file_present={d['kill']['kill_file_present']} kv={d['kill']['kv_flag']}[/bold red]")
    return Group(*parts)


def render_daily(d: dict[str, Any]) -> Group:
    eq = d["equity"]
    head = Table(title=f"Daily report {d['day']}  (T-{d['hours_to_deadline']:.0f} h)")
    for col in ("equity now", "start of day", "change today", "change total", "rank", "rank 24h", "members"):
        head.add_column(col, justify="right")
    rh = d["rank_history"]
    head.add_row(_n(eq["now"]), _n(eq["start_of_day"]), _n(eq["change_today"], sign=True), _n(eq["change_total"], sign=True), _n(rh["rank"]), _n(rh["rank_change_24h"], sign=True), _n(rh["member_count"]))

    pnl = Table(title="P&L by strategy (YES-equivalent average-cost ledger; marks from the exchange)")
    for col in ("mode", "strategy", "fills", "shares", "markets", "open", "realised", "unrealised", "fees", "total"):
        pnl.add_column(col, justify="right" if col not in ("mode", "strategy") else "left")
    for r in d["pnl"]:
        pnl.add_row(r["mode"], r["strategy"], str(r["n_fills"]), str(r["shares"]), str(r["markets"]), _n(r["open_shares"]), _n(r["realised"], 1, True), _n(r["unrealised"], 1, True), _n(r["fees"], 1), _n(r["total"], 1, True))
    if not d["pnl"]:
        pnl.add_row("(no fills yet)", *[""] * 9)

    cal = d["calibration"]
    basis = f"last {cal['window_days']} d, {cal['n_signals']} signals" if cal.get("source", "signals") == "signals" else f"current fair values, {cal['n_signals']} markets"
    ct = Table(title=f"Calibration ({basis}; fair-vs-ref n={cal['fair_vs_ref']['n']} mean gap={_n(cal['fair_vs_ref']['mean_gap'], 3, True)})")
    for col in ("strategy", "n", "mean edge", "mean |edge|", "size"):
        ct.add_column(col, justify="right" if col != "strategy" else "left")
    for k, v in cal["by_strategy"].items():
        ct.add_row(k, str(v["n"]), _n(v["mean_edge"], 3, True), _n(v["mean_abs_edge"], 3), str(v["size"]))
    for k, v in cal["buckets"].items():
        ct.add_row(f"fair {k}", str(v["n"]), f"fair {v['mean_fair']:.3f}", f"price {v['mean_price']:.3f}", "")
    if not cal["by_strategy"]:
        ct.add_row("(no signals yet)" if not cal.get("error") else f"(fair values unavailable: {cal['error'][:60]})", "", "", "", "")

    sr = d["sig_vs_ref"]
    st = Table(title=f"SIG vs reference mid (n={sr['n']} mean gap={_n(sr['mean_gap'], 3, True)} mean |gap|={_n(sr['mean_abs_gap'], 3)})")
    for col in ("race", "party", "SIG", "ref", "source", "gap"):
        st.add_column(col, justify="right" if col not in ("race", "party", "source") else "left")
    for g in sr["largest"]:
        st.add_row(str(g["race_id"]), str(g["party"]), f"{g['sig_mid']:.3f}", f"{g['ref_mid']:.3f}", g["source"], f"{g['gap']:+.3f}")
    if not sr["largest"]:
        st.add_row("(no crosswalked quotes yet)", "", "", "", "", "")

    pr = d.get("portfolio_pnl")
    pt: Table | None = None
    if pr is not None:
        pt = Table(
            title=f"Portfolio settlement P&L (correlated simulation, n={pr['n_sims']}; vs cost basis; fair values from the model, "
            f"marks where missing; today's unrealised {_n(pr['unrealised_at_mark'], 0, True)})"
        )
        loss_col = f"P(loss > {pr['loss_threshold']:.0f})"
        for col in ("positions", "simulated", "mean", "p5", "p50", "p95", loss_col):
            pt.add_column(col, justify="right")
        if pr["mean"] is not None:
            pt.add_row(
                str(pr["n_positions"]), str(pr["n_simulated"]), _n(pr["mean"], 0, True), _n(pr["p05"], 0, True),
                _n(pr["p50"], 0, True), _n(pr["p95"], 0, True), f"{pr['p_loss']:.1%}",
            )
        elif pr.get("error"):
            pt.add_row(f"(simulation unavailable: {pr['error'][:60]})", "", "", "", "", "", "")
        elif not pr["n_positions"]:
            pt.add_row("(no positions)", "", "", "", "", "", "")
        else:
            pt.add_row(str(pr["n_positions"]), "0", "-", "-", "-", "-", "-")
        if pr.get("unsimulated"):
            pt.add_row(f"not simulated: {', '.join(u['exchange_id'] for u in pr['unsimulated'][:6])}", "", "", "", "", "", "")

    rk = Table(title="Rank history")
    for col in ("day", "rank", "total", "members", "top 3"):
        rk.add_column(col, justify="right" if col != "day" else "left")
    for p in rh["points"][-14:]:
        rk.add_row(p["day"], _n(p["rank"]), _n(p["our_total"]), _n(p["member_count"]), ", ".join(f"{t:.0f}" for t in p["top3"]))
    if not rh["points"]:
        rk.add_row("(no leaderboard snapshots yet)", "", "", "", "")

    gate = clvmod.render_gate(d.get("clv_gate") or [], d.get("clv_gate_min_fills") or clvmod.CLV_GATE_MIN_FILLS)
    parts: list[Any] = [head, pnl, clvmod.render_table(d["clv"]), gate, ct]
    if pt is not None:
        parts.append(pt)
    return Group(*parts, st, rk)
