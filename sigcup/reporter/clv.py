"""Closing-line value per fill: price 24 h after the fill and the last price before settlement/close,
signed in our favour (buy YES at p, later price q -> q - p; sell YES -> p - q).

CLV is the process scorecard (practitioners_and_data.md: "grade yourself on process, not on which side
resolved"). Every fill is converted to its YES-equivalent (buy NO at p == sell YES at 1 - p) so one
formula covers all four side/action combinations::

    clv = direction * (later_yes_mid - yes_price)      direction = +1 long YES, -1 short YES

``compute(session, now)`` fills the ``clv`` table incrementally:

* ``price_24h``  the SIG mid as of ``fill.ts + 24 h``: the last Quote row at or before that moment
  (BookState writes a row only when the top of book changes, so on a quiet book that row can be
  hours or days old and is still the book's state). Only when no row exists at or before the target
  is the first quote within 1 h after it used. Computed only once the target has passed (never from a
  pass that runs before it); left null if the market has no quote by then.
* ``price_close`` the last quote at or before the market's close: its ``settlement_date`` when known,
  capped at the competition deadline (no order can be sent after it, so later prices are not our
  closing line); only computed once that moment has passed.

``aggregate_by_strategy(session)`` returns one plain dict per strategy tag (plus an ``all`` row) for the
daily report; ``render_table(rows)`` turns them into a rich table.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from rich.table import Table
from sqlalchemy import select
from sqlalchemy.orm import Session

from sigcup.common.time import ensure_utc
from sigcup.db.models import CLV, Fill, Market, Quote
from sigcup.settings import config

HORIZON = timedelta(hours=24)
WINDOW = timedelta(hours=1)
UNKNOWN_STRATEGY = "unknown"


# ---------------------------------------------------------------------------------------------
# fill arithmetic
# ---------------------------------------------------------------------------------------------
def quote_mid(q: Quote | None) -> float | None:
    """Mid of a Quote row (bid/ask mid, else last, else the one-sided quote); None when empty."""
    if q is None:
        return None
    if q.bid is not None and q.ask is not None:
        return (float(q.bid) + float(q.ask)) / 2.0
    if q.last is not None:
        return float(q.last)
    if q.bid is not None:
        return float(q.bid)
    if q.ask is not None:
        return float(q.ask)
    return None


def fill_direction(side: str | None, action: str | None) -> int:
    """+1 when the fill makes us longer YES (buy YES / sell NO), -1 otherwise."""
    s = (side or "yes").lower()
    a = (action or "buy").lower()
    return 1 if (s == "yes") == (a == "buy") else -1


def fill_yes_price(side: str | None, price: float) -> float:
    """The fill price expressed in YES terms (a NO fill at p is a YES trade at 1 - p)."""
    return float(price) if (side or "yes").lower() == "yes" else 1.0 - float(price)


def clv_value(side: str | None, action: str | None, price: float, later_mid: float | None) -> float | None:
    """Closing-line value of one fill, signed in our favour, in price units (SUSQies per share)."""
    if later_mid is None:
        return None
    return fill_direction(side, action) * (float(later_mid) - fill_yes_price(side, price))


# ---------------------------------------------------------------------------------------------
# quote lookups
# ---------------------------------------------------------------------------------------------
def nearest_quote(session: Session, exchange_id: str, target: datetime, window: timedelta = WINDOW) -> Quote | None:
    """The Quote row for ``exchange_id`` closest to ``target`` within +/- ``window`` (None if none)."""
    target = ensure_utc(target)
    rows = (
        session.execute(
            select(Quote).where(
                Quote.exchange_id == exchange_id,
                Quote.ts >= target - window,
                Quote.ts <= target + window,
            )
        )
        .scalars()
        .all()
    )
    best: Quote | None = None
    best_gap: timedelta | None = None
    for q in rows:
        if quote_mid(q) is None:
            continue
        gap = abs(ensure_utc(q.ts) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = q, gap
    return best


def last_quote_before(session: Session, exchange_id: str, cutoff: datetime) -> Quote | None:
    """The most recent Quote row for ``exchange_id`` at or before ``cutoff`` that carries a price."""
    cutoff = ensure_utc(cutoff)
    rows = (
        session.execute(
            select(Quote)
            .where(Quote.exchange_id == exchange_id, Quote.ts <= cutoff)
            .order_by(Quote.ts.desc(), Quote.id.desc())
            .limit(50)
        )
        .scalars()
        .all()
    )
    for q in rows:
        if quote_mid(q) is not None:
            return q
    return None


def quote_as_of(session: Session, exchange_id: str, target: datetime, window: timedelta = WINDOW) -> Quote | None:
    """The book as of ``target``: the last priced Quote at or before it; when there is none, the first
    priced Quote within ``window`` after it (None if neither exists)."""
    q = last_quote_before(session, exchange_id, target)
    if q is not None:
        return q
    target = ensure_utc(target)
    rows = (
        session.execute(
            select(Quote)
            .where(Quote.exchange_id == exchange_id, Quote.ts > target, Quote.ts <= target + window)
            .order_by(Quote.ts, Quote.id)
        )
        .scalars()
        .all()
    )
    return next((r for r in rows if quote_mid(r) is not None), None)


def market_close(session: Session, exchange_id: str, deadline: datetime, cache: dict[str, datetime] | None = None) -> datetime:
    """When a market's closing line is fixed: its settlement_date if known (capped at the trading
    deadline, after which no order can be sent), else the deadline."""
    if cache is not None and exchange_id in cache:
        return cache[exchange_id]
    close = ensure_utc(deadline)
    m = session.execute(select(Market).where(Market.exchange_id == exchange_id)).scalars().first()
    if m is not None and m.settlement_date is not None:
        close = min(close, ensure_utc(m.settlement_date))
    if cache is not None:
        cache[exchange_id] = close
    return close


# ---------------------------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------------------------
@dataclass
class ComputeStats:
    """What one ``compute`` pass did."""

    examined: int = 0
    created: int = 0
    updated_24h: int = 0
    updated_close: int = 0
    pending: int = 0  # fills still waiting for a quote (window not closed yet)


def _pending_rows(session: Session) -> list[tuple[Fill, CLV | None]]:
    """Fills without a CLV row, or whose 24 h / close price is still null."""
    stmt = (
        select(Fill, CLV)
        .outerjoin(CLV, CLV.fill_id == Fill.fill_id)
        .where((CLV.fill_id.is_(None)) | (CLV.price_24h.is_(None)) | (CLV.price_close.is_(None)))
        .order_by(Fill.ts)
    )
    return [(f, c) for f, c in session.execute(stmt).all()]


def compute(
    session: Session,
    now: datetime,
    *,
    horizon: timedelta = HORIZON,
    window: timedelta = WINDOW,
    deadline: datetime | None = None,
) -> ComputeStats:
    """Create/complete CLV rows for every fill that still lacks a 24 h or closing price.

    Idempotent and incremental: call it hourly. ``now`` decides which prices can exist yet.
    """
    now = ensure_utc(now)
    deadline = ensure_utc(deadline or config().deadline_utc)
    stats = ComputeStats()
    close_cache: dict[str, datetime] = {}
    for fill, row in _pending_rows(session):
        stats.examined += 1
        ts = ensure_utc(fill.ts)
        if row is None:
            row = CLV(fill_id=fill.fill_id, computed_at=now)
            session.add(row)
            stats.created += 1
        waiting = False

        # -- 24 h price ---------------------------------------------------------------------
        if row.price_24h is None:
            target = ts + horizon
            if now >= target:
                q = quote_as_of(session, fill.exchange_id, target, window)
                mid = quote_mid(q)
                if mid is not None:
                    row.price_24h = mid
                    row.clv_24h = clv_value(fill.side, fill.action, fill.price, mid)
                    stats.updated_24h += 1
                elif now <= target + window:
                    waiting = True
            else:
                waiting = True

        # -- closing price ------------------------------------------------------------------
        if row.price_close is None:
            close = market_close(session, fill.exchange_id, deadline, close_cache)
            if now >= close:
                q = last_quote_before(session, fill.exchange_id, close)
                mid = quote_mid(q)
                if mid is not None:
                    row.price_close = mid
                    row.clv_close = clv_value(fill.side, fill.action, fill.price, mid)
                    stats.updated_close += 1
            else:
                waiting = True
        row.computed_at = now
        if waiting:
            stats.pending += 1
    session.flush()
    return stats


# ---------------------------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------------------------
def _mean(xs: Iterable[float]) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _summary(pairs: list[tuple[Fill, CLV | None]]) -> dict:
    """One aggregate row from (fill, clv) pairs."""
    n = len(pairs)
    shares = sum(int(f.quantity) for f, _ in pairs)
    h24 = [(f, c) for f, c in pairs if c is not None and c.clv_24h is not None]
    close = [(f, c) for f, c in pairs if c is not None and c.clv_close is not None]

    def weighted(items: list[tuple[Fill, CLV]], attr: str) -> float | None:
        q = sum(int(f.quantity) for f, _ in items)
        return sum(float(getattr(c, attr)) * int(f.quantity) for f, c in items) / q if q else None

    return {
        "n_fills": n,
        "shares": shares,
        "n_24h": len(h24),
        "clv_24h_mean": _mean(float(c.clv_24h) for _, c in h24),
        "clv_24h_weighted": weighted(h24, "clv_24h"),
        "clv_24h_total": sum(float(c.clv_24h) * int(f.quantity) for f, c in h24),
        "hit_24h": (sum(1 for _, c in h24 if float(c.clv_24h) > 0) / len(h24)) if h24 else None,
        "n_close": len(close),
        "clv_close_mean": _mean(float(c.clv_close) for _, c in close),
        "clv_close_weighted": weighted(close, "clv_close"),
        "clv_close_total": sum(float(c.clv_close) * int(f.quantity) for f, c in close),
        "hit_close": (sum(1 for _, c in close if float(c.clv_close) > 0) / len(close)) if close else None,
    }


def aggregate_by_strategy(session: Session, *, paper: bool | None = None) -> list[dict]:
    """CLV summary rows per strategy tag (sorted), followed by an ``all`` row.

    ``paper`` filters fills to paper (True) or live (False); None pools both. Fills without a
    strategy tag (adopted orphans) are reported under ``unknown``.
    """
    stmt = select(Fill, CLV).outerjoin(CLV, CLV.fill_id == Fill.fill_id)
    if paper is not None:
        stmt = stmt.where(Fill.paper.is_(bool(paper)))
    pairs = [(f, c) for f, c in session.execute(stmt).all()]
    groups: dict[str, list[tuple[Fill, CLV | None]]] = {}
    for f, c in pairs:
        groups.setdefault(str(f.strategy or UNKNOWN_STRATEGY), []).append((f, c))
    rows = [{"strategy": k, **_summary(v)} for k, v in sorted(groups.items())]
    if pairs:
        rows.append({"strategy": "all", **_summary(pairs)})
    return rows


def _fmt(v: float | None, pct: bool = False, digits: int = 3) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.0f}%" if pct else f"{v:+.{digits}f}"


def render_table(rows: list[dict], title: str = "Closing-line value by strategy") -> Table:
    """Rich table for ``aggregate_by_strategy`` rows (CLV in price units; totals in SUSQies). Rows that
    carry a ``mode`` key (paper / live, as the daily report builds them) get a leading mode column."""
    t = Table(title=title, expand=False)
    with_mode = any("mode" in r for r in rows)
    cols = (("mode",) if with_mode else ()) + ("strategy", "fills", "shares", "n24h", "clv24h", "wclv24h", "sum24h", "hit24h", "nclose", "clvclose", "sumclose", "hitclose")
    for col in cols:
        t.add_column(col, justify="left" if col in ("mode", "strategy") else "right")
    for r in rows:
        t.add_row(
            *((str(r.get("mode", "-")),) if with_mode else ()),
            str(r["strategy"]),
            str(r["n_fills"]),
            str(r["shares"]),
            str(r["n_24h"]),
            _fmt(r["clv_24h_mean"]),
            _fmt(r["clv_24h_weighted"]),
            _fmt(r["clv_24h_total"], digits=1),
            _fmt(r["hit_24h"], pct=True),
            str(r["n_close"]),
            _fmt(r["clv_close_mean"]),
            _fmt(r["clv_close_total"], digits=1),
            _fmt(r["hit_close"], pct=True),
        )
    if not rows:
        t.add_row("(no fills yet)", *[""] * (len(cols) - 1))
    return t
