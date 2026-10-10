"""Lead-lag cross-correlation of feature changes vs SIG/reference price changes at 1-24 h lags and the
information coefficient (Spearman of z-score vs next-day price change) per platform.

This is the gate for the ``fade`` / ``follow`` strategies (literature.md section on sentiment): they stay
disabled until the measured IC is positive on at least a week of data. Everything here is descriptive
and tolerant of sparse data: missing platforms, races without a Democratic leg, hours without quotes
and tiny samples all produce ``None`` entries rather than exceptions.

Series construction
* **signal**: ``Feature.z_score`` per (platform, race, day). The day-``D`` value is time-stamped at the
  END of day D (``D+1 00:00 UTC``) so a lag-``k`` test never lets the signal see posts that were not yet
  written (no look-ahead). Hourly changes ``dz`` are therefore non-zero only at day boundaries, which is
  exactly where a daily signal can move a price.
* **price**: the race's Democratic leg (P(D wins) rises when ``net_stance_dem`` rises); when a race has
  only a Republican leg its YES price is used with the sign flipped. SIG mids come from ``Quote``
  (bid/ask mid, else last), reference mids from ``RefPrice`` via the market's Kalshi / Polymarket keys.
  Both are resampled to hourly closes (last value in the hour, forward-filled) and differenced.
* **cross-correlation** at lag ``k`` = Pearson corr of ``dz(t)`` with ``dp(t + k)`` pooled over races.
* **IC** = Spearman rho of the day-``D`` z-score vs the day ``D+1`` close-to-close price change (daily
  closes are the last quote at or before each 00:00 UTC), pooled over races per platform.

``compute(session, now)`` returns a plain dict; ``render(result)`` turns it into rich tables.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from rich.console import Group
from rich.table import Table
from scipy import stats
from sqlalchemy import select
from sqlalchemy.orm import Session

from sigcup.common.time import ensure_utc
from sigcup.db.models import Feature, Market, Quote, RefPrice
from sigcup.reporter.clv import quote_mid

MAX_LAG_H = 24
MIN_PAIRS = 5  # fewer aligned pairs than this: report None
DEFAULT_DAYS = 14

Point = tuple[datetime, float]


# ---------------------------------------------------------------------------------------------
# pure numerics
# ---------------------------------------------------------------------------------------------
def pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation; None when fewer than 3 points or either side is constant."""
    if len(x) != len(y) or len(x) < 3:
        return None
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        if len(a) < 3:
            return None
    if a.std() == 0 or b.std() == 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return r if math.isfinite(r) else None


def spearman(x: Sequence[float], y: Sequence[float]) -> tuple[float, float] | None:
    """(rho, p-value) of Spearman's rank correlation; None when undefined (n < 3 or constant input)."""
    if len(x) != len(y) or len(x) < 3:
        return None
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    res = stats.spearmanr(a, b)
    rho, p = float(res.statistic), float(res.pvalue)  # type: ignore[attr-defined]
    if not math.isfinite(rho):
        return None
    return rho, (p if math.isfinite(p) else 1.0)


def cross_correlation(x: Sequence[float], y: Sequence[float], max_lag: int = MAX_LAG_H, min_pairs: int = MIN_PAIRS) -> dict[int, dict[str, Any]]:
    """``{lag: {"r": corr(x[t], y[t+lag]), "n": pairs}}`` for lag 1..max_lag (x leading y).

    Designed for one aligned pair of equally spaced series; ``pooled_cross_correlation`` handles many.
    """
    return pooled_cross_correlation([(list(x), list(y))], max_lag=max_lag, min_pairs=min_pairs)


def pooled_cross_correlation(
    pairs: Iterable[tuple[Sequence[float], Sequence[float]]],
    max_lag: int = MAX_LAG_H,
    min_pairs: int = MIN_PAIRS,
) -> dict[int, dict[str, Any]]:
    """Cross-correlation per lag with (x, y) points pooled across several aligned series."""
    pairs = [(np.asarray(x, dtype=float), np.asarray(y, dtype=float)) for x, y in pairs]
    out: dict[int, dict[str, Any]] = {}
    for lag in range(1, max_lag + 1):
        xs: list[float] = []
        ys: list[float] = []
        for x, y in pairs:
            n = min(len(x), len(y))
            if n <= lag:
                continue
            xs.extend(x[: n - lag].tolist())
            ys.extend(y[lag:n].tolist())
        r = pearson(xs, ys) if len(xs) >= min_pairs else None
        out[lag] = {"r": r, "n": len(xs)}
    return out


def best_lag(xcorr: dict[int, dict[str, Any]]) -> int | None:
    """The lag with the largest positive correlation (None when nothing is defined)."""
    cands = [(v["r"], k) for k, v in xcorr.items() if v.get("r") is not None]
    if not cands:
        return None
    r, k = max(cands)
    return k if r > 0 else None


def floor_hour(ts: datetime) -> datetime:
    return ensure_utc(ts).replace(minute=0, second=0, microsecond=0)


def floor_day(ts: datetime) -> datetime:
    return ensure_utc(ts).replace(hour=0, minute=0, second=0, microsecond=0)


def resample(points: Iterable[Point], start: datetime, end: datetime, step: timedelta) -> list[float | None]:
    """Bucket (ts, value) points on a grid ``start, start+step, ...`` up to ``end`` (exclusive).

    Each slot takes the LAST value observed at or before its end; slots without any observation yet
    are ``None`` (no back-filling), later slots are forward-filled.
    """
    start, end = ensure_utc(start), ensure_utc(end)
    n = max(0, int((end - start) / step))
    pts = sorted(((ensure_utc(t), float(v)) for t, v in points if v is not None), key=lambda p: p[0])
    out: list[float | None] = []
    i = 0
    last: float | None = None
    for slot in range(n):
        slot_end = start + step * (slot + 1)
        while i < len(pts) and pts[i][0] < slot_end:
            last = pts[i][1]
            i += 1
        out.append(last)
    return out


def diff(series: Sequence[float | None]) -> list[float]:
    """First differences; a missing neighbour yields 0 (no change observed)."""
    out: list[float] = []
    for a, b in zip(series[:-1], series[1:], strict=False):
        out.append(float(b) - float(a) if a is not None and b is not None else 0.0)
    return out


# ---------------------------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------------------------
def race_legs(session: Session) -> dict[str, tuple[Market, int]]:
    """race_id -> (market, sign): the Democratic leg (+1) or, failing that, the Republican leg (-1)."""
    out: dict[str, tuple[Market, int]] = {}
    for m in session.execute(select(Market)).scalars().all():
        party = (m.party or "").upper()
        if party == "D":
            out[m.race_id] = (m, 1)
        elif party == "R" and m.race_id not in out:
            out[m.race_id] = (m, -1)
    return out


def sig_points(session: Session, exchange_id: str, start: datetime, end: datetime) -> list[Point]:
    rows = (
        session.execute(
            select(Quote).where(Quote.exchange_id == exchange_id, Quote.ts >= start, Quote.ts < end).order_by(Quote.ts, Quote.id)
        )
        .scalars()
        .all()
    )
    pts: list[Point] = []
    for q in rows:
        mid = quote_mid(q)
        if mid is not None:
            pts.append((ensure_utc(q.ts), mid))
    return pts


def _ref_mid(r: RefPrice) -> float | None:
    if r.yes_mid is not None:
        return float(r.yes_mid)
    if r.yes_bid is not None and r.yes_ask is not None:
        return (float(r.yes_bid) + float(r.yes_ask)) / 2.0
    return None


def ref_points(session: Session, market: Market, start: datetime, end: datetime) -> list[Point]:
    """Reference mids for a market from the source with the most rows in the window."""
    conds = []
    if market.market_id:
        conds.append(RefPrice.market_id == market.market_id)
    if market.kalshi_ticker:
        conds.append((RefPrice.source == "kalshi") & (RefPrice.key == market.kalshi_ticker))
    if market.polymarket_condition_id:
        conds.append((RefPrice.source == "polymarket") & (RefPrice.key == market.polymarket_condition_id))
    if not conds:
        return []
    cond = conds[0]
    for c in conds[1:]:
        cond = cond | c
    rows = session.execute(select(RefPrice).where(cond, RefPrice.ts >= start, RefPrice.ts < end).order_by(RefPrice.ts)).scalars().all()
    by_source: dict[str, list[Point]] = {}
    for r in rows:
        mid = _ref_mid(r)
        if mid is not None:
            by_source.setdefault(r.source, []).append((ensure_utc(r.ts), mid))
    if not by_source:
        return []
    return max(by_source.values(), key=len)


def feature_rows(session: Session, start_day: str) -> dict[str, dict[str, dict[str, float]]]:
    """platform -> race_id -> day -> z_score (rows without a z-score are skipped)."""
    rows = session.execute(select(Feature).where(Feature.day >= start_day)).scalars().all()
    out: dict[str, dict[str, dict[str, float]]] = {}
    for f in rows:
        if f.z_score is None:
            continue
        out.setdefault(f.platform, {}).setdefault(f.race_id, {})[f.day] = float(f.z_score)
    return out


def _day_end(day: str) -> datetime:
    return floor_day(datetime.fromisoformat(day + "T00:00:00+00:00")) + timedelta(days=1)


# ---------------------------------------------------------------------------------------------
# compute
# ---------------------------------------------------------------------------------------------
def compute(session: Session, now: datetime, *, days: int = DEFAULT_DAYS, max_lag_h: int = MAX_LAG_H) -> dict[str, Any]:
    """Lead-lag and IC per platform over the trailing ``days``; never raises on sparse data."""
    now = ensure_utc(now)
    end = floor_hour(now)
    start = floor_day(end - timedelta(days=days))
    hours = int((end - start) / timedelta(hours=1))
    start_day = start.strftime("%Y-%m-%d")
    feats = feature_rows(session, start_day)
    legs = race_legs(session)
    result: dict[str, Any] = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": days,
        "max_lag_h": max_lag_h,
        "n_platforms": len(feats),
        "platforms": {},
    }
    if hours < 2:
        return result

    # price series are shared across platforms: cache per race
    price_cache: dict[str, dict[str, Any]] = {}

    def race_prices(race_id: str) -> dict[str, Any] | None:
        if race_id in price_cache:
            return price_cache[race_id]
        leg = legs.get(race_id)
        if leg is None:
            price_cache[race_id] = {}
            return None
        market, sign = leg
        sig = sig_points(session, market.exchange_id, start, end)
        ref = ref_points(session, market, start, end)
        sig_h = resample(sig, start, end, timedelta(hours=1))
        ref_h = resample(ref, start, end, timedelta(hours=1))
        entry = {
            "sign": sign,
            "sig_dp": [sign * d for d in diff(sig_h)] if sig else [],
            "ref_dp": [sign * d for d in diff(ref_h)] if ref else [],
            "sig_daily": _daily_closes(sig, start, end, sign) if sig else {},
            "ref_daily": _daily_closes(ref, start, end, sign) if ref else {},
        }
        price_cache[race_id] = entry
        return entry

    for platform, races in sorted(feats.items()):
        sig_pairs: list[tuple[list[float], list[float]]] = []
        ref_pairs: list[tuple[list[float], list[float]]] = []
        ic_sig: tuple[list[float], list[float]] = ([], [])
        ic_ref: tuple[list[float], list[float]] = ([], [])
        n_features = 0
        n_races_used = 0
        for race_id, by_day in races.items():
            n_features += len(by_day)
            prices = race_prices(race_id)
            if not prices:
                continue
            n_races_used += 1
            z_points: list[Point] = [(_day_end(day), z) for day, z in by_day.items()]
            z_h = resample(z_points, start, end, timedelta(hours=1))
            dz = diff(z_h)
            if prices["sig_dp"]:
                sig_pairs.append((dz, prices["sig_dp"]))
            if prices["ref_dp"]:
                ref_pairs.append((dz, prices["ref_dp"]))
            for day, z in by_day.items():
                nxt = (datetime.fromisoformat(day) + timedelta(days=1)).strftime("%Y-%m-%d")
                for daily, acc in ((prices["sig_daily"], ic_sig), (prices["ref_daily"], ic_ref)):
                    change = daily.get(nxt)
                    if change is not None:
                        acc[0].append(z)
                        acc[1].append(change)
        xs = pooled_cross_correlation(sig_pairs, max_lag=max_lag_h) if sig_pairs else {}
        xr = pooled_cross_correlation(ref_pairs, max_lag=max_lag_h) if ref_pairs else {}
        result["platforms"][platform] = {
            "n_features": n_features,
            "n_races": len(races),
            "n_races_with_prices": n_races_used,
            "xcorr_sig": xs,
            "xcorr_ref": xr,
            "best_lag_sig": best_lag(xs),
            "best_lag_ref": best_lag(xr),
            "ic_sig": _ic(ic_sig),
            "ic_ref": _ic(ic_ref),
        }
    return result


def _daily_closes(points: list[Point], start: datetime, end: datetime, sign: int) -> dict[str, float]:
    """day -> signed close-to-close change for that day (close = last price at or before day end)."""
    n_days = int((end - start) / timedelta(days=1))
    closes = resample(points, start, start + timedelta(days=n_days), timedelta(days=1))
    out: dict[str, float] = {}
    for i in range(1, len(closes)):
        a, b = closes[i - 1], closes[i]
        if a is None or b is None:
            continue
        day = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        out[day] = sign * (b - a)
    return out


def _ic(acc: tuple[list[float], list[float]]) -> dict[str, Any]:
    z, dp = acc
    res = spearman(z, dp) if len(z) >= MIN_PAIRS else None
    return {"rho": res[0] if res else None, "p": res[1] if res else None, "n": len(z)}


# ---------------------------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------------------------
def _f(v: float | None, digits: int = 3) -> str:
    return "-" if v is None else f"{v:+.{digits}f}"


def render(result: dict[str, Any]) -> Group:
    """Rich renderable: one IC table and one cross-correlation table (lags as rows) per platform."""
    parts: list[Any] = []
    ic = Table(title=f"Information coefficient (Spearman z vs next-day change) {result.get('start', '')[:10]} .. {result.get('end', '')[:10]}")
    for col in ("platform", "features", "races", "priced", "IC sig", "p", "n", "IC ref", "p", "n", "best lag sig", "best lag ref"):
        ic.add_column(col, justify="right" if col != "platform" else "left")
    platforms: dict[str, Any] = result.get("platforms", {})
    for name, p in platforms.items():
        ic.add_row(
            name,
            str(p["n_features"]),
            str(p["n_races"]),
            str(p["n_races_with_prices"]),
            _f(p["ic_sig"]["rho"]),
            _f(p["ic_sig"]["p"], 2),
            str(p["ic_sig"]["n"]),
            _f(p["ic_ref"]["rho"]),
            _f(p["ic_ref"]["p"], 2),
            str(p["ic_ref"]["n"]),
            "-" if p["best_lag_sig"] is None else f"{p['best_lag_sig']} h",
            "-" if p["best_lag_ref"] is None else f"{p['best_lag_ref']} h",
        )
    if not platforms:
        ic.add_row("(no feature rows yet)", *[""] * 11)
    parts.append(ic)
    if platforms:
        xc = Table(title="Cross-correlation of signal changes vs later price changes (r, n)")
        xc.add_column("lag", justify="right")
        for name in platforms:
            xc.add_column(f"{name} sig", justify="right")
            xc.add_column(f"{name} ref", justify="right")
        max_lag = int(result.get("max_lag_h", MAX_LAG_H))
        for lag in range(1, max_lag + 1):
            cells = [f"{lag} h"]
            for p in platforms.values():
                for key in ("xcorr_sig", "xcorr_ref"):
                    v = p[key].get(lag) if p[key] else None
                    cells.append("-" if v is None or v.get("r") is None else f"{v['r']:+.2f} ({v['n']})")
            xc.add_row(*cells)
        parts.append(xc)
    return Group(*parts)
