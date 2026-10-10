"""Hybrid objective: reads the latest LeaderboardSnap; in the final endgame.final_hours decides the convex
sleeve budget (sleeve_frac_if_reachable / sleeve_frac_if_placed) from rank and the gap to the
rank_target total; otherwise returns convex_sleeve.budget_frac.

Decision table (``sleeve_fraction``)::

    endgame disabled, or more than final_hours to the deadline  -> convex_sleeve.budget_frac
    no usable / stale leaderboard                               -> convex_sleeve.budget_frac
    rank <= rank_target ("placed")                              -> sleeve_frac_if_placed
    rank unknown and our total >= the rank_target total         -> sleeve_frac_if_placed
    (target total - ours) / ours <= reachable_gap_frac          -> sleeve_frac_if_reachable
    otherwise (out of reach)                                    -> convex_sleeve.budget_frac

The interim leaderboard marks positions at recent SIG trades (noisy, manipulable by others), so the
decision uses the snapshot's ``our_total`` when present and our own ``equity`` otherwise, and ignores a
snapshot older than ``MAX_LEADERBOARD_AGE``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from sigcup.common.log import get_logger
from sigcup.common.time import ensure_utc, hours_to
from sigcup.settings import StrategyConfig

__all__ = ["MAX_LEADERBOARD_AGE", "in_final_window", "sleeve_fraction", "target_total"]

log = get_logger(__name__)

MAX_LEADERBOARD_AGE = timedelta(hours=2)


def _get(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping):
            if obj.get(name) is not None:
                return obj[name]
            continue
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return None


def in_final_window(cfg: StrategyConfig, now: datetime) -> bool:
    """True inside the last ``endgame.final_hours`` before the deadline (and not past it)."""
    h = hours_to(cfg.deadline_utc, ensure_utc(now))
    return 0 < h <= cfg.endgame.final_hours


def target_total(leaderboard: Any, rank_target: int) -> float | None:
    """The total of the account currently holding ``rank_target`` (from ``top_totals``, best first)."""
    tops = _get(leaderboard, "top_totals", "topTotals", "totals")
    if not tops:
        return None
    try:
        vals = sorted((float(t) for t in tops if t is not None), reverse=True)
    except (TypeError, ValueError):
        return None
    if len(vals) < rank_target:
        return None
    return vals[rank_target - 1]


def sleeve_fraction(cfg: StrategyConfig, now: datetime, leaderboard: Any | None, equity: float) -> float:
    """Convex-sleeve budget as a fraction of the bankroll (see the module docstring)."""
    default = float(cfg.convex_sleeve.budget_frac)
    if not cfg.endgame.enabled or not in_final_window(cfg, now):
        return default
    if leaderboard is None:
        return default
    ts = _get(leaderboard, "ts", "timestamp")
    if isinstance(ts, datetime) and ensure_utc(now) - ensure_utc(ts) > MAX_LEADERBOARD_AGE:
        log.info("endgame: leaderboard snapshot stale (%s); keeping default sleeve", ts)
        return default
    rank_target = max(1, int(cfg.endgame.rank_target))
    rank = _get(leaderboard, "rank", "my_rank")
    try:
        rank_i = int(rank) if rank is not None else None
    except (TypeError, ValueError):
        rank_i = None
    if rank_i is not None and rank_i <= rank_target:
        log.info("endgame: rank %d <= target %d -> placed, sleeve %.2f", rank_i, rank_target, cfg.endgame.sleeve_frac_if_placed)
        return float(min(1.0, max(0.0, cfg.endgame.sleeve_frac_if_placed)))
    ours = _get(leaderboard, "our_total", "my_total")
    try:
        ours_f = float(ours) if ours is not None else float(equity)
    except (TypeError, ValueError):
        ours_f = float(equity)
    target = target_total(leaderboard, rank_target)
    if target is None or ours_f <= 0:
        return default
    gap = (target - ours_f) / ours_f
    if gap <= 0:
        # rank unknown (lenient leaderboard parse) but our total already matches the target's: placed
        log.info("endgame: total %.0f >= rank-%d total %.0f -> placed, sleeve %.2f", ours_f, rank_target, target, cfg.endgame.sleeve_frac_if_placed)
        return float(min(1.0, max(0.0, cfg.endgame.sleeve_frac_if_placed)))
    if gap <= cfg.endgame.reachable_gap_frac:
        log.info("endgame: gap %.3f to rank %d within reach -> sleeve %.2f", gap, rank_target, cfg.endgame.sleeve_frac_if_reachable)
        return float(min(1.0, max(0.0, cfg.endgame.sleeve_frac_if_reachable)))
    log.info("endgame: gap %.3f to rank %d out of reach -> default sleeve %.2f", gap, rank_target, default)
    return default
