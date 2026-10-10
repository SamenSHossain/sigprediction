"""Probability <-> margin math shared by fair value and risk. Pure functions, no I/O."""

from __future__ import annotations

from scipy.stats import t as student_t

from sigcup.settings import StrategyConfig


def sigma_for(chamber: str, cfg: StrategyConfig) -> float:
    return cfg.fairvalue.sigma_pts.get(chamber, cfg.fairvalue.sigma_pts.get("senate", 7.0))


def prob_from_margin(margin_pts: float, chamber: str, cfg: StrategyConfig) -> float:
    """P(candidate wins) given an expected margin in points and a chamber-specific error scale."""
    return float(student_t.cdf(margin_pts / sigma_for(chamber, cfg), df=cfg.fairvalue.t_df))


def margin_from_prob(p: float, chamber: str, cfg: StrategyConfig) -> float:
    p = min(0.999, max(0.001, p))
    return float(student_t.ppf(p, df=cfg.fairvalue.t_df) * sigma_for(chamber, cfg))


def shift_probability(p: float, swing_pts: float, chamber: str, cfg: StrategyConfig) -> float:
    """Probability after a uniform margin shift of `swing_pts` (positive = toward the YES side)."""
    return prob_from_margin(margin_from_prob(p, chamber, cfg) + swing_pts, chamber, cfg)


def implied_margin_shift(p_before: float, p_after: float, chamber: str, cfg: StrategyConfig) -> float:
    """Margin change (points) that a price move implies; used to judge over-reaction."""
    return margin_from_prob(p_after, chamber, cfg) - margin_from_prob(p_before, chamber, cfg)
