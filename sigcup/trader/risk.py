"""Sizing and limits.

- kelly_fraction(fair, price, side) for a binary contract; fractional multipliers per strategy.
- Caps: per race (netted across party legs: +YES on D == -YES on R economically), per state, reserve
  (liquid_reserve_frac, higher in the final 72 h), max position shares.
- Stress: uniform national swing of risk.national_swing_pts through the T-distribution fair-value map must
  cost <= max_swing_loss_frac of bankroll; a candidate order set is rejected or scaled down otherwise.
- Theta pricing: bid <= min(ask, min(fair, theta.fair_cap) / (1 + theta.min_yield)).
- Daily loss limit evaluation.
"""

from __future__ import annotations

raise NotImplementedError("implemented in the execution workstream")
