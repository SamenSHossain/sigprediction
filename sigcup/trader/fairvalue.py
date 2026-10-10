"""Fair value per market: precision-weighted blend of (a) raw real-money reference prices (Kalshi,
Polymarket; weight scaled by liquidity; NO favourite-longshot stretch applied), (b) polls+fundamentals
P(win) = T_df(margin / sigma_chamber) with the poll average shrunk toward a fundamentals prior (PVI,
generic ballot, incumbency), (c) ratings-implied probability. Returns FairValue(exchange_id, fair,
components, uncertainty). Also exposes implied_margin_shift(fair_before, fair_after, chamber) used by
the regime classifier."""

from __future__ import annotations

raise NotImplementedError("implemented in the model workstream")
