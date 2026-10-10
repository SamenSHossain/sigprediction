"""Dutch-book scanner: YES+NO on one market, YES-sum / NO-sum across the legs of a race, and control vs
seat implied bounds. Only races whose exclusivity+exhaustiveness is confirmed by the relationships API
(Race.exclusive and Race.exhaustive True) are eligible when arb.require_api_exclusivity is set. Returns
ArbOpportunity(legs=[OrderRequest...], profit_per_set, max_sets_by_depth)."""

from __future__ import annotations

raise NotImplementedError("implemented in the model workstream")
