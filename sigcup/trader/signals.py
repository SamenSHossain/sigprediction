"""Turns fair values, books, positions and config into the desired order set, tagged by strategy:
arb (from consistency), theta (favourites by yield), value (|fair-mid| > min_edge inside the band),
convex (toss-up/control sleeve), mm (optional), fade/follow (only when sentiment.enable_fade_follow and an
Event of the right kind is active), election (election-night taker rule). Returns DesiredOrder list with
priority and size BEFORE risk scaling."""

from __future__ import annotations

raise NotImplementedError("implemented in the model workstream")
