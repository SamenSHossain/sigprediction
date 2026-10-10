"""In-memory view of the exchange: best bid/ask/last per exchange_id, depth cache with timestamps,
our positions/open orders, cash and reservation. Updated by Realtime pushes and REST polls; the OMS
and the signal engine read from here, never from the DB, during a cycle. Writes Quote rows on change."""

from __future__ import annotations

raise NotImplementedError("implemented in the execution workstream")
