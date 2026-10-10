"""Supabase Realtime consumer for order-book pushes (POST /realtime/token -> private broadcast channels).

Responsibilities (owner: sig workstream):
- Token refresh before the 3-hour expiry; reconnect with full rejoin of channels.
- Subscribe to `tournament:{tid}:market:{mid}` for up to config.limits.realtime_channels markets plus `user:{profileId}`.
- Parse `market_batch` (books + trades, delivery.revision, marketSettled, resyncRequired), `book_dirty`,
  `market_settled`, `account_batch` (fills). Detect revision gaps and emit a resync request.
- Expose an async iterator of normalized events: BookUpdate(exchange_id, bids, asks, ts, revision),
  FillEvent(FillInfo), SettledEvent(exchange_id), ResyncNeeded(exchange_id|None).
The trader's book_state consumes these; REST reconciliation covers dropped batches.
"""

from __future__ import annotations

raise NotImplementedError("implemented in the sig workstream")
