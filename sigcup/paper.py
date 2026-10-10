"""PaperAdapter: implements ExchangeAdapter without touching the exchange.

- Wraps a read-only live client (or recorded quotes from the DB) for market data.
- Orders rest in memory; a buy fills when its price >= best ask (sell when price <= best bid) at the
  resting price; partial fills honour displayed size; expiry honoured.
- Tracks cash, positions, fills with ids `paper:<n>`; cancel/cancel_all supported.
- Persists intents/fills with paper=True so reports can separate paper from live.
"""

from __future__ import annotations

raise NotImplementedError("implemented in the execution workstream")
