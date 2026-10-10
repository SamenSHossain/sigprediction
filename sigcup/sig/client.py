"""Live client for the documented SIG v1 API (https://sig.thesuper.market/api/v1/docs).

Responsibilities (owner: sig workstream):
- httpx.AsyncClient with Bearer auth, base URL from settings, read/write timeouts from config.limits.
- Two shared limiters (reads 90/min, writes 25/min) wrapping EVERY call; 429 -> freeze both for Retry-After.
- Records the server clock offset from the HTTP Date header (sigcup.common.time).
- Every write carries `tournamentId` and `idempotencyKey`; raises AuthError if the tournament id is unknown.
- Error classification per sigcup.sig.errors; OrderResult.uncertain=True on timeout/502.
- Cursor pagination helper for list endpoints (pagination.nextCursor / hasMore).
- Never logs headers or keys.
Implements sigcup.sig.adapter.ExchangeAdapter.
"""

from __future__ import annotations

raise NotImplementedError("implemented in the sig workstream")
