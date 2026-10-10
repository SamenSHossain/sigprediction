"""Order management: desired orders -> exchange, with an idempotency state machine and a write budget.

Rules (owner: execution workstream):
- Persist OrderIntent(NEW) BEFORE sending; SENT while in flight; ACKED on 2xx; UNCERTAIN on timeout/502;
  on UNCERTAIN re-POST the SAME idempotency key (409 -> wait out the lease, then poll /orders and /fills).
- On startup: reconcile every UNCERTAIN/SENT intent against open orders and fills before quoting anything.
- Reconcile every config.cadence.reconcile_s: fills ledger == positions, intents == open orders,
  reservation <= cash; log and repair discrepancies.
- Write budget per minute: priority 0 arb/election takes (burst), 1 theta, 2 value, 3 mm, 4 housekeeping;
  re-quote only when |new - old| >= value.requote_threshold with a per-market cooldown; prefer letting an
  order expire over cancel+replace (1 write vs 2).
- Every order gets expires_at = now + risk.order_expiry_s; nothing is sent after config.deadline_utc.
- Kill switch: if the kill file exists or the daily-loss limit trips -> cancel_all, mark state, stop.
- Internal cash reservation for resting buys (exchange does not reserve).
"""

from __future__ import annotations

raise NotImplementedError("implemented in the execution workstream")
