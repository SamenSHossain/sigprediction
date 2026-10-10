"""The `trader` process entry point (asyncio). Owns the ONLY exchange client in the system.
Cycle every cadence.model_s: refresh books (Realtime + bulk REST), load fair values, run consistency +
signals, apply risk, hand desired orders to the OMS; reconcile every cadence.reconcile_s; snapshot the
leaderboard every cadence.leaderboard_s; heartbeat. `sigcup trade [--paper]` calls run()."""

from __future__ import annotations

raise NotImplementedError("implemented in the execution workstream")
