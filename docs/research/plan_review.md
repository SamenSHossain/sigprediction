# Adversarial review of the implementation plan

_Research report produced 2026-10-10 by a web-research agent for the SIG Predictions Cup trading plan. Facts come from search-engine snippets and public repositories; items marked UNVERIFIED must be confirmed before relying on them._

# Review: SIG Cup plan (read 2026-10-10)

Facts from the research JSONL the plan under-weights: an Oct-7 audit found the tournament-scoped **relationship graph empty**; a fusion winner "counts for every party on that ticket"; runoffs wait for official results; `settlementDate` = 2026-11-04T17:00Z (noon **EST**, DST ends Nov 1); 502 `ORDER_STATUS_UNKNOWN`, 409 `REQUEST_IN_FLIGHT` with a 90-s key lease; writes took 14–36 s at the open; Realtime batches are sent once, ~100 channels/connection, joins 3.5–6 s each; final rank = balance after all settlements, interim leaderboard = SIG marks.

## 1. Arithmetic

**(a) Writes.** 27/min = 1,620/h. Cancel+replace = 2 writes; letting an order expire and re-placing = 1. Decisive unknown: does a ≤50-order batch count as one write? Test day 1; if yes, writes stop binding. Assuming 1 order = 1 write: the ~85 favorite legs (117 races, ~70% non-toss-up) are laid in ~4 min and rarely need re-quoting if the trigger is ≥2¢ with a 90-s per-market cooldown. The plan's 1¢ trigger will thrash on election night (Kalshi ticks 1¢ every few seconds → 429 → **all** writes frozen 60 s). Allocation per minute: 8 reserved for arb/event takes (bucket bursts to 25), 6 theta/longshot-NO placement and refresh (book fully re-laid every ~15 min), 8 re-quotes of mid-band value orders (≈8 markets at 1/min or 40 at 5 min — not 237), 3 reconciliation cancels. Hard ceiling 25/min sliding window. Two-sided `mm` on dozens of markets (4 writes per reprice) is unaffordable; the priority order should put it last.

**(b) Reads.** 90/min. Bulk prices every 15 s = 12; Realtime for ≤94 markets (the user channel counts) at zero reads — run 3 connections if the cap is per connection (verify) and drop bulk polling to 30 s; on-demand depth 30 (one market per 2 s; full sweep of the 143 non-Realtime markets ≈ 5 min); positions + open orders + fills every 60 s = 4; pnl/constraints every 5 min ≈ 1; leaderboard every 15 min; revision-gap resyncs ≈ 5. Total ≈ 55, ~35 headroom. This only holds with **one** token bucket; six pm2 processes each importing the adapter will each believe they own 90/27. Every SIG REST call belongs in one process.

**(c) Capital.** Theta at average 0.90 with true p ≈ 0.975 (not 1.00): per 3k position, 3,333 shares, +333 right / −3,000 wrong, EV ≈ +250 → 60k ≈ +5k (8%). P(≥1 upset in 20) = 40%; one upset → +3.3k, two → ≈0. Realistic terminal 104–106k. Two bugs make it worse: applying the 1.64 stretch to Kalshi turns 0.97 into Φ(1.64·1.88) = 0.999, and `bid = min(fair−1¢, ask)` then lifts favorites to 0.98 for a 1% yield with 2.5% ruin per leg. Use bid ≤ min(ask, fair/(1+min_yield)), min_yield 5–6%, rank by yield, cap fair at 0.985, and net the per-market cap across both legs (YES-favorite ≡ NO-longshot). Verified Dutch books without relationship collateral yield ~2% on 0.98 locked — dominated by theta; the plan's "nearly capital-free" rank depends on a day-1 `/relationships` check that one competitor already failed.

Reserve: idle cash earns 0 and favorites converge only upward, so deploying early is right. Election-night capturable stale liquidity is bounded by depth (DMM 20 shares/level), not cash — speculation: 5–15k per bot. Keep 20% liquid through Oct 31, rebuild to ~30% in the last 72 h by letting arbs/short orders roll off, not by selling theta into an 8-pt spread. But +5% does not place in a rank-based field of hundreds (speculation: top-3 needs ≥ +25%). If the prize is the objective, decide on day 2 whether a 20–30% convex sleeve (toss-ups and control contracts, the only SIG books with 4–5k shares at top) is funded now rather than in the last 72 h when spreads are worst and everyone piles in.

## 2. Under-specified risks

- **Exclusivity**: RI fusion (both party legs can resolve YES → NO+NO pays 0); NE/MT/RI-Gov independents (YES+YES pays 0); GA runoffs settle in December, N/A refunds "refundable cost". Gate Dutch books on the relationships API reporting `mutually_exclusive` + `isExhaustive`, never titles. Verify whether LA's 2026 closed-primary law removes the general runoff (speculation).
- **Deadline/DST**: a hardcoded UTC-4 closes you an hour early. House control is Freeform/manual and may be uncalled on Nov 4 — fine for final rank, dead capital for interim.
- **Order state machine** is absent: persist intent + idempotency key *before* POST; on timeout/502 mark UNCERTAIN and re-POST the same key (409 → wait out the lease, then poll `/orders` and `/fills`); on restart, reconcile UNCERTAIN intents before quoting; pm2 `max_restarts` so a crash loop's cancel-all + re-lay does not burn the budget; write timeout ≥ 45 s. Resting orders do not reserve cash, so keep an internal reservation ≤ cash. Use order expiry as a dead-man's switch (a sleeping laptop leaves quotes live).
- **Kill switch** as written only refuses to start; it must cancel-all, then stop.
- **Realtime**: handle revision gaps, `resyncRequired`, the 3-h token, and a reconnect that "rejoins nothing"; ~60 s to warm 94 channels.
- **Latency**: `model` every 5 min → SQLite → `executor` every 60 s cannot snipe stale quotes or act on a 10-s ref feed. Merge into one asyncio `trader` process; the 60-s loop becomes reconciliation. Three processes, one SQLite writer per table family, `busy_timeout`, `synchronous=NORMAL`, store quotes on change (6-s snapshots = 3.4M rows/day).
- **Leaderboard**: others' mark-painting cannot steal the settled prize, but makes the interim rank the endgame trigger relies on noisy by ±10% (speculation). Never use SIG prints as fair; election-night takes trigger on Kalshi ≥ 0.97 or settlement, not a SIG print.
- **Secrets/clock**: read-scope key for collectors/reporter, trade key only in `trader`, chmod 600, no headers in logs; derive server offset from the HTTP `Date` header before discarding "stale" books. A missing `tournamentId` routes orders to the public book — test it.

## 3. Scope

The Day-2 gate ("replay a recorded day") is impossible on Day 2. Minimum viable system (~4 days): SIG client + single limiter + idempotent OMS + reconciliation + kill switch; Kalshi/Polymarket collector + crosswalk; hand-edited ratings YAML as fallback fair; theta/longshot-NO bids ranked by yield; relationship-verified Dutch books; P&L/CLV by tag; election-night mode dry-run by Oct 28. Cut in this order: all social collectors and the Claude stance pipeline, lead-lag/IC, regime classifier, cohort-skew regression, FEC/VoteHub/RCP polls model, the 40k-sim Monte Carlo (replace with a 4-pt uniform-swing shift through T₅ plus state/party group caps), two-sided `mm`, the queue-aware paper adapter (dry-run logging plus cross-the-best fills).

## 4. Strategy

- The cohort-skew fade is mostly favorite-longshot bias in disguise: with safe-D seats dominating, (price − fair) regressed on party loads on favorite status. Test it on toss-ups only (~12 races), where a 3–5¢ tilt is indistinguishable from Kalshi's own partisan error (2022, 2024). Fold into `value`; drop the weekly basket.
- Sentiment before week 3: no. The IC gate cannot pass before ~Oct 25; real-money prices are a faster, cleaner event detector than posts, and the plan's own evidence says the Polymarket→SIG lead-lag was inconclusive. Put the honest analysis in STRATEGY.md; implement at most GDELT+RSS as a "news spike → widen/cancel" defensive flag once the base is stable.
- "Selling longshots adds the mirror" double-counts the favorite trade; an in-race NO+NO is theta plus an 8× overpriced hedge unless collateral is live.

## 5. Testing

Five that matter: (1) idempotent retry/restart — kill between intent-persist and response against a fixture returning 502 then the stored response; zero duplicates; (2) shared limiter under burst — concurrent tasks cannot exceed 25/90 in any sliding 60 s; (3) exclusivity gate — RI fusion, NE independent and a GA runoff market rejected without API relationship evidence; (4) deadline + kill switch — no POST after 2026-11-04T17:00Z; kill file → cancel-all within one cycle; (5) reconciliation invariants — fills ledger equals `/positions`, OMS intents equal `/orders?status=open`, reservation ≤ cash. Replace the replay gate with a live canary: one market, 10 shares, verify fill, cancel, a forced 429 recovery, then widen caps.

### Critical Files for Implementation
- /home/user/sigprediction/sigcup/execution/oms.py (intent persistence, idempotency state machine, single token bucket)
- /home/user/sigprediction/sigcup/execution/adapter_sigcup.py (timeouts, 409/502 handling, Realtime resync, tournamentId)
- /home/user/sigprediction/sigcup/execution/risk.py (yield-based theta pricing, per-race netting, cash reservation, kill switch = cancel-all)
- /home/user/sigprediction/sigcup/model/consistency.py (relationship-API exclusivity gate for fusion/independent/runoff races)
- /root/.claude/plans/i-am-trying-to-cached-sunrise.md (phasing and verification sections to revise)
