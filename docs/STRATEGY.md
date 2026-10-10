# sigcup strategy memo: SIG Predictions Cup, 2026 Midterm Elections Competition

_Status: approved strategy as implemented in `sigcup/` and parameterised in `config/settings.yaml`. Written 2026-10-10 (day 10 of 35). Sources are the four research reports in `docs/research/` and the reviewer's notes in `docs/research/plan_review.md`; full citations in section 12. Anything the research could not confirm against an official page is marked **UNVERIFIED** here too._

## 1. Objective and constraints

**What is scored.** Final rank is by verified SUSQies balance after every market settles: no Sharpe, no drawdown adjustment. Prizes $30,000 / $5,000 / $2,500 for places 1–3, ties split. Everyone started with 100,000 SUSQies on 2026-10-01 12:00 ET; orders must be received before **2026-11-04 12:00 ET**, which is `deadline_utc: 2026-11-04T17:00:00Z` because US daylight time ends on Nov 1 (a hard-coded UTC−4 would stop us an hour early). The interim leaderboard marks positions at recent trades, so it is noisy and paintable; only the settled balance counts.

**Hybrid objective.** Rank-based payoffs reward variance when trailing (Browne 1999; Ottaviani & Sørensen 2006; Lichtendahl & Winkler 2007), but the field size is unknown and both the reviewer's "top-3 needs ≥ +25%" and a competitor's "~4.5×" are guesses. So a **steady-edge engine** (fractional Kelly, hard caps, closing-line-value scorecard) runs throughout, plus a **convex sleeve** of `convex_sleeve.budget_frac: 0.15` in the only deep SIG books (toss-ups and chamber control), plus an **endgame rule**: in the final `endgame.final_hours: 72` the sleeve rises to `0.30` if third place is within `reachable_gap_frac: 0.15` of our total and falls to `0.0` if we are already placed. Variance is bought in a bounded sleeve from day 2, not in the last 72 hours when spreads are worst.

**Binding constraints.**

| Constraint | Value | Consequence |
|---|---|---|
| One account, unlimited bots | rules | all SIG REST traffic in one process, one token bucket |
| Rate limits | 100 reads + 30 writes/min across all keys; 429 → `Retry-After: 60` | we use `writes_per_min: 25`, `reads_per_min: 90`; one 429 costs every strategy a minute |
| Play money, no interest | — | holding a 0.93 favourite to settlement costs only opportunity |
| Horizon | 25 days; party markets settle `2026-11-04T17:00Z` | theta converges only in the last 48 h; capital locked until then |
| Fees | reported zero by two bots; **UNVERIFIED** | day-1 probe; a Kalshi-style 0.07·p·(1−p) fee is 0.3¢ at 0.95 and harmless, a PredictIt-style 10% profit fee kills theta |
| Position limits | in the rules, numbers unknown; 45,662-share orders observed | own cap `max_position_shares: 50000` |

## 2. Platform facts that matter

A white-label play-money venue ("The Super Market", operator **UNVERIFIED**) with a continuous YES-quoted limit order book per market. Prices 0.005–0.995 in 0.005 ticks (`round_tick` on every price we send); limit orders only, optional expiry; resting orders **do not reserve cash** (the OMS keeps its own reservation ≤ cash); an unbacked sell fills as the opposite-side buy, so YES and NO of one market are one book. Atomic placement of ≤10 legs (placement atomic, fills not), batches of ≤50, cancel-all. Every order must carry `tournamentId` or it hits the public book. Error semantics: `409 REQUEST_IN_FLIGHT` (90-s idempotency lease), `502 ORDER_STATUS_UNKNOWN` (retry the same key), `503 TX_CONFLICT`; writes took 14–36 s at the open (`write_timeout_s: 45`). Supabase Realtime pushes books and fills (~250 ms batching, ~100 channels per connection, batches sent once, so REST resync stays necessary) and does not count against REST limits.

**Collateral and exclusivity.** "Relationship collateral" reduces the cash needed for linked positions (NO+NO across a race) by the guaranteed minimum payout, and relationships are typed `mutually_exclusive` with `isExhaustive`; an Oct 7 audit found the tournament relationship graph **empty**, so Dutch books currently lock full cash.

**Universe and settlement.** 237 markets: 4 chamber-control + 70 Senate + 71 Governor + 92 House, one market per party per race plus independent legs (NE-Sen, MT-Sen, RI-Gov). A fusion-ticket winner counts for every party on the ticket (RI); Georgia runoffs settle in December; N/A outcomes refund cost; control markets are Freeform, settled manually once networks call them; the sponsor may change outcomes after the fact. Ohio Senate and Governor were missing from the Sept 30 list.

**Days 1–9 as competitors saw them** (public GitHub repos, not official):

| Observation | Source |
|---|---|
| A maker posts 20-share levels around the mid in ~230 of 237 markets at an ~8-pt spread; whether it is SIG's DMM tier (600/200 limits) is inferred | IamYanis, samyakbaid, grissou |
| Favourites 85–94 where Kalshi/Polymarket say 97–99 (RI-Sen D 84.5 vs 99; MA-D 93.5 vs 98; NC-D 94 vs 97; House control D 86 vs 91–94); longshots 8–15 vs 1–3 | Lnvqntx, grissou |
| Mid-band gaps small: TX-D 63 vs 66–67, KS-D 35.5 vs 36–37, AK-D 71.5 vs 74–76 | Lnvqntx |
| Race bids summed > 1 in ~30% of snapshots (52% on Oct 3) by 0.5–2¢, median life ~2 min; asks < 1 in ~9.6% | grissou |
| Post-fill drift −0.01 to −0.08¢ (adverse selection ≈ 0); takes earned ~+7.5¢/share, maker value fills +2–2.6¢ | grissou |
| Pure market making: "tens of units a day on a 20k reserve"; >90% of capital tied up by day 3; undercut by a tick within hours | grissou |
| Directional sniping lost money out of sample; Polymarket→SIG 30-s lead-lag inconclusive, "executable crossing negative" | nullif1ed, DanielCWeng |
| RI-Sen D bids swung 0.96 → 0.16 → back as one bot pulled a side; tournament-wide D tilt grew 1.7% → 14% in three days | nullif1ed, grissou |
| Toss-up and control books carry 4–5k shares at the top; most others are thin | IamYanis |

Row 2 is why theta is the primary engine, row 4 why the arb scanner exists, row 5 why we are a maker, row 7 why we never cross a spread on a SIG print.

## 3. Where the edge comes from, ranked

### 3.1 Theta / favourite-longshot book (primary P&L engine)

**Evidence.** Political markets are persistently under-confident: Le (2026), 353M Kalshi + Polymarket trades, finds prices compressed toward 50% on both venues (a 70¢ politics contract a week out implies ≈83%). Page & Clemen (2013), 500k+ Intrade transactions, show the distortion grows with time to expiry (4.7–10.9 pp on long-dated contracts) and unwinds toward resolution. Kalshi's calibration study (2026): close-horizon Brier 0.0635 below $10k volume vs 0.0099 above $200k. Caps and thin books make it bigger: Zitzewitz (2026a) found PredictIt bids persistently summing above 110; Lott's 805 PredictIt probabilities had 10–20% longshots winning 4.5% and the 0–20% bucket priced 5.4% vs 1.7% realised. Bürgi, Deng & Whelan (2025): a 20% event is priced ≈23%. Cardozo & Rivero-Wildemauwe (2026): Polymarket sub-10¢ buys lose 6.3% equal- / 19.4% dollar-weighted, ≥90¢ contracts earn +0.28% / +0.83%, in politics but not sports. Snowberg & Wolfers (2010): the cause is probability misperception, so it survives without a bookmaker take and, we assume, without money. The SIG book shows exactly this (section 2).

**Rule** (`signals.py`, `risk.theta_bid`). For every leg with `fair ≥ theta.min_fair: 0.85`, rest `bid = round_tick(min(ask, min(fair, fair_cap) / (1 + min_yield)))` with `fair_cap: 0.985`, `min_yield: 0.055`, stepping one tick down if the grid would breach the yield bound. At the cap, 0.985/1.055 = 0.9336 → **0.930** is the highest bid the engine ever posts. YES-favourite and NO-longshot are one position: only the better-yielding leg per race is emitted and both are netted under one per-race cap. Candidates are ranked by yield per SUSQie locked. `kelly_fraction_theta: 0.5` is nominal; the per-race cap binds (section 5).

**Size and confidence.** Reviewer arithmetic: average fill 0.90, true probability ≈0.975 (not 1.00); a 3,000-SUSQie position is 3,333 shares, +333 if right, −3,000 if wrong, EV ≈ +250; 60k in 20 such positions ≈ **+5k (+8%)**. P(≥1 upset in 20) = 1 − 0.975²⁰ = 40%; one upset leaves ≈ +3.3k, two (P ≈ 9%) ≈ 0. Realistic terminal 104–106k. High confidence on direction, medium on size: grissou's fitted bias went 0.046 → 0.128 → 0.092 in week 1, so the gap is already being competed.

**Falsifiers.** Yields in the 85–95 band below 5.5% (the engine then stops bidding); negative closing-line value on theta fills at settlement. A correlated 2022-style miss is the section 5 stress, not a falsification.

### 3.2 Dutch books (NO+NO, YES+YES, multi-leg sums)

**Evidence.** Zitzewitz (2026a); Saggese et al. (2025): ~100 arbitrage windows per Polymarket market, both ΣYES>1 and ΣYES<1, with leg risk; Gebele, Mutzel & Matthes (2026): ≈$1.12M executable, clustering where the protocol cannot natively convert. On SIG: Delaware Senate bids 0.95 + 0.08 = 1.03 on Sept 30; race bids > 1 in ~30% of snapshots.

**Rule** (`consistency.py`). `race_no_sum`: best bids over a race's legs sum to > 1 + `arb.min_profit: 0.005` → sell YES (buy NO) on every leg at its bid. `race_yes_sum`: best asks sum to < 1 − 0.005 → buy YES on every leg. Legs share a `group_id`, go out as one atomic multi-leg order (`max_legs: 10`), `take=True`, priority 0. `max_sets` is bounded by displayed size at every leg's best level; unknown size bounds it at zero, because an unfilled leg is a directional position, not an arbitrage. Control-vs-seats Fréchet violations are reported, never traded.

**Why exclusivity must come from the API.** NO+NO pays 1 only if at most one leg resolves YES: a Rhode Island fusion winner resolves both party legs YES and the set pays 0. YES+YES pays ≥1 only if some listed leg resolves YES: an independent win in NE/MT/RI-Gov pays 0. A Georgia runoff settles in December; an N/A refund returns cost. Hence `arb.require_api_exclusivity: true`: eligible only when the relationships API reports `mutually_exclusive` **and** `isExhaustive`, never from titles. With the graph empty, expect zero arb volume until SIG populates it; bootstrap reports it.

**Size and confidence.** 0.5–2¢ per set, median life 2 min. Without collateral a NO+NO at bids 0.95 + 0.08 locks 0.97 per set for 3¢ (≈3%) until Nov 4, dominated by theta at 5.5%+; with collateral it is near capital-free and ranks first. High confidence that violations exist, low on capturable volume.

### 3.3 Value vs real-money fair in the 15–85 band

**Evidence.** Le's compression holds in the mid band; Angelini & De Angelis (2026) find ≈0.64-for-1 contemporaneous pass-through of benchmark changes on Kalshi with drift afterward, larger in thin markets; Kalshi and Polymarket agree within ~1 pt on 988 matched races (samyakbaid), so their blend is a usable anchor; grissou: the top 80 markets carry ~90% of measurable edge.

**Rule.** Inside `value.band: [0.15, 0.85]`, when `|fair − mid| > value.min_edge: 0.05`, rest at `fair ∓ quote_offset: 0.02` (buy YES if fair > mid, else sell YES), capped at the touch. Re-quote only on a `requote_threshold: 0.02` fair move with `requote_cooldown_s: 90`. Skipped for 24 h after a hard event if the fair value's inputs predate it. `kelly_fraction_default: 0.25`: a 5¢ edge on a 50¢ contract gives f* = 0.10, so 2.5% of bankroll.

**Size and confidence.** 3–8¢ where it fires, but the observed mid-band gaps are mostly 1–5¢, below threshold, so in calm markets it fires rarely by design. Medium. Falsifier: 24-h CLV ≤ 0 over the first 50 value fills.

### 3.4 Election night (the convex event)

**Evidence.** Markets digest returns faster than networks (ACX's Virginia anecdote; Kalshi digests AP calls in seconds) but overreact to early counts: PredictIt's Biden contract fell ~59¢ → ~42¢ on early returns in 2020 ("clearly overreacted", Zvi); Pennsylvania's Republican odds went 61% → 0% in six hours in 2022. That thin student quotes with 20-share levels lag the real-money venues is the hypothesis; there is no prior edition, so its size is **UNVERIFIED**.

**Rule** (`election_night.py`, section 8). From `start_utc: 2026-11-03T23:00Z` the reference feed runs every `ref_cadence_s: 10`; the executor may *take* SIG liquidity only when the reference ≥ `take_when_ref_ge: 0.97` (≤ 0.03 for the NO mirror) and SIG lags by ≥ `min_lag: 0.08`, never on a SIG print; size bounded by displayed depth and `reserve_deploy_frac: 0.8` of the liquid reserve. Capturable size is bounded by depth, not cash (reviewer's speculation: 5–15k per bot). Medium confidence; if SIG reprices within seconds of Kalshi there is nothing to take and nothing lost.

### 3.5 News regime: fade soft, follow hard

**Evidence.** Clinton & Huang (2025): 58% of Polymarket national markets negatively serially correlated in 2024, spikes reversed next day. Tsang & Yang (2026): the debate jump mostly reversed, the assassination-attempt repricing persisted, the dropout produced two-sided volume with little net move. Brown, Reade & Vaughan Williams (2019): the first wave after a poll reacts to its *incidence*, not content. Angelini & De Angelis (2026): hard information under-reacts (~0.64) then drifts. Vojtko & Dujava (2026): mean reversion pays only with passive execution. Gelman et al. (2016): most poll swings are differential non-response. Rhode & Strumpf (2008): manipulations revert. Practice agrees: the NYC 2025 Emerson flip, and the Aug 2026 "Median Strategies" fake polls that moved Kalshi within 15 minutes.

**Rule.** A > 8¢/day SIG move is tested against its implied margin shift through the T₅ map: on a 50/50 Senate race (σ = 7) a 1-pt poll surprise is worth 5.4¢; an 8¢ move implies 1.5 pts, a 15¢ move 2.9 pts. Soft (single poll, debate, ad, one big order) → `fade`: rest 4.5¢ from the pre-event price on the far side of the move. Hard (dropout, death, indictment, ballot ruling) → `follow`: quote toward pre-event price + move/0.64. Both ship **disabled** (`sentiment.enable_fade_follow: false`) until the signal's IC is positive on ≥ 1 week of data. 2–5¢ per soft event, 3–10¢ per rare hard one; medium confidence. The costliest error is fading hard news, which is why classification precedes action.

### 3.6 Market making (last, and off)

Makers win on real-money venues (Akey et al. 2026: the top 1% take 76.5% of profits with resting orders; Bürgi et al.: takers lose ≈32%), but on SIG pure market making earned ~0.1%/day, bound >90% of capital by day 3, was undercut within hours and costs 4 writes per two-sided reprice against a 25/min budget. `mm.enabled: false`; if enabled, only where `spread ≥ min_spread: 0.06`, at `mid ∓ half_spread: 0.02`, `size: 20`, never through our own fair, priority 3, only when the write budget is idle.

### 3.7 Cohort tilt (folded into value)

Wishful-thinking bias inside a trader population is real (Forsythe et al. 1992, 1999; Becker, McGurk & LoGrasso 2025) but has no a priori direction (Zitzewitz 2026b: no left or right overpricing in any subsample 1880–2025; 2022 markets too Republican, 2024 Polymarket too Trump-heavy). grissou measured a D tilt growing to 14%, but the reviewer's objection stands: with safe-D seats dominating, (price − fair) regressed on party loads on favourite status, i.e. favourite-longshot bias in disguise. The tilt is tested only on the ~12 toss-ups, where a 3–5¢ residual is indistinguishable from Kalshi's own partisan error in 2022 and 2024, and traded only through `value` where it exceeds 5¢. No weekly basket.

## 4. Fair-value model

`fairvalue.compute_all` blends three components with `fairvalue.weights: {ref: 0.6, polls: 0.3, ratings: 0.1}`, each scaled by availability and renormalised over those present; `sd` is component disagreement plus a floor.

**(a) Reference prices, raw.** Liquidity-weighted mean of the latest Kalshi and Polymarket YES mids, each weighted `min(1, liquidity / ref_min_liquidity: 1000)` (unknown liquidity 0.5). No favourite-longshot stretch. Rothschild (2009) debiases with a Leigh et al. transform remembered as Pr = Φ(1.64·Φ⁻¹(price)); the constant is **UNVERIFIED**, and the reviewer showed its effect: Kalshi 0.97 becomes Φ(1.64·1.88) = 0.999, after which a bid at fair − 1¢ lifts favourites to 0.98 for a 1% yield with 2.5% ruin per leg. The student book carries the bias; the real-money venues agree within ~1 pt and are the best estimate of truth we have. The residual compression Le finds in real-money politics is handled by the 0.985 cap and the 5.5% yield, not by stretching.

**(b) Polls + fundamentals.** Recency-weighted poll margin (half-life `poll_halflife_days: 7`) shrunk toward a fundamentals prior (generic ballot − PVI + incumbency, D-minus-R) with weight `max(fundamentals_shrink: 0.3, 1/(1 + n_eff))`, so one poll is shrunk half-way (Erikson & Wlezien 2008). P(D wins) = T₅(margin / σ_chamber), `t_df: 5` as in FiveThirtyEight, `sigma_pts: {senate: 7.0, governor: 6.5, house: 9.0, control: 5.0}`.

The sigmas come from final-weeks polling error: FiveThirtyEight's all-cycle averages Governor 5.2, Senate 6.1, House 9.3 (since 1998: Senate 5.4, House 6.3); 2022 was 4.8 with almost no bias; Silver Bulletin's 2024 Senate error 4.1 with ≈3 pts Democratic bias versus "almost completely unbiased" 2018 and 2022 midterms; Shirani-Mehr et al. (2018), 4,221 polls in 608 state races: actual error roughly double the nominal margin, election-level bias ≈2.1 (Senate) / 2.3 (Governor) per a secondary citation (**UNVERIFIED**). For a normal MAE ≈ 0.8σ, so σ ≈ 1.25×MAE: Senate 6.5–7.5, Governor ≈6.5, House 8–11, where polls are sparse and the prior does most of the work. The implemented map:

| D−R margin (pts) | Senate σ=7 | Governor σ=6.5 | House σ=9 |
|---|---|---|---|
| +1 | 0.554 | — | — |
| +3 | 0.657 | 0.668 | 0.624 |
| +5 | 0.746 | 0.762 | 0.699 |
| +8 | 0.848 | 0.863 | 0.793 |
| +10 | 0.894 | 0.908 | 0.841 |
| +15 | 0.957 | 0.965 | 0.922 |
| +20 | 0.982 | — | 0.962 |

Check: since 1998 Senate candidates leading by > 5.5 pts 30 days out went 68–0 and 3–6-pt leaders 9–3 (FiveThirtyEight), so the t₅ tails are conservative on 5–8-pt leads, which is what the yield rule wants. The national bias draw has zero mean: presidential-year polls under-estimated Republicans three cycles running, midterms were unbiased, and direction is unpredictable ex ante.

**(c) Ratings.** `ratings_prob: {safe: 0.97, likely: 0.85, lean: 0.68, tilt: 0.58, tossup: 0.50}` (Split Ticket bands: Lean ≥ 60%, Likely ≥ 80%, Safe ≥ 95%), averaged over the latest rating per source (Cook, Sabato, Inside Elections, Split Ticket, RacetotheWH); `config/ratings_manual.yaml` is the day-one fallback. Independent legs use `ref` only, or `1 − D − R`.

**Correlated simulation** (`simulate.py`): `n_sims: 10000`, layers `national_sigma_pts: 2.75`, `state_sigma_pts: 2.5`, `race_sigma_pts: {senate: 4.5, governor: 4.5, house: 6.0}`, with a shared chi-square tail scaler so each race's marginal is exactly σ_chamber·t₅ (outputs reproduce inputs) and fat-tail events are national, after FiveThirtyEight, Silver's FLIPR, Heidemanns, Gelman & Morris (2020) and Linzer (2013). Used for three things only: the chamber-control fair (blended with the control market's reference at the polls+ratings weight; 203 Democratic House seats and 34 Senate seats off the ballot are documented constants to verify against the live list), Fréchet bounds for the informational control-vs-seats check, and the portfolio P&L distribution in reports. Sizing uses the swing stress below; the effective number of independent bets in 237 markets is 10–20, not 237.

## 5. Sizing and risk

**Kelly.** Bought at p with belief q, f* = (q − p)/(1 − p); selling YES, (p − q)/p (MacLean, Thorp & Ziemba 2011: half Kelly keeps ~75% of the growth at a fraction of the variance). Multipliers `kelly_fraction_default: 0.25`, `kelly_fraction_theta: 0.5`, `kelly_fraction_arb: 0.5`, sleeve `0.5`. For theta Kelly does not bind: fair 0.975 at 0.90 gives f* = 0.75, half Kelly 37.5% of bankroll per position. The caps do: `per_race_cap_frac: 0.03` (3,000 SUSQies, netted across legs; exposure is the worst-case loss over the race's outcomes, so buy-YES-D and buy-NO-R add while buy-YES-D and buy-YES-R offset), `per_state_cap_frac: 0.08`, `max_position_shares: 50000`. The reviewer's arithmetic in 3.1 is the realistic theta outcome under these caps.

**Swing stress.** Every candidate order set must satisfy: a uniform `national_swing_pts: 4.0` shift against our net party exposure, through the T₅ map, costs ≤ `max_swing_loss_frac: 0.25` of bankroll, else it is scaled down by bisection. A −4 swing moves Senate fair 0.985 → 0.970, 0.975 → 0.949, 0.90 → 0.796, 0.68 → 0.472, 0.50 → 0.296: a 0.90 bid on a 0.975 favourite is still in the money, a toss-up loses ~20¢ a share. So the stress binds on the sleeve and on value, as intended; a 2022-style correlated miss (every market favoured the GOP for the Senate; Masters 53¢ vs Kelly 48¢) is the one risk theta cannot diversify.

**Reserve.** Idle cash earns nothing and favourites converge upward, so deploy early. Keep `liquid_reserve_frac: 0.20` through Oct 31 and `liquid_reserve_frac_final72h: 0.30` after, rebuilt by letting arbs and expiring orders roll off, never by selling theta into an 8-pt spread. Election night may spend `reserve_deploy_frac: 0.8` of it (24% of bankroll), below the practitioner 30–50% because capturable size is bounded by depth.

**Sleeve and endgame.** `convex_sleeve: {budget_frac: 0.15, kelly_fraction: 0.5, min_edge: 0.05}` on toss-ups (fair 0.3–0.7) and control. `endgame.py` reads the leaderboard every `leaderboard_s: 900`, uses our own equity when the snapshot lacks our total and ignores stale snapshots, because interim marks are paintable and noisy (±10%, speculation).

**Kill switch.** A kill file or a day at ≤ −`daily_loss_limit_frac: 0.05` → cancel-all, mark state, stop (not merely refuse to start). Every resting order carries `order_expiry_s: 900` as a dead-man's switch. No POST after `deadline_utc`.

## 6. Execution

**Maker first.** Resting orders are the edge on every venue studied (Akey et al.; Bürgi et al.; Vojtko & Dujava) and SIG's post-fill drift is ≈0, so resting carries no adverse-selection tax. Takes are limited to two cases where the price is known rather than estimated: arbitrage legs (riskless if all legs fill, so they cross together atomically) and election-night takes at a reference ≥ 0.97. Value orders rest even at a 5¢ edge: DanielCWeng found executable crossing on the Polymarket→SIG lag negative.

**Write budget per minute** (ceiling 25 of 30, sliding 60 s, one slot always free for a kill): 8 for arb legs and election takes (priority 0, may burst to 25), 6 for theta placement and refresh (priority 1; the ~85 favourite legs are laid in ~4 minutes and re-laid every ~15 minutes as orders expire), 8 for value re-quotes (priority 2; ≈40 markets at a 5-minute cadence, not 237), 3 for reconciliation cancels. Re-quote only on a ≥ 2¢ fair move with a 90-s cooldown: a 1¢ trigger thrashes on election night when Kalshi ticks every few seconds → 429 → all writes frozen 60 s. Prefer expiry-and-replace (1 write) to cancel-and-replace (2). Day-1 probe: if a ≤50-order batch counts as one write, the allocation is re-planned.

**Reads (~55 of 90).** Realtime covers the `realtime_channels: 94` hottest markets at zero reads; bulk prices for the rest every `bulk_prices_s: 30`; depth on demand ≤ 30/min; positions, orders and fills every `reconcile_s: 60`; constraints and leaderboard ≈ 2; resyncs ≈ 5. All SIG REST calls live in `trader`, so there is exactly one limiter; `feeds` and `reporter` hold a read-scope key only.

**Idempotency and reconciliation** (`oms.py`). Persist `OrderIntent(NEW)` with its idempotency key *before* the POST; SENT in flight; ACKED on 2xx; timeout/502/503 → UNCERTAIN and re-POST the same key; 409 → wait out the 90-s lease, poll `/orders` and `/fills`; 4xx → REJECTED. On startup reconcile every UNCERTAIN/SENT intent before quoting. Every 60 s: fills ledger == `/positions`, intents == `/orders?status=open`, reservation ≤ cash, discrepancies logged and repaired. pm2 `max_restarts` keeps a crash loop from burning the budget on cancel-all/re-lay cycles; the server clock offset comes from the HTTP `Date` header before any book is judged stale.

## 7. The role of social-media sentiment

**Verdict on levels.** Every rigorous test of sentiment or volume *levels* fails the trivial baseline: Metaxas, Mustafaraj & Gayo-Avello (2011) found published Twitter methods on the 2010 Senate races no better than chance and worse than incumbency; Gayo-Avello (2012, 2013) showed the famous results were post hoc; Jungherr et al. (2012) showed Tumasjan et al. (2010) depended on dropping one party; Skoric, Liu & Jaidka's (2020) meta-analysis of 74 studies puts average accuracy below survey benchmarks; Caballero (2021) underperformed polls in 2020; a 2021 review puts volume at 17.1% MAE and sentiment at 7.6% against 2–3% for polls. No race is priced from sentiment; the blend has no sentiment term.

**What survives.** (1) *Changes* in opinion share track polls with a lead: Bovet, Morone & Makse (2018), r = 0.93–0.96 against the NYT average, Twitter ≈10 days ahead of published polls; Beauchamp (2017) interpolates state polls from text at sub-day resolution. (2) Social streams surface *events* before wire and RSS (one non-peer-reviewed project: median 32-minute lead on 5 stories) and probably hours before a thin student market reprices. (3) Crowd stance turning against a leader weakly precedes reversals: Sultan & Morstatter (2026) see it in 23/27 Polymarket leadership changes, but it added nothing beyond market-state features. Expected IC 0.02–0.05.

**Stance, not sentiment.** Anger at the opponent is negative sentiment and positive stance; lexicons fail on politics. Pipeline (`feeds/nlp/`): gazetteer from the 237 market titles (full names and handles 0.95, hashtags 0.90, aliases 0.85, surname only with a co-occurring race token 0.60; common-word surnames never match alone) → heuristic bot score (account age, posting rate, duplicate text, handle pattern; > 0.7 dropped; Bessi & Ferrara 2016: ≈15% of accounts and ≈20% of tweets in 2016 were bots) → Claude Haiku 5.5 stance, 25 posts per request, Batch API, cached by text hash, `unclear` escalated to Sonnet 5.5 (LLM consensus matched the human majority > 90% on 2024 X political stance; ≈ $0.09 per 10k posts) → daily `net_stance_dem` per (platform, race) → z-score against the trailing `baseline_days: 14` → event when `|z| ≥ spike_z: 2.5` on ≥ `min_platforms: 2` the same day or on a GDELT volume spike → an LLM labels it hard / soft / unknown with the race context.

**Platform skews and z-scoring.** Bluesky is journalist-heavy and left-leaning (69% of left-leaning news influencers joined it vs 15% of conservatives; 82% remain on X); X skews right (14% of Republicans vs 10% of Democrats get news there); Reddit the reverse (7% vs 12%). Levels across platforms are incomparable, but each (platform, race) series z-scored against its *own* 14-day baseline measures only its own change, and the two-platform requirement removes single-platform noise.

**IC gate.** `reporter/leadlag.py` computes cross-correlation of feature changes against SIG and reference price changes at 1–24 h lags and the Spearman IC of z-score vs next-day price change per platform; `fade`/`follow` stay off until IC > 0 on ≥ 1 week of data, which cannot happen before ~Oct 25. The real-money feed remains the faster event detector, so the first live use of the pipeline is defensive: value quotes are suppressed for 24 h on a race with a fresh hard event.

**Data-access reality (Oct 2026).**

| Platform | Status here | Why |
|---|---|---|
| X | vendor adapter (SocialData ≈ $0.20/1k, twitterapi.io ≈ $0.15/1k); official v2 pay-per-use at $5/1k is a config switch | official is 25× the vendor; vendors breach X ToS, a risk the user accepted; ~300k posts ≈ $45–60 |
| Bluesky | primary free feed: Jetstream firehose (no auth) + `searchPosts` backfill with an app password | unauthenticated search 403s since mid-2026; left/journalist skew handled by z-scoring |
| GDELT 2.0 DOC | primary news-event detector, 15-min updates, ~1 request/5 s | free, sanctioned; docs **UNVERIFIED** this session |
| RSS | local papers, Substacks, Crystal Ball feed | free, high precision, low volume |
| YouTube Data API | comments on local-station and candidate channels via `playlistItems` + `commentThreads`, never `search.list` (100 units) | 10,000 units/day |
| Reddit | scaffolded, **disabled** | Responsible Builder Policy closed self-service OAuth; approval 2–4 weeks; no credentials |
| LinkedIn | skipped | no keyword API, scraping litigated, near-zero electoral signal |
| Instagram / Facebook | skipped | Graph API reads own pages only; CrowdTangle dead; Meta Content Library academic-only; Apify is a ToS breach the user declined |
| TikTok, Truth Social, Google Trends | skipped | Research API academic-only and weeks away; no public Truth API (institutional feed reportedly ~$100k/month); pytrends dead, official Trends API in alpha |

## 8. Election-night playbook

| When (UTC) | What |
|---|---|
| Oct 20–28 | `election_night.py` and `endgame.py` dry run on paper; settlement caveats in `races.yaml` (GA runoff, RI fusion, NE/MT/RI-Gov independents, N/A refunds); confirm Ohio presence |
| Nov 1 | DST ends; verify the deadline clock reads 17:00Z |
| Nov 3 before 23:00 | reserve rebuilt to 30% by roll-off; value re-quotes on the 2¢/90-s rule |
| Nov 3 23:00 (6 pm EST) | `election_night.is_active`; `ref_markets` cadence 120 s → 10 s; OMS priority 0 may burst to the full 25 writes |
| Nov 3–4 overnight | taker rule only: ref ≥ 0.97 and SIG ask ≤ ref − 0.08 → buy YES at the ask; ref ≤ 0.03 and SIG bid ≥ ref + 0.08 → buy NO; best edge first, bounded by displayed size and the deployable reserve; races with a settlement caveat sized down |
| Nov 4 17:00 | hard stop; control markets may still be uncalled (dead capital for interim marks, irrelevant to final rank) |

The 0.97 threshold is the red-mirage filter: it means Kalshi has digested a network call, not an early return, the lesson of 2020 (Biden 59 → 42 on early counts) and 2022 (Pennsylvania 61 → 0 in six hours). We never trigger on a SIG print, never fade a called race, and hold the theta book to settlement rather than sell it into a 20-share book.

## 9. Scorecard

- **Closing-line value** (`reporter/clv.py`): every fill scored against the price 24 h later and the last price before settlement, signed in our favour. Over 237 markets in 3.5 weeks this is the only statistic with a sample (Miller & Davidow; Peabody; Duke on "resulting"). Expect theta ≈ +2–4¢ at close and value ≥ +1¢ at 24 h; a tag whose 24-h CLV is negative over 50 fills is cut.
- **P&L by tag** (`theta`, `arb`, `value`, `convex`, `election`, `fade`, `follow`, `mm`), daily, paper and live separated.
- **Calibration**: fair values vs reference prices daily and vs resolution after Nov 4; simulation marginals vs inputs (`tests/test_model_simulate.py`).
- **Operations**: budget utilisation, 429 count (target zero), UNCERTAIN intents resolved, reconciliation discrepancies, rank history every 15 minutes.
- **Sentiment**: lead-lag and IC per platform weekly, the gate for 3.5.

## 10. What we deliberately do not do

- No market orders into longshots or into any SIG print; the Kalshi/Polymarket loser profile is "take liquidity at extreme prices".
- No sentiment levels in fair value; no fade/follow until the IC gate passes.
- No debiasing stretch on real-money prices; the student book is never used as fair.
- No exclusivity inferred from titles; no Dutch book on RI, NE, MT, RI-Gov or Georgia without API evidence.
- No undocumented web-app endpoints, no browser automation, one account, no MNPI.
- No two-sided market making by default; no separate cohort-tilt basket.
- No selling theta to raise reserve; no hedging of the national factor on SIG prints.
- No reliance on PredictIt (2026 status **UNVERIFIED**), Reddit, LinkedIn, Meta properties, TikTok or Google Trends.
- No 1¢ re-quote triggers, no cancel-and-replace where expiry will do, no second process holding a trade key.

## 11. Risks and failure modes

1. **Correlated national miss.** The 4-pt stress is calibrated to 2016/2020-style errors (polls 3–4 pts too Democratic) and the midterm pattern of near-zero bias; a miss against the markets' direction hits theta and the sleeve together. Loss bounded at 25% of bankroll by construction.
2. **Settlement discretion.** The sponsor may change outcomes after the fact; fusion, runoff and independent legs can resolve on different terms from the Kalshi contract we price them against; control markets may be uncalled on Nov 4.
3. **Platform behaviour.** One 429 freezes all writes for 60 s; a 502 leaves an order UNKNOWN (observed); Realtime batches are sent once; writes took 14–36 s at the open. The OMS state machine, the 25/90 ceilings and the 45-s write timeout exist for this.
4. **Competition decay.** grissou's bias parameter fell from 0.128 to 0.092 within week 1; bots undercut by a tick within hours. Yields may fall below 5.5% before our book is full, leaving cash idle.
5. **Capital lock.** Theta locks ~60% of bankroll until Nov 4 for ~8%; competitors had >90% tied up by day 3. The 20%/30% reserve is the only election-night dry powder.
6. **Leaderboard noise.** Marks come from recent trades, can be painted, and the endgame rule reads them. Mitigation: own equity when the snapshot is missing, stale snapshots ignored, thresholds in config.
7. **Model inputs.** PVI seeds are ±2 pts from memory; the House baseline (203 D seats off the market list) must be checked against the live list and any redistricting; Ohio may be absent; the Rothschild constant, Manski bound and several secondary figures are **UNVERIFIED**.
8. **Fake and single polls.** Median Strategies moved Kalshi in 15 minutes in Aug 2026; a fake poll that moves the *reference* feed moves our fair value with weight 0.6. Hard events freeze value quotes for 24 h; nothing freezes on a single poll.
9. **Rank objective.** +5–8% from theta may not place in a field of hundreds; the sleeve can lose 15–30% of bankroll in a correlated miss. This is the trade-off the hybrid objective accepts.
10. **Data access.** The X vendor can be cut off at any time; Bluesky is a poor read on GOP-side opinion; Reddit is off. Sentiment is an alerting layer, so this degrades alerts, not fair value.

## 12. References

Academic and research papers

- Akey, Grégoire, Harvie & Martineau (2026), "Who Wins and Who Loses in Prediction Markets? Evidence from Polymarket", CEPR DP21615. https://cepr.org/publications/dp21615
- Angelini & De Angelis (2026), "When Do Markets Fully Process Public Information?", arXiv:2606.07811. https://arxiv.org/abs/2606.07811
- Beauchamp (2017), "Predicting and Interpolating State-Level Polls Using Twitter Textual Data", AJPS 61(2). https://onlinelibrary.wiley.com/doi/abs/10.1111/ajps.12274
- Becker, McGurk & LoGrasso (2025), Scottish Journal of Political Economy 72(4). https://onlinelibrary.wiley.com/doi/abs/10.1111/sjpe.70007
- Berg, Nelson & Rietz (2008), "Prediction market accuracy in the long run", IJF 24(2). https://www.biz.uiowa.edu/faculty/trietz/papers/long%20run%20accuracy.pdf
- Berg & Rietz (2019), IJF 35(1). https://www.sciencedirect.com/science/article/abs/pii/S0169207018300499
- Bessi & Ferrara (2016), "Social bots distort the 2016 U.S. Presidential election online discussion", First Monday 21(11). https://firstmonday.org/ojs/index.php/fm/article/view/7090
- Bovet, Morone & Makse (2018), Scientific Reports 8:8673. https://www.nature.com/articles/s41598-018-26951-y
- Brown, Reade & Vaughan Williams (2019), "When are prediction market prices most informative?", IJF 35(1). https://www.sciencedirect.com/science/article/abs/pii/S0169207018300852
- Browne (1999), "Reaching goals by a deadline", Advances in Applied Probability 31. https://doi.org/10.2139/ssrn.703
- Bürgi, Deng & Whelan (2025), "Makers and Takers: The Economics of the Kalshi Prediction Market", UCD WP 2025/19. https://www.ucd.ie/economics/t4media/WP2025_19.pdf
- Busseti, Ryu & Boyd (2016), "Risk-constrained Kelly gambling". https://stanford.edu/~boyd//papers/pdf/kelly.pdf
- Caballero (2021), arXiv:2107.09640. https://arxiv.org/abs/2107.09640
- Cardozo & Rivero-Wildemauwe (2026), "The Favorite–Longshot Bias in Prediction Markets: Evidence from Polymarket", arXiv:2609.12878. https://arxiv.org/abs/2609.12878
- Clayton (2019), Quantitative Finance 19(11). https://www.tandfonline.com/doi/full/10.1080/14697688.2019.1639802
- Clinton & Huang (2025), SocArXiv. https://ideas.repec.org/p/osf/socarx/d5yx2_v1.html
- Erikson & Wlezien (2008), POQ 72(2). https://doi.org/10.1093/poq/nfn010
- Forsythe, Nelson, Neumann & Wright (1992), AER 82(5). https://econpapers.repec.org/article/aeaaecrev/v_3a82_3ay_3a1992_3ai_3a5_3ap_3a1142-61.htm
- Forsythe, Rietz & Ross (1999), "Wishes, expectations and actions", JEBO. https://www.biz.uiowa.edu/faculty/trietz/papers/wishes.pdf
- Gayo-Avello (2012), arXiv:1204.6441. https://arxiv.org/pdf/1204.6441 ; (2013), SSCR 31(6). https://journals.sagepub.com/doi/abs/10.1177/0894439313493979
- Gebele, Mutzel & Matthes (2026), "Executable Arbitrage and Market Efficiency in Prediction Markets", arXiv:2608.00666. https://arxiv.org/abs/2608.00666
- Gelman, Goel, Rivers & Rothschild (2016), "The Mythical Swing Voter", QJPS 11(1). https://www.nowpublishers.com/article/Details/QJPS-15031
- Heidemanns, Gelman & Morris (2020), HDSR. https://hdsr.mitpress.mit.edu/pub/nw1dzd02
- Jacobson (2015), JOP 77(3); incumbency series. https://www.degruyterbrill.com/document/doi/10.1515/for-2023-2003/html
- Jungherr, Jürgens & Schoen (2012), SSCR 30(2); Tumasjan et al. (2010), ICWSM.
- Kalshi Research (2026), "Calibration in Prediction Markets: Theory and Evidence". https://kalshi.com/research/kalshi-research-calibration.pdf
- Lauderdale & Linzer (2015), IJF 31(3). https://www.sciencedirect.com/science/article/abs/pii/S0169207015000102
- Le (2026), "Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets", arXiv:2602.19520. https://arxiv.org/abs/2602.19520
- Lichtendahl & Winkler (2007), Management Science 53(11). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=961001
- Linzer (2013), JASA 108(501).
- MacLean, Thorp & Ziemba (2011), The Kelly Capital Growth Investment Criterion. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1797366
- Manski (2006), Economics Letters 91(3). https://www.nber.org/papers/w10359
- Metaxas, Mustafaraj & Gayo-Avello (2011), "How (Not) to Predict Elections", IEEE SocialCom. https://www.researchgate.net/publication/220876090_How_Not_to_Predict_Elections
- Ottaviani & Sørensen (2006), JFE 81(2). https://web.econ.ku.dk/sorensen/papers/spf.pdf
- Page & Clemen (2013), Economic Journal 123(568). https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1468-0297.2012.02561.x
- Rhode & Strumpf (2008), NBER w14377. https://www.nber.org/papers/w14377
- Rothschild (2009), POQ 73(5). https://academic.oup.com/poq/article/73/5/895/1868587
- Rothschild & Sethi (2016), J. Prediction Markets 10(1). https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2322420
- Saggese et al. (2025), "Unravelling the Probabilistic Forest", AFT 2025, arXiv:2508.03474. https://arxiv.org/abs/2508.03474
- Shirani-Mehr, Rothschild, Goel & Gelman (2018), JASA 113(522). https://econpapers.repec.org/RePEc:taf:jnlasa:v:113:y:2018:i:522:p:607-614
- Skoric, Liu & Jaidka (2020), Information 11(4):187. https://doi.org/10.3390/info11040187
- Snowberg & Wolfers (2010), JPE 118(4). https://www.nber.org/papers/w15923
- Sultan & Morstatter (2026), arXiv:2609.28965. https://arxiv.org/abs/2609.28965
- Taleb (2018), Quantitative Finance 18(1), arXiv:1703.06351. https://arxiv.org/abs/1703.06351
- Tsang & Yang (2026), "Political Shocks and Price Discovery in Prediction Markets", arXiv:2603.03152. https://arxiv.org/abs/2603.03152
- Vojtko & Dujava (2026), SSRN 6726362. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6726362
- Whelan (2023), Quantitative Finance 23(11). https://www.tandfonline.com/doi/full/10.1080/14697688.2023.2257756
- Whitrow (2007), JRSS-C 56(5). https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9876.2007.00594.x
- Wolfers & Zitzewitz (2006), NBER w12200. https://www.nber.org/papers/w12200
- Zitzewitz (2026a), "Limits to Arbitrage and Prediction Market Efficiency: the Case of PredictIt", NBER w35845. https://www.nber.org/papers/w35845
- Zitzewitz (2026b), "Are Prediction Markets Politically Biased?", NBER w35846. https://www.nber.org/papers/w35846
- LLM stance: arXiv:2412.02712 https://arxiv.org/abs/2412.02712 ; PolitiSky24 arXiv:2506.07606 https://arxiv.org/abs/2506.07606 ; zero-shot stance arXiv:2403.00236 https://arxiv.org/pdf/2403.00236
- Social-media review (2021). https://link.springer.com/article/10.1186/s40537-021-00525-8 ; Social Media + Society (2024) counterpoint. https://journals.sagepub.com/doi/10.1177/20563051241298449

Polling error, models and ratings

- FiveThirtyEight, Senate poll accuracy (2022). https://fivethirtyeight.com/features/how-accurate-have-senate-polls-been-and-what-could-that-mean-for-november/ ; 2022 polling accuracy. https://fivethirtyeight.com/features/2022-election-polling-accuracy ; polling-average leads. https://fivethirtyeight.com/features/the-uncanny-accuracy-of-polling-averages-part-ii-what-the-numbers-say/ ; 2024 methodology (t with 5 df). https://abcnews.com/538/538s-2024-presidential-election-forecast-works/story?id=113068753
- Silver Bulletin, 2024 poll review. https://www.natesilver.net/p/so-how-did-the-polls-do-in-2024-its ; FLIPR methodology. https://www.natesilver.net/p/flipr-midterms-model-methodology ; generic ballot. https://www.natesilver.net/p/generic-ballot-average-2026-nate-silver-bulletin-congress-polls
- AAPOR Task Force on 2024 Pre-Election Polling. https://aapor.org/wp-content/uploads/2025/10/AAPOR-Task-Force-on-2024-Pre-Election-Polling_Report.pdf
- Split Ticket 2026 model (The Argument). https://www.theargumentmag.com/p/split-ticket-2026-midterms-model ; 270toWin Split Ticket Senate mirror. https://www.270towin.com/2026-senate-election/split-ticket-2026-senate-ratings
- Cook Political Report ratings. https://www.cookpolitical.com/ratings/senate-race-ratings ; Sabato's Crystal Ball 2026 changes. https://centerforpolitics.org/crystalball/2026-rating-changes/ ; RacetotheWH. https://www.racetothewh.com/senate/26 ; DDHQ. https://votes.decisiondeskhq.com/forecast/2026/governor

Platform and competitor sources

- Predictions Cup rules. https://predictionscup.com/rules/ ; platform guide. https://sig.thesuper.market/docs ; API reference. https://sig.thesuper.market/api/v1/docs ; press release (Business Wire, 2026-09-22). https://www.businesswire.com/news/home/20260922468550/en/Susquehanna-Predictions-Launches-Predictions-Cup-for-University-Students
- grissou/prediction-market-maker (incl. LIT_REVIEW). https://github.com/grissou/prediction-market-maker
- nullif1ed/sigprediction. https://github.com/nullif1ed/sigprediction
- samyakbaid/Susquehanna-Super-Market (MATH.md). https://github.com/samyakbaid/Susquehanna-Super-Market/blob/main/MATH.md
- Lnvqntx market watchlist. https://github.com/Lnvqntx/Susquehanna-Predictions-Cup/blob/main/research/market_watchlist.md
- IamYanis API/settlement audit. https://github.com/IamYanis/Susquehanna-Prediction-Cup-Strategy/blob/main/API_SETTLEMENT_AUDIT.md
- DanielCWeng/predictions-cup. https://github.com/DanielCWeng/predictions-cup ; benjamingao93/SIG-prediction-bot. https://github.com/benjamingao93/SIG-prediction-bot ; nl2992/sig-arb crosswalk. https://raw.githubusercontent.com/nl2992/sig-arb/main/docs/market-links.csv
- Kalshi market data API. https://docs.kalshi.com/getting_started/quick_start_market_data ; Polymarket docs. https://docs.polymarket.com/market-data/overview

Practitioners and reporting

- Domer interviews. https://www.thendobetter.com/investing/2025/12/11/notes-on-domer-investment-trading-process-and-philosophy ; https://www.onchaintimes.com/a-chat-with-domer-the-1-trader-on-polymarket/
- Zvi Mowshowitz, 2020 post-mortem. https://www.lesswrong.com/posts/WcHbFLgn3fmFn79mg/2020-election-prediction-markets-versus-polling-modeling ; "Free Money at PredictIt". https://thezvi.substack.com/p/free-money-at-predictit-2020-general-election
- Maxim Lott, election betting vs 538. https://www.maximumtruth.org/p/deep-dive-on-predicting-elections
- Calhoun (Forbes, 2022), markets failed their midterm exams. https://www.forbes.com/sites/georgecalhoun/2022/11/14/the-un-wisdom-of-crowds-prediction-markets-failed-their-midterm-exams/
- Johnson (Asterisk), prediction markets have an elections problem. https://asteriskmag.com/issues/05/prediction-markets-have-an-elections-problem-jeremiah-johnson
- ACX Prediction Market FAQ. https://www.astralcodexten.com/p/prediction-market-faq ; Mantic Monday 10/17/22. https://www.astralcodexten.com/p/mantic-monday-101722
- NPR on the Aug 2026 fake polls. https://www.npr.org/nx-s1-5945290 ; NYC 2025 poll shock. https://nexteventhorizon.substack.com/p/kalshi-moved-the-goalposts-on-nyc-mayor
- Thin congressional markets (Reuters via IDI). https://en.idi.org.il/articles/66410 ; CNN on midterm market volumes. https://www.cnn.com/2026/09/24/politics/midterm-elections-prediction-markets-kalshi-polymarket
- Miller & Davidow, The Logic of Sports Betting (review). https://bettably.com/blog/the-logic-of-sports-betting-review ; Peabody Q&A. https://www.rotowire.com/football/article/beating-the-book-qa-with-rufus-peabody-40172
- Pew, Bluesky vs X news influencers. https://www.pewresearch.org/short-reads/2025/05/29/bluesky-has-caught-on-with-many-news-influencers-but-x-remains-popular/ ; social media and news fact sheet. https://www.pewresearch.org/journalism/fact-sheet/social-media-and-news-fact-sheet/
- X API pricing (2026). https://www.medianama.com/2026/02/223-x-developer-api-pricing-pay-per-use-model/ ; SocialData pricing. https://docs.socialdata.tools/getting-started/pricing/ ; Reddit Data API 2026. https://www.socialcrawl.dev/blog/reddit-data-api-2026 ; Bluesky search 403. https://github.com/bluesky-social/bsky-docs/issues/332 ; YouTube quota. https://developers.google.com/youtube/v3/determine_quota_cost ; GDELT DOC client. https://pypi.org/project/gdeltdoc/
