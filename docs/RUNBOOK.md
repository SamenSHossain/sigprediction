# sigcup RUNBOOK

Operating guide for the SIG Predictions Cup trading system (2026 US midterms; trading closes
**2026-11-04 17:00 UTC** = noon EST after DST ends on Nov 1; never hard-code UTC-4). The strategy
itself is in `docs/STRATEGY.md`; this file is about running it safely.

Three long-running processes share one SQLite database (WAL mode):

| pm2 app    | command                           | key used        | what it does |
|------------|-----------------------------------|-----------------|--------------|
| `trader`   | `python -m sigcup.cli trade`      | `SIG_API_KEY` (trade) | the ONLY process that talks to the exchange with a write-capable key: books, fair values, signals, risk, OMS, reconciliation, kill switch |
| `feeds`    | `python -m sigcup.cli feeds`      | none (public APIs) + `SIG_READ_KEY` is *not* needed | Kalshi/Polymarket reference prices, ratings/polls, GDELT/RSS news, Bluesky/X/YouTube posts, stance classification, features, events |
| `reporter` | `python -m sigcup.cli report loop`| none (database only) | hourly status line + CLV, daily report at 12:00 UTC, heartbeat |

Everything else (`sigcup status`, `sigcup report …`, `sigcup canary`, `sigcup kill`) is a one-off
command you run by hand.

---

## 1. Install

Requirements: Python 3.12+ (3.13 tested), [`uv`](https://docs.astral.sh/uv/), Node 18+ for pm2.

```bash
git clone <repo> sigprediction && cd sigprediction
uv sync --extra dev --extra social          # creates .venv with every dependency
cp .env.example .env && chmod 600 .env      # fill in keys (section 2)
.venv/bin/sigcup --help
.venv/bin/sigcup db init                    # creates data/sigcup.db (idempotent)
npm install -g pm2
```

`.venv/bin/sigcup` and `.venv/bin/python -m sigcup.cli` are the same thing; the pm2 file uses the
latter so it never depends on `PATH`.

Run the tests once on the machine that will trade (no network is needed):

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check sigcup tests scripts
```

## 2. Keys and `.env`

### Creating the SIG API key

1. Sign in at `https://sig.thesuper.market`, confirm your e-mail, **register for the tournament**
   (`2026 Midterm Elections Competition`) and accept the current Terms. Until all three are done every
   API call returns 403.
2. **My Profile → API Keys → create key** with the `read` **and** `trade` scopes. Copy it once; it is
   not shown again. Put it in `SIG_API_KEY`. (Optionally create a second, read-only key for
   `SIG_READ_KEY`; the feeds/reporter processes do not need it today but the paper trader uses it as
   a live market-data source when present.)
3. **Enrolment step: the account shows "Pending enrolment: make one trade to join" until one trade
   has been made.** Place one tiny trade in the web app (or run `sigcup canary --live --take` once
   `LIVE=1`), otherwise the leaderboard/portfolio endpoints stay empty and you are not ranked.
4. Rate limits are **per account across all keys**: 100 reads + 30 writes per minute. Two keys do
   not double the budget, which is why only `trader` ever sends writes.

### `.env` fields (see `.env.example`)

| variable | used by | meaning |
|---|---|---|
| `SIG_BASE_URL` | trader, canary, bootstrap | documented v1 API base (`https://sig.thesuper.market/api/v1`) |
| `SIG_API_KEY` | **trader**, `canary --live`, `bootstrap --probe-writes`, `kill --cancel-now` | trade-scope key. Never put it on another machine. |
| `SIG_READ_KEY` | paper trader (market data), bootstrap (read calls) | read-scope key; optional |
| `SIG_TOURNAMENT_SLUG` | trader, bootstrap | e.g. `2026-midterm-elections`; `sigcup bootstrap` reports the slugs it sees |
| `LIVE` | everything that can send an order | `1` = real orders allowed; anything else = paper. **Both `LIVE=1` and `SIG_API_KEY` are required** for `trade --live`, `canary --live`, `bootstrap --probe-writes`, `kill --cancel-now`. |
| `KILL_FILE` | trader, `sigcup kill` | default `data/KILL` |
| `DSN` / `DATA_DIR` | all | database and data directory (`sqlite:///data/sigcup.db`, `data`) |
| `ANTHROPIC_API_KEY` | feeds (stance classification) | without it the `nlp_stance` collector is skipped (features/lead-lag stay empty) |
| `X_VENDOR`, `X_VENDOR_KEY` | feeds | `none` (default, X skipped) \| `socialdata` \| `twitterapi` \| `official` |
| `BLUESKY_HANDLE`, `BLUESKY_APP_PASSWORD` | feeds | optional (Jetstream needs no auth; search does) |
| `YOUTUBE_API_KEY` | feeds | optional; without it YouTube is skipped |
| `FEC_API_KEY` | feeds | `DEMO_KEY` works at low volume |
| `REDDIT_*` | feeds | leave empty: Reddit is disabled (section 9) |

Secrets are never logged; the client never logs headers. Keep `.env` at mode 600.

## 3. Bootstrap the universe

```bash
.venv/bin/sigcup bootstrap                 # SIG market list + relationships -> config/races.yaml + DB
.venv/bin/sigcup bootstrap -- --dry-run    # print what would change, write nothing
.venv/bin/sigcup bootstrap -- --skip-crosswalk
.venv/bin/sigcup universe                  # table of races / markets / exclusivity / crosswalk coverage
```

`bootstrap` also runs the day-1 probes: fee presence in fills, whether the tournament relationship
graph is empty (if it is, **no Dutch book is riskless**: the `arb` strategy stays idle because it
requires API-verified `mutually_exclusive` + `isExhaustive`), and the batch-write-count probe.
The write probe is only described unless you pass `--probe-writes` (`LIVE=1` + trade key: it places
and cancels two 1-share bids at 0.005).

`config/races.yaml` is hand-editable; bootstrap overwrites only exchange-owned fields.

## 4. pm2 operations

```bash
pm2 start ecosystem.config.js        # trader (paper unless LIVE=1), feeds, reporter
pm2 status
pm2 logs trader --lines 100          # also: feeds, reporter; files live in logs/
pm2 restart trader                   # on SIGINT/SIGTERM the LIVE trader cancels its resting orders first
pm2 stop trader                      # stops quoting; resting orders expire within risk.order_expiry_s (15 min)
pm2 stop all && pm2 delete all       # full teardown
pm2 save                             # remember the process list ...
pm2 startup                          # ... and print the command that makes pm2 start at boot (run it)
```

The ecosystem file sets `max_restarts: 20`, `restart_delay: 5000` and `kill_timeout: 15000` on
purpose: a crash loop must not burn the write budget with cancel-all / re-lay cycles, and the trader
needs a few seconds to cancel quotes on shutdown. If a process is flapping, `pm2 stop` it and read
the log before restarting.

Changing `.env` or `config/settings.yaml` requires `pm2 restart <app>`.

## 5. Go-live sequence: paper → canary → small caps → full caps

1. **Paper (≥ 1 day).** `LIVE=0`. `pm2 start ecosystem.config.js`. After an hour: `sigcup status`
   shows three fresh heartbeats, `trader` status `paper`, open orders by strategy; `sigcup report
   daily` shows paper P&L and CLV. Paper fills happen when our resting price crosses the real best
   quote. Nothing is sent to the exchange.
2. **Canary (one order, 10 shares).**
   ```bash
   # in .env: LIVE=1, SIG_API_KEY=...   (pm2 trader still stopped or in paper)
   .venv/bin/sigcup canary --live                 # joins the best bid, waits 30 s, cancels
   .venv/bin/sigcup canary --live --take          # crosses the spread so it fills (also completes enrolment)
   ```
   The table must end with `verdict OK` and a known final state (`filled`, `cancelled`,
   `partial-cancelled`, `expired`). `uncertain` means the exchange answered 502
   `ORDER_STATUS_UNKNOWN` / timed out: do **not** retry blindly; check the web app, then
   `sigcup status` (the intent is stored as UNCERTAIN and the trader reconciles it at startup).
   A canary never bids above 0.50 unless `--take` is given, so the worst case is 5 SUSQies.
3. **Forced 429 recovery (optional but recommended).** Run `sigcup canary --live` four times inside a
   minute while the trader is live: the limiter must queue, not 429. If a 429 does appear, the log
   shows `RateLimited` and every write is frozen for `Retry-After` (60 s) — that is the designed
   behaviour, verify nothing else happened.
4. **Small caps (one hour).** In `config/settings.yaml` set `risk.per_race_cap_frac: 0.005` and
   `risk.per_state_cap_frac: 0.02`, then `pm2 restart trader` with `LIVE=1`. Watch `pm2 logs
   trader` for `cycle:` lines (placed/cancelled counts), `sigcup status` for open orders and the
   reservation ≤ cash invariant, and `sigcup report daily` for fills tagged by strategy.
5. **Full caps.** Restore `0.03` / `0.08`, `pm2 restart trader`. Keep the daily checklist.

Switching back to paper at any time: `LIVE=0` in `.env` + `pm2 restart trader`
(`sigcup trade --paper` ignores `LIVE`).

## 6. Daily checklist (5 minutes)

```bash
.venv/bin/sigcup status              # exit code 1 means there are alerts: read them
.venv/bin/sigcup report daily        # P&L by strategy, CLV, calibration, SIG-vs-ref gaps, rank history
.venv/bin/sigcup report leadlag      # only matters once features exist (>= 1 week of posts)
pm2 logs trader --lines 50 --nostream
```

What to look at, in order:

1. **Heartbeats**: all three fresh (`trader` < 90 s, `feeds` < 4 min, `reporter` < 75 min). A
   stale trader means no quotes are being maintained; resting orders expire within 15 min.
2. **Rank**: `rank/members` from the last leaderboard snapshot (every 15 min). The interim
   leaderboard marks at recent trades and is noisy; the endgame module uses it only in the last 72 h.
3. **Reserve**: `free` (cash − reservation) must stay above 20% of the bankroll (30% in the final
   72 h). `status` alerts when it does not. The exchange does not reserve cash for resting orders; the
   OMS keeps an internal reservation ≤ cash and sheds buys if it is violated (`reconcile repair` lines).
4. **Open orders**: count and notional by strategy; `uncertain > 0` must be 0 after the next
   reconcile (60 s). `arb` orders only exist if the relationships probe found exclusive groups.
5. **CLV**: `clv24h` per strategy should be ≥ 0 on average; a persistently negative `theta` CLV
   means we are buying favourites too high (raise `theta.min_yield`); a negative `value` CLV means
   the fair value is worse than the market (check `sig_vs_ref` and the ref feed heartbeat).
6. **P&L**: realised + unrealised by strategy and by mode (paper/live are never mixed). Judge
   process (CLV), not outcome.
7. **Reports on disk**: `data/reports/<day>.json` (the reporter writes one per day at 12:00 UTC).

## 7. Kill switch

```bash
.venv/bin/sigcup kill                 # touch data/KILL and print what happens next
.venv/bin/sigcup kill --cancel-now    # also cancel-all immediately through a direct client (LIVE=1 + key)
.venv/bin/sigcup kill --clear         # remove the file + clear the DB flag; then: pm2 restart trader
```

What the trader does when the file appears (checked every 5 s and on every cycle): cancel-all,
write a `killed` flag to the KV table, log `KILL SWITCH`, exit. pm2 restarts it but it refuses to
trade while the file or the flag exists. The same path fires automatically on the daily loss limit
(`risk.daily_loss_limit_frac`, 5%). `kill` warns when the trader heartbeat is stale, because then
nothing is running to cancel orders: use `--cancel-now` or cancel in the web app.

Other automatic stops: no order is sent after `deadline_utc` minus 30 s; every resting order carries
an expiry (`risk.order_expiry_s`) as a dead-man's switch; a 401/403 marks the OMS `auth_failed` and
stops sends until restart.

## 8. Election night (Nov 3 evening → Nov 4 noon ET)

Before (by Oct 28, full dry run in paper):
- `config/settings.yaml`: `election_night.start_utc` (default 2026-11-03T23:00Z), `take_when_ref_ge`
  0.97, `min_lag` 0.08, `reserve_deploy_frac` 0.8; `endgame.final_hours` 72.
- Feeds switch the reference cadence to 10 s automatically while `election_night.is_active`.
- Rebuild the liquid reserve to 30% by letting arbs/short orders roll off, not by selling theta.

Checklist for the night:
1. `sigcup status` every hour; confirm `feeds` fresh (reference prices are the trigger — never a SIG
   print) and `trader` fresh.
2. Watch `pm2 logs trader | grep -E "take|election"`: takes only when Kalshi/Polymarket ≥ 0.97 and
   SIG lags by ≥ 8¢.
3. Expect 429s if the book moves fast: writes freeze 60 s, then resume. Do not restart the trader
   during a freeze.
4. Keep the kill command ready in a second terminal.
5. Settlement edge cases to remember: GA runoffs settle in December (dead capital for interim
   marks), RI fusion tickets count for every party, independents (NE/MT Senate, RI Gov) break
   party-leg exclusivity, N/A markets refund cost, House control may be uncalled on Nov 4.
6. After 17:00 UTC Nov 4 nothing is sent; leave the processes running so fills/positions are
   reconciled and the final report is produced, then `pm2 stop all`.

## 9. What is deliberately not collected, and the X note

- **LinkedIn**: no keyword search API, scraping is litigated, near-zero electoral signal. Skipped.
- **Instagram / Facebook**: the Graph API reads only pages you own, CrowdTangle is gone and the
  academic replacement is not available to us; Apify-style scraping breaches ToS. Skipped.
- **Reddit**: needs OAuth credentials with a 2–4 week manual approval; none available, so the
  collector is scaffolded but disabled (`REDDIT_*` empty → skipped with a log line).
- **X**: the official pay-per-use API is ~$5 per 1k posts. The collector is vendor-agnostic
  (`X_VENDOR=socialdata|twitterapi|official`). Third-party vendors are ~25x cheaper but **buying
  X data from a scraper vendor breaches X's Terms of Service**; the user accepted that risk. Set
  `X_VENDOR=none` to opt out at any time.
- Sentiment *levels* are never used as a forecast (the literature says they fail the polls
  baseline); the pipeline is an event detector and stance-change signal gated by the measured IC
  (`sigcup report leadlag`). `fade`/`follow` stay disabled until IC > 0 on ≥ 1 week of data.

## 10. Cost estimates

| item | estimate |
|---|---|
| X via vendor | ~$0.20 per 1k posts; ~300k posts over the window ≈ $60 |
| X official API | ~$5 per 1k posts (not recommended) |
| Claude Haiku stance classification (batch, 25 posts per request, cached by text hash) | ~$0.10–0.20 per 10k posts; escalations to Sonnet add a few percent |
| Kalshi, Polymarket, GDELT, RSS, Bluesky Jetstream, YouTube Data API (10k units/day), FEC | free |
| SIG API | free; play money |

## 11. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| every SIG call 401/403 (`AuthError`), trader logs `auth_failed` | key missing/wrong scope, e-mail unconfirmed, tournament not joined, Terms not accepted, or "pending enrolment" | check `.env`, redo section 2 steps 1–3, make one trade, `pm2 restart trader` |
| `RateLimited` / 429 in logs, no orders for 60 s | the 100/30 per-minute budget was exceeded (another bot or the web app on the same account counts) | nothing to do: all writes are frozen for `Retry-After`; stop other clients; do not restart |
| `UNCERTAIN` intents in `sigcup status` | 502 `ORDER_STATUS_UNKNOWN` or a timeout: the order may or may not exist | wait for the next reconcile (60 s): the OMS re-POSTs with the SAME idempotency key, 409 `REQUEST_IN_FLIGHT` means the 90 s lease is still held and it polls `/orders` + `/fills` instead. Never re-send by hand. |
| `reservation > cash` discrepancy | a fill arrived between cycles | automatic: the OMS cancels the lowest-priority buys (`reconcile repair`) |
| `tournamentId required` / orders landing on the public book | `SIG_TOURNAMENT_SLUG` empty or wrong | run `sigcup bootstrap -- --dry-run`, set the slug, restart |
| `kill file present` on every start | you killed it | `sigcup kill --clear` after investigating |
| `report leadlag` prints `(no feature rows yet)` | no stance features (no `ANTHROPIC_API_KEY`, or < 3 baseline days) | expected for the first days |
| `sigcup status` exit code 1 | alerts present | read them; exit 0 means clean |
| `universe` empty | bootstrap not run | `sigcup bootstrap`; `--source yaml` shows the seed |
| trader starts but `fair=0` every cycle | no reference prices / ratings yet | check `feeds` heartbeat and `data/` for `ref_prices` rows (`sigcup db stats`) |
| `database is locked` | another writer held SQLite > 30 s | WAL + busy_timeout handle normal contention; stop stray `sigcup` shells |
| clock drift warnings | local clock vs exchange `Date` header | the client measures the offset; fix NTP on the host |

Log files: `logs/<app>.out.log` / `logs/<app>.err.log` (pm2), `data/reports/*.json` (daily),
`data/sigcup.db` (everything; copy it with `sqlite3 .backup` or while the processes are stopped).
