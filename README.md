# sigcup — automated trading for SIG's Predictions Cup (2026 US midterms)

A Python 3.13 trading system for the SIG Predictions Cup (237 play-money political markets on the 2026
House, Senate and Governor races, trading until 2026-11-04 12:00 ET). It runs 24/7 under pm2 as three
processes (`trader`, `feeds`, `reporter`) over one SQLite database.

- `docs/STRATEGY.md` — the strategy memo: where the edge comes from, with citations.
- `docs/RUNBOOK.md` — install, keys, pm2 operations, daily checklist, kill switch.
- `docs/research/` — the research reports the strategy is built on.

Quick start (on your own machine; see the RUNBOOK for details):

```bash
uv sync --extra dev --extra social
cp .env.example .env            # fill in keys; keep LIVE=0 for paper mode
.venv/bin/sigcup db init
.venv/bin/sigcup bootstrap       # market universe + crosswalk + day-1 probes
npm i -g pm2 && pm2 start ecosystem.config.js
.venv/bin/sigcup status
```
