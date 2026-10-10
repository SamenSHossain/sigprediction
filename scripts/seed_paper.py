"""Seed an OFFLINE paper-trading database so ``sigcup trade --paper --once`` runs a full cycle with no
network access and no API key (before ``scripts/bootstrap_races.py`` has ever talked to the exchange).

What it writes (all idempotent; nothing already present is overwritten):

1. ``Race`` rows from config/races.yaml (``sigcup.feeds.polls.ensure_race_rows``).
2. ``Market`` rows for every leg of races.yaml. Legs that already carry a SIG id use it; legs without one
   get the synthetic ids ``paper:<race_id>:<party>`` (market id) / ``paper-<race_id>-<party>`` (exchange
   id), which the bootstrap replaces once it runs (it keys on the SIG ids, so delete the synthetic rows
   first: ``--purge`` does that).
3. ``Rating`` rows from config/ratings_manual.yaml (``polls.run_once`` with every network fetch off).
4. ``Quote`` rows (``source='seed'``) for every market: from ``--quotes PATH`` (YAML/JSON mapping of
   exchange id or ``RACE-ID:PARTY`` to ``{bid, ask, last, bid_size, ask_size}``), else synthesised around
   the ratings-implied probability (D leg = mean P(D) of the stored ratings, R leg = the complement,
   other legs ``--minor-prob``) with ``--half-spread`` and ``--size`` on each side.

The PaperAdapter (DB-quote mode) then serves these as the market, so the cycle produces fair values,
desired orders, paper intents/fills, an AccountSnap and the ``trader`` heartbeat.

Usage::

    python scripts/seed_paper.py                      # DSN from the environment / .env
    DSN=sqlite:///data/it.db python scripts/seed_paper.py --half-spread 0.03
    python scripts/seed_paper.py --quotes tests/fixtures/integration/quotes.yaml
    python scripts/seed_paper.py --purge               # drop the synthetic rows (before bootstrap)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import delete, select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `python scripts/seed_paper.py` without installing the package
    sys.path.insert(0, str(ROOT))

from sigcup.common.log import get_logger, setup_logging  # noqa: E402
from sigcup.common.time import utc_now  # noqa: E402
from sigcup.db import engine as dbengine  # noqa: E402
from sigcup.db.models import Market, Quote, Race, Rating  # noqa: E402
from sigcup.feeds import polls  # noqa: E402
from sigcup.feeds import universe as U  # noqa: E402
from sigcup.settings import Secrets, StrategyConfig, config  # noqa: E402
from sigcup.settings import secrets as load_secrets  # noqa: E402
from sigcup.sig.schemas import PRICE_MAX, PRICE_MIN, round_tick  # noqa: E402

log = get_logger(__name__)

SessionFactory = Callable[[], Any]
SYNTH_MARKET_PREFIX = "paper:"
SYNTH_EXCHANGE_PREFIX = "paper-"
QUOTE_SOURCE = "seed"


@dataclass
class SeedReport:
    races_inserted: int = 0
    markets_inserted: int = 0
    markets_existing: int = 0
    ratings_written: int = 0
    quotes_written: int = 0
    purged_markets: int = 0
    purged_quotes: int = 0
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [
            f"races inserted:   {self.races_inserted}",
            f"markets inserted: {self.markets_inserted} (existing kept: {self.markets_existing})",
            f"ratings written:  {self.ratings_written}",
            f"quotes written:   {self.quotes_written}",
        ]
        if self.purged_markets or self.purged_quotes:
            lines.append(f"purged: {self.purged_markets} synthetic markets, {self.purged_quotes} seed quotes")
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


def synthetic_ids(race_id: str, party: str) -> tuple[str, str]:
    """(market_id, exchange_id) used for a leg that has no SIG ids yet."""
    return f"{SYNTH_MARKET_PREFIX}{race_id}:{party}", f"{SYNTH_EXCHANGE_PREFIX}{race_id}-{party}"


def _clean(v: Any) -> str | None:
    s = str(v).strip() if v is not None else ""
    return s or None


def seed_markets(session_factory: SessionFactory, data: Mapping[str, Any], now: datetime) -> tuple[int, int]:
    """Insert a Market row per races.yaml leg that has none; returns (inserted, existing)."""
    inserted = existing = 0
    with session_factory() as s:
        have_race = set(s.execute(select(Race.race_id)).scalars().all())
        by_leg = {(m.race_id, m.party): m for m in s.execute(select(Market)).scalars().all()}
        taken_eids = {m.exchange_id for m in by_leg.values()}
        taken_mids = {m.market_id for m in by_leg.values()}
        for race_id, party, entry, leg in U.market_legs(dict(data)):
            if race_id not in have_race:
                continue
            if (race_id, party) in by_leg:
                existing += 1
                continue
            mid, eid = synthetic_ids(race_id, party)
            market_id = _clean(leg.get("sig_market_id")) or mid
            exchange_id = _clean(leg.get("sig_exchange_id")) or eid
            if market_id in taken_mids or exchange_id in taken_eids:
                existing += 1
                continue
            label = str(entry.get("label") or race_id)
            row = Market(
                market_id=market_id,
                exchange_id=exchange_id,
                race_id=race_id,
                party=party,
                title=_clean(leg.get("title")) or f"{label} - {party} wins",
                status=_clean(leg.get("status")) or "open",
                kalshi_ticker=_clean(leg.get("kalshi_ticker")),
                polymarket_condition_id=_clean(leg.get("polymarket_condition_id")),
                polymarket_token_id=_clean(leg.get("polymarket_token_id")),
                predictit_id=_clean(leg.get("predictit_id")),
                updated_at=now,
            )
            s.add(row)
            by_leg[(race_id, party)] = row
            taken_eids.add(exchange_id)
            taken_mids.add(market_id)
            inserted += 1
    return inserted, existing


def ratings_pdem(session_factory: SessionFactory) -> dict[str, float]:
    """Mean P(D) over the latest stored rating of each source, per race."""
    latest: dict[tuple[str, str], Rating] = {}
    with session_factory() as s:
        for r in s.execute(select(Rating)).scalars().all():
            key = (r.race_id, r.source)
            cur = latest.get(key)
            if cur is None or (r.ts, r.id) > (cur.ts, cur.id):
                latest[key] = r
    acc: dict[str, list[float]] = {}
    for (race_id, _), r in latest.items():
        if r.prob_dem is not None:
            acc.setdefault(race_id, []).append(float(r.prob_dem))
    return {rid: sum(v) / len(v) for rid, v in acc.items() if v}


def _clip(p: float) -> float:
    return round_tick(min(PRICE_MAX, max(PRICE_MIN, p)))


def synth_quotes(
    markets: Sequence[Market], pdem: Mapping[str, float], *, half_spread: float, size: int, minor_prob: float
) -> dict[str, dict[str, Any]]:
    """A two-sided quote per market around the ratings-implied probability of its leg."""
    by_race: dict[str, list[Market]] = {}
    for m in markets:
        by_race.setdefault(m.race_id, []).append(m)
    out: dict[str, dict[str, Any]] = {}
    for race_id, legs in by_race.items():
        p = float(pdem.get(race_id, 0.5))
        parties = {m.party for m in legs}
        minors = [m for m in legs if m.party not in ("D", "R")]
        minor_total = minor_prob * len(minors)
        for m in legs:
            if m.party == "D":
                mid = p
            elif m.party == "R":
                mid = 1.0 - p - (minor_total if "D" in parties else 0.0)
            else:
                mid = minor_prob
            mid = min(0.99, max(0.01, mid))
            out[m.exchange_id] = {
                "bid": _clip(mid - half_spread),
                "ask": _clip(mid + half_spread),
                "last": _clip(mid),
                "bid_size": size,
                "ask_size": size,
            }
    return out


def load_quotes_file(path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text()) if path.suffix in (".yaml", ".yml") else json.loads(path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: expected a mapping of exchange id (or RACE-ID:PARTY) to quote fields")
    return {str(k): dict(v) for k, v in raw.items() if isinstance(v, Mapping)}


def write_quotes(
    session_factory: SessionFactory, quotes: Mapping[str, Mapping[str, Any]], markets: Sequence[Market], now: datetime
) -> int:
    """Insert one Quote row per market present in ``quotes`` (keys may be exchange ids or RACE-ID:PARTY)."""
    by_eid = {m.exchange_id: m for m in markets}
    by_leg = {f"{m.race_id}:{m.party}": m for m in markets}
    n = 0
    with session_factory() as s:
        for key, q in quotes.items():
            m = by_eid.get(key) or by_leg.get(key)
            if m is None:
                log.warning("seed: quote key %s matches no market; skipped", key)
                continue
            bid = q.get("bid")
            ask = q.get("ask")
            last = q.get("last")
            s.add(
                Quote(
                    ts=now,
                    exchange_id=m.exchange_id,
                    bid=_clip(float(bid)) if bid is not None else None,
                    ask=_clip(float(ask)) if ask is not None else None,
                    last=_clip(float(last)) if last is not None else None,
                    bid_size=int(q.get("bid_size") or 0) or None,
                    ask_size=int(q.get("ask_size") or 0) or None,
                    source=QUOTE_SOURCE,
                )
            )
            n += 1
    return n


def purge(session_factory: SessionFactory) -> tuple[int, int]:
    """Delete the synthetic Market rows and every seed quote (run before the bootstrap)."""
    with session_factory() as s:
        nq = s.execute(delete(Quote).where(Quote.source == QUOTE_SOURCE)).rowcount
        nm = s.execute(delete(Market).where(Market.market_id.like(f"{SYNTH_MARKET_PREFIX}%"))).rowcount
    return int(nm or 0), int(nq or 0)


async def seed(
    *,
    cfg: StrategyConfig | None = None,
    sec: Secrets | None = None,
    session_factory: SessionFactory | None = None,
    races_yaml: Path | None = None,
    quotes_path: Path | None = None,
    half_spread: float = 0.02,
    size: int = 200,
    minor_prob: float = 0.03,
    now: datetime | None = None,
) -> SeedReport:
    """Run every seeding step; see the module docstring."""
    cfg = cfg or config()
    sec = sec or load_secrets()
    sf = session_factory or dbengine.session
    now = now or utc_now()
    rep = SeedReport()
    data = U.load_races_yaml(races_yaml)
    rep.races_inserted = polls.ensure_race_rows(sf, data)
    rep.markets_inserted, rep.markets_existing = seed_markets(sf, data, now)
    rep.ratings_written = await polls.run_once(sf, cfg, sec, now=now, fetch_web=False, fetch_fec=False)
    with sf() as s:
        markets = list(s.execute(select(Market)).scalars().all())
    if quotes_path is not None:
        quotes = load_quotes_file(quotes_path)
        rep.notes.append(f"quotes from {quotes_path}")
    else:
        pdem = ratings_pdem(sf)
        quotes = synth_quotes(markets, pdem, half_spread=half_spread, size=size, minor_prob=minor_prob)
        rep.notes.append(f"quotes synthesised around ratings for {len(pdem)} races (others at 0.50)")
    rep.quotes_written = write_quotes(sf, quotes, markets, now)
    return rep


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seed_paper", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--races-yaml", type=Path, default=None, help="races.yaml to read (default config/races.yaml)")
    parser.add_argument("--quotes", type=Path, default=None, help="YAML/JSON quotes fixture instead of synthesised quotes")
    parser.add_argument("--half-spread", type=float, default=0.02)
    parser.add_argument("--size", type=int, default=200, help="displayed size on each side")
    parser.add_argument("--minor-prob", type=float, default=0.03, help="mid for independent/other legs")
    parser.add_argument("--purge", action="store_true", help="delete synthetic markets and seed quotes, then exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    dbengine.init_db()
    if args.purge:
        nm, nq = purge(dbengine.session)
        print(f"purged: {nm} synthetic markets, {nq} seed quotes")
        return 0
    rep = asyncio.run(
        seed(
            races_yaml=args.races_yaml,
            quotes_path=args.quotes,
            half_spread=args.half_spread,
            size=args.size,
            minor_prob=args.minor_prob,
        )
    )
    print(rep.text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
