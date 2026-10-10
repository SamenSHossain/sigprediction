"""Pull the tournament market list + relationships from the SIG API, map each market to a race_id/party,
crosswalk to Kalshi tickers and Polymarket condition ids by title matching, and write config/races.yaml
(preserving hand edits) and Race/Market rows. Also runs the day-1 probes: fee check, batch-write-count
probe, relationship graph emptiness.

Steps (``python scripts/bootstrap_races.py``; see ``--help``):

1. ``list_markets`` on the read key (SIG_READ_KEY, else SIG_API_KEY); parse every title with
   sigcup.feeds.universe.parse_title; unparseable titles are reported, never guessed.
2. Merge into config/races.yaml: exchange-owned fields (sig_market_id, sig_exchange_id, title, status,
   settlement_date) are overwritten, every hand-set field (pvi, incumbent, candidates, notes, crosswalk
   ids) is preserved; new races are created from the title.
3. ``get_relationships``: ``exclusive`` / ``exhaustive`` are set ONLY when a mutually_exclusive group
   covers every SIG leg of the race (exhaustive only when the group is exactly the race's legs and the
   API says isExhaustive); otherwise they are reset to null. Titles never decide.
4. Crosswalk: Kalshi tickers from the documented patterns, verified by fetching (batch GET
   /markets?tickers=) and rejecting titles that contradict the race; Polymarket via Gamma slug lookups
   then /public-search, accepted only above a title-similarity score. Existing ids are kept.
5. Save races.yaml and upsert Race / Market rows (``--dry-run`` skips both, ``--no-db`` skips the DB).
6. Probes: fee presence in the fills sample, relationship-graph emptiness, and the batch-write-count
   probe, which is a *dry* description unless ``--probe-writes`` is passed (then it places a 2-order
   batch of 1-share YES bids at 0.005 with a 120 s expiry and cancels them; needs the trade key).

The SIG client is imported lazily (it is written concurrently) and everything is coded against
sigcup.sig.adapter.ExchangeAdapter so tests can inject a fake.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # allow `python scripts/bootstrap_races.py` without installing the package
    sys.path.insert(0, str(ROOT))

from sigcup.common.ids import new_idempotency_key  # noqa: E402
from sigcup.common.log import get_logger, setup_logging  # noqa: E402
from sigcup.common.time import ensure_utc, utc_now  # noqa: E402
from sigcup.db import engine as dbengine  # noqa: E402
from sigcup.db.models import Market, Race  # noqa: E402
from sigcup.feeds import ref_markets as RM  # noqa: E402
from sigcup.feeds import universe as U  # noqa: E402
from sigcup.settings import Secrets, StrategyConfig, config  # noqa: E402
from sigcup.settings import secrets as load_secrets  # noqa: E402
from sigcup.sig.adapter import ExchangeAdapter  # noqa: E402
from sigcup.sig.ratelimit import Budget  # noqa: E402
from sigcup.sig.schemas import PRICE_MIN, MarketInfo, OrderRequest, Relationship, round_tick  # noqa: E402

log = get_logger("bootstrap")

KALSHI_MAX_CANDIDATES = 8
POLY_MAX_SLUGS = 6
POLY_MIN_SCORE = 0.6
PAUSE_S = 0.15  # between external calls (public APIs: be polite)
PROBE_ORDER_QTY = 1
PROBE_ORDER_TTL_S = 120

BATCH_PROBE_PROCEDURE = (
    "Places ONE POST /orders/batch carrying 2 harmless orders (buy YES, 1 share, price 0.005, 120 s expiry, "
    "on the two open markets with the highest YES ask so they cannot fill), then cancels both. Our limiter "
    "charges one write for the batch call; the exchange's own accounting is read from its rate-limit headers "
    "(any *ratelimit*remaining* header on GET /account before and after). If the remaining-writes counter "
    "drops by 1 the batch counts as ONE write (writes stop binding: re-plan the budget); if it drops by 2 "
    "(or more) each order is a write. Without such headers the probe is inconclusive and the fallback is "
    "to count batched orders until an early 429 appears (do that by hand, it freezes writes for 60 s)."
)


# ---------------------------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------------------------


@dataclass
class MappedMarket:
    market: MarketInfo
    parsed: U.ParsedTitle


@dataclass
class BootstrapReport:
    sig_markets: int = 0
    mapped: int = 0
    unmapped: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    member_count: int | None = None
    legs_added: int = 0
    legs_updated: int = 0
    relationships: int = 0
    races_exclusive: int = 0
    races_exhaustive: int = 0
    partial_relationships: list[str] = field(default_factory=list)
    kalshi_found: int = 0
    kalshi_missing: int = 0
    polymarket_found: int = 0
    polymarket_missing: int = 0
    db_races: int = 0
    db_markets: int = 0
    probes: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    stats: U.UniverseStats | None = None

    def text(self) -> str:
        st = self.stats or U.UniverseStats()
        lines = [
            "== bootstrap report ==",
            f"SIG markets listed: {self.sig_markets}  mapped: {self.mapped}  unmapped: {len(self.unmapped)}  "
            f"duplicates: {len(self.duplicates)}  memberCount: {self.member_count}",
            f"races.yaml: {st.races} races / {st.legs} legs ({st.by_chamber}); legs with SIG id {st.with_sig}, "
            f"Kalshi {st.with_kalshi}, Polymarket {st.with_polymarket}",
            f"merge: legs added {self.legs_added}, updated {self.legs_updated}",
            f"relationships: {self.relationships} groups -> exclusive races {self.races_exclusive}, exhaustive "
            f"{self.races_exhaustive}, partial coverage {len(self.partial_relationships)}",
            f"crosswalk: kalshi found {self.kalshi_found} missing {self.kalshi_missing}; polymarket found "
            f"{self.polymarket_found} missing {self.polymarket_missing}",
            f"db: {self.db_races} races, {self.db_markets} markets upserted",
        ]
        for name, probe in self.probes.items():
            lines.append(f"probe {name}: {probe.get('verdict', '?')}")
            for k, v in probe.items():
                if k != "verdict":
                    lines.append(f"    {k}: {v}")
        if self.unmapped:
            lines.append("unmapped titles:")
            lines.extend(f"    {t}" for t in self.unmapped[:50])
        if self.duplicates:
            lines.append("duplicate (race, party) mappings (first kept):")
            lines.extend(f"    {t}" for t in self.duplicates[:50])
        if self.partial_relationships:
            lines.append("relationship groups that cover only part of a race (flags left null):")
            lines.extend(f"    {t}" for t in self.partial_relationships[:50])
        lines.extend(f"note: {n}" for n in self.notes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# 1-2. map titles and merge into races.yaml
# ---------------------------------------------------------------------------------------------


def map_markets(markets: Iterable[MarketInfo]) -> tuple[list[MappedMarket], list[MarketInfo]]:
    """Parse every SIG title; returns (mapped, unmapped). Nothing is guessed for unparseable titles."""
    mapped: list[MappedMarket] = []
    unmapped: list[MarketInfo] = []
    for m in markets:
        parsed = U.parse_title(m.title)
        if parsed is None:
            unmapped.append(m)
        else:
            mapped.append(MappedMarket(m, parsed))
    return mapped, unmapped


def merge_markets(data: dict[str, Any], mapped: Iterable[MappedMarket], *, now: datetime | None = None) -> tuple[int, int, list[str]]:
    """Write the exchange-owned leg fields into the universe, preserving every hand-set value.
    Returns (legs added, legs updated, duplicate descriptions)."""
    added = updated = 0
    seen: dict[tuple[str, str], str] = {}
    dups: list[str] = []
    for mm in mapped:
        rid, party = mm.parsed.race_id, mm.parsed.party
        key = (rid, party)
        if key in seen:
            dups.append(f"{rid}/{party}: market {mm.market.id} {mm.market.title!r} (kept {seen[key]})")
            continue
        seen[key] = str(mm.market.id)
        entry = U.race_entry(data, rid, create=True, special=mm.parsed.special)
        if entry is None:
            continue
        if mm.parsed.special and not entry.get("special"):
            entry["special"] = True
        if mm.parsed.candidate:
            cands = entry.setdefault("candidates", {})
            c = cands.get(party)
            if not isinstance(c, dict):
                c = {"name": "", "aliases": [], "handles": [], "hashtags": []}
                cands[party] = c
            if not c.get("name"):
                c["name"] = mm.parsed.candidate
        leg = U.market_entry(entry, party)
        before = {k: leg.get(k) for k in U.EXCHANGE_OWNED_FIELDS}
        leg["sig_market_id"] = str(mm.market.id)
        leg["sig_exchange_id"] = str(mm.market.exchange_id) if mm.market.exchange_id else ""
        leg["title"] = mm.market.title
        leg["status"] = mm.market.status or ""
        leg["settlement_date"] = ensure_utc(mm.market.settlement_date).isoformat() if mm.market.settlement_date else ""
        if not before.get("sig_exchange_id") and not before.get("sig_market_id"):
            added += 1
        elif any(leg.get(k) != before.get(k) for k in U.EXCHANGE_OWNED_FIELDS):
            updated += 1
    if now is not None:
        data["bootstrapped_at"] = now.isoformat()
    return added, updated, dups


# ---------------------------------------------------------------------------------------------
# 3. relationships -> exclusive / exhaustive (API evidence only)
# ---------------------------------------------------------------------------------------------


def _is_exclusive_kind(rel: Relationship) -> bool:
    kind = (rel.kind or "").lower().replace("-", "_")
    return "exclusive" in kind or kind in ("one_of", "exactly_one", "at_most_one")


def apply_relationships(data: dict[str, Any], relationships: Sequence[Relationship]) -> dict[str, Any]:
    """Set every race's ``exclusive`` / ``exhaustive`` from the relationships API and nothing else.

    A race is exclusive when one mutually-exclusive group contains all of its SIG legs; it is exhaustive
    only when that group is exactly its legs and the API marks the group exhaustive (a larger exhaustive
    group means something outside the race's legs can win -> False). Races without such evidence get
    null, even if a previous run or a hand edit said otherwise.
    """
    legs_by_race: dict[str, set[str]] = {}
    race_of_eid: dict[str, str] = {}
    for rid, _party, _entry, leg in U.market_legs(data):
        eid = str(leg.get("sig_exchange_id") or "")
        if eid:
            legs_by_race.setdefault(rid, set()).add(eid)
            race_of_eid[eid] = rid
    for entry in (data.get("races") or {}).values():
        if isinstance(entry, dict):
            entry["exclusive"] = None
            entry["exhaustive"] = None
    partial: list[str] = []
    exclusive = exhaustive = 0
    for rel in relationships:
        if not _is_exclusive_kind(rel):
            continue
        ids = {str(x) for x in rel.exchange_ids}
        touched = {race_of_eid[e] for e in ids if e in race_of_eid}
        for rid in sorted(touched):
            legs = legs_by_race.get(rid, set())
            entry = (data.get("races") or {}).get(rid)
            if not isinstance(entry, dict) or not legs:
                continue
            if legs <= ids:
                entry["exclusive"] = True
                exclusive += 1
                if ids == legs:
                    entry["exhaustive"] = rel.is_exhaustive
                else:
                    entry["exhaustive"] = False if rel.is_exhaustive else None
                if entry["exhaustive"]:
                    exhaustive += 1
            else:
                partial.append(f"{rid}: group {rel.id or '?'} covers {sorted(legs & ids)} of {sorted(legs)}")
    return {
        "relationships": len(relationships),
        "exclusive_races": exclusive,
        "exhaustive_races": exhaustive,
        "partial": partial,
    }


# ---------------------------------------------------------------------------------------------
# 4. crosswalk
# ---------------------------------------------------------------------------------------------


async def crosswalk_kalshi(
    http: httpx.AsyncClient,
    base: str,
    data: dict[str, Any],
    *,
    only_missing: bool = True,
    max_candidates: int = KALSHI_MAX_CANDIDATES,
) -> tuple[int, int]:
    """Fill ``kalshi_ticker`` for every leg from the candidate patterns, verified by fetching the tickers
    in batches and rejecting any whose title contradicts the race. Returns (found, missing)."""
    wanted: dict[tuple[str, str], list[str]] = {}
    legs: dict[tuple[str, str], dict[str, Any]] = {}
    for rid, party, _entry, leg in U.market_legs(data):
        if only_missing and leg.get("kalshi_ticker"):
            continue
        cands = U.kalshi_ticker_candidates(rid, party)[:max_candidates]
        if cands:
            wanted[(rid, party)] = cands
            legs[(rid, party)] = leg
    all_tickers = [t for cands in wanted.values() for t in cands]
    quotes = await RM.fetch_kalshi(http, base, all_tickers) if all_tickers else {}
    found = missing = 0
    for key, cands in wanted.items():
        rid, party = key
        chosen: str | None = None
        for t in cands:
            q = quotes.get(t)
            if q is None or not q.is_open or U.title_contradicts(rid, party, q.title):
                continue
            chosen = q.key
            break
        if chosen:
            legs[key]["kalshi_ticker"] = chosen
            found += 1
        else:
            missing += 1
    pre = sum(1 for _r, _p, _e, leg in U.market_legs(data) if leg.get("kalshi_ticker")) - found
    log.info("kalshi crosswalk: %d found, %d missing (%d already set)", found, missing, pre)
    return found, missing


def _poly_score(rid: str, party: str, gm: RM.GammaMarket) -> float:
    best = U.crosswalk_match_score(rid, party, gm.title)
    if gm.event_title and gm.question:
        best = max(best, U.crosswalk_match_score(rid, party, f"{gm.event_title} {gm.question}"))
    return best


def _set_poly(leg: dict[str, Any], gm: RM.GammaMarket) -> None:
    leg["polymarket_condition_id"] = gm.condition_id
    leg["polymarket_token_id"] = gm.yes_token_id or ""
    leg["polymarket_slug"] = gm.slug or ""
    leg["polymarket_gamma_id"] = gm.gamma_id or ""


async def crosswalk_polymarket(
    http: httpx.AsyncClient,
    gamma: str,
    data: dict[str, Any],
    *,
    only_missing: bool = True,
    min_score: float = POLY_MIN_SCORE,
    max_slugs: int = POLY_MAX_SLUGS,
    pause_s: float | None = None,
) -> tuple[int, int]:
    """Fill Polymarket ids: slug candidates first, then /public-search, accepted only when the title
    scores >= min_score for (race, party). Returns (found, missing). ``pause_s`` (default PAUSE_S)
    sleeps between Gamma calls."""
    if pause_s is None:
        pause_s = PAUSE_S
    found = missing = 0
    search_cache: dict[str, list[RM.GammaMarket]] = {}
    for rid, party, _entry, leg in U.market_legs(data):
        if only_missing and leg.get("polymarket_condition_id"):
            continue
        best: RM.GammaMarket | None = None
        best_score = 0.0
        for slug in U.polymarket_slug_candidates(rid, party)[:max_slugs]:
            gm = await RM.fetch_gamma_by_slug(http, gamma, slug)
            if pause_s:
                await asyncio.sleep(pause_s)
            if gm is None or gm.closed:
                continue
            score = _poly_score(rid, party, gm)
            if score >= min_score:
                best, best_score = gm, score
                break
        if best is None:
            query = U.polymarket_search_query(rid)
            if query not in search_cache:
                search_cache[query] = await RM.fetch_gamma_search(http, gamma, query)
                if pause_s:
                    await asyncio.sleep(pause_s)
            for gm in search_cache[query]:
                if gm.closed:
                    continue
                score = _poly_score(rid, party, gm)
                if score > best_score:
                    best, best_score = gm, score
            if best is not None and best_score < min_score:
                best = None
        if best is not None:
            _set_poly(leg, best)
            found += 1
        else:
            missing += 1
    log.info("polymarket crosswalk: %d found, %d missing", found, missing)
    return found, missing


# ---------------------------------------------------------------------------------------------
# 5. DB upsert
# ---------------------------------------------------------------------------------------------


def _parse_iso(v: Any) -> datetime | None:
    if isinstance(v, datetime):
        return ensure_utc(v)
    if not v:
        return None
    try:
        return ensure_utc(datetime.fromisoformat(str(v).replace("Z", "+00:00")))
    except ValueError:
        return None


def upsert_db(
    session_factory: Any,
    data: Mapping[str, Any],
    *,
    now: datetime | None = None,
    resolutions: Mapping[str, str] | None = None,
) -> tuple[int, int]:
    """Upsert Race rows for every race in the universe and Market rows for every leg with a SIG id."""
    now = now or utc_now()
    resolutions = resolutions or {}
    n_races = n_markets = 0
    with session_factory() as s:
        for rid, entry in (data.get("races") or {}).items():
            vals = U.race_row_values(str(rid), entry if isinstance(entry, Mapping) else None)
            if vals is None:
                continue
            row = s.get(Race, vals["race_id"])
            if row is None:
                s.add(Race(**vals))
            else:
                for k, v in vals.items():
                    setattr(row, k, v)
            n_races += 1
        s.flush()
        for rid, party, _entry, leg in U.market_legs(data):
            eid = str(leg.get("sig_exchange_id") or "")
            if not eid:
                continue
            mid = str(leg.get("sig_market_id") or eid)
            vals = {
                "market_id": mid,
                "exchange_id": eid,
                "race_id": rid,
                "party": party,
                "title": (leg.get("title") or U.expected_title(rid, party))[:256],
                "status": (leg.get("status") or "open")[:32],
                "settlement_date": _parse_iso(leg.get("settlement_date")),
                "kalshi_ticker": leg.get("kalshi_ticker") or None,
                "polymarket_condition_id": leg.get("polymarket_condition_id") or None,
                "polymarket_token_id": leg.get("polymarket_token_id") or None,
                "predictit_id": (str(leg["predictit_id"]) if leg.get("predictit_id") else None),
                "updated_at": now,
            }
            if eid in resolutions or mid in resolutions:
                vals["resolution_text"] = resolutions.get(eid) or resolutions.get(mid)
            row = s.get(Market, mid)
            if row is None:
                row = s.execute(select(Market).where(Market.exchange_id == eid)).scalar_one_or_none()
            if row is None:
                s.add(Market(**vals))
            else:
                for k, v in vals.items():
                    if k == "market_id" and row.market_id != v:
                        continue  # primary key stays; the exchange id is the stable link
                    setattr(row, k, v)
            n_markets += 1
    return n_races, n_markets


# ---------------------------------------------------------------------------------------------
# 6. probes
# ---------------------------------------------------------------------------------------------


async def probe_fees(adapter: ExchangeAdapter) -> dict[str, Any]:
    """Count fills carrying a non-zero fee (fees are reported zero; this verifies it on our own fills)."""
    try:
        fills = await adapter.get_fills()
    except Exception as e:  # noqa: BLE001
        return {"verdict": "inconclusive (fills unavailable)", "error": f"{type(e).__name__}: {e}"[:200]}
    n = len(fills)
    with_fee = [f for f in fills if (f.fee or 0.0) > 0]
    total_fee = sum(float(f.fee or 0.0) for f in fills)
    if n == 0:
        verdict = "inconclusive: no fills yet (re-run after the canary trade)"
    elif with_fee:
        verdict = f"FEES PRESENT: {len(with_fee)} of {n} fills carry a fee (total {total_fee:.4f})"
    else:
        verdict = f"zero fees on all {n} sampled fills"
    return {"verdict": verdict, "fills": n, "fills_with_fee": len(with_fee), "total_fee": round(total_fee, 6)}


def probe_relationship_graph(relationships: Sequence[Relationship], data: Mapping[str, Any]) -> dict[str, Any]:
    """Is the tournament-scoped relationship graph empty (an Oct-7 audit found it so)?"""
    races = data.get("races") or {}
    covered = sum(1 for e in races.values() if isinstance(e, dict) and e.get("exclusive") is True)
    exhaustive = sum(1 for e in races.values() if isinstance(e, dict) and e.get("exhaustive") is True)
    if not relationships:
        verdict = "EMPTY: no relationships returned -> Dutch books stay disabled (arb.require_api_exclusivity)"
    else:
        verdict = f"{len(relationships)} groups; {covered} races exclusive, {exhaustive} exhaustive"
    return {"verdict": verdict, "relationships": len(relationships), "races_exclusive": covered, "races_exhaustive": exhaustive}


async def _ratelimit_remaining(adapter: Any) -> dict[str, int] | None:
    """Best effort: GET /account through the adapter's raw http client and return any
    ``*ratelimit*remaining*`` headers (lower-cased name -> int). None when unavailable."""
    http = getattr(adapter, "_http", None)
    budget = getattr(adapter, "budget", None)
    if http is None or not hasattr(http, "get"):
        return None
    try:
        if budget is not None:
            await budget.reads.wait(1)
        resp = await http.get("/account")
    except Exception as e:  # noqa: BLE001
        log.warning("rate-limit header probe failed: %s", type(e).__name__)
        return None
    out: dict[str, int] = {}
    for name, value in resp.headers.items():
        n = name.lower()
        if "ratelimit" in n.replace("-", "").replace("_", "") and "remaining" in n:
            try:
                out[n] = int(float(value))
            except ValueError:
                continue
    return out or None


def pick_probe_markets(markets: Sequence[MarketInfo], prices: Mapping[str, Any] | None = None, n: int = 2) -> list[MarketInfo]:
    """Open markets with the highest YES ask (so a 0.005 bid cannot fill); falls back to the first open ones."""
    open_markets = [m for m in markets if m.exchange_id and (m.status or "open").lower() in ("open", "active", "trading")]

    def ask(m: MarketInfo) -> float:
        p = (prices or {}).get(m.exchange_id or "")
        a = getattr(p, "best_ask", None) if p is not None else None
        return float(a) if a is not None else 0.0

    return sorted(open_markets, key=ask, reverse=True)[:n]


async def probe_batch_writes(
    adapter: ExchangeAdapter, markets: Sequence[MarketInfo], *, execute: bool, now: datetime | None = None
) -> dict[str, Any]:
    """Does POST /orders/batch count as one write? Dry (documented) unless ``execute``."""
    if not execute:
        return {"verdict": "skipped (dry): pass --probe-writes to place the 2-order probe batch", "procedure": BATCH_PROBE_PROCEDURE}
    place_batch = getattr(adapter, "place_batch", None)
    if not callable(place_batch):
        return {"verdict": "adapter has no place_batch; cannot probe", "procedure": BATCH_PROBE_PROCEDURE}
    now = now or utc_now()
    try:
        prices = await adapter.get_prices([m.exchange_id for m in markets if m.exchange_id][:100])
    except Exception:  # noqa: BLE001
        prices = {}
    targets = pick_probe_markets(markets, prices)
    if len(targets) < 2:
        return {"verdict": "inconclusive: fewer than two open markets", "procedure": BATCH_PROBE_PROCEDURE}
    reqs = [
        OrderRequest(
            exchange_id=str(m.exchange_id),
            side="yes",
            action="buy",
            price=round_tick(PRICE_MIN),
            quantity=PROBE_ORDER_QTY,
            idempotency_key=new_idempotency_key(),
            expires_at=now + timedelta(seconds=PROBE_ORDER_TTL_S),
            strategy="manual",
        )
        for m in targets
    ]
    budget = getattr(adapter, "budget", None)
    used_before = budget.writes.used() if budget is not None else None
    hdr_before = await _ratelimit_remaining(adapter)
    try:
        results = await place_batch(reqs)
    except Exception as e:  # noqa: BLE001
        return {"verdict": f"batch failed: {type(e).__name__}: {e}"[:200], "procedure": BATCH_PROBE_PROCEDURE}
    hdr_after = await _ratelimit_remaining(adapter)
    used_after = budget.writes.used() if budget is not None else None
    order_ids = [r.order_id for r in results if r.order_id]
    cancelled = 0
    for oid in order_ids:
        try:
            cancelled += int(bool(await adapter.cancel_order(oid)))
        except Exception as e:  # noqa: BLE001
            log.warning("probe cancel %s failed: %s", oid, type(e).__name__)
    out: dict[str, Any] = {
        "orders_in_batch": len(reqs),
        "orders_acked": len(order_ids),
        "uncertain": sum(1 for r in results if r.uncertain),
        "cancelled": cancelled,
        "our_limiter_writes_for_batch": (used_after - used_before) if used_before is not None and used_after is not None else None,
        "ratelimit_headers_before": hdr_before,
        "ratelimit_headers_after": hdr_after,
        "procedure": BATCH_PROBE_PROCEDURE,
    }
    if hdr_before and hdr_after:
        deltas = {k: hdr_before[k] - hdr_after.get(k, hdr_before[k]) for k in hdr_before}
        drop = max(deltas.values()) if deltas else 0
        if drop <= 1:
            out["verdict"] = f"batch of {len(reqs)} counted as {drop} write(s): ONE write per batch -> writes stop binding, re-plan"
        elif drop >= len(reqs):
            out["verdict"] = f"batch of {len(reqs)} counted as {drop} writes: one write per ORDER"
        else:
            out["verdict"] = f"inconclusive: remaining-writes dropped by {drop} for {len(reqs)} orders"
    else:
        out["verdict"] = "inconclusive: the exchange sent no rate-limit headers (see procedure)"
    return out


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------


def make_adapter(sec: Secrets, cfg: StrategyConfig, *, trade: bool = False) -> tuple[ExchangeAdapter | None, str]:
    """A SigClient on the read key (trade key when ``trade``); (None, reason) when impossible."""
    key = (sec.sig_api_key if trade else (sec.sig_read_key or sec.sig_api_key)) or ""
    if not key:
        return None, "no SIG key (set SIG_READ_KEY / SIG_API_KEY)"
    try:
        from sigcup.sig.client import SigClient
    except Exception as e:  # noqa: BLE001
        return None, f"SigClient unavailable: {type(e).__name__}: {e}"
    budget = Budget(cfg.limits.reads_per_min, cfg.limits.writes_per_min)
    try:
        return SigClient(key, budget, cfg=cfg), ""
    except Exception as e:  # noqa: BLE001
        return None, f"SigClient init failed: {type(e).__name__}: {e}"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bootstrap_races", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--races-yaml", type=Path, default=None, help="path to races.yaml (default config/races.yaml)")
    p.add_argument("--dry-run", action="store_true", help="print the report; write neither races.yaml nor the DB")
    p.add_argument("--no-db", action="store_true", help="write races.yaml but do not touch the database")
    p.add_argument("--skip-sig", action="store_true", help="do not call the SIG API (crosswalk / DB only)")
    p.add_argument("--skip-crosswalk", action="store_true", help="do not call Kalshi / Polymarket")
    p.add_argument("--recheck-crosswalk", action="store_true", help="re-verify legs that already have ids")
    p.add_argument("--probe-writes", action="store_true", help="run the batch-write-count probe (places and cancels 2 tiny orders)")
    p.add_argument("--log-level", default="INFO")
    return p


async def run(
    args: argparse.Namespace,
    *,
    cfg: StrategyConfig | None = None,
    sec: Secrets | None = None,
    adapter: ExchangeAdapter | None = None,
    http: httpx.AsyncClient | None = None,
    session_factory: Any = None,
) -> BootstrapReport:
    cfg = cfg or config()
    sec = sec or load_secrets()
    now = utc_now()
    report = BootstrapReport()
    data = U.load_races_yaml(args.races_yaml)
    resolutions: dict[str, str] = {}
    markets: list[MarketInfo] = []
    relationships: list[Relationship] = []
    owns_adapter = False

    if not args.skip_sig:
        if adapter is None:
            adapter, reason = make_adapter(sec, cfg, trade=bool(args.probe_writes))
            owns_adapter = adapter is not None
            if adapter is None:
                report.notes.append(f"SIG step skipped: {reason}")
        if adapter is not None:
            try:
                try:
                    t = await adapter.get_tournament()
                    report.member_count = t.member_count
                except Exception as e:  # noqa: BLE001
                    report.notes.append(f"get_tournament failed: {type(e).__name__}: {e}"[:200])
                markets = list(await adapter.list_markets())
                report.sig_markets = len(markets)
                mapped, unmapped = map_markets(markets)
                report.mapped = len(mapped)
                report.unmapped = [f"{m.id}: {m.title}" for m in unmapped]
                for mm in mapped:
                    if mm.market.resolution and mm.market.exchange_id:
                        resolutions[str(mm.market.exchange_id)] = str(mm.market.resolution)
                report.legs_added, report.legs_updated, report.duplicates = merge_markets(data, mapped, now=now)
                try:
                    relationships = list(await adapter.get_relationships())
                    rels_ok = True
                except Exception as e:  # noqa: BLE001
                    rels_ok = False
                    report.notes.append(f"get_relationships failed, exclusivity flags left untouched: {type(e).__name__}: {e}"[:200])
                if rels_ok:
                    rel = apply_relationships(data, relationships)
                    report.relationships = rel["relationships"]
                    report.races_exclusive = rel["exclusive_races"]
                    report.races_exhaustive = rel["exhaustive_races"]
                    report.partial_relationships = rel["partial"]
                for rid in ("SEN-OH", "GOV-OH"):
                    entry = (data.get("races") or {}).get(rid) or {}
                    legs = entry.get("markets") or {}
                    if not any(isinstance(leg, dict) and leg.get("sig_exchange_id") for leg in legs.values()):
                        report.notes.append(f"{rid} has no SIG market in the live list (absent on Sept 30 too)")
                report.probes["fees"] = await probe_fees(adapter)
                if rels_ok:
                    report.probes["relationship_graph"] = probe_relationship_graph(relationships, data)
                report.probes["batch_writes"] = await probe_batch_writes(adapter, markets, execute=bool(args.probe_writes), now=now)
            finally:
                if owns_adapter and hasattr(adapter, "aclose"):
                    await adapter.aclose()

    if not args.skip_crosswalk:
        client = http or httpx.AsyncClient(timeout=RM.HTTP_TIMEOUT_S, headers={"User-Agent": "sigcup/0.1 (+bootstrap)"})
        try:
            only_missing = not args.recheck_crosswalk
            try:
                report.kalshi_found, report.kalshi_missing = await crosswalk_kalshi(client, cfg.feeds.kalshi_base, data, only_missing=only_missing)
            except Exception as e:  # noqa: BLE001
                report.notes.append(f"kalshi crosswalk failed: {type(e).__name__}: {e}"[:200])
            try:
                report.polymarket_found, report.polymarket_missing = await crosswalk_polymarket(
                    client, cfg.feeds.polymarket_gamma, data, only_missing=only_missing
                )
            except Exception as e:  # noqa: BLE001
                report.notes.append(f"polymarket crosswalk failed: {type(e).__name__}: {e}"[:200])
        finally:
            if http is None:
                await client.aclose()

    report.stats = U.universe_stats(data)
    if args.dry_run:
        report.notes.append("dry run: races.yaml and the database were not written")
        return report
    U.save_races_yaml(data, args.races_yaml)
    if not args.no_db:
        if session_factory is None:
            dbengine.init_db()
            session_factory = dbengine.session
        report.db_races, report.db_markets = upsert_db(session_factory, data, now=now, resolutions=resolutions)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    report = asyncio.run(run(args))
    print(report.text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
