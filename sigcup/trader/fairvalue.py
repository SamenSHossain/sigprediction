"""Fair value per market: precision-weighted blend of (a) raw real-money reference prices (Kalshi,
Polymarket; weight scaled by liquidity; NO favourite-longshot stretch applied), (b) polls+fundamentals
P(win) = T_df(margin / sigma_chamber) with the poll average shrunk toward a fundamentals prior (PVI,
generic ballot, incumbency), (c) ratings-implied probability. Returns FairValue(exchange_id, fair,
components, uncertainty). Also exposes implied_margin_shift(fair_before, fair_after, chamber) used by
the regime classifier.

Contract (see ``sigcup.trader.types``)::

    load_races(session) -> dict[race_id, RaceInfo]            # Race + Market rows, party -> exchange_id
    compute_all(session, cfg, markets, now) -> dict[exchange_id, FairValue]

Model notes
-----------
* ``ref``: liquidity-weighted mean of the latest Kalshi / Polymarket YES mids mapped through
  ``Market.kalshi_ticker`` / ``Market.polymarket_condition_id`` that are younger than ``ref_max_age``
  (``REF_MAX_AGE``; on election night ``NIGHT_REF_MAX_AGE_CADENCES`` collector cadences, at least
  ``NIGHT_REF_MIN_AGE``). Each source's weight is
  ``min(1, liquidity / fairvalue.ref_min_liquidity)`` (unknown liquidity counts half). The component's
  availability is ``min(1, sum of source weights)`` so an illiquid single source carries less precision.
  Real-money prices are used *raw*: no Rothschild/Leigh-style debiasing stretch (the student book, not
  the real-money book, is the one with favourite-longshot bias).
* ``polls``: recency-weighted poll margin (half-life ``fairvalue.poll_halflife_days``) shrunk toward a
  fundamentals prior ``generic_ballot - 2 * PVI + incumbency`` (D-minus-R margin points; Cook's PVI is a
  two-party vote-*share* lean, positive = Republican, so the margin lean is twice it). The shrink weight is ``max(fundamentals_shrink, 1 / (1 + n_eff))`` so a single poll
  is shrunk half-way. With no polls the prior is used alone at half availability. The two-party frame
  maps margins to P(D wins) through ``probmath.prob_from_margin``; the Republican leg is the complement.
* ``ratings``: mean P(D) across the latest rating per source, parsed from the rating text through
  ``fairvalue.ratings_prob`` (``Lean D``, ``Likely R``, ``Toss-up``, ``Tilt D``, ``Safe R`` ...), falling
  back to the stored ``prob_dem``.
* Independent / third-party legs: when the race also has D and R legs, the two-party components do not
  describe them (they would imply ~0), so only ``ref`` is used; if the race lacks a D (or R) leg the
  independent leg takes that role in the two-party frame. A major-party leg whose party fields no
  candidate counts as missing: ``Race.candidates`` is trusted as the complete field when it lists two or
  more parties and the race has an independent leg (NE Senate: R and I only). Such an empty leg is priced
  from its own ``ref`` alone (no ref -> no fair value) and is left out of the simulation. When the
  independent stands in for a major party the D-vs-R fundamentals prior (PVI, generic ballot) is not
  used; real polls and ratings still are. ``1 - D - R`` is used for the ``ref`` component when an
  independent leg has no crosswalk of its own but both major-party legs have references.
* Blend: ``weights = fairvalue.weights[c] * availability[c]`` over the components that are present,
  renormalised; ``fair`` clamped to the tick range; ``sd`` from the weighted component disagreement plus a
  floor (larger when only one component is present or when the components that *apply* to the leg are
  not fully available: ref/polls/ratings for a two-party leg, ref for an independent leg, ref/sim for a
  control market).
* Election night (``election_night.start_utc`` until the deadline): polls, ratings and the simulation
  cannot see results, so a leg with a fresh reference is priced from the reference alone; a leg whose
  reference has gone stale keeps only its slow components (no ``ref``), which the taker rule reads as
  "unconfirmed".
* Control markets (chamber ``control``) blend the ``ref`` component with the correlated simulation's
  control probability (``simulate.simulate`` on the blended race fairs); the simulation stands in for the
  polls+ratings weight. Its availability is the squared coverage of the chamber's seat races (missing
  seats are assumed held by the incumbent party, so a partial universe says little about control).
  Either component alone is used when the other is missing.
* ``FairValue.ts`` is the freshest *input* timestamp (reference price, poll fetch, rating; for a control
  market the reference or the newest seat fair value the simulation used), never ``now``, so callers can
  tell whether a fair value predates a hard news event.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from sigcup.common.log import get_logger
from sigcup.common.time import ensure_utc
from sigcup.db.models import GenericBallot, Market, Poll, Race, Rating
from sigcup.db.repo import latest_ref_prices
from sigcup.settings import StrategyConfig
from sigcup.sig.schemas import PRICE_MAX, PRICE_MIN
from sigcup.trader.probmath import (
    implied_margin_shift,
    margin_from_prob,
    prob_from_margin,
    shift_probability,
)
from sigcup.trader.types import FairValue, MarketView, RaceInfo

__all__ = [
    "NIGHT_REF_MAX_AGE_CADENCES",
    "NIGHT_REF_MIN_AGE",
    "REF_MAX_AGE",
    "compute_all",
    "election_night_active",
    "fundamentals_margin",
    "implied_margin_shift",
    "load_races",
    "margin_from_prob",
    "parse_rating",
    "poll_average",
    "prob_from_margin",
    "ref_max_age",
    "shift_probability",
]

log = get_logger(__name__)

# -- tunables that are not (yet) in settings.yaml -------------------------------------------------
REF_MAX_AGE = timedelta(hours=24)  # reference prices older than this are ignored
NIGHT_REF_MAX_AGE_CADENCES = 6  # on election night: references older than this many collector cadences ...
NIGHT_REF_MIN_AGE = timedelta(seconds=60)  # ... (but never less than this) are stale
POLL_MAX_AGE = timedelta(days=60)  # polls older than this are ignored outright
RATING_MAX_AGE = timedelta(days=60)
GENERIC_BALLOT_MAX_AGE = timedelta(days=30)
INCUMBENCY_BUMP_PTS = 1.5  # Jacobson (2015): incumbency advantage ~1-2 pts today
UNKNOWN_LIQUIDITY_WEIGHT = 0.5  # weight for a reference price whose liquidity is unknown
FUNDAMENTALS_ONLY_AVAILABILITY = 0.5  # precision of the polls component with no polls at all
INDEPENDENT_FLOOR = 0.02  # P(YES) floor for an independent leg derived as 1 - D - R
SD_FLOOR = 0.02
SD_FLOOR_SINGLE = 0.04  # when only one component is present
SD_AVAILABILITY_PENALTY = 0.05  # extra sd when the available precision is low
TWO_PARTY_COMPONENTS = ("ref", "polls", "ratings")
REF_ONLY = ("ref",)
CONTROL_COMPONENTS = ("ref", "sim")
_EPS = 1e-9


# -- helpers --------------------------------------------------------------------------------------
def _role(party: str | None) -> str:
    """Normalise a party label to the two-party frame: 'D', 'R' or 'I' (anything else)."""
    p = (party or "").strip().upper()
    if p.startswith("D"):
        return "D"
    if p.startswith("R"):
        return "R"
    return "I"


def _clamp(p: float) -> float:
    return min(PRICE_MAX, max(PRICE_MIN, float(p)))


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    """Lenient attribute / key getter: first present, non-None candidate."""
    for name in names:
        if isinstance(obj, Mapping):
            if name in obj and obj[name] is not None:
                return obj[name]
            continue
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return default


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _latest(a: datetime | None, b: datetime | None) -> datetime | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def election_night_active(cfg: StrategyConfig, now: datetime) -> bool:
    """True from ``election_night.start_utc`` (inclusive) until the deadline (exclusive); the same window as
    ``election_night.is_active`` (kept here to avoid an import cycle)."""
    now = ensure_utc(now)
    return ensure_utc(cfg.election_night.start_utc) <= now < ensure_utc(cfg.deadline_utc)


def ref_max_age(cfg: StrategyConfig, now: datetime) -> timedelta:
    """Age beyond which a reference price is ignored: ``REF_MAX_AGE``, or on election night
    ``NIGHT_REF_MAX_AGE_CADENCES`` collector cadences (at least ``NIGHT_REF_MIN_AGE``)."""
    if election_night_active(cfg, now):
        night = timedelta(seconds=NIGHT_REF_MAX_AGE_CADENCES * float(cfg.election_night.ref_cadence_s))
        return max(NIGHT_REF_MIN_AGE, night)
    return REF_MAX_AGE


def control_chamber(race: RaceInfo) -> str | None:
    """Which chamber a control race refers to ('senate' | 'house'), from its id / label."""
    text = f"{race.race_id} {race.label}".lower()
    if "senate" in text or "sen" in text.split("-"):
        return "senate"
    if "house" in text:
        return "house"
    return None


# -- ratings --------------------------------------------------------------------------------------
_LEVEL_ALIASES = {
    "safe": "safe",
    "solid": "safe",
    "strong": "likely",
    "likely": "likely",
    "lean": "lean",
    "leans": "lean",
    "leaning": "lean",
    "tilt": "tilt",
    "tilts": "tilt",
    "tilting": "tilt",
}
_PARTY_ALIASES = {
    "d": "D",
    "dem": "D",
    "dems": "D",
    "democrat": "D",
    "democrats": "D",
    "democratic": "D",
    "r": "R",
    "rep": "R",
    "gop": "R",
    "republican": "R",
    "republicans": "R",
    "i": "I",
    "ind": "I",
    "independent": "I",
}


def parse_rating(rating: str | None, cfg: StrategyConfig) -> float | None:
    """P(Democrat wins) implied by a rating label through ``fairvalue.ratings_prob``.

    Understands ``Safe/Solid``, ``Likely``, ``Lean(s)``, ``Tilt(s)`` followed or preceded by a party
    (``D``, ``Dem``, ``R``, ``GOP`` ...) and any spelling of ``Toss-up``. Returns None when the label cannot
    be mapped (unknown level, independent-favoured rating, empty string).
    """
    if not rating:
        return None
    table = cfg.fairvalue.ratings_prob
    s = rating.strip().lower().replace("_", " ").replace("-", " ").replace("/", " ")
    s = s.replace("(", " ").replace(")", " ")
    s = " ".join(s.split())
    if not s:
        return None
    if "toss" in s or s in ("tu", "t u"):
        return table.get("tossup", table.get("toss-up", 0.5))
    level: str | None = None
    party: str | None = None
    for tok in s.split():
        if tok in _LEVEL_ALIASES and level is None:
            level = _LEVEL_ALIASES[tok]
        elif tok in _PARTY_ALIASES and party is None:
            party = _PARTY_ALIASES[tok]
    if level is None or party is None:
        return None
    p = table.get(level)
    if p is None:
        return None
    p = min(1.0, max(0.0, float(p)))
    if party == "D":
        return p
    if party == "R":
        return 1.0 - p
    return None


def _ratings_pdem(rows: Iterable[Rating], cfg: StrategyConfig, now: datetime) -> tuple[float | None, datetime | None]:
    """Mean P(D) over the latest rating per source (ignoring stale rows)."""
    latest: dict[str, Rating] = {}
    for r in rows:
        ts = ensure_utc(r.ts) if r.ts is not None else None
        if ts is not None and now - ts > RATING_MAX_AGE:
            continue
        prev = latest.get(r.source)
        if prev is None or (ts is not None and (prev.ts is None or ts > ensure_utc(prev.ts))):
            latest[r.source] = r
    vals: list[float] = []
    newest: datetime | None = None
    for r in latest.values():
        p = parse_rating(r.rating, cfg)
        if p is None:
            p = _to_float(r.prob_dem)
            if p is None or not 0.0 <= p <= 1.0:
                continue
        vals.append(p)
        if r.ts is not None:
            newest = _latest(newest, ensure_utc(r.ts))
    if not vals:
        return None, None
    return sum(vals) / len(vals), newest


# -- polls + fundamentals -------------------------------------------------------------------------
def poll_average(polls: Iterable[Any], now: datetime, halflife_days: float) -> tuple[float | None, float]:
    """Recency-weighted mean margin (D - R, points) and the effective number of polls.

    Each poll's weight is ``0.5 ** (age_days / halflife_days)`` with age measured from its end date.
    Accepts Poll rows or any objects/dicts exposing ``margin`` and ``end_date``.
    """
    num = 0.0
    den = 0.0
    hl = max(0.1, float(halflife_days))
    for p in polls:
        m = _to_float(_get(p, "margin"))
        if m is None:
            dem, rep = _to_float(_get(p, "dem")), _to_float(_get(p, "rep"))
            if dem is None or rep is None:
                continue
            m = dem - rep
        end = _get(p, "end_date", "end", "date")
        if isinstance(end, datetime):
            age_days = max(0.0, (now - ensure_utc(end)).total_seconds() / 86400.0)
        else:
            age_days = 0.0
        w = 0.5 ** (age_days / hl)
        num += w * m
        den += w
    if den <= 0:
        return None, 0.0
    return num / den, den


def fundamentals_margin(
    pvi: float | None,
    generic_margin: float | None,
    incumbent_party: str | None,
    bump_pts: float = INCUMBENCY_BUMP_PTS,
) -> float | None:
    """Fundamentals prior for the D - R margin (points): ``generic - 2 * PVI + incumbency``.

    PVI is Cook's convention: positive = Republican lean, measured in two-party vote *share* (R+10 means
    the Republican share runs 10 points above the nation), so its margin equivalent is twice the PVI; the
    generic ballot is already a D-minus-R margin. Returns None when neither PVI nor a generic ballot is
    known.
    """
    if pvi is None and generic_margin is None:
        return None
    m = float(generic_margin or 0.0) - 2.0 * float(pvi or 0.0)
    role = _role(incumbent_party) if incumbent_party else "I"
    if role == "D":
        m += bump_pts
    elif role == "R":
        m -= bump_pts
    return m


def _polls_pdem(
    race: RaceInfo,
    polls: list[Poll],
    generic: float | None,
    cfg: StrategyConfig,
    now: datetime,
    fundamentals: bool = True,
) -> tuple[float | None, float, datetime | None]:
    """(P(D), availability, freshest fetch time) from polls shrunk toward fundamentals.

    ``fundamentals=False`` drops the PVI / generic-ballot prior (it describes a D-vs-R contest only).
    """
    avg, n_eff = poll_average(polls, now, cfg.fairvalue.poll_halflife_days)
    fund = fundamentals_margin(race.pvi, generic, race.incumbent_party) if fundamentals else None
    newest: datetime | None = None
    for p in polls:
        if p.fetched_at is not None:
            newest = _latest(newest, ensure_utc(p.fetched_at))
    if avg is None and fund is None:
        return None, 0.0, None
    if avg is None:
        margin = float(fund)  # type: ignore[arg-type]
        avail = FUNDAMENTALS_ONLY_AVAILABILITY
    elif fund is None:
        margin = avg
        avail = 1.0
    else:
        s = max(float(cfg.fairvalue.fundamentals_shrink), 1.0 / (1.0 + n_eff))
        s = min(1.0, max(0.0, s))
        margin = (1.0 - s) * avg + s * fund
        avail = 1.0
    chamber = race.chamber if race.chamber in cfg.fairvalue.sigma_pts else "senate"
    return prob_from_margin(margin, chamber, cfg), avail, newest


# -- reference prices -----------------------------------------------------------------------------
@dataclass
class _Ref:
    mid: float
    weight: float  # min(1, liquidity / ref_min_liquidity)
    liquidity: float | None
    ts: datetime | None


def _ref_mid(row: Any) -> float | None:
    mid = _to_float(_get(row, "yes_mid", "mid"))
    if mid is None:
        bid, ask = _to_float(_get(row, "yes_bid", "bid")), _to_float(_get(row, "yes_ask", "ask"))
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
        elif bid is not None or ask is not None:
            mid = bid if bid is not None else ask
    if mid is None or not 0.0 <= mid <= 1.0:
        return None
    return mid


def _refs_for(
    mrow: Market | None,
    refs: Mapping[tuple[str, str], Any],
    cfg: StrategyConfig,
    now: datetime,
    max_age: timedelta | None = None,
) -> list[_Ref]:
    out: list[_Ref] = []
    if mrow is None:
        return out
    max_age = REF_MAX_AGE if max_age is None else max_age
    keys = (("kalshi", mrow.kalshi_ticker), ("polymarket", mrow.polymarket_condition_id), ("predictit", mrow.predictit_id))
    for source, key in keys:
        if not key:
            continue
        row = refs.get((source, key))
        if row is None:
            continue
        ts = _get(row, "ts")
        ts = ensure_utc(ts) if isinstance(ts, datetime) else None
        if ts is not None and now - ts > max_age:
            continue
        mid = _ref_mid(row)
        if mid is None:
            continue
        liq = _to_float(_get(row, "liquidity"))
        if liq is None:
            w = UNKNOWN_LIQUIDITY_WEIGHT
        else:
            w = min(1.0, max(0.0, liq / max(_EPS, float(cfg.fairvalue.ref_min_liquidity))))
        out.append(_Ref(mid=mid, weight=w, liquidity=liq, ts=ts))
    return out


def _ref_component(refs: list[_Ref]) -> tuple[float | None, float, float | None, datetime | None]:
    """(liquidity-weighted mid, availability, total liquidity, freshest ts)."""
    if not refs:
        return None, 0.0, None, None
    wsum = sum(r.weight for r in refs)
    if wsum <= _EPS:
        # every source illiquid: fall back to a plain mean with minimal precision
        mid = sum(r.mid for r in refs) / len(refs)
        avail = 0.05
    else:
        mid = sum(r.weight * r.mid for r in refs) / wsum
        avail = min(1.0, wsum)
    liq = sum(r.liquidity for r in refs if r.liquidity is not None)
    ts: datetime | None = None
    for r in refs:
        ts = _latest(ts, r.ts)
    return mid, avail, (liq if any(r.liquidity is not None for r in refs) else None), ts


# -- blend ----------------------------------------------------------------------------------------
def blend(
    components: Mapping[str, float],
    availability: Mapping[str, float],
    cfg: StrategyConfig,
    weight_overrides: Mapping[str, float] | None = None,
    applicable: Iterable[str] | None = None,
) -> tuple[float, float, dict[str, float]] | None:
    """Precision-weighted blend of the present components.

    Returns ``(fair, sd, normalised_weights)`` or None when nothing carries weight. Base weights come
    from ``fairvalue.weights`` (``weight_overrides`` can add keys such as ``sim``) and are scaled by each
    component's availability in [0, 1] before renormalising over the components present. The
    availability penalty in ``sd`` compares the carried weight with the full weight of the ``applicable``
    components (every base key when omitted), so a leg is not penalised for inputs it can never have.
    """
    base = dict(cfg.fairvalue.weights)
    if weight_overrides:
        base.update(weight_overrides)
    weights: dict[str, float] = {}
    for name, value in components.items():
        if value is None:
            continue
        w = float(base.get(name, 0.0)) * float(min(1.0, max(0.0, availability.get(name, 1.0))))
        if w > _EPS:
            weights[name] = w
    total = sum(weights.values())
    if total <= _EPS:
        return None
    norm = {k: v / total for k, v in weights.items()}
    fair = sum(norm[k] * float(components[k]) for k in norm)
    disagreement = math.sqrt(sum(norm[k] * (float(components[k]) - fair) ** 2 for k in norm))
    keys = list(base) if applicable is None else [k for k in applicable if k in base]
    full = sum(float(base.get(k, 0.0)) for k in keys) or 1.0
    avail_total = min(1.0, total / full)
    floor = SD_FLOOR if len(norm) >= 2 else SD_FLOOR_SINGLE
    sd = math.sqrt(disagreement**2 + floor**2 + (SD_AVAILABILITY_PENALTY * (1.0 - avail_total)) ** 2)
    return _clamp(fair), sd, norm


def _empty_major_roles(candidates: Any, roles: Mapping[str, str]) -> set[str]:
    """Major-party roles ('D' / 'R') that have a leg in the race but field no candidate.

    ``Race.candidates`` usually lists only the incumbent, so it is trusted as the complete field only when
    it names candidates for two or more parties, and only in a race with an independent leg (which takes
    the empty party's place in the two-party frame), e.g. NE Senate (R and I, no Democrat).
    """
    if not isinstance(candidates, Mapping):
        return set()
    fielded = {_role(str(party)) for party, who in candidates.items() if who}
    if len(fielded) < 2 or "I" not in roles.values():
        return set()
    return {r for r in ("D", "R") if r in roles.values() and r not in fielded}


# -- public API -----------------------------------------------------------------------------------
def load_races(session: Session) -> dict[str, RaceInfo]:
    """RaceInfo per race from the Race and Market tables (``markets`` maps party -> exchange_id)."""
    races: dict[str, RaceInfo] = {}
    for r in session.execute(select(Race)).scalars().all():
        races[r.race_id] = RaceInfo(
            race_id=r.race_id,
            chamber=r.chamber,
            state=r.state,
            label=r.label,
            exclusive=r.exclusive,
            exhaustive=r.exhaustive,
            pvi=r.pvi,
            incumbent_party=r.incumbent_party,
            settlement_note=r.settlement_note,
        )
    for m in session.execute(select(Market)).scalars().all():
        race = races.get(m.race_id)
        if race is not None and m.exchange_id:
            race.markets[m.party] = m.exchange_id
    return races


def _generic_ballot(session: Session, now: datetime) -> float | None:
    rows = session.execute(select(GenericBallot).where(GenericBallot.ts >= now - GENERIC_BALLOT_MAX_AGE)).scalars().all()
    latest: dict[str, GenericBallot] = {}
    for g in rows:
        prev = latest.get(g.source)
        if prev is None or ensure_utc(g.ts) > ensure_utc(prev.ts):
            latest[g.source] = g
    vals = [float(g.margin) for g in latest.values() if _to_float(g.margin) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _iter_views(markets: Mapping[str, MarketView] | Iterable[MarketView]) -> list[MarketView]:
    if isinstance(markets, Mapping):
        return list(markets.values())
    return list(markets)


def compute_all(
    session: Session,
    cfg: StrategyConfig,
    markets: Mapping[str, MarketView] | Iterable[MarketView],
    now: datetime,
) -> dict[str, FairValue]:
    """Fair value for every market view with a known race.

    ``markets`` is the ``dict[exchange_id, MarketView]`` from ``BookState.views`` (a list is accepted).
    Markets whose race is unknown to the DB are skipped; markets with no usable component are skipped
    (callers treat a missing fair as "no opinion").
    """
    now = ensure_utc(now)
    night = election_night_active(cfg, now)
    max_age = ref_max_age(cfg, now)
    views = _iter_views(markets)
    races = load_races(session)
    candidates = {rid: cands for rid, cands in session.execute(select(Race.race_id, Race.candidates)).all()}
    mrows = {m.exchange_id: m for m in session.execute(select(Market)).scalars().all()}
    refs = latest_ref_prices(session)
    generic = _generic_ballot(session, now)
    polls_by_race: dict[str, list[Poll]] = {}
    for p in session.execute(select(Poll).where(Poll.end_date >= now - POLL_MAX_AGE)).scalars().all():
        polls_by_race.setdefault(p.race_id, []).append(p)
    ratings_by_race: dict[str, list[Rating]] = {}
    for r in session.execute(select(Rating)).scalars().all():
        ratings_by_race.setdefault(r.race_id, []).append(r)

    # group the views by race, resolving race/party from the Market row when the view is unmapped
    by_race: dict[str, dict[str, MarketView]] = {}
    for v in views:
        race_id, party = v.race_id, v.party
        if race_id not in races:
            mrow = mrows.get(v.exchange_id)
            if mrow is None or mrow.race_id not in races:
                continue
            race_id, party = mrow.race_id, mrow.party
        by_race.setdefault(race_id, {})[v.exchange_id] = MarketView(
            exchange_id=v.exchange_id,
            race_id=race_id,
            party=party,
            chamber=races[race_id].chamber,
            state=races[race_id].state,
            bid=v.bid,
            ask=v.ask,
            last=v.last,
            status=v.status,
        )

    out: dict[str, FairValue] = {}
    race_fairs: dict[str, dict[str, float]] = {}
    control_views: dict[str, dict[str, MarketView]] = {}

    seat_ts: dict[str, datetime] = {}  # chamber -> newest input time among the fairs the simulation uses
    for race_id, legs in by_race.items():
        race = races[race_id]
        if race.chamber == "control":
            control_views[race_id] = legs
            continue
        roles = {eid: _role(v.party) for eid, v in legs.items()}
        empty = _empty_major_roles(candidates.get(race_id), roles)
        present_roles = {r for r in roles.values() if r not in empty}
        # a major-party leg without a candidate has no place in the two-party frame
        role_map = {eid: ("-" if r in empty else r) for eid, r in roles.items()}
        # the independent leg takes a missing major party's role in the two-party frame
        substituted = False
        if "D" not in present_roles and "I" in present_roles:
            for eid, role in roles.items():
                if role == "I":
                    role_map[eid] = "D"
                    substituted = True
                    break
        elif "R" not in present_roles and "I" in present_roles:
            for eid, role in roles.items():
                if role == "I":
                    role_map[eid] = "R"
                    substituted = True
                    break
        p_dem_polls, avail_polls, ts_polls = _polls_pdem(
            race, polls_by_race.get(race_id, []), generic, cfg, now, fundamentals=not substituted
        )
        p_dem_rat, ts_rat = _ratings_pdem(ratings_by_race.get(race_id, []), cfg, now)
        ref_by_eid: dict[str, tuple[float | None, float, float | None, datetime | None]] = {
            eid: _ref_component(_refs_for(mrows.get(eid), refs, cfg, now, max_age)) for eid in legs
        }
        # by the leg's own party label: 1 - D - R prices an independent leg without a crosswalk
        ref_by_label = {roles[eid]: ref_by_eid[eid][0] for eid in legs if ref_by_eid[eid][0] is not None}
        for eid, v in legs.items():
            role = role_map[eid]
            comps: dict[str, float] = {}
            avail: dict[str, float] = {}
            ts: datetime | None = None
            ref_mid, ref_avail, ref_liq, ref_ts = ref_by_eid[eid]
            if ref_mid is None and roles[eid] == "I" and "D" in ref_by_label and "R" in ref_by_label:
                ref_mid = max(INDEPENDENT_FLOOR, 1.0 - ref_by_label["D"] - ref_by_label["R"])
                ref_avail = 0.5
            if ref_mid is not None:
                comps["ref"] = ref_mid
                avail["ref"] = ref_avail
                ts = _latest(ts, ref_ts)
            two_party = role in ("D", "R") and not (night and ref_mid is not None)
            if two_party:
                if p_dem_polls is not None:
                    comps["polls"] = p_dem_polls if role == "D" else 1.0 - p_dem_polls
                    avail["polls"] = avail_polls
                    ts = _latest(ts, ts_polls)
                if p_dem_rat is not None:
                    comps["ratings"] = p_dem_rat if role == "D" else 1.0 - p_dem_rat
                    avail["ratings"] = 1.0
                    ts = _latest(ts, ts_rat)
            res = blend(comps, avail, cfg, applicable=TWO_PARTY_COMPONENTS if two_party else REF_ONLY)
            if res is None:
                continue
            fair, sd, weights = res
            out[eid] = FairValue(
                exchange_id=eid,
                fair=fair,
                sd=sd,
                components={k: float(x) for k, x in comps.items()},
                weights=weights,
                ts=ts,
                ref_liquidity=ref_liq,
            )
            if role == "-":
                continue  # an empty party cannot win the seat: not part of the simulation
            race_fairs.setdefault(race_id, {})[v.party] = fair
            if ts is not None:
                seat_ts[race.chamber] = _latest(seat_ts.get(race.chamber), ts)  # type: ignore[assignment]

    # control markets: simulation (on the blended race fairs) blended with the real-money reference
    sim_control: dict[str, float] = {}
    sim_avail: dict[str, float] = {}
    if control_views and race_fairs:
        try:
            from sigcup.trader import simulate as _sim

            sim = _sim.simulate(race_fairs, races, cfg, n=cfg.simulation.n_sims, seed=cfg.simulation.seed)
            sim_control = dict(sim.control_probs)
            for chamber_name, (_k, ids) in sim.requirements.items():
                sim_avail[chamber_name] = _sim.coverage(chamber_name, len(ids)) ** 2
        except Exception as e:  # noqa: BLE001 - the simulation is optional for control fair values
            log.warning("control simulation failed: %s", e)
    sim_weight = float(cfg.fairvalue.weights.get("polls", 0.0)) + float(cfg.fairvalue.weights.get("ratings", 0.0))
    for race_id, legs in control_views.items():
        race = races[race_id]
        chamber = control_chamber(race)
        for eid, v in legs.items():
            role = _role(v.party)
            comps = {}
            avail = {}
            ts = None
            ref_mid, ref_avail, ref_liq, ref_ts = _ref_component(_refs_for(mrows.get(eid), refs, cfg, now, max_age))
            if ref_mid is not None:
                comps["ref"] = ref_mid
                avail["ref"] = ref_avail
                ts = _latest(ts, ref_ts)
            use_sim = not (night and ref_mid is not None)
            if use_sim and chamber in sim_control and role in ("D", "R"):
                p = float(sim_control[chamber])
                comps["sim"] = p if role == "D" else 1.0 - p
                avail["sim"] = sim_avail.get(chamber, 0.0)
                ts = _latest(ts, seat_ts.get(chamber or ""))
            res = blend(comps, avail, cfg, weight_overrides={"sim": sim_weight}, applicable=CONTROL_COMPONENTS if use_sim else REF_ONLY)
            if res is None:
                continue
            fair, sd, weights = res
            out[eid] = FairValue(
                exchange_id=eid,
                fair=fair,
                sd=sd,
                components={k: float(x) for k, x in comps.items()},
                weights=weights,
                ts=ts,
                ref_liquidity=ref_liq,
            )
    return out
