"""Correlated Monte Carlo over races: national + state + race error layers with t(df) tails.
Outputs per-race win probabilities (should reproduce the input marginals), chamber-control
probabilities and Frechet bounds, and the portfolio P&L distribution for a candidate position set.

Construction
------------
Every race is a two-party contest in margin space. The input fair values of the race's legs are
renormalised to sum to one; the Democratic two-party share ``p_D / (p_D + p_R)`` is mapped to an
expected margin ``m`` with ``probmath.margin_from_prob`` (the inverse of the T_df(m / sigma_chamber)
fair-value map). The realised margin is ``m + eps`` with::

    eps = sigma_chamber * sqrt(df / W) * (a * Z_national + b * Z_state + c * Z_race)

where ``Z`` are standard normals (one national draw per simulation, one per state, one per race),
``W ~ chi2(df)`` is a *shared* tail scaler per simulation and ``a^2 + b^2 + c^2 = 1`` with the layer
proportions taken from ``simulation.national_sigma_pts / state_sigma_pts / race_sigma_pts``. Because
the normal combination has unit variance and the scaler is the same for all races, each ``eps`` is
exactly ``sigma_chamber * t_df`` so ``P(m + eps > 0) = T_df(m / sigma_chamber)`` = the input fair value
(the output marginals reproduce the inputs up to Monte-Carlo error; ``tests/test_model_simulate.py``
checks this bias). The shared scaler also makes fat-tail events national rather than independent.
Third-party legs win with their (renormalised) fair probability, independently of the two-party
margin draw, and the D/R outcome is drawn conditional on no third-party win.

Chamber control (documented assumptions, exposed as constants)
--------------------------------------------------------------
* Senate: 53R-47D today with 35 seats up (33 class II + FL and OH specials), 13 of them D-held
  (``SENATE_D_HELD_UP``). Democrats keep ``SENATE_D_NOT_UP = 34`` seats that are not on the ballot and
  need ``SENATE_MAJORITY = 51`` because the Vice-President (R) breaks a 50-50 tie. A D-held seat without
  a market in the simulation is assumed held; an R-held seat without a market is assumed held by the
  Republicans; an independent winner does not count toward either party.
* House: the 46 district markets cover the competitive seats. ``HOUSE_D_SEATS_NOT_IN_MARKETS = 203``
  Democratic seats are treated as fixed (215 D seats after 2024 minus the 12 D-held districts among the
  46 market districts). Revise against the live market list and any mid-decade redistricting; the
  constant can be overridden per call (``house_baseline``). Majority is ``HOUSE_MAJORITY = 218``. A
  D-held market district that is in the race universe (``meta``, ``incumbent_party == 'D'``) but has no
  fair value this cycle is assumed held, as on the Senate side, rather than counted as a loss.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sigcup.common.log import get_logger
from sigcup.settings import StrategyConfig
from sigcup.trader.probmath import margin_from_prob, sigma_for
from sigcup.trader.types import RaceInfo

__all__ = [
    "EXPECTED_RACES",
    "HOUSE_D_SEATS_NOT_IN_MARKETS",
    "HOUSE_MAJORITY",
    "SENATE_D_HELD_UP",
    "SENATE_D_NOT_UP",
    "SENATE_MAJORITY",
    "SimResult",
    "control_requirements",
    "coverage",
    "frechet_bounds",
    "portfolio_pnl",
    "simulate",
]

log = get_logger(__name__)

# -- Senate baseline (see module docstring) --------------------------------------------------------
SENATE_MAJORITY = 51
SENATE_SEATS_UP = 35
SENATE_D_HELD_UP = frozenset({"CO", "DE", "GA", "IL", "MA", "MI", "MN", "NH", "NJ", "NM", "OR", "RI", "VA"})
SENATE_D_NOT_UP = 47 - len(SENATE_D_HELD_UP)  # 34

# -- House baseline --------------------------------------------------------------------------------
HOUSE_MAJORITY = 218
HOUSE_MARKET_DISTRICTS = 46
HOUSE_D_SEATS_NOT_IN_MARKETS = 203

# races a chamber's control calculation expects to see; fewer means the fixed baseline is doing the work
EXPECTED_RACES = {"senate": SENATE_SEATS_UP, "house": HOUSE_MARKET_DISTRICTS}

_EPS = 1e-12


@dataclass
class SimResult:
    """Output of :func:`simulate`.

    ``race_probs``: race_id -> party -> P(win); ``control_probs``: chamber -> P(Democratic control);
    ``samples``: exchange_id (or ``race_id:party`` when unmapped) -> uint8 array (1 = YES) per
    simulation; ``race_samples``: race_id -> party -> the same arrays; ``seats``: chamber -> Democratic
    seat count per simulation; ``requirements``: chamber -> (D wins needed among simulated races,
    simulated race ids).
    """

    race_probs: dict[str, dict[str, float]]
    control_probs: dict[str, float]
    samples: dict[str, np.ndarray]
    n: int
    race_samples: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    seats: dict[str, np.ndarray] = field(default_factory=dict)
    requirements: dict[str, tuple[int, list[str]]] = field(default_factory=dict)


# -- helpers --------------------------------------------------------------------------------------
def _role(party: str | None) -> str:
    p = (party or "").strip().upper()
    if p.startswith("D"):
        return "D"
    if p.startswith("R"):
        return "R"
    return "I"


def _roles(legs: Mapping[str, float]) -> tuple[str | None, str | None, list[str]]:
    """Which leg plays D, which plays R, and the remaining (third-party) legs."""
    d = next((p for p in legs if _role(p) == "D"), None)
    r = next((p for p in legs if _role(p) == "R"), None)
    others = [p for p in legs if p not in (d, r)]
    if d is None and others:
        d = others.pop(0)
    if r is None and others:
        r = others.pop(0)
    return d, r, others


def _normalised(legs: Mapping[str, float]) -> dict[str, float]:
    vals = {p: min(1.0, max(0.0, float(v))) for p, v in legs.items() if v is not None and math.isfinite(float(v))}
    if not vals:
        return {}
    if len(vals) == 1:
        return vals
    total = sum(vals.values())
    if total <= _EPS:
        return {p: 1.0 / len(vals) for p in vals}
    return {p: v / total for p, v in vals.items()}


def control_requirements(
    meta: Mapping[str, RaceInfo],
    house_baseline: int | None = None,
    universe: Mapping[str, RaceInfo] | None = None,
) -> dict[str, tuple[int, list[str]]]:
    """Democratic wins needed among the simulated races per chamber, with the race ids involved.

    Senate: ``51 - 34 - (D-held seats up without a market)``. House: ``218 - baseline - (D-held districts
    of ``universe`` that are not in ``meta``)``: ``universe`` is the full race universe when ``meta`` only
    holds the races being simulated, so a D-held district without a fair value counts as held.
    """
    senate = sorted(rid for rid, r in meta.items() if r.chamber == "senate")
    house = sorted(rid for rid, r in meta.items() if r.chamber == "house")
    out: dict[str, tuple[int, list[str]]] = {}
    if senate:
        states = {meta[rid].state for rid in senate if meta[rid].state}
        holds = len([s for s in SENATE_D_HELD_UP if s not in states])
        out["senate"] = (SENATE_MAJORITY - SENATE_D_NOT_UP - holds, senate)
    if house:
        base = HOUSE_D_SEATS_NOT_IN_MARKETS if house_baseline is None else int(house_baseline)
        present = set(house)
        held = 0
        if universe is not None:
            held = sum(
                1
                for rid, r in universe.items()
                if r.chamber == "house" and rid not in present and r.incumbent_party and _role(r.incumbent_party) == "D"
            )
        out["house"] = (HOUSE_MAJORITY - base - held, house)
    return out


def coverage(chamber: str, n_present: int) -> float:
    """Fraction of the chamber's expected races that are present (1.0 when complete)."""
    expected = EXPECTED_RACES.get(chamber)
    if not expected:
        return 1.0
    return min(1.0, max(0.0, n_present / expected))


def frechet_bounds(seat_probs: Sequence[float], k: int) -> tuple[float, float]:
    """Sharp bounds on P(at least ``k`` of the events occur) given only their marginal probabilities.

    With ``N`` the number of events that occur (``E[N] = sum p``): the lower bound is the larger of
    ``(E[N] - (k-1)) / (n-k+1)`` and ``sum(top-k p) - (k-1)`` (the Frechet lower bound on the k most
    likely events all occurring); the upper bound is the smaller of ``E[N] / k`` and the sum of the
    ``n-k+1`` least likely probabilities (at least one of them must occur). The ``k == n`` case is the
    classic ``[max(0, sum p - (n-1)), min p]``.
    """
    p = sorted((min(1.0, max(0.0, float(x))) for x in seat_probs), reverse=True)
    n = len(p)
    if k <= 0:
        return 1.0, 1.0
    if k > n:
        return 0.0, 0.0
    total = sum(p)
    lo = max(0.0, (total - (k - 1)) / (n - k + 1), sum(p[:k]) - (k - 1))
    hi = min(1.0, total / k, sum(p[k - 1 :]))
    lo = min(lo, hi)
    return lo, hi


# -- the simulation -------------------------------------------------------------------------------
def simulate(
    races: Mapping[str, Mapping[str, float]],
    meta: Mapping[str, RaceInfo],
    cfg: StrategyConfig,
    n: int | None = None,
    seed: int | None = None,
    house_baseline: int | None = None,
) -> SimResult:
    """Correlated draw of every race outcome (see the module docstring).

    ``races``: race_id -> party -> fair P(YES); ``meta``: RaceInfo per race (chamber, state, markets).
    Control races (``chamber == 'control'``) are not simulated; their probabilities are outputs.
    """
    n = int(n or cfg.simulation.n_sims)
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(cfg.simulation.seed if seed is None else seed)
    df = max(1, int(cfg.fairvalue.t_df))
    tail = np.sqrt(df / rng.chisquare(df, size=n))
    z_nat = rng.standard_normal(n)
    z_state: dict[str, np.ndarray] = {}
    s_nat = float(cfg.simulation.national_sigma_pts)
    s_state = float(cfg.simulation.state_sigma_pts)

    race_probs: dict[str, dict[str, float]] = {}
    race_samples: dict[str, dict[str, np.ndarray]] = {}
    samples: dict[str, np.ndarray] = {}
    d_wins: dict[str, list[np.ndarray]] = {"senate": [], "house": []}
    sim_meta: dict[str, RaceInfo] = {}

    for race_id in sorted(races):
        info = meta.get(race_id)
        chamber = (info.chamber if info else "senate") or "senate"
        if chamber == "control":
            continue
        legs = _normalised(races[race_id])
        if not legs:
            continue
        d_leg, r_leg, others = _roles(legs)
        if d_leg is None:
            continue
        p_d = legs.get(d_leg, 0.0)
        p_r = legs.get(r_leg, 0.0) if r_leg is not None else max(0.0, 1.0 - p_d - sum(legs[o] for o in others))
        q_other = sum(legs[o] for o in others)
        two_party = p_d + p_r
        if two_party <= _EPS:
            continue
        pd2 = p_d / two_party
        m = margin_from_prob(pd2, chamber if chamber in cfg.fairvalue.sigma_pts else "senate", cfg)
        sigma = sigma_for(chamber, cfg)
        s_race = float(cfg.simulation.race_sigma_pts.get(chamber, cfg.simulation.race_sigma_pts.get("senate", 4.5)))
        norm = math.sqrt(s_nat**2 + s_state**2 + s_race**2) or 1.0
        state_key = (info.state if info and info.state else race_id) or race_id
        zs = z_state.get(state_key)
        if zs is None:
            zs = rng.standard_normal(n)
            z_state[state_key] = zs
        zr = rng.standard_normal(n)
        eps = sigma * tail * ((s_nat * z_nat + s_state * zs + s_race * zr) / norm)
        d_two = (m + eps) > 0.0
        if q_other > _EPS:
            u = rng.random(n)
            other_hit = u < q_other
            d_win = d_two & ~other_hit
            r_win = ~d_two & ~other_hit
            out_parties: dict[str, np.ndarray] = {d_leg: d_win}
            if r_leg is not None:
                out_parties[r_leg] = r_win
            cum = 0.0
            for o in others:
                lo, hi = cum, cum + legs[o]
                out_parties[o] = other_hit & (u >= lo) & (u < hi)
                cum = hi
        else:
            out_parties = {d_leg: d_two}
            if r_leg is not None:
                out_parties[r_leg] = ~d_two
        race_samples[race_id] = {p: arr.astype(np.uint8) for p, arr in out_parties.items()}
        race_probs[race_id] = {p: float(arr.mean()) for p, arr in out_parties.items()}
        sim_meta[race_id] = info or RaceInfo(race_id=race_id, chamber=chamber, state=None, label=race_id, exclusive=None, exhaustive=None)
        markets = info.markets if info else {}
        for p, arr in race_samples[race_id].items():
            key = markets.get(p) or f"{race_id}:{p}"
            samples[key] = arr
        d_role_party = d_leg if _role(d_leg) == "D" else None
        if chamber in d_wins and d_role_party is not None:
            d_wins[chamber].append(out_parties[d_role_party])
        elif chamber in d_wins:
            d_wins[chamber].append(np.zeros(n, dtype=bool))

    reqs = control_requirements(sim_meta, house_baseline, universe=meta)
    seats: dict[str, np.ndarray] = {}
    control: dict[str, float] = {}
    if reqs.get("senate"):
        k, _ = reqs["senate"]
        won = np.sum(np.stack(d_wins["senate"]), axis=0) if d_wins["senate"] else np.zeros(n)
        seats["senate"] = (SENATE_MAJORITY - k) + won
        control["senate"] = float(np.mean(seats["senate"] >= SENATE_MAJORITY))
    if reqs.get("house"):
        k, _ = reqs["house"]
        won = np.sum(np.stack(d_wins["house"]), axis=0) if d_wins["house"] else np.zeros(n)
        seats["house"] = (HOUSE_MAJORITY - k) + won
        control["house"] = float(np.mean(seats["house"] >= HOUSE_MAJORITY))
    return SimResult(
        race_probs=race_probs,
        control_probs=control,
        samples=samples,
        n=n,
        race_samples=race_samples,
        seats=seats,
        requirements=reqs,
    )


# -- portfolio ------------------------------------------------------------------------------------
def _qty_and_basis(v: Any) -> tuple[float, float | None]:
    if isinstance(v, int | float):
        return float(v), None
    if isinstance(v, list | tuple) and v:
        q = float(v[0])
        b = float(v[1]) if len(v) > 1 and v[1] is not None else None
        return q, b
    q = getattr(v, "quantity", None)
    if q is None and isinstance(v, Mapping):
        q = v.get("quantity", v.get("qty"))
    b = getattr(v, "avg_price", None)
    if b is None:
        b = getattr(v, "mark", None)
    if b is None and isinstance(v, Mapping):
        b = v.get("avg_price", v.get("mark", v.get("price")))
    return float(q or 0.0), (float(b) if b is not None else None)


def portfolio_pnl(positions: Mapping[str, Any], sim: SimResult, marks: Mapping[str, float] | None = None) -> np.ndarray:
    """Settlement P&L per simulation for signed YES positions keyed by exchange_id.

    ``positions`` values may be signed share counts, ``(qty, basis)`` pairs, or objects / dicts with
    ``quantity`` and ``avg_price``/``mark``. The basis per market is, in order: ``marks[eid]``, the
    position's own basis, the simulated probability (so the mean P&L of an unmarked book is ~0 and the
    array is a pure risk view). NO positions are negative YES quantities, as in ``Position.quantity``.
    """
    pnl = np.zeros(sim.n)
    for eid, v in positions.items():
        q, basis = _qty_and_basis(v)
        if q == 0.0:
            continue
        y = sim.samples.get(eid)
        if y is None:
            continue
        if marks is not None and eid in marks and marks[eid] is not None:
            basis = float(marks[eid])
        if basis is None:
            basis = float(y.mean())
        pnl += q * (y.astype(float) - basis)
    return pnl
