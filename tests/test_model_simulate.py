"""Correlated simulation: marginal bias, correlation, control bounds, portfolio P&L."""

from __future__ import annotations

import numpy as np
import pytest

from sigcup.trader import simulate as S
from sigcup.trader.types import RaceInfo
from tests.test_model_support import race

SENATE_STATES = [
    "AK", "IA", "KS", "ME", "MI", "OH", "TX", "NH", "GA", "MN", "NC", "CO", "DE", "IL", "MA", "NJ", "NM",
    "OR", "RI", "VA", "AL", "AR", "ID", "KY", "LA", "MS", "MT", "NE", "OK", "SC", "SD", "TN", "WV", "WY", "FL",
]


def _senate_universe(probs=None):
    meta: dict[str, RaceInfo] = {}
    races: dict[str, dict[str, float]] = {}
    ps = np.linspace(0.05, 0.95, len(SENATE_STATES)) if probs is None else probs
    for st, p in zip(SENATE_STATES, ps, strict=True):
        rid = f"SEN-{st}"
        meta[rid] = race(rid, "senate", st)
        races[rid] = {"D": float(p), "R": float(1 - p)}
    return races, meta


def test_marginals_reproduce_inputs(cfg):
    """Bias check: with symmetric t noise scaled to sigma_chamber the output P(win) equals the input
    fair value up to Monte-Carlo error (sd <= 0.0035 at n=20000; tolerance 0.02 ~ 5.7 sd)."""
    races, meta = _senate_universe()
    res = S.simulate(races, meta, cfg, n=20000, seed=7)
    for rid, legs in races.items():
        assert res.race_probs[rid]["D"] == pytest.approx(legs["D"], abs=0.02)
        assert res.race_probs[rid]["R"] == pytest.approx(legs["R"], abs=0.02)
        assert res.race_probs[rid]["D"] + res.race_probs[rid]["R"] == pytest.approx(1.0)
    # also for a governor and a house race (different sigma / race layer)
    g = race("GOV-GA", "governor", "GA")
    h = race("HOUSE-PA-08", "house", "PA")
    res2 = S.simulate({"GOV-GA": {"D": 0.45, "R": 0.55}, "HOUSE-PA-08": {"D": 0.62, "R": 0.38}}, {g.race_id: g, h.race_id: h}, cfg, n=20000, seed=3)
    assert res2.race_probs["GOV-GA"]["D"] == pytest.approx(0.45, abs=0.02)
    assert res2.race_probs["HOUSE-PA-08"]["D"] == pytest.approx(0.62, abs=0.02)


def test_samples_are_keyed_by_exchange_id_and_sum_to_one_per_race(cfg):
    races, meta = _senate_universe()
    res = S.simulate(races, meta, cfg, n=2000, seed=1)
    assert res.n == 2000
    assert "SEN-TX-D" in res.samples and "SEN-TX-R" in res.samples
    assert np.all(res.samples["SEN-TX-D"] + res.samples["SEN-TX-R"] == 1)


def test_correlated_joint_exceeds_independence_and_control_within_frechet(cfg):
    races, meta = _senate_universe()
    res = S.simulate(races, meta, cfg, n=20000, seed=11)
    # P(D wins all of five 60%-ish seats) >= product of marginals (shared national swing)
    picks = [rid for rid in races if 0.55 <= races[rid]["D"] <= 0.75][:5]
    assert len(picks) >= 3
    joint = np.mean(np.all(np.stack([res.race_samples[r]["D"] for r in picks]).astype(bool), axis=0))
    product = float(np.prod([races[r]["D"] for r in picks]))
    assert joint >= product * 0.98
    assert joint > product + 0.02  # clearly correlated, not just MC noise
    # control probability inside the Frechet bounds implied by the seat marginals
    k, ids = res.requirements["senate"]
    lo, hi = S.frechet_bounds([races[r]["D"] for r in ids], k)
    assert lo <= res.control_probs["senate"] <= hi
    assert "senate" in res.seats and res.seats["senate"].shape == (20000,)


def test_control_requirements_senate_and_house():
    _, meta = _senate_universe()
    reqs = S.control_requirements(meta)
    assert reqs["senate"][0] == S.SENATE_MAJORITY - S.SENATE_D_NOT_UP  # all 35 seats present -> 17
    assert len(reqs["senate"][1]) == 35
    # a D-held seat without a market is assumed held: one fewer win needed
    del meta["SEN-VA"]
    assert S.control_requirements(meta)["senate"][0] == 16
    house = {f"HOUSE-{i:02d}": race(f"HOUSE-{i:02d}", "house", "PA") for i in range(46)}
    assert S.control_requirements(house)["house"][0] == S.HOUSE_MAJORITY - S.HOUSE_D_SEATS_NOT_IN_MARKETS
    assert S.control_requirements(house, house_baseline=210)["house"][0] == 8
    assert S.SENATE_D_NOT_UP == 34 and S.HOUSE_MAJORITY == 218


def test_house_control_probability(cfg):
    house = {}
    fairs = {}
    for i in range(46):
        rid = f"HOUSE-{i:02d}"
        house[rid] = race(rid, "house", f"S{i % 20:02d}")
        fairs[rid] = {"D": 0.5, "R": 0.5}
    res = S.simulate(fairs, house, cfg, n=5000, seed=2)
    # D needs 15 of 46 coin flips: ~0.994 if independent; the shared national / state errors make
    # the count far more dispersed, so the probability is clearly lower but still high
    from scipy.stats import binom

    independent = float(binom.sf(14, 46, 0.5))
    assert 0.7 < res.control_probs["house"] < independent
    assert res.seats["house"].min() >= S.HOUSE_D_SEATS_NOT_IN_MARKETS


def test_frechet_bounds():
    # k == n: classic [max(0, sum p - (n-1)), min p]
    assert S.frechet_bounds([0.9, 0.8], 2) == pytest.approx((0.7, 0.8))
    # k == 1: at least one -> [max p, min(1, sum p)]
    lo, hi = S.frechet_bounds([0.2, 0.3], 1)
    assert lo == pytest.approx(0.3) and hi == pytest.approx(0.5)
    assert S.frechet_bounds([0.5, 0.5, 0.5], 0) == (1.0, 1.0)
    assert S.frechet_bounds([0.5], 2) == (0.0, 0.0)
    lo, hi = S.frechet_bounds([0.6] * 10, 5)
    assert 0.0 <= lo <= hi <= 1.0
    # independence product sits inside the bounds for k == n
    lo, hi = S.frechet_bounds([0.6, 0.7, 0.8], 3)
    assert lo <= 0.6 * 0.7 * 0.8 <= hi


def test_three_party_race_renormalises_and_reproduces(cfg):
    ne = race("SEN-NE", "senate", "NE", parties=("D", "R", "I"))
    res = S.simulate({"SEN-NE": {"D": 0.03, "R": 0.73, "I": 0.24}}, {"SEN-NE": ne}, cfg, n=20000, seed=5)
    p = res.race_probs["SEN-NE"]
    assert p["D"] == pytest.approx(0.03, abs=0.01)
    assert p["R"] == pytest.approx(0.73, abs=0.02)
    assert p["I"] == pytest.approx(0.24, abs=0.02)
    assert np.all(res.samples["SEN-NE-D"] + res.samples["SEN-NE-R"] + res.samples["SEN-NE-I"] == 1)
    # an independent win does not count as a Democratic seat
    assert res.seats["senate"].max() <= S.SENATE_MAJORITY - res.requirements["senate"][0] + 1


def test_single_leg_and_control_races_are_handled(cfg):
    only_d = race("GOV-X", "governor", "XX", parties=("D",))
    ctl = race("CONTROL-SENATE", "control", None, label="Senate control")
    res = S.simulate({"GOV-X": {"D": 0.3}, "CONTROL-SENATE": {"D": 0.6, "R": 0.4}}, {"GOV-X": only_d, "CONTROL-SENATE": ctl}, cfg, n=8000, seed=9)
    assert res.race_probs["GOV-X"]["D"] == pytest.approx(0.3, abs=0.03)
    assert "CONTROL-SENATE" not in res.race_probs
    assert res.control_probs == {}  # no senate seat races were simulated


def test_seed_determinism(cfg):
    races, meta = _senate_universe()
    a = S.simulate(races, meta, cfg, n=1000, seed=42)
    b = S.simulate(races, meta, cfg, n=1000, seed=42)
    c = S.simulate(races, meta, cfg, n=1000, seed=43)
    assert a.race_probs == b.race_probs
    assert a.race_probs != c.race_probs


def test_portfolio_pnl(cfg):
    races, meta = _senate_universe()
    res = S.simulate(races, meta, cfg, n=10000, seed=4)
    p = res.race_probs["SEN-TX"]["D"]
    # long YES at the simulated probability: mean ~ 0, support {-p, 1-p} * q
    pnl = S.portfolio_pnl({"SEN-TX-D": 100}, res)
    assert pnl.shape == (10000,)
    assert abs(pnl.mean()) < 1e-9
    assert set(np.round(np.unique(pnl), 6)) <= {round(100 * (1 - p), 6), round(-100 * p, 6)}
    # NO position (negative quantity) with an explicit basis: wins 1 - basis when YES loses
    pnl_no = S.portfolio_pnl({"SEN-TX-D": (-100, 0.4)}, res)
    assert pnl_no.max() == pytest.approx(100 * 0.4)
    assert pnl_no.min() == pytest.approx(-100 * 0.6)
    # marks override; objects with quantity / avg_price work; unknown ids are ignored
    class Pos:
        quantity = 50
        avg_price = 0.5

    pnl_obj = S.portfolio_pnl({"SEN-TX-D": Pos(), "nope": 10}, res, marks={"SEN-TX-D": 0.25})
    assert pnl_obj.max() == pytest.approx(50 * 0.75)
