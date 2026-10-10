"""Regression tests for review findings F12 (endgame: rank unknown but already at the target total) and
F13 (simulate: House districts without a fair value counted as Republican wins)."""

from __future__ import annotations

from datetime import timedelta

from sigcup.trader import endgame as EG
from sigcup.trader import simulate as S
from tests.test_model_support import race


def test_f12_unknown_rank_at_or_above_target_total_is_placed(cfg):
    now = cfg.deadline_utc - timedelta(hours=48)
    snap = {"ts": now, "rank": None, "our_total": 125_000, "top_totals": [130_000, 125_000, 120_000]}
    assert EG.sleeve_fraction(cfg, now, snap, 125_000) == cfg.endgame.sleeve_frac_if_placed
    # still below the target total but within reach: the larger sleeve
    snap = {"ts": now, "rank": None, "our_total": 110_000, "top_totals": [130_000, 125_000, 120_000]}
    assert EG.sleeve_fraction(cfg, now, snap, 110_000) == cfg.endgame.sleeve_frac_if_reachable


def _house(n_d_held: int = 12):
    meta = {}
    for i in range(S.HOUSE_MARKET_DISTRICTS):
        rid = f"HOUSE-ZZ-{i:02d}"
        meta[rid] = race(rid, "house", f"Z{i % 20:02d}", incumbent_party="D" if i < n_d_held else "R")
    return meta


def test_f13_missing_d_held_districts_are_assumed_held(cfg):
    meta = _house()
    fairs = {rid: {"D": 0.5, "R": 0.5} for rid in meta}
    k_full = S.simulate(fairs, meta, cfg, n=500, seed=1).requirements["house"][0]
    assert k_full == S.HOUSE_MAJORITY - S.HOUSE_D_SEATS_NOT_IN_MARKETS
    # two D-held districts have no fair value this cycle: they count as held, not as losses
    del fairs["HOUSE-ZZ-00"], fairs["HOUSE-ZZ-01"]
    res = S.simulate(fairs, meta, cfg, n=500, seed=1)
    assert res.requirements["house"][0] == k_full - 2
    assert len(res.requirements["house"][1]) == S.HOUSE_MARKET_DISTRICTS - 2
    # a missing R-held district changes nothing
    fairs = {rid: {"D": 0.5, "R": 0.5} for rid in meta if rid != "HOUSE-ZZ-40"}
    assert S.simulate(fairs, meta, cfg, n=500, seed=1).requirements["house"][0] == k_full
    # the helper itself, with and without the universe
    sub = {rid: r for rid, r in meta.items() if rid not in ("HOUSE-ZZ-00", "HOUSE-ZZ-01")}
    assert S.control_requirements(sub)["house"][0] == k_full
    assert S.control_requirements(sub, universe=meta)["house"][0] == k_full - 2
