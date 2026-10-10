"""Regression tests for review findings on fairvalue.py: F1 (election-night fair value held by stale
polls), F2 (Cook PVI is a vote-share lean: the margin lean is 2 * PVI), F3 (a party leg with no candidate
priced from the two-party frame) and F14 (control-market timestamp, availability over applicable inputs)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sigcup.db import engine as dbengine
from sigcup.db.models import GenericBallot, Market, Poll, Race, Rating, RefPrice
from sigcup.trader import fairvalue as FV
from sigcup.trader import signals as SG
from sigcup.trader.probmath import prob_from_margin
from tests.test_model_support import NOW, ctx, race, view

NIGHT = datetime(2026, 11, 4, 2, 0, tzinfo=UTC)


def _race(s, race_id, chamber="senate", state="TX", parties=("D", "R"), pvi=None, incumbent=None, candidates=None, tickers=None):
    s.add(Race(race_id=race_id, chamber=chamber, state=state, district=None, label=f"{state} {chamber}", incumbent_party=incumbent, pvi=pvi, exclusive=None, exhaustive=None, candidates=candidates))
    for p in parties:
        s.add(
            Market(
                market_id=f"m-{race_id}-{p}",
                exchange_id=f"{race_id}-{p}",
                race_id=race_id,
                party=p,
                title=f"Will the {p} party win {race_id}?",
                status="open",
                kalshi_ticker=(tickers or {}).get(p, f"K-{race_id}-{p}"),
            )
        )


def _ref(s, key, mid, ts, liquidity=50_000):
    s.add(RefPrice(ts=ts, source="kalshi", key=key, yes_bid=mid - 0.005, yes_ask=mid + 0.005, yes_mid=mid, liquidity=liquidity))


def _views(info, bid=0.45, ask=0.55):
    return {info.markets[p]: view(info, p, bid=bid, ask=ask) for p in info.markets}


# -- F2 ---------------------------------------------------------------------------------------------
def test_f2_fundamentals_margin_uses_twice_the_pvi():
    # R+10 district, generic ballot D+8, R incumbent: 8 - 2 * 10 - 1.5
    assert FV.fundamentals_margin(10.0, 8.0, "R") == pytest.approx(-13.5)
    assert FV.fundamentals_margin(-3.0, 0.0, None) == pytest.approx(6.0)


def test_f2_fundamentals_prior_keeps_the_partisan_lean(db, cfg):
    info = race("HOUSE-XX-01", "house", "XX")
    with dbengine.session() as s:
        _race(s, "HOUSE-XX-01", chamber="house", state="XX", pvi=10.0, incumbent="R")
        s.add(GenericBallot(source="x", ts=NOW - timedelta(days=1), margin=8.0))
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    p = out["HOUSE-XX-01-D"].components["polls"]
    assert p == pytest.approx(prob_from_margin(-13.5, "house", cfg))
    assert p < 0.15  # a likely-R seat, not a near toss-up


# -- F3 ---------------------------------------------------------------------------------------------
def _seed_ne(s, candidates):
    _race(s, "SEN-NE", state="NE", parties=("D", "R", "I"), pvi=13.0, incumbent="R", candidates=candidates, tickers={"D": None})
    s.add(GenericBallot(source="x", ts=NOW - timedelta(days=1), margin=8.0))
    s.add(Rating(race_id="SEN-NE", source="inside", rating="Lean R", prob_dem=0.5, ts=NOW - timedelta(days=1)))
    _ref(s, "K-SEN-NE-R", 0.73, NOW - timedelta(minutes=5))
    _ref(s, "K-SEN-NE-I", 0.26, NOW - timedelta(minutes=5))


def test_f3_party_leg_without_a_candidate_gets_no_two_party_components(db, cfg):
    info = race("SEN-NE", "senate", "NE", parties=("D", "R", "I"))
    with dbengine.session() as s:
        _seed_ne(s, {"R": {"name": "Pete Ricketts"}, "I": {"name": "Dan Osborn"}})
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    # no Democrat on the ballot and no reference for the D leg: no opinion (never ~0.25 from the R-vs-D frame)
    assert "SEN-NE-D" not in out
    lean_r = cfg.fairvalue.ratings_prob["lean"]
    # the independent takes the Democratic role in the two-party frame
    assert out["SEN-NE-I"].components["ratings"] == pytest.approx(1 - lean_r)
    assert out["SEN-NE-I"].components["ref"] == pytest.approx(0.26)
    # the D-vs-R fundamentals prior (PVI, generic ballot) does not describe an R-vs-I race
    assert "polls" not in out["SEN-NE-R"].components and "polls" not in out["SEN-NE-I"].components
    assert out["SEN-NE-R"].components["ratings"] == pytest.approx(lean_r)
    # nothing buys the empty D leg
    views = list(_views(info, bid=0.01, ask=0.03).values())
    c = ctx(cfg, [info], views, list(out.values()))
    assert [d for d in SG.generate(c, []) if d.exchange_id == "SEN-NE-D" and d.is_long_yes] == []


def test_f3_partial_candidate_list_is_not_treated_as_the_full_field(db, cfg):
    info = race("SEN-NE", "senate", "NE", parties=("D", "R", "I"))
    with dbengine.session() as s:
        _seed_ne(s, {"R": {"name": "Pete Ricketts"}})  # one listed candidate: unknown field, D keeps its role
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    assert "ratings" in out["SEN-NE-D"].components
    assert set(out["SEN-NE-I"].components) == {"ref"}


# -- F1 ---------------------------------------------------------------------------------------------
def test_f1_election_night_fair_is_the_fresh_reference(db, cfg):
    info = race("SEN-IA", "senate", "IA")
    ref_ts = NIGHT - timedelta(seconds=20)
    with dbengine.session() as s:
        _race(s, "SEN-IA", state="IA", pvi=0.0)
        _ref(s, "K-SEN-IA-D", 0.98, ref_ts)
        _ref(s, "K-SEN-IA-R", 0.02, ref_ts)
        for i in range(5):
            end = NIGHT - timedelta(days=2 + i)
            s.add(Poll(race_id="SEN-IA", pollster=f"p{i}", start_date=end - timedelta(days=3), end_date=end, sample_size=800, population="LV", dem=48, rep=48, margin=0.0, grade="A", fetched_at=NIGHT - timedelta(hours=2)))
        s.add(Rating(race_id="SEN-IA", source="cook", rating="Toss-up", prob_dem=0.5, ts=NIGHT - timedelta(days=3)))
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info, 0.92, 0.94), NIGHT)
    fv = out["SEN-IA-D"]
    assert set(fv.components) == {"ref"}
    assert fv.fair == pytest.approx(0.98)
    assert fv.ts == ref_ts
    assert out["SEN-IA-R"].fair == pytest.approx(0.02)
    # a reference that stopped updating on election night is not a fair value
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info, 0.92, 0.94), NIGHT + timedelta(hours=1))
    assert "ref" not in out["SEN-IA-D"].components


# -- F14 --------------------------------------------------------------------------------------------
def test_f14_control_timestamp_is_an_input_time_not_now(db, cfg):
    cfg.simulation.n_sims = 500
    seat = race("SEN-TX", "senate", "TX")
    ctl = race("CONTROL-SENATE", "control", None, label="Senate control")
    seat_ts = NOW - timedelta(minutes=7)
    with dbengine.session() as s:
        _race(s, "SEN-TX")
        _race(s, "CONTROL-SENATE", chamber="control", state=None)
        _ref(s, "K-SEN-TX-D", 0.6, seat_ts)
        _ref(s, "K-SEN-TX-R", 0.4, seat_ts)
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, {**_views(seat), **_views(ctl)}, NOW)
    fv = out["CONTROL-SENATE-D"]
    assert set(fv.components) == {"sim"}
    assert fv.ts == seat_ts


def test_f14_availability_penalty_only_over_applicable_components(db, cfg):
    info = race("SEN-NE", "senate", "NE", parties=("D", "R", "I"))
    with dbengine.session() as s:
        _race(s, "SEN-NE", state="NE", parties=("D", "R", "I"))
        for p, mid in (("D", 0.03), ("R", 0.73), ("I", 0.24)):
            _ref(s, f"K-SEN-NE-{p}", mid, NOW - timedelta(minutes=5), liquidity=5000)
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    # an independent leg can only ever have a reference: a fully liquid one carries no availability penalty
    assert set(out["SEN-NE-I"].components) == {"ref"}
    assert out["SEN-NE-I"].sd == pytest.approx(FV.SD_FLOOR_SINGLE)
    # a control market with both of its inputs fully available: no penalty either (full = ref + sim, not 1.4)
    res = FV.blend({"ref": 0.6, "sim": 0.5}, {"ref": 1.0, "sim": 1.0}, cfg, weight_overrides={"sim": 0.4}, applicable=("ref", "sim"))
    assert res is not None
    fair, sd, w = res
    disagreement2 = w["ref"] * (0.6 - fair) ** 2 + w["sim"] * (0.5 - fair) ** 2
    assert sd == pytest.approx((disagreement2 + FV.SD_FLOOR**2) ** 0.5)
