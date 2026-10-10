"""Fair-value blend on synthetic DB rows: ref-only, polls-only, ratings-only, all three; ratings parsing;
independent legs; control markets; input-timestamp semantics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sigcup.db import engine as dbengine
from sigcup.db.models import GenericBallot, Market, Poll, Race, Rating, RefPrice
from sigcup.trader import fairvalue as FV
from sigcup.trader.probmath import prob_from_margin
from tests.test_model_support import NOW, race, view

D_EID, R_EID = "SEN-TX-D", "SEN-TX-R"


def _seed_race(s, race_id="SEN-TX", chamber="senate", state="TX", parties=("D", "R"), pvi=None, incumbent=None, kalshi=True, poly=True):
    s.add(Race(race_id=race_id, chamber=chamber, state=state, district=None, label=f"{state} {chamber}", incumbent_party=incumbent, pvi=pvi, exclusive=True, exhaustive=True))
    for p in parties:
        s.add(
            Market(
                market_id=f"m-{race_id}-{p}",
                exchange_id=f"{race_id}-{p}",
                race_id=race_id,
                party=p,
                title=f"Will the {p} party win {race_id}?",
                status="open",
                kalshi_ticker=f"K-{race_id}-{p}" if kalshi else None,
                polymarket_condition_id=f"P-{race_id}-{p}" if poly else None,
            )
        )


def _ref(s, source, key, mid, liquidity, ts=NOW - timedelta(minutes=5)):
    s.add(RefPrice(ts=ts, source=source, key=key, yes_bid=mid - 0.01, yes_ask=mid + 0.01, yes_mid=mid, liquidity=liquidity))


def _poll(s, race_id, margin, days_ago, pollster="A", fetched=None):
    end = NOW - timedelta(days=days_ago)
    s.add(Poll(race_id=race_id, pollster=pollster, start_date=end - timedelta(days=3), end_date=end, sample_size=800, population="LV", dem=50 + margin / 2, rep=50 - margin / 2, margin=margin, grade="A", fetched_at=fetched or (NOW - timedelta(hours=1))))


def _rating(s, race_id, source, rating, prob_dem=0.5, ts=None):
    s.add(Rating(race_id=race_id, source=source, rating=rating, prob_dem=prob_dem, ts=ts or (NOW - timedelta(days=1))))


def _views(info):
    return {info.markets[p]: view(info, p, bid=0.5, ask=0.6) for p in info.markets}


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Lean D", "lean"),
        ("Likely R", "likely_r"),
        ("Toss-up", "tossup"),
        ("Tilt D", "tilt"),
        ("Safe R", "safe_r"),
        ("Solid Dem", "safe"),
        ("Leans Republican", "lean_r"),
        ("TOSSUP", "tossup"),
        ("Pure Toss Up", "tossup"),
        ("Likely Democratic", "likely"),
        ("tilt_r", "tilt_r"),
    ],
)
def test_parse_rating(cfg, label, expected):
    table = cfg.fairvalue.ratings_prob
    p = FV.parse_rating(label, cfg)
    assert p is not None
    if expected.endswith("_r"):
        assert p == pytest.approx(1 - table[expected[:-2]])
    else:
        assert p == pytest.approx(table[expected])


def test_parse_rating_unparseable(cfg):
    assert FV.parse_rating("", cfg) is None
    assert FV.parse_rating("Lean I", cfg) is None
    assert FV.parse_rating("Competitive", cfg) is None
    assert FV.parse_rating(None, cfg) is None


def test_load_races_maps_party_to_exchange_id(db):
    with dbengine.session() as s:
        _seed_race(s, pvi=3.0, incumbent="R")
    with dbengine.session() as s:
        races = FV.load_races(s)
    assert races["SEN-TX"].markets == {"D": D_EID, "R": R_EID}
    assert races["SEN-TX"].pvi == 3.0 and races["SEN-TX"].incumbent_party == "R"
    assert races["SEN-TX"].exclusive is True


def test_ref_only_liquidity_weighted_no_stretch(db, cfg):
    info = race("SEN-TX", "senate", "TX")
    with dbengine.session() as s:
        _seed_race(s)
        _ref(s, "kalshi", "K-SEN-TX-D", 0.97, 5000)  # full weight
        _ref(s, "polymarket", "P-SEN-TX-D", 0.93, 500)  # half weight
        _ref(s, "kalshi", "K-SEN-TX-R", 0.03, 5000)
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    fv = out[D_EID]
    assert set(fv.components) == {"ref"}
    expected = (1.0 * 0.97 + 0.5 * 0.93) / 1.5
    assert fv.components["ref"] == pytest.approx(expected)
    assert fv.fair == pytest.approx(expected)  # raw real-money price: no favourite-longshot stretch
    assert fv.weights == {"ref": pytest.approx(1.0)}
    assert fv.ref_liquidity == pytest.approx(5500)
    assert fv.sd >= FV.SD_FLOOR_SINGLE
    assert fv.ts == NOW - timedelta(minutes=5)
    assert out[R_EID].fair == pytest.approx(0.03)


def test_stale_reference_ignored(db, cfg):
    info = race("SEN-TX", "senate", "TX")
    with dbengine.session() as s:
        _seed_race(s)
        _ref(s, "kalshi", "K-SEN-TX-D", 0.97, 5000, ts=NOW - FV.REF_MAX_AGE - timedelta(hours=1))
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    assert D_EID not in out


def test_polls_only_recency_weighted_and_shrunk(db, cfg):
    info = race("SEN-TX", "senate", "TX")
    with dbengine.session() as s:
        _seed_race(s, pvi=5.0, incumbent="R")  # fundamentals: generic - 2 * 5 - 1.5 (PVI is a vote-share lean)
        s.add(GenericBallot(source="x", ts=NOW - timedelta(days=1), margin=8.0))
        _poll(s, "SEN-TX", 4.0, days_ago=0, pollster="fresh")
        _poll(s, "SEN-TX", -4.0, days_ago=7, pollster="old")  # half weight
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    fv = out[D_EID]
    assert set(fv.components) == {"polls"}
    avg = (4.0 * 1.0 + (-4.0) * 0.5) / 1.5
    n_eff = 1.5
    fund = 8.0 - 2 * 5.0 - FV.INCUMBENCY_BUMP_PTS
    s_eff = max(cfg.fairvalue.fundamentals_shrink, 1 / (1 + n_eff))
    margin = (1 - s_eff) * avg + s_eff * fund
    assert fv.components["polls"] == pytest.approx(prob_from_margin(margin, "senate", cfg))
    assert out[R_EID].components["polls"] == pytest.approx(1 - fv.components["polls"])
    assert fv.ts == NOW - timedelta(hours=1)  # the poll fetch time, not now


def test_fundamentals_only_when_no_polls(db, cfg):
    info = race("HOUSE-PA-08", "house", "PA")
    with dbengine.session() as s:
        _seed_race(s, race_id="HOUSE-PA-08", chamber="house", state="PA", pvi=4.0, incumbent="R")
        s.add(GenericBallot(source="x", ts=NOW - timedelta(days=1), margin=8.0))
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    fv = out["HOUSE-PA-08-D"]
    margin = 8.0 - 2 * 4.0 - FV.INCUMBENCY_BUMP_PTS
    assert fv.components["polls"] == pytest.approx(prob_from_margin(margin, "house", cfg))
    assert fv.sd > FV.SD_FLOOR_SINGLE  # lower availability -> wider


def test_ratings_only_mean_over_sources(db, cfg):
    info = race("SEN-TX", "senate", "TX")
    table = cfg.fairvalue.ratings_prob
    with dbengine.session() as s:
        _seed_race(s)
        _rating(s, "SEN-TX", "cook", "Lean D")
        _rating(s, "SEN-TX", "sabato", "Toss-up")
        _rating(s, "SEN-TX", "cook", "Likely R", ts=NOW - timedelta(days=10))  # older cook row ignored
        _rating(s, "SEN-TX", "manual", "??", prob_dem=0.7)  # unparseable -> stored prob_dem
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    fv = out[D_EID]
    assert set(fv.components) == {"ratings"}
    assert fv.components["ratings"] == pytest.approx((table["lean"] + table["tossup"] + 0.7) / 3)
    assert out[R_EID].fair == pytest.approx(1 - fv.fair)


def test_all_three_components_blend_with_renormalised_weights(db, cfg):
    info = race("SEN-TX", "senate", "TX")
    with dbengine.session() as s:
        _seed_race(s, pvi=0.0)
        _ref(s, "kalshi", "K-SEN-TX-D", 0.66, 5000)
        _ref(s, "kalshi", "K-SEN-TX-R", 0.34, 5000)
        _poll(s, "SEN-TX", 2.0, days_ago=1)
        _rating(s, "SEN-TX", "cook", "Toss-up")
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    fv = out[D_EID]
    assert set(fv.components) == {"ref", "polls", "ratings"}
    w = cfg.fairvalue.weights
    assert fv.weights["ref"] == pytest.approx(w["ref"])
    assert fv.weights["polls"] == pytest.approx(w["polls"])
    assert fv.weights["ratings"] == pytest.approx(w["ratings"])
    expect = sum(fv.weights[k] * fv.components[k] for k in fv.components)
    assert fv.fair == pytest.approx(expect)
    assert 0.005 <= fv.fair <= 0.995
    assert fv.sd >= FV.SD_FLOOR
    # the R leg mirrors the D leg exactly when every component is two-party
    assert out[R_EID].fair == pytest.approx(1 - fv.fair, abs=1e-6)


def test_availability_scales_weights(db, cfg):
    info = race("SEN-TX", "senate", "TX")
    with dbengine.session() as s:
        _seed_race(s)
        _ref(s, "kalshi", "K-SEN-TX-D", 0.90, 100)  # weight 0.1
        _rating(s, "SEN-TX", "cook", "Safe D")
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    fv = out[D_EID]
    w_ref = cfg.fairvalue.weights["ref"] * 0.1
    w_rat = cfg.fairvalue.weights["ratings"]
    assert fv.weights["ref"] == pytest.approx(w_ref / (w_ref + w_rat))
    assert fv.components["ref"] == pytest.approx(0.90)
    assert fv.components["ratings"] == pytest.approx(cfg.fairvalue.ratings_prob["safe"])


def test_independent_leg(db, cfg):
    info = race("SEN-NE", "senate", "NE", parties=("D", "R", "I"))
    with dbengine.session() as s:
        _seed_race(s, race_id="SEN-NE", state="NE", parties=("D", "R", "I"), poly=False)
        _ref(s, "kalshi", "K-SEN-NE-D", 0.03, 5000)
        _ref(s, "kalshi", "K-SEN-NE-R", 0.73, 5000)
        _rating(s, "SEN-NE", "cook", "Lean R")
        s.execute(Market.__table__.update().where(Market.exchange_id == "SEN-NE-I").values(kalshi_ticker=None))
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info), NOW)
    # independent: ref from 1 - D - R, no two-party components
    assert set(out["SEN-NE-I"].components) == {"ref"}
    assert out["SEN-NE-I"].components["ref"] == pytest.approx(0.24)
    assert "ratings" in out["SEN-NE-D"].components
    # when the race has no D leg the independent takes the D role
    info2 = race("GOV-RI", "governor", "RI", parties=("R", "I"))
    with dbengine.session() as s:
        _seed_race(s, race_id="GOV-RI", chamber="governor", state="RI", parties=("R", "I"), kalshi=False, poly=False)
        _rating(s, "GOV-RI", "cook", "Likely R")
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, _views(info2), NOW)
    assert out["GOV-RI-I"].components["ratings"] == pytest.approx(1 - cfg.fairvalue.ratings_prob["likely"])
    assert out["GOV-RI-R"].components["ratings"] == pytest.approx(cfg.fairvalue.ratings_prob["likely"])


def test_control_market_blends_simulation_and_ref(db, cfg):
    cfg.simulation.n_sims = 3000
    from sigcup.trader import simulate as S

    states = sorted(S.SENATE_D_HELD_UP | {"AK", "IA", "KS", "ME", "MI", "OH", "TX", "NC", "AL", "AR", "ID", "KY", "LA", "MS", "MT", "NE", "OK", "SC", "SD", "TN", "WV", "WY", "FL"})
    assert len(states) == S.SENATE_SEATS_UP
    seats = [race(f"SEN-{st}", "senate", st) for st in states]
    ctl = race("CONTROL-SENATE", "control", None, label="Senate control")
    with dbengine.session() as s:
        for st in states:
            _seed_race(s, race_id=f"SEN-{st}", state=st, poly=False)
            _ref(s, "kalshi", f"K-SEN-{st}-D", 0.5, 5000)
            _ref(s, "kalshi", f"K-SEN-{st}-R", 0.5, 5000)
        _seed_race(s, race_id="CONTROL-SENATE", chamber="control", state=None, poly=False)
        _ref(s, "kalshi", "K-CONTROL-SENATE-D", 0.63, 50000)
        _ref(s, "kalshi", "K-CONTROL-SENATE-R", 0.37, 50000)
    views = {}
    for info in (*seats, ctl):
        views.update(_views(info))
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, views, NOW)
    fv = out["CONTROL-SENATE-D"]
    assert set(fv.components) == {"ref", "sim"}
    assert fv.components["ref"] == pytest.approx(0.63)
    assert 0.05 < fv.components["sim"] < 0.6  # D needs 17 of 35 coin flips under correlated noise
    sim_w = cfg.fairvalue.weights["polls"] + cfg.fairvalue.weights["ratings"]
    assert fv.weights["sim"] == pytest.approx(sim_w / (sim_w + cfg.fairvalue.weights["ref"]))  # full coverage
    assert out["CONTROL-SENATE-R"].components["sim"] == pytest.approx(1 - fv.components["sim"])
    assert out["CONTROL-SENATE-R"].fair == pytest.approx(1 - fv.fair, abs=1e-6)


def test_control_market_partial_coverage_trusts_the_reference(db, cfg):
    cfg.simulation.n_sims = 2000
    senate = race("SEN-TX", "senate", "TX")
    ctl = race("CONTROL-SENATE", "control", None, label="Senate control")
    with dbengine.session() as s:
        _seed_race(s)
        _seed_race(s, race_id="CONTROL-SENATE", chamber="control", state=None, poly=False)
        _ref(s, "kalshi", "K-SEN-TX-D", 0.66, 5000)
        _ref(s, "kalshi", "K-CONTROL-SENATE-D", 0.63, 50000)
    views = {**_views(senate), **_views(ctl)}
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, views, NOW)
    fv = out["CONTROL-SENATE-D"]
    # 1 of 35 seat races: the simulation (which says ~0 with missing seats assumed held) barely counts
    assert fv.weights["sim"] < 0.01
    assert fv.fair == pytest.approx(0.63, abs=0.01)
    # control with no reference and no seat races -> no fair (no opinion)
    with dbengine.session() as s:
        s.query(RefPrice).delete()
    with dbengine.session() as s:
        out2 = FV.compute_all(s, cfg, _views(ctl), NOW)
    assert "CONTROL-SENATE-D" not in out2


def test_unknown_market_skipped_and_mapping_from_db(db, cfg):
    info = race("SEN-TX", "senate", "TX")
    with dbengine.session() as s:
        _seed_race(s)
        _ref(s, "kalshi", "K-SEN-TX-D", 0.9, 5000)
    unmapped = view(info, "D")
    unmapped.race_id, unmapped.party, unmapped.chamber = "SEN-TX-D", "?", "unknown"  # as BookState emits without meta
    stranger = view(race("SEN-ZZ", "senate", "ZZ"), "D")
    with dbengine.session() as s:
        out = FV.compute_all(s, cfg, [unmapped, stranger], NOW)
    assert D_EID in out and out[D_EID].fair == pytest.approx(0.9)
    assert "SEN-ZZ-D" not in out


def test_reexports():
    assert FV.prob_from_margin is prob_from_margin
    assert callable(FV.implied_margin_shift) and callable(FV.margin_from_prob) and callable(FV.shift_probability)
    assert FV.fundamentals_margin(None, None, "D") is None
    assert FV.fundamentals_margin(5.0, 8.0, "D") == pytest.approx(8.0 - 2 * 5.0 + FV.INCUMBENCY_BUMP_PTS)
    avg, n = FV.poll_average([{"margin": 2.0, "end_date": NOW}, {"margin": 6.0, "end_date": NOW - timedelta(days=7)}], NOW, 7)
    assert n == pytest.approx(1.5) and avg == pytest.approx((2.0 + 3.0) / 1.5)
    assert FV.control_chamber(race("CONTROL-HOUSE", "control", None, label="House control")) == "house"
    assert FV.control_chamber(race("CTL-1", "control", None, label="Which party controls the Senate?")) == "senate"
    assert datetime.now(UTC).tzinfo is UTC
