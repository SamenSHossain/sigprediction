"""A3: no theta / value / convex trading on guessed fair values.

77 of the 131 rows in config/ratings_manual.yaml are "seed guess" placeholders, stored as Rating rows with
source ``manual``. A fair value with no real-money reference (``ref`` component) whose ratings component
comes only from such rows (and whose polls component is the PVI / generic-ballot prior) is a guess: the
resting strategies must not trade it. A reference component, or a rating from a published source, is
enough."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta

from sigcup.db import engine as dbengine
from sigcup.db.models import Race, Rating, Strategy
from sigcup.trader import signals as SG
from sigcup.trader.types import FairValue
from tests.test_model_support import NOW, ctx, race, view

RESTING = {Strategy.THETA, Strategy.VALUE, Strategy.CONVEX}


def _fv(eid: str, p: float, *, ref: bool) -> FairValue:
    comps = {"polls": p, "ratings": p}
    weights = {"polls": 0.6, "ratings": 0.4}
    if ref:
        comps["ref"] = p
        weights = {"ref": 0.6, "polls": 0.3, "ratings": 0.1}
    return FairValue(exchange_id=eid, fair=p, sd=0.03, components=comps, weights=weights, ts=NOW)


def _seed_ratings(rows: list[tuple[str, str, str, float]], age: timedelta = timedelta(days=1)) -> None:
    with dbengine.session() as s:
        for rid, _, _, _ in rows:
            if s.get(Race, rid) is None:
                s.add(Race(race_id=rid, chamber="senate", state=rid[-2:], label=rid))
        s.flush()
        for rid, source, label, p in rows:
            s.add(Rating(race_id=rid, source=source, rating=label, prob_dem=p, ts=NOW - age))


def _favourite_and_tossup():
    fav, toss = race("SEN-WY", "senate", "WY"), race("SEN-AK", "senate", "AK")
    views = [
        view(fav, "D", bid=0.02, ask=0.05), view(fav, "R", bid=0.86, ask=0.90),
        view(toss, "D", bid=0.39, ask=0.41), view(toss, "R", bid=0.57, ask=0.59),
    ]
    return [fav, toss], views


def _resting(orders) -> list:
    return [d for d in orders if d.strategy in RESTING]


def test_seed_guess_only_fair_values_do_not_trade(db, cfg):
    _seed_ratings([("SEN-WY", "manual", "Safe R", 0.03), ("SEN-AK", "manual", "Toss-up", 0.50)])
    races, views = _favourite_and_tossup()
    fairs = [
        _fv("SEN-WY-D", 0.03, ref=False), _fv("SEN-WY-R", 0.97, ref=False),
        _fv("SEN-AK-D", 0.50, ref=False), _fv("SEN-AK-R", 0.50, ref=False),
    ]
    c = ctx(cfg, races, views, fairs)
    assert _resting(SG.generate(c, [])) == []


def test_a_reference_component_is_enough(db, cfg):
    _seed_ratings([("SEN-WY", "manual", "Safe R", 0.03), ("SEN-AK", "manual", "Toss-up", 0.50)])
    races, views = _favourite_and_tossup()
    fairs = [
        _fv("SEN-WY-D", 0.03, ref=True), _fv("SEN-WY-R", 0.97, ref=True),
        _fv("SEN-AK-D", 0.50, ref=True), _fv("SEN-AK-R", 0.50, ref=True),
    ]
    c = ctx(cfg, races, views, fairs)
    out = _resting(SG.generate(c, []))
    assert {d.strategy for d in out} == RESTING
    assert {d.exchange_id[:6] for d in out} == {"SEN-WY", "SEN-AK"}


def test_a_published_rating_is_enough_without_a_reference(db, cfg):
    # WY: a seed guess plus a published Cook rating; AK: seed guess only
    _seed_ratings([
        ("SEN-WY", "manual", "Safe R", 0.03), ("SEN-WY", "cook", "Solid R", 0.03),
        ("SEN-AK", "manual", "Toss-up", 0.50),
    ])
    races, views = _favourite_and_tossup()
    fairs = [
        _fv("SEN-WY-D", 0.03, ref=False), _fv("SEN-WY-R", 0.97, ref=False),
        _fv("SEN-AK-D", 0.50, ref=False), _fv("SEN-AK-R", 0.50, ref=False),
    ]
    c = ctx(cfg, races, views, fairs)
    out = _resting(SG.generate(c, []))
    assert out and {d.exchange_id[:6] for d in out} == {"SEN-WY"}


def test_a_stale_published_rating_does_not_count(db, cfg):
    # fairvalue ignores ratings older than 60 days, so such a row cannot be what the ratings component used
    _seed_ratings([("SEN-WY", "cook", "Solid R", 0.03)], age=timedelta(days=90))
    _seed_ratings([("SEN-WY", "manual", "Safe R", 0.03)])
    races, views = _favourite_and_tossup()
    c = ctx(cfg, races, views, [_fv("SEN-WY-D", 0.03, ref=False), _fv("SEN-WY-R", 0.97, ref=False)])
    assert _resting(SG.generate(c, [])) == []


def test_fundamentals_only_fair_value_does_not_trade(db, cfg):
    races, views = _favourite_and_tossup()
    fairs = [FairValue(exchange_id="SEN-WY-R", fair=0.97, sd=0.04, components={"polls": 0.97}, weights={"polls": 1.0}, ts=NOW)]
    c = ctx(cfg, races, views, fairs)
    assert _resting(SG.generate(c, [])) == []


def test_unreadable_ratings_table_means_no_trading_without_a_reference(db, cfg, monkeypatch):
    @contextmanager
    def broken():
        raise RuntimeError("database is locked")
        yield  # pragma: no cover

    races, views = _favourite_and_tossup()
    fairs = [_fv("SEN-WY-R", 0.97, ref=False), _fv("SEN-AK-D", 0.50, ref=True)]
    c = ctx(cfg, races, views, fairs)
    monkeypatch.setattr(dbengine, "session", broken)
    out = _resting(SG.generate(c, []))
    assert {d.exchange_id for d in out} == {"SEN-AK-D"}
