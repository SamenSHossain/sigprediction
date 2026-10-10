"""Regression tests for the bootstrap exclusivity review finding (F10).

Exclusive / exhaustive flags may come ONLY from a relationships fetch made in the same run:
* relationship kinds are matched exactly ('non_exclusive' / 'not_mutually_exclusive' are not exclusive);
* inactive relationships are ignored;
* with --skip-sig, or when /relationships fails, the DB gets NULL flags (no evidence), never the stale or
  hand-set values still sitting in races.yaml.
Nothing touches the network."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from sigcup.db import engine as dbengine
from sigcup.db.models import Race
from sigcup.feeds import universe as U
from sigcup.settings import Secrets
from sigcup.sig.schemas import Relationship
from tests.test_feeds_bootstrap import B, FakeAdapter, _merged, fresh_universe


@pytest.mark.parametrize("kind", ["non_exclusive", "not_mutually_exclusive", "NOT-MUTUALLY-EXCLUSIVE", "inclusive"])
def test_negated_kinds_are_not_exclusive(tmp_path, kind):
    data = _merged(tmp_path)
    out = B.apply_relationships(data, [Relationship(id="x", kind=kind, isExhaustive=True, exchange_ids=["1101", "1102"])])
    assert data["races"]["SEN-TX"]["exclusive"] is None and data["races"]["SEN-TX"]["exhaustive"] is None
    assert out["exclusive_races"] == 0


@pytest.mark.parametrize("kind", ["mutually_exclusive", "MUTUALLY-EXCLUSIVE", "exactly_one", "at_most_one", "one_of"])
def test_exclusive_kinds_still_match(tmp_path, kind):
    data = _merged(tmp_path)
    B.apply_relationships(data, [Relationship(id="x", kind=kind, isExhaustive=True, exchange_ids=["1101", "1102"])])
    assert data["races"]["SEN-TX"]["exclusive"] is True


@pytest.mark.parametrize("active", [False, "false", 0])
def test_inactive_relationships_are_ignored(tmp_path, active):
    data = _merged(tmp_path)
    rel = Relationship(id="x", kind="mutually_exclusive", isExhaustive=True, exchange_ids=["1101", "1102"], active=active)
    out = B.apply_relationships(data, [rel])
    assert data["races"]["SEN-TX"]["exclusive"] is None and out["exclusive_races"] == 0
    live = Relationship(id="y", kind="mutually_exclusive", isExhaustive=True, exchange_ids=["1101", "1102"], active=True)
    B.apply_relationships(data, [live])
    assert data["races"]["SEN-TX"]["exclusive"] is True


def _yaml_with_stale_flags(tmp_path):
    path = fresh_universe(tmp_path)
    data = U.load_races_yaml(path)
    data["races"]["SEN-TX"]["exclusive"] = True  # stale / hand-set: the exchange may have withdrawn it
    data["races"]["SEN-TX"]["exhaustive"] = True
    U.save_races_yaml(data, path)
    return path


async def test_skip_sig_does_not_write_yaml_exclusivity_to_the_db(db, tmp_path, cfg):
    path = _yaml_with_stale_flags(tmp_path)
    args = B.build_parser().parse_args(["--races-yaml", str(path), "--skip-sig", "--skip-crosswalk"])
    await B.run(args, cfg=cfg, sec=Secrets(_env_file=None), session_factory=dbengine.session)
    with dbengine.session() as s:
        tx = s.get(Race, "SEN-TX")
        assert (tx.exclusive, tx.exhaustive) == (None, None)
        assert all(r.exclusive is None for r in s.execute(select(Race)).scalars())


async def test_failed_relationships_call_does_not_write_yaml_exclusivity_to_the_db(db, tmp_path, cfg):
    path = _yaml_with_stale_flags(tmp_path)
    fake = FakeAdapter(fail_relationships=True)
    args = B.build_parser().parse_args(["--races-yaml", str(path), "--skip-crosswalk"])
    report = await B.run(args, cfg=cfg, sec=Secrets(_env_file=None), adapter=fake, session_factory=dbengine.session)
    assert any("get_relationships failed" in n for n in report.notes)
    with dbengine.session() as s:
        tx = s.get(Race, "SEN-TX")
        assert (tx.exclusive, tx.exhaustive) == (None, None)


def test_upsert_db_without_a_relationships_fetch_stores_null_flags(db, tmp_path):
    data = _merged(tmp_path)
    B.apply_relationships(data, [Relationship(id="tx", kind="mutually_exclusive", isExhaustive=True, exchange_ids=["1101", "1102"])])
    B.upsert_db(dbengine.session, data, relationships_fetched=False)
    with dbengine.session() as s:
        assert s.get(Race, "SEN-TX").exclusive is None
    B.upsert_db(dbengine.session, data)  # default: the flags in `data` are this run's API evidence
    with dbengine.session() as s:
        assert s.get(Race, "SEN-TX").exclusive is True
