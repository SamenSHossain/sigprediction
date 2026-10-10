"""Endgame sleeve transitions: outside the final window, placed, reachable, out of reach, stale."""

from __future__ import annotations

from datetime import timedelta

import pytest

from sigcup.db.models import LeaderboardSnap
from sigcup.trader import endgame as EG
from tests.test_model_support import NOW


def _snap(ts, rank, ours, tops):
    return LeaderboardSnap(ts=ts, rank=rank, our_total=ours, top_totals=tops, member_count=500)


def test_not_in_final_window_returns_budget_frac(cfg):
    final = cfg.deadline_utc - timedelta(hours=cfg.endgame.final_hours + 1)
    snap = _snap(final, 1, 200_000, [200_000, 150_000, 120_000])
    assert EG.sleeve_fraction(cfg, NOW, snap, 100_000) == cfg.convex_sleeve.budget_frac
    assert EG.sleeve_fraction(cfg, final, snap, 100_000) == cfg.convex_sleeve.budget_frac
    assert not EG.in_final_window(cfg, final)
    assert EG.in_final_window(cfg, cfg.deadline_utc - timedelta(hours=10))
    assert not EG.in_final_window(cfg, cfg.deadline_utc + timedelta(minutes=1))


def test_placed_cuts_sleeve(cfg):
    now = cfg.deadline_utc - timedelta(hours=48)
    snap = _snap(now - timedelta(minutes=10), 2, 150_000, [160_000, 150_000, 120_000, 110_000])
    assert EG.sleeve_fraction(cfg, now, snap, 150_000) == cfg.endgame.sleeve_frac_if_placed
    snap.rank = cfg.endgame.rank_target
    assert EG.sleeve_fraction(cfg, now, snap, 150_000) == cfg.endgame.sleeve_frac_if_placed


def test_reachable_raises_sleeve(cfg):
    now = cfg.deadline_utc - timedelta(hours=48)
    # third place 120k, ours 110k: gap 9% <= 15% -> reachable
    snap = _snap(now - timedelta(minutes=10), 9, 110_000, [160_000, 150_000, 120_000, 115_000])
    assert EG.sleeve_fraction(cfg, now, snap, 110_000) == cfg.endgame.sleeve_frac_if_reachable
    # exactly at the threshold still counts
    snap = _snap(now, 9, 100_000, [160_000, 150_000, 115_000])
    assert EG.sleeve_fraction(cfg, now, snap, 100_000) == cfg.endgame.sleeve_frac_if_reachable
    # our_total missing -> equity is used
    snap = _snap(now, 9, None, [160_000, 150_000, 115_000])
    assert EG.sleeve_fraction(cfg, now, snap, 100_000) == cfg.endgame.sleeve_frac_if_reachable
    assert EG.sleeve_fraction(cfg, now, snap, 50_000) == cfg.convex_sleeve.budget_frac


def test_out_of_reach_keeps_default(cfg):
    now = cfg.deadline_utc - timedelta(hours=48)
    snap = _snap(now, 40, 100_000, [400_000, 300_000, 250_000])
    assert EG.sleeve_fraction(cfg, now, snap, 100_000) == cfg.convex_sleeve.budget_frac
    assert EG.target_total(snap, 3) == 250_000
    assert EG.target_total(snap, 4) is None


def test_missing_stale_or_disabled(cfg):
    now = cfg.deadline_utc - timedelta(hours=48)
    assert EG.sleeve_fraction(cfg, now, None, 100_000) == cfg.convex_sleeve.budget_frac
    stale = _snap(now - EG.MAX_LEADERBOARD_AGE - timedelta(minutes=1), 1, 200_000, [200_000])
    assert EG.sleeve_fraction(cfg, now, stale, 100_000) == cfg.convex_sleeve.budget_frac
    empty = _snap(now, None, None, None)
    assert EG.sleeve_fraction(cfg, now, empty, 100_000) == cfg.convex_sleeve.budget_frac
    cfg.endgame.enabled = False
    placed = _snap(now, 1, 200_000, [200_000])
    assert EG.sleeve_fraction(cfg, now, placed, 100_000) == cfg.convex_sleeve.budget_frac


def test_duck_typed_leaderboard(cfg):
    now = cfg.deadline_utc - timedelta(hours=1)
    snap = {"ts": now, "rank": 7, "our_total": 100_000, "top_totals": [130_000, 120_000, 110_000]}
    assert EG.sleeve_fraction(cfg, now, snap, 100_000) == pytest.approx(cfg.endgame.sleeve_frac_if_reachable)
