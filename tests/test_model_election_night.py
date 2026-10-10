"""Election-night mode: activation window, the taker rule on reference prices only, size bounds."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sigcup.db.models import Strategy
from sigcup.trader import election_night as EN
from sigcup.trader.types import PRIORITY_ARB
from tests.test_model_support import book, ctx, race, view

NIGHT = datetime(2026, 11, 4, 2, 0, tzinfo=UTC)


def test_is_active_window(cfg):
    assert not EN.is_active(cfg, cfg.election_night.start_utc - timedelta(seconds=1))
    assert EN.is_active(cfg, cfg.election_night.start_utc)
    assert EN.is_active(cfg, NIGHT)
    assert not EN.is_active(cfg, cfg.deadline_utc)
    assert not EN.is_active(cfg, cfg.deadline_utc + timedelta(hours=1))
    assert EN.is_active(cfg, datetime(2026, 11, 4, 1, 0))  # naive -> UTC


def _night_ctx(cfg, races, views, **kw):
    kw.setdefault("election_night", True)
    return ctx(cfg, races, views, now=NIGHT, **kw)


def test_yes_take_rule(cfg):
    info = race("SEN-TX", "senate", "TX")
    # ref 0.98, SIG ask 0.89 <= 0.98 - 0.08 -> take; ask 0.91 -> lag too small
    v = view(info, "D", bid=0.85, ask=0.89, ask_size=120, last=0.50)
    c = _night_ctx(cfg, [info], [v])
    out = EN.taker_orders(c, {"SEN-TX-D": 0.98})
    assert len(out) == 1
    d = out[0]
    assert d.side == "yes" and d.action == "buy" and d.price == pytest.approx(0.89)
    assert d.take is True and d.priority == PRIORITY_ARB and d.strategy == Strategy.ELECTION
    assert d.quantity == 120 and d.fair == pytest.approx(0.98)
    v.ask = 0.91
    assert EN.taker_orders(c, {"SEN-TX-D": 0.98}) == []
    # exactly at the lag boundary counts
    v.ask = 0.90
    assert len(EN.taker_orders(c, {"SEN-TX-D": 0.98})) == 1
    # ref below the trigger: never
    v.ask = 0.50
    assert EN.taker_orders(c, {"SEN-TX-D": 0.96}) == []


def test_no_take_rule_mirror(cfg):
    info = race("SEN-TX", "senate", "TX")
    # ref 0.02 (YES dead): NO ref 0.98; SIG bid 0.12 -> NO ask 0.88 <= 0.98 - 0.08 -> buy NO at 0.88
    v = view(info, "R", bid=0.12, ask=0.16, bid_size=75)
    c = _night_ctx(cfg, [info], [v])
    out = EN.taker_orders(c, {"SEN-TX-R": 0.02})
    assert len(out) == 1
    d = out[0]
    assert d.side == "no" and d.action == "buy" and d.price == pytest.approx(0.88)
    assert d.quantity == 75 and d.take and d.edge == pytest.approx(0.10)
    # bid 0.09: lag 0.07 < 0.08 -> nothing
    v.bid = 0.09
    assert EN.taker_orders(c, {"SEN-TX-R": 0.02}) == []


def test_never_triggers_on_sig_prints(cfg):
    info = race("SEN-TX", "senate", "TX")
    v = view(info, "D", bid=0.85, ask=0.89, ask_size=100, last=0.99)  # a SIG print at 0.99 is not a reference
    c = _night_ctx(cfg, [info], [v], ref_prices={"SEN-TX-D": 0.99})
    assert EN.taker_orders(c, {}) == []
    assert EN.taker_orders(c, {"SEN-TX-D": 0.50}) == []
    assert len(EN.taker_orders(c, {"SEN-TX-D": 0.99})) == 1


def test_inactive_outside_window(cfg):
    info = race("SEN-TX", "senate", "TX")
    v = view(info, "D", bid=0.85, ask=0.89, ask_size=100)
    c = ctx(cfg, [info], [v], now=datetime(2026, 10, 20, tzinfo=UTC))
    assert EN.taker_orders(c, {"SEN-TX-D": 0.99}) == []
    c = ctx(cfg, [info], [v], now=cfg.deadline_utc + timedelta(minutes=1))
    assert EN.taker_orders(c, {"SEN-TX-D": 0.99}) == []


def test_sizes_bounded_by_depth_and_reserve(cfg):
    a = race("SEN-A", "senate", "AA")
    b = race("SEN-B", "senate", "BB")
    va = view(a, "D", bid=0.80, ask=0.85, ask_size=None)  # no displayed size -> no order
    vb = view(b, "D", bid=0.80, ask=0.85, ask_size=None, book=book("SEN-B-D", bids=[(0.80, 50)], asks=[(0.85, 40_000), (0.90, 10)]))
    c = _night_ctx(cfg, [a, b], [va, vb], cash=100_000, reserved=10_000)
    out = EN.taker_orders(c, {"SEN-A-D": 0.99, "SEN-B-D": 0.99})
    assert [d.exchange_id for d in out] == ["SEN-B-D"]
    # final-72h reserve 30% of 100k = 30k; 80% deployable = 24k; 24k / 0.85 = 28235 < 40000 displayed
    budget = cfg.election_night.reserve_deploy_frac * cfg.risk.liquid_reserve_frac_final72h * cfg.bankroll_start
    assert EN.deployable_cash(c) == pytest.approx(budget)
    assert out[0].quantity == int(budget / 0.85 + 1e-9)
    # free cash binds when lower
    c = _night_ctx(cfg, [a, b], [va, vb], cash=5_000, reserved=1_000)
    out = EN.taker_orders(c, {"SEN-B-D": 0.99})
    assert out[0].quantity == int(4_000 / 0.85 + 1e-9)
    # best edge first, budget shared across takes
    vc = view(race("SEN-C", "senate", "CC"), "D", bid=0.70, ask=0.75, ask_size=100)
    c = _night_ctx(cfg, [a, b, race("SEN-C", "senate", "CC")], [va, vb, vc], cash=1_000, reserved=0)
    out = EN.taker_orders(c, {"SEN-B-D": 0.99, "SEN-C-D": 0.99})
    assert [d.exchange_id for d in out] == ["SEN-C-D", "SEN-B-D"]
    assert sum(d.quantity * d.price for d in out) <= 1_000 + 1e-9


def test_settlement_caveats_halve_size(cfg):
    ga = race("SEN-GA", "senate", "GA", exclusive=None, exhaustive=None, settlement_note="Runoff: settles on the official December result")
    v = view(ga, "D", bid=0.85, ask=0.89, ask_size=100)
    c = _night_ctx(cfg, [ga], [v])
    assert EN.settlement_caveats(ga) == ["runoff"]
    out = EN.taker_orders(c, {"SEN-GA-D": 0.98})
    assert out[0].quantity == int(100 * EN.CAVEAT_SIZE_MULTIPLIER)
    plain = race("SEN-TX", "senate", "TX")
    assert EN.settlement_caveats(plain) == [] and EN.settlement_caveats(None) == []
    assert "not exclusive/exhaustive" in EN.settlement_caveats(race("GOV-RI", "governor", "RI", exclusive=False))


def test_closed_market_and_bad_ref_ignored(cfg):
    info = race("SEN-TX", "senate", "TX")
    v = view(info, "D", bid=0.85, ask=0.89, ask_size=100, status="settled")
    c = _night_ctx(cfg, [info], [v])
    assert EN.taker_orders(c, {"SEN-TX-D": 0.99}) == []
    v.status = "open"
    assert EN.taker_orders(c, {"SEN-TX-D": "n/a", "nope": 0.99}) == []
    assert EN.taker_orders(c, {"SEN-TX-D": 1.5}) == []
