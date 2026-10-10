"""A4: fade only an over-reaction. After a soft event the move from the pre-event price is converted to the
margin shift it implies (probmath.implied_margin_shift through the T5 map); the fade fires only when that
shift exceeds what the event class plausibly justifies (soft event 1.5 pts, a single poll 2.0 pts). follow is
unchanged, and fade / follow stay disabled by default."""

from __future__ import annotations

from datetime import timedelta

import pytest

from sigcup.db import engine as dbengine
from sigcup.db.models import Quote, Strategy
from sigcup.settings import StrategyConfig, load_strategy_config
from sigcup.trader import signals as SG
from sigcup.trader.probmath import implied_margin_shift
from tests.test_model_support import NOW, ctx, event, race, view

PRE_TS = NOW - timedelta(hours=2)


def _setup(mid: float, kind: str = "soft", description: str = "debate reaction"):
    info = race("SEN-TX", "senate", "TX")
    with dbengine.session() as s:
        s.add(Quote(ts=PRE_TS - timedelta(minutes=1), exchange_id="SEN-TX-D", bid=0.49, ask=0.51, last=0.50, source="rest"))
    views = [view(info, "D", bid=mid - 0.01, ask=mid + 0.01), view(info, "R", bid=0.99 - mid - 0.01, ask=0.99 - mid + 0.01)]
    return info, views, event("SEN-TX", kind, PRE_TS, description)


def _event_orders(cfg, mid: float, kind: str = "soft", description: str = "debate reaction"):
    info, views, e = _setup(mid, kind, description)
    cfg.sentiment.enable_fade_follow = True
    c = ctx(cfg, [info], views, [], events=[e])
    return [d for d in SG.generate(c, []) if d.strategy in (Strategy.FADE, Strategy.FOLLOW)]


def test_shift_constants_and_the_t5_map(cfg):
    assert SG.SOFT_EVENT_MAX_SHIFT_PTS == pytest.approx(1.5)
    assert SG.POLL_EVENT_MAX_SHIFT_PTS == pytest.approx(2.0)
    assert implied_margin_shift(0.50, 0.62, "senate", cfg) > 1.5
    assert abs(implied_margin_shift(0.50, 0.54, "senate", cfg)) < 1.5


def test_twelve_cent_move_after_a_soft_event_is_faded(db, cfg):
    out = _event_orders(cfg, 0.62)
    assert len(out) == 1 and out[0].strategy == Strategy.FADE and out[0].action == "sell"


def test_twelve_cent_drop_after_a_soft_event_is_faded_from_below(db, cfg):
    out = _event_orders(cfg, 0.38)
    assert len(out) == 1 and out[0].strategy == Strategy.FADE and out[0].action == "buy"


def test_four_cent_move_after_a_soft_event_is_justified_and_not_faded(db, cfg):
    # 4c on a 50/50 Senate race implies ~0.74 pts: well inside what a soft event can move
    assert _event_orders(cfg, 0.54) == []
    assert _event_orders(cfg, 0.46) == []


def test_poll_events_are_allowed_a_larger_shift(db, cfg):
    # 10c implies ~1.87 pts: an over-reaction to a debate, a plausible reaction to a single poll
    assert implied_margin_shift(0.50, 0.60, "senate", cfg) == pytest.approx(1.87, abs=0.01)
    assert _event_orders(cfg, 0.60, kind="poll", description="new Emerson poll") == []
    # 15c implies ~2.9 pts: more than one poll justifies -> fade
    out = _event_orders(cfg, 0.65, kind="poll", description="new Emerson poll")
    assert len(out) == 1 and out[0].strategy == Strategy.FADE and out[0].action == "sell"


def test_follow_is_unchanged_and_fade_follow_ship_disabled(db, cfg):
    info, views, soft = _setup(0.60)
    assert load_strategy_config().sentiment.enable_fade_follow is False
    assert StrategyConfig().sentiment.enable_fade_follow is False
    assert SG.generate(ctx(StrategyConfig(), [info], views, [], events=[soft]), []) == []
    # hard event: follow toward pre + move / pass-through, whatever the implied shift
    cfg.sentiment.enable_fade_follow = True
    hard = event("SEN-TX", "hard", PRE_TS, "indictment")
    out = [d for d in SG.generate(ctx(cfg, [info], views, [], events=[hard]), []) if d.strategy == Strategy.FOLLOW]
    assert len(out) == 1 and out[0].action == "buy"
    assert out[0].fair == pytest.approx(0.50 + 0.10 / SG.HARD_PASS_THROUGH)
