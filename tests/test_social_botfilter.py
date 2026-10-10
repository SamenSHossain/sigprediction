"""Bot filter: bounds and monotonicity in every risk factor."""

from __future__ import annotations

from datetime import timedelta

from sigcup.feeds.nlp import botfilter
from tests.test_social_helpers import NOW


def post(**kw):
    base = {"platform": "x", "author_handle": "normal_person", "author_created_at": NOW - timedelta(days=400), "text": "a plain opinion about the race"}
    base.update(kw)
    return base


def test_bounds_and_clean_post():
    assert 0.0 <= botfilter.score(post(), now=NOW) <= 0.2
    worst = post(author_handle="maga48291734", author_created_at=NOW - timedelta(days=1),
                 text="#a #b #c #d #e #f @x @y @z @w @v https://t.co/x")
    assert botfilter.score(worst, dup_count=100, author_posts_24h=1000, now=NOW) == 1.0


def test_monotone_in_duplicates():
    scores = [botfilter.score(post(), dup_count=d, now=NOW) for d in (0, 1, 3, 10, 30, 100)]
    assert scores == sorted(scores)
    assert scores[-1] > scores[0]


def test_monotone_in_account_age():
    ages = (1, 10, 60, 120, 400)
    scores = [botfilter.score(post(author_created_at=NOW - timedelta(days=a)), now=NOW) for a in ages]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1]
    unknown = botfilter.score(post(author_created_at=None), now=NOW)
    assert scores[-1] <= unknown <= scores[0]


def test_monotone_in_posting_rate():
    scores = [botfilter.score(post(), author_posts_24h=r, now=NOW) for r in (0, 10, 50, 100, 500)]
    assert scores == sorted(scores)


def test_handle_patterns():
    assert botfilter.handle_component("jon_smith") == 0.0
    assert botfilter.handle_component("patriot_mom1776") > 0.0
    assert botfilter.handle_component("jd93847561") == 0.25
    assert botfilter.handle_component("maga_patriot48291734") == 0.25
    assert botfilter.handle_component("@maga_patriot48291734") == botfilter.handle_component("maga_patriot48291734")
    assert botfilter.handle_component(None) == 0.0


def test_spammy_text_raises_score():
    plain = botfilter.score(post(), now=NOW)
    tags = botfilter.score(post(text="#a #b #c #d #e vote now"), now=NOW)
    link_only = botfilter.score(post(text="https://t.co/abc"), now=NOW)
    shouting = botfilter.score(post(text="WAKE UP AMERICA THIS IS THE MOST IMPORTANT ELECTION EVER"), now=NOW)
    assert tags > plain and link_only > plain and shouting > plain


def test_news_platforms_are_never_bots():
    assert botfilter.score({"platform": "rss", "author_handle": "x1234567"}, dup_count=50, now=NOW) == 0.0
    assert botfilter.score({"platform": "gdelt"}, dup_count=50, now=NOW) == 0.0


def test_accepts_objects_and_rounding():
    class P:
        platform = "bluesky"
        author_handle = "someone.bsky.social"
        author_created_at = NOW - timedelta(days=3)
        text = "hello"

    s = botfilter.score(P(), now=NOW)
    assert s == round(s, 4) and s > 0.3
