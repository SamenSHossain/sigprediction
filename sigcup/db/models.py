"""SQLAlchemy 2.0 models. All timestamps are timezone-aware UTC. Prices are floats in [0, 1]."""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Strategy(enum.StrEnum):
    ARB = "arb"
    THETA = "theta"
    VALUE = "value"
    MM = "mm"
    FADE = "fade"
    FOLLOW = "follow"
    ELECTION = "election"
    CONVEX = "convex"
    MANUAL = "manual"


class IntentState(enum.StrEnum):
    NEW = "NEW"              # persisted, not yet sent
    SENT = "SENT"            # request in flight
    UNCERTAIN = "UNCERTAIN"  # timeout / 502 ORDER_STATUS_UNKNOWN: must reconcile before anything else
    ACKED = "ACKED"          # exchange accepted; resting or partially filled
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class Race(Base):
    __tablename__ = "races"
    race_id: Mapped[str] = mapped_column(String(32), primary_key=True)  # e.g. SEN-TX, GOV-GA, HOUSE-PA-08, CONTROL-SENATE
    chamber: Mapped[str] = mapped_column(String(16), index=True)  # senate | governor | house | control
    state: Mapped[str | None] = mapped_column(String(2), index=True)
    district: Mapped[str | None] = mapped_column(String(8))
    label: Mapped[str] = mapped_column(String(128))
    incumbent_party: Mapped[str | None] = mapped_column(String(8))
    pvi: Mapped[float | None] = mapped_column(Float)  # Cook PVI, positive = Republican lean (pts)
    exclusive: Mapped[bool | None] = mapped_column(Boolean)  # from relationships API, never from titles
    exhaustive: Mapped[bool | None] = mapped_column(Boolean)
    settlement_note: Mapped[str | None] = mapped_column(Text)  # runoff / fusion / independent caveats
    candidates: Mapped[dict | None] = mapped_column(JSON)  # {party: {name, aliases[], handles[], hashtags[]}}


class Market(Base):
    __tablename__ = "markets"
    market_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    exchange_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    race_id: Mapped[str] = mapped_column(ForeignKey("races.race_id"), index=True)
    party: Mapped[str] = mapped_column(String(8))  # D | R | I | other
    title: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32), default="open")
    settlement_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    resolution_text: Mapped[str | None] = mapped_column(Text)
    kalshi_ticker: Mapped[str | None] = mapped_column(String(64))
    polymarket_condition_id: Mapped[str | None] = mapped_column(String(80))
    polymarket_token_id: Mapped[str | None] = mapped_column(String(80))
    predictit_id: Mapped[str | None] = mapped_column(String(32))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Quote(Base):
    """Top-of-book snapshot, stored on change."""

    __tablename__ = "quotes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    exchange_id: Mapped[str] = mapped_column(String(64), index=True)
    bid: Mapped[float | None] = mapped_column(Float)
    ask: Mapped[float | None] = mapped_column(Float)
    last: Mapped[float | None] = mapped_column(Float)
    bid_size: Mapped[int | None] = mapped_column(Integer)
    ask_size: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16), default="rest")  # rest | realtime | paper (DB-quote paper echo) | seed
    __table_args__ = (Index("ix_quotes_exchange_ts", "exchange_id", "ts"),)


class RefPrice(Base):
    __tablename__ = "ref_prices"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    source: Mapped[str] = mapped_column(String(16), index=True)  # kalshi | polymarket | predictit
    key: Mapped[str] = mapped_column(String(96), index=True)  # ticker / condition id
    market_id: Mapped[str | None] = mapped_column(String(64), index=True)
    yes_bid: Mapped[float | None] = mapped_column(Float)
    yes_ask: Mapped[float | None] = mapped_column(Float)
    yes_mid: Mapped[float | None] = mapped_column(Float)
    liquidity: Mapped[float | None] = mapped_column(Float)  # $ volume or open interest, source-specific
    __table_args__ = (Index("ix_ref_source_key_ts", "source", "key", "ts"),)


class Poll(Base):
    __tablename__ = "polls"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(ForeignKey("races.race_id"), index=True)
    pollster: Mapped[str] = mapped_column(String(128))
    start_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    end_date: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sample_size: Mapped[int | None] = mapped_column(Integer)
    population: Mapped[str | None] = mapped_column(String(8))  # LV | RV | A
    dem: Mapped[float | None] = mapped_column(Float)
    rep: Mapped[float | None] = mapped_column(Float)
    margin: Mapped[float] = mapped_column(Float)  # dem - rep, points
    grade: Mapped[str | None] = mapped_column(String(8))
    source_url: Mapped[str | None] = mapped_column(String(512))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("race_id", "pollster", "end_date", "margin", name="uq_poll"),)


class Rating(Base):
    __tablename__ = "ratings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    race_id: Mapped[str] = mapped_column(ForeignKey("races.race_id"), index=True)
    source: Mapped[str] = mapped_column(String(32))  # cook | sabato | inside | split | rttwh | manual
    rating: Mapped[str] = mapped_column(String(32))  # e.g. "Lean D", "Toss-up", "Safe R"
    prob_dem: Mapped[float] = mapped_column(Float)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("race_id", "source", "rating", "ts", name="uq_rating"),)


class GenericBallot(Base):
    __tablename__ = "generic_ballot"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    margin: Mapped[float] = mapped_column(Float)  # dem - rep


class Post(Base):
    __tablename__ = "posts"
    post_id: Mapped[str] = mapped_column(String(128), primary_key=True)  # "<platform>:<native id>"
    platform: Mapped[str] = mapped_column(String(16), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    author_id: Mapped[str | None] = mapped_column(String(128))
    author_handle: Mapped[str | None] = mapped_column(String(128))
    author_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    text: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(32), index=True)
    url: Mapped[str | None] = mapped_column(String(512))
    race_ids: Mapped[list | None] = mapped_column(JSON)
    bot_score: Mapped[float | None] = mapped_column(Float)
    classified: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Stance(Base):
    __tablename__ = "stances"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    post_id: Mapped[str] = mapped_column(ForeignKey("posts.post_id"), index=True)
    race_id: Mapped[str] = mapped_column(String(32), index=True)
    party: Mapped[str] = mapped_column(String(8))  # party of the candidate the stance is about
    stance: Mapped[str] = mapped_column(String(8))  # support | oppose | neutral | unclear
    confidence: Mapped[float] = mapped_column(Float)
    is_news_or_poll: Mapped[bool] = mapped_column(Boolean, default=False)
    model: Mapped[str] = mapped_column(String(48))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Feature(Base):
    """Daily per-(platform, race) stance features."""

    __tablename__ = "features"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    platform: Mapped[str] = mapped_column(String(16))
    race_id: Mapped[str] = mapped_column(String(32), index=True)
    day: Mapped[str] = mapped_column(String(10))  # YYYY-MM-DD (UTC)
    n_posts: Mapped[int] = mapped_column(Integer)
    net_stance_dem: Mapped[float] = mapped_column(Float)  # (support_D + oppose_R - oppose_D - support_R) / n
    z_score: Mapped[float | None] = mapped_column(Float)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    __table_args__ = (UniqueConstraint("platform", "race_id", "day", name="uq_feature"),)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    race_id: Mapped[str] = mapped_column(String(32), index=True)
    kind: Mapped[str] = mapped_column(String(8))  # soft | hard | unknown
    description: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[float] = mapped_column(Float)
    acted: Mapped[bool] = mapped_column(Boolean, default=False)


class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    exchange_id: Mapped[str] = mapped_column(String(64), index=True)
    strategy: Mapped[str] = mapped_column(String(16))
    fair: Mapped[float | None] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float)
    edge: Mapped[float | None] = mapped_column(Float)
    size: Mapped[int | None] = mapped_column(Integer)
    meta: Mapped[dict | None] = mapped_column(JSON)


class OrderIntent(Base):
    """Our record of an order, persisted BEFORE the request is sent (idempotency state machine)."""

    __tablename__ = "order_intents"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # == idempotency key
    exchange_id: Mapped[str] = mapped_column(String(64), index=True)
    side: Mapped[str] = mapped_column(String(4))  # yes | no
    action: Mapped[str] = mapped_column(String(4))  # buy | sell
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[int] = mapped_column(Integer)
    strategy: Mapped[str] = mapped_column(String(16), index=True)
    state: Mapped[str] = mapped_column(String(12), index=True, default=IntentState.NEW.value)
    sig_order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    group_id: Mapped[str | None] = mapped_column(String(32), index=True)  # multi-leg group
    signal_id: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    filled_qty: Mapped[int] = mapped_column(Integer, default=0)
    avg_fill: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(Text)
    paper: Mapped[bool] = mapped_column(Boolean, default=False)


class Fill(Base):
    __tablename__ = "fills"
    fill_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sig_order_id: Mapped[str | None] = mapped_column(String(64), index=True)
    intent_id: Mapped[str | None] = mapped_column(String(32), index=True)
    exchange_id: Mapped[str] = mapped_column(String(64), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    side: Mapped[str] = mapped_column(String(4))
    action: Mapped[str] = mapped_column(String(4))
    price: Mapped[float] = mapped_column(Float)
    quantity: Mapped[int] = mapped_column(Integer)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    strategy: Mapped[str | None] = mapped_column(String(16), index=True)
    paper: Mapped[bool] = mapped_column(Boolean, default=False)


class Position(Base):
    __tablename__ = "positions"
    exchange_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    quantity: Mapped[int] = mapped_column(Integer)  # signed: +YES shares, -NO shares
    avg_price: Mapped[float | None] = mapped_column(Float)
    mark: Mapped[float | None] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CLV(Base):
    __tablename__ = "clv"
    fill_id: Mapped[str] = mapped_column(ForeignKey("fills.fill_id"), primary_key=True)
    price_24h: Mapped[float | None] = mapped_column(Float)
    price_close: Mapped[float | None] = mapped_column(Float)
    clv_24h: Mapped[float | None] = mapped_column(Float)  # signed in our favour, in price units
    clv_close: Mapped[float | None] = mapped_column(Float)
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class LeaderboardSnap(Base):
    __tablename__ = "leaderboard"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    rank: Mapped[int | None] = mapped_column(Integer)
    our_total: Mapped[float | None] = mapped_column(Float)
    top_totals: Mapped[list | None] = mapped_column(JSON)
    member_count: Mapped[int | None] = mapped_column(Integer)


class AccountSnap(Base):
    __tablename__ = "account"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    cash: Mapped[float] = mapped_column(Float)
    positions_value: Mapped[float] = mapped_column(Float)
    total: Mapped[float] = mapped_column(Float)
    reserved: Mapped[float] = mapped_column(Float, default=0.0)


class Heartbeat(Base):
    __tablename__ = "heartbeats"
    process: Mapped[str] = mapped_column(String(32), primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), default="ok")
    detail: Mapped[str | None] = mapped_column(Text)


class KV(Base):
    """Small key-value store for process state (cursors, last-seen ids, flags)."""

    __tablename__ = "kv"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict | None] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
