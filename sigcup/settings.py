"""Configuration: secrets from .env / environment, strategy parameters from config/settings.yaml."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"


class Secrets(BaseSettings):
    """Environment-backed settings. Never log this object."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    sig_base_url: str = "https://sig.thesuper.market/api/v1"
    sig_api_key: str = ""
    sig_read_key: str = ""
    sig_tournament_slug: str = ""
    live: bool = False
    kill_file: str = "data/KILL"
    dsn: str = "sqlite:///data/sigcup.db"
    data_dir: str = "data"
    anthropic_api_key: str = ""
    x_vendor: str = "none"
    x_vendor_key: str = ""
    bluesky_handle: str = ""
    bluesky_app_password: str = ""
    youtube_api_key: str = ""
    fec_api_key: str = "DEMO_KEY"
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = ""

    @property
    def kill_path(self) -> Path:
        p = Path(self.kill_file)
        return p if p.is_absolute() else ROOT / p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else ROOT / p


class Limits(BaseModel):
    writes_per_min: int = 25
    reads_per_min: int = 90
    realtime_channels: int = 94
    write_timeout_s: float = 45
    read_timeout_s: float = 20


class Cadence(BaseModel):
    bulk_prices_s: int = 30
    reconcile_s: int = 60
    leaderboard_s: int = 900
    model_s: int = 60
    depth_refresh_s: int = 300


class Risk(BaseModel):
    kelly_fraction_default: float = 0.25
    kelly_fraction_theta: float = 0.5
    kelly_fraction_arb: float = 0.5
    per_race_cap_frac: float = 0.03
    per_state_cap_frac: float = 0.08
    national_swing_pts: float = 4.0
    max_swing_loss_frac: float = 0.25
    liquid_reserve_frac: float = 0.20
    liquid_reserve_frac_final72h: float = 0.30
    daily_loss_limit_frac: float = 0.05
    order_expiry_s: int = 900
    max_position_shares: int = 50000


class Theta(BaseModel):
    min_fair: float = 0.85
    fair_cap: float = 0.985
    min_yield: float = 0.055


class Value(BaseModel):
    band: tuple[float, float] = (0.15, 0.85)
    min_edge: float = 0.05
    quote_offset: float = 0.02
    requote_threshold: float = 0.02
    requote_cooldown_s: int = 90


class Arb(BaseModel):
    min_profit: float = 0.005
    require_api_exclusivity: bool = True
    max_legs: int = 10


class MM(BaseModel):
    enabled: bool = False
    min_spread: float = 0.06
    half_spread: float = 0.02
    size: int = 20


class ConvexSleeve(BaseModel):
    budget_frac: float = 0.15
    kelly_fraction: float = 0.5
    min_edge: float = 0.05


class Endgame(BaseModel):
    enabled: bool = True
    final_hours: int = 72
    rank_target: int = 3
    reachable_gap_frac: float = 0.15
    sleeve_frac_if_reachable: float = 0.30
    sleeve_frac_if_placed: float = 0.0


class ElectionNight(BaseModel):
    start_utc: datetime = datetime(2026, 11, 3, 23, 0, tzinfo=UTC)
    ref_cadence_s: int = 10
    take_when_ref_ge: float = 0.97
    min_lag: float = 0.08
    reserve_deploy_frac: float = 0.8


class FairValue(BaseModel):
    t_df: int = 5
    sigma_pts: dict[str, float] = Field(default_factory=lambda: {"senate": 7.0, "governor": 6.5, "house": 9.0, "control": 5.0})
    ratings_prob: dict[str, float] = Field(
        default_factory=lambda: {"safe": 0.97, "likely": 0.85, "lean": 0.68, "tilt": 0.58, "tossup": 0.50}
    )
    weights: dict[str, float] = Field(default_factory=lambda: {"ref": 0.6, "polls": 0.3, "ratings": 0.1})
    ref_min_liquidity: float = 1000
    poll_halflife_days: float = 7
    fundamentals_shrink: float = 0.3


class Simulation(BaseModel):
    n_sims: int = 10000
    national_sigma_pts: float = 2.75
    state_sigma_pts: float = 2.5
    race_sigma_pts: dict[str, float] = Field(default_factory=lambda: {"senate": 4.5, "governor": 4.5, "house": 6.0})
    seed: int = 2026


class Sentiment(BaseModel):
    baseline_days: int = 14
    spike_z: float = 2.5
    min_platforms: int = 2
    enable_fade_follow: bool = False
    max_posts_per_batch: int = 25
    classifier_model: str = "claude-haiku-5-5"
    escalation_model: str = "claude-sonnet-5-5"


class Feeds(BaseModel):
    kalshi_base: str = "https://api.elections.kalshi.com/trade-api/v2"
    polymarket_gamma: str = "https://gamma-api.polymarket.com"
    polymarket_clob: str = "https://clob.polymarket.com"
    gdelt_doc: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    ref_cadence_s: int = 120
    polls_cadence_s: int = 21600
    news_cadence_s: int = 900
    social_cadence_s: int = 900
    youtube_cadence_s: int = 3600


class StrategyConfig(BaseModel):
    bankroll_start: float = 100000
    deadline_utc: datetime = datetime(2026, 11, 4, 17, 0, tzinfo=UTC)
    limits: Limits = Field(default_factory=Limits)
    cadence: Cadence = Field(default_factory=Cadence)
    risk: Risk = Field(default_factory=Risk)
    theta: Theta = Field(default_factory=Theta)
    value: Value = Field(default_factory=Value)
    arb: Arb = Field(default_factory=Arb)
    mm: MM = Field(default_factory=MM)
    convex_sleeve: ConvexSleeve = Field(default_factory=ConvexSleeve)
    endgame: Endgame = Field(default_factory=Endgame)
    election_night: ElectionNight = Field(default_factory=ElectionNight)
    fairvalue: FairValue = Field(default_factory=FairValue)
    simulation: Simulation = Field(default_factory=Simulation)
    sentiment: Sentiment = Field(default_factory=Sentiment)
    feeds: Feeds = Field(default_factory=Feeds)


def load_strategy_config(path: Path | None = None) -> StrategyConfig:
    path = path or CONFIG_DIR / "settings.yaml"
    raw: dict[str, Any] = yaml.safe_load(path.read_text()) if path.exists() else {}
    return StrategyConfig.model_validate(raw or {})


@lru_cache(maxsize=1)
def secrets() -> Secrets:
    return Secrets()


@lru_cache(maxsize=1)
def config() -> StrategyConfig:
    return load_strategy_config()


def load_yaml(name: str) -> Any:
    """Load config/<name>.yaml (returns None if absent)."""
    p = CONFIG_DIR / f"{name}.yaml"
    return yaml.safe_load(p.read_text()) if p.exists() else None
