"""Shared fixtures: an in-memory SQLite engine wired into sigcup.db.engine, and a default StrategyConfig."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from sigcup.db import engine as dbengine
from sigcup.db.models import Base
from sigcup.settings import StrategyConfig


@pytest.fixture()
def db():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool, future=True)
    Base.metadata.create_all(eng)
    dbengine.configure(eng)
    yield eng
    eng.dispose()


@pytest.fixture()
def cfg() -> StrategyConfig:
    return StrategyConfig()
