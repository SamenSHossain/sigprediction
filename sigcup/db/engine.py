"""Engine/session factory. SQLite WAL with busy timeout by default; any SQLAlchemy DSN works."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from sigcup.db.models import Base
from sigcup.settings import ROOT, secrets

_engine: Engine | None = None
_Session: sessionmaker[Session] | None = None


def _resolve_sqlite_path(dsn: str) -> str:
    if dsn.startswith("sqlite:///") and not dsn.startswith("sqlite:////"):
        rel = dsn[len("sqlite:///"):]
        p = Path(rel)
        if not p.is_absolute():
            p = ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{p}"
    return dsn


def make_engine(dsn: str | None = None) -> Engine:
    dsn = _resolve_sqlite_path(dsn or secrets().dsn)
    kwargs: dict = {"future": True}
    if dsn.startswith("sqlite"):
        kwargs["connect_args"] = {"timeout": 30, "check_same_thread": False}
    eng = create_engine(dsn, **kwargs)
    if dsn.startswith("sqlite"):

        @event.listens_for(eng, "connect")
        def _pragmas(dbapi_conn, _):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.execute("PRAGMA foreign_keys=ON")
            cur.close()

    return eng


def get_engine() -> Engine:
    global _engine, _Session
    if _engine is None:
        _engine = make_engine()
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def init_db(engine: Engine | None = None) -> None:
    engine = engine or get_engine()
    Base.metadata.create_all(engine)


def configure(engine: Engine) -> None:
    """Point the module-level session factory at a specific engine (tests use in-memory SQLite)."""
    global _engine, _Session
    _engine = engine
    _Session = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session() -> Iterator[Session]:
    get_engine()
    assert _Session is not None
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def ping(engine: Engine | None = None) -> bool:
    engine = engine or get_engine()
    with engine.connect() as c:
        return c.execute(text("select 1")).scalar() == 1
