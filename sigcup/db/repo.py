"""Small repository helpers shared by processes (heartbeats, kv, latest quotes)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from sigcup.common.time import utc_now
from sigcup.db.models import KV, Heartbeat, Quote, RefPrice


def heartbeat(s: Session, process: str, status: str = "ok", detail: str | None = None) -> None:
    hb = s.get(Heartbeat, process)
    if hb is None:
        hb = Heartbeat(process=process, ts=utc_now(), status=status, detail=detail)
        s.add(hb)
    else:
        hb.ts, hb.status, hb.detail = utc_now(), status, detail


def kv_get(s: Session, key: str, default: Any = None) -> Any:
    row = s.get(KV, key)
    return row.value if row is not None else default


def kv_set(s: Session, key: str, value: Any) -> None:
    row = s.get(KV, key)
    if row is None:
        s.add(KV(key=key, value=value, updated_at=utc_now()))
    else:
        row.value, row.updated_at = value, utc_now()


def latest_quotes(s: Session) -> dict[str, Quote]:
    """Most recent quote row per exchange_id."""
    sub = select(Quote.exchange_id, func.max(Quote.id).label("mid")).group_by(Quote.exchange_id).subquery()
    rows = s.execute(select(Quote).join(sub, Quote.id == sub.c.mid)).scalars().all()
    return {q.exchange_id: q for q in rows}


def latest_ref_prices(s: Session, since: datetime | None = None) -> dict[tuple[str, str], RefPrice]:
    """Most recent reference price per (source, key)."""
    sub = select(RefPrice.source, RefPrice.key, func.max(RefPrice.id).label("mid"))
    if since is not None:
        sub = sub.where(RefPrice.ts >= since)
    sub = sub.group_by(RefPrice.source, RefPrice.key).subquery()
    rows = s.execute(select(RefPrice).join(sub, RefPrice.id == sub.c.mid)).scalars().all()
    return {(r.source, r.key): r for r in rows}
