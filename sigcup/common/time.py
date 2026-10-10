"""UTC time helpers and the competition deadline guard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

_server_offset = timedelta(0)


def utc_now() -> datetime:
    """Current UTC time corrected by the measured exchange clock offset."""
    return datetime.now(UTC) + _server_offset


def set_server_offset_from_date_header(date_header: str | None) -> None:
    """Record (server time - local time) from an HTTP Date header."""
    global _server_offset
    if not date_header:
        return
    try:
        server = parsedate_to_datetime(date_header)
    except (TypeError, ValueError):
        return
    if server.tzinfo is None:
        server = server.replace(tzinfo=UTC)
    _server_offset = server - datetime.now(UTC)


def server_offset() -> timedelta:
    return _server_offset


def ensure_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def parse_iso(s: str) -> datetime:
    return ensure_utc(datetime.fromisoformat(s.replace("Z", "+00:00")))


def before_deadline(deadline: datetime, now: datetime | None = None, margin: timedelta = timedelta(seconds=30)) -> bool:
    """True if an order may still be sent (strictly before deadline minus a safety margin)."""
    now = now or utc_now()
    return now < ensure_utc(deadline) - margin


def hours_to(deadline: datetime, now: datetime | None = None) -> float:
    now = now or utc_now()
    return (ensure_utc(deadline) - now).total_seconds() / 3600.0
