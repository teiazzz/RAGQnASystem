"""Application time helpers.

The database keeps timezone-aware timestamps, while API responses are formatted
as China-local seconds for easier reading in the UI and database clients.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from app.core.config import settings

APP_TZ = ZoneInfo(settings.APP_TIMEZONE)


def now_app_timezone() -> datetime:
    """Return current time in the configured application timezone."""
    return datetime.now(APP_TZ)


def format_datetime(value: datetime | None) -> str | None:
    """Format a datetime as ``YYYY-MM-DD HH:mm:ss`` in app timezone."""
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=APP_TZ)
    return value.astimezone(APP_TZ).strftime("%Y-%m-%d %H:%M:%S")
