"""UTC cron for the digest workflow from the profile's local send time (spec 6.8, 13.2).
GitHub cron is UTC and best-effort; the offset used is the one in force on `now`."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from .profile import Delivery

_WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def cron_utc(delivery: Delivery, now: datetime) -> str:
    zone = ZoneInfo(delivery.timezone)
    hour, minute = (int(p) for p in delivery.send_time_local.split(":"))
    local_day = now.astimezone(zone).date()
    local = datetime.combine(local_day, time(hour, minute), tzinfo=zone)
    utc = local.astimezone(ZoneInfo("UTC"))
    if delivery.frequency == "daily":
        dow = "*"
    else:
        assert delivery.weekday is not None
        shift = (utc.date() - local.date()).days
        dow = str((_WEEKDAYS.index(delivery.weekday) + 1 + shift) % 7)  # cron: 0 = Sunday
    return f"{utc.minute} {utc.hour} * * {dow}"
