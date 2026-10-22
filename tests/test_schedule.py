from __future__ import annotations

import pytest

from digest.schedule import cron_utc
from tests.conftest import NOW, make_profile


def cron(**over: str | int) -> str:
    return cron_utc(make_profile(delivery=dict(over)).delivery, NOW)


def test_shipped_profile_matches_the_digest_workflow_cron() -> None:
    assert cron() == "0 2 * * *"  # Asia/Kolkata 07:30 -> 02:00 UTC, the line in digest.yml


@pytest.mark.parametrize(
    ("tz", "time", "expected"),
    [
        ("UTC", "07:30", "30 7 * * *"),
        ("America/New_York", "07:30", "30 11 * * *"),  # EDT on 2026-10-20
        ("Asia/Tokyo", "08:00", "0 23 * * *"),
        ("Pacific/Auckland", "07:00", "0 18 * * *"),
    ],
)
def test_daily(tz: str, time: str, expected: str) -> None:
    assert cron(timezone=tz, send_time_local=time) == expected


def test_weekly_weekday_shifts_across_midnight() -> None:
    assert (
        cron(frequency="weekly", weekday="mon", timezone="UTC", send_time_local="07:00")
        == "0 7 * * 1"
    )
    # Monday 07:30 in Kolkata is Monday 02:00 UTC; Monday 01:00 in Tokyo is Sunday 16:00 UTC
    assert cron(frequency="weekly", weekday="mon") == "0 2 * * 1"
    assert (
        cron(frequency="weekly", weekday="mon", timezone="Asia/Tokyo", send_time_local="01:00")
        == "0 16 * * 0"
    )
    assert (
        cron(
            frequency="weekly", weekday="sun", timezone="Pacific/Auckland", send_time_local="05:00"
        )
        == "0 16 * * 6"
    )
