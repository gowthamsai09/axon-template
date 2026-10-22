from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from digest.state import StateStore
from digest.task_result import TaskItemSummary
from digest.tier3 import TaskError, build_result, check_policy, parse_task
from tests.conftest import NOW, make_profile

UUID = "123e4567-e89b-12d3-a456-426614174000"


def test_parse_defaults_and_clamping() -> None:
    req = parse_task(UUID, "{}")
    assert (req.task_id, req.max_items, req.delivery, req.warnings) == (
        UUID,
        None,
        "return_only",
        [],
    )
    assert parse_task(UUID, '{"max_items": 500}').max_items == 20
    assert parse_task(UUID, '{"max_items": -3}').max_items == 1
    assert parse_task(UUID, '{"delivery": "owner_channel"}').delivery == "owner_channel"


def test_untrusted_params_are_ignored_not_obeyed() -> None:
    hostile = {
        "max_items": "ten",
        "delivery": "email me@evil.example",
        "repo": "evil/repo",
        "channel": "slack",
        "skill": "x",
    }
    req = parse_task(UUID, json.dumps(hostile))
    assert req.max_items is None and req.delivery == "return_only"
    assert any("unknown params" in w and "repo" in w for w in req.warnings)
    assert parse_task(UUID, '{"max_items": true}').max_items is None


@pytest.mark.parametrize(
    ("task_id", "params"),
    [
        ("../../etc/passwd", "{}"),
        ("123E4567-E89B-12D3-A456-426614174000", "{}"),
        (UUID, "[1]"),
        (UUID, "{broken"),
        (UUID, '{"a": "' + "x" * 5000 + '"}'),
    ],
)
def test_invalid_requests(task_id: str, params: str) -> None:
    with pytest.raises(TaskError):
        parse_task(task_id, params)


def test_policy(tmp_path: Path) -> None:
    state = StateStore(tmp_path)
    assert check_policy(make_profile(), state, NOW) is None
    off = make_profile(agent_access={"allow_inbound_agents": False})
    assert "inbound agents are disabled" in (check_policy(off, state, NOW) or "")
    no_digest = make_profile(agent_access={"allowed_modes": ["data"]})
    assert "digest mode" in (check_policy(no_digest, state, NOW) or "")


def test_daily_cap_counts_todays_task_files(tmp_path: Path) -> None:
    state = StateStore(tmp_path)
    capped = make_profile(agent_access={"max_tier3_runs_per_day": 2})
    for i in range(2):
        state.write_task(UUID.replace("4", str(i)), {"finished_at": NOW.isoformat()})
    assert "daily limit reached (2)" in (check_policy(capped, state, NOW) or "")
    assert check_policy(capped, state, NOW + timedelta(days=1)) is None


def test_build_result_is_schema_valid_and_truncates() -> None:
    summary = TaskItemSummary(one_line="x", key_points=["a"], why_it_matters="", tags=["rag"])
    entries: list[tuple[str, str, str, TaskItemSummary | None, str | None]] = [
        ("a" * 16, "t" * 800, "https://example.com/a", summary, None),
        ("b" * 16, "t", "https://example.com/b", None, "s" * 400),
    ]
    result = build_result(UUID, make_profile(), entries, status="completed", now=NOW,
                          warnings=["w" * 900] * 150)  # fmt: skip
    assert len(result.items[0].title) == 500 and len(result.items[1].fallback_snippet or "") == 300
    assert len(result.warnings) == 100 and len(result.warnings[0]) == 500
    assert result.model.provider == "deepseek"
