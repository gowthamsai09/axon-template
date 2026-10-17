from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from digest.task_result import TaskResult


def result(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "completed",
        "finished_at": "2026-10-15T07:31:12Z",
        "items": [
            {
                "id": "9c1e4a7b2d3f5a60",
                "title": "t",
                "url": "https://example.com/post",
                "summary": {
                    "one_line": "x",
                    "key_points": ["a"],
                    "why_it_matters": "y",
                    "tags": ["rag"],
                },
            }
        ],
        "model": {"provider": "deepseek", "model": "deepseek-chat"},
        "warnings": [],
    }
    base.update(over)
    return base


def test_valid_and_fallback_item() -> None:
    TaskResult.model_validate(result())
    fb = {
        "id": "9c1e4a7b2d3f5a60",
        "title": "t",
        "url": "https://example.com/p",
        "fallback_snippet": "s",
    }
    TaskResult.model_validate(result(items=[fb]))


@pytest.mark.parametrize(
    "over",
    [
        {"task_id": "not-a-uuid"},
        {"status": "queued"},
        {"finished_at": "2026-10-15T07:31:12"},
        {"items": [{}] * 26},
        {"model": {"provider": "openai", "model": "x"}},
        {"warnings": ["w"] * 101},
        {"extra": 1},
        {"items": [{"id": "9c1e4a7b2d3f5a60", "title": "t", "url": "https://example.com/p"}]},
    ],
)
def test_invalid(over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TaskResult.model_validate(result(**over))


def test_both_summary_and_fallback_rejected() -> None:
    item = result()["items"][0] | {"fallback_snippet": "s"}
    with pytest.raises(ValidationError):
        TaskResult.model_validate(result(items=[item]))
