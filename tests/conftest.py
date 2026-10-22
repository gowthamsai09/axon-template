"""Shared helpers: fixtures from real feed output, a fake chat client, profile builders."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from digest.profile import Profile
from digest.summarizers.base import ChatResult, LLMError, Usage

ROOT = Path(__file__).resolve().parent.parent
FEED = Path(__file__).resolve().parent / "fixtures" / "feed-out"
NOW = datetime(2026, 10, 20, 6, 0, tzinfo=UTC)


def profile_data() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / "profile.yaml").read_text(encoding="utf-8"))
    return copy.deepcopy(data)


def make_profile(**sections: dict[str, Any]) -> Profile:
    data = profile_data()
    for name, over in sections.items():
        data[name].update(over)
    return Profile.model_validate(data)


@pytest.fixture
def feed_dir() -> Path:
    return FEED


@pytest.fixture
def profile() -> Profile:
    return make_profile(
        interests={"topics": ["agents-tooling"], "keywords_exclude": []},
        delivery={"timezone": "UTC", "max_items": 5},
    )


def task_field(user: str, name: str) -> str:
    m = re.search(rf"^{name}: (.*)$", user, re.MULTILINE)
    assert m is not None
    return m.group(1)


def summary_json(user: str, **over: Any) -> str:
    """A valid model answer for the task block found in `user`."""
    data: dict[str, Any] = {
        "status": "ok",
        "title": "A title",
        "one_line": "One line about the item.",
        "key_points": ["First point.", "Second point."],
        "why_it_matters": "It matters for testing.",
        "tags": ["agents-tooling"],
        "entities": ["Laminar"],
        "notable_figures": [],
        "source_url": task_field(user, "source_url"),
        "injection_flag": False,
    }
    data.update(over)
    return json.dumps(data)


class FakeChat:
    """ChatClient test double. `script` items are strings, callables(user) -> str, or errors."""

    def __init__(
        self,
        script: list[str | Callable[[str], str] | LLMError] | None = None,
        *,
        provider: str = "deepseek",
        model: str = "deepseek-v4-flash",
    ) -> None:
        self.provider = provider
        self.model = model
        self.script = list(script or [])
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> ChatResult:
        self.calls.append((system, user))
        step = self.script.pop(0) if self.script else summary_json
        if isinstance(step, LLMError):
            raise step
        text = step(user) if callable(step) else step
        return ChatResult(text, Usage(prompt_tokens=100, completion_tokens=40, calls=1))


def empty_json(user: str, status: str, **over: Any) -> str:
    """A model answer with the given non-ok status and all content fields empty."""
    return summary_json(
        user, status=status, title="", one_line="", key_points=[], why_it_matters="",
        tags=[], entities=[], **over,
    )  # fmt: skip
