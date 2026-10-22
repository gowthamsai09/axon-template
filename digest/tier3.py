"""Tier 3 request handling (spec 9.3): untrusted params, owner policy, result building.

Params arrive as one JSON string from the gateway and are never trusted: unknown keys are
ignored (with a warning), max_items is clamped, delivery is an enum. The owner's profile
decides what is allowed; a caller can never change secrets, destination or profile."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .profile import Profile
from .state import StateStore
from .task_result import UUID_PATTERN, ModelInfo, TaskItem, TaskItemSummary, TaskResult

MAX_PARAMS_BYTES = 4096
MAX_ITEMS_CAP = 20
DELIVERIES = ("return_only", "owner_channel")


class TaskError(Exception):
    """Invalid task id or params (exit code 2)."""


@dataclass
class TaskRequest:
    task_id: str
    max_items: int | None = None
    delivery: str = "return_only"
    warnings: list[str] = field(default_factory=list)


def parse_task(task_id: str, params_json: str) -> TaskRequest:
    if not re.fullmatch(UUID_PATTERN, task_id):
        raise TaskError("TASK_ID is not a lowercase UUID")
    if len(params_json.encode("utf-8")) > MAX_PARAMS_BYTES:
        raise TaskError("TASK_PARAMS is too large")
    try:
        params: Any = json.loads(params_json or "{}")
    except ValueError:
        raise TaskError("TASK_PARAMS is not valid JSON") from None
    if not isinstance(params, dict):
        raise TaskError("TASK_PARAMS must be a JSON object")
    req = TaskRequest(task_id)
    unknown = sorted(str(k)[:32] for k in params if k not in {"max_items", "delivery"})
    if unknown:
        req.warnings.append(
            "ignored unknown params: " + ", ".join(re.sub(r"\W", "_", u) for u in unknown[:10])
        )
    value = params.get("max_items")
    if isinstance(value, int) and not isinstance(value, bool):
        req.max_items = max(1, min(MAX_ITEMS_CAP, value))
    elif value is not None:
        req.warnings.append("ignored max_items (not an integer)")
    delivery = params.get("delivery", "return_only")
    if delivery in DELIVERIES:
        req.delivery = str(delivery)
    else:
        req.warnings.append("delivery not recognised; using return_only")
    return req


def check_policy(profile: Profile, state: StateStore, now: datetime) -> str | None:
    """None if the owner allows this run, otherwise an `owner_policy: ...` reason."""
    access = profile.agent_access
    if not access.allow_inbound_agents:
        return "owner_policy: inbound agents are disabled"
    if "digest" not in access.allowed_modes:
        return "owner_policy: digest mode is not allowed"
    used = state.tasks_on(now)
    if used >= access.max_tier3_runs_per_day:
        return f"owner_policy: daily limit reached ({access.max_tier3_runs_per_day})"
    return None


def build_result(
    task_id: str,
    profile: Profile,
    entries: Sequence[tuple[str, str, str, TaskItemSummary | None, str | None]],
    *,
    status: str,
    now: datetime,
    warnings: list[str],
    model: tuple[str, str] | None = None,
) -> TaskResult:
    """entries: (id, title, url, summary, fallback_snippet)."""
    provider, name = model or (profile.summarizer.provider, profile.summarizer.model)
    items = [
        TaskItem(
            id=i, title=title[:500] or "(untitled)", url=url, summary=summary,
            fallback_snippet=fallback[:300] if fallback is not None else None,
        )
        for i, title, url, summary, fallback in entries[:25]
    ]  # fmt: skip
    return TaskResult.model_validate(
        {
            "task_id": task_id,
            "status": status,
            "finished_at": now,
            "items": [it.model_dump() for it in items],
            "model": ModelInfo.model_validate({"provider": provider, "model": name}).model_dump(),
            "warnings": [w[:500] for w in warnings[:100]],
        }
    )
