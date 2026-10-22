"""The state branch on disk (spec 6.5): seen.json, summaries/<cachekey>.json, tasks/<id>.json,
last_run.json. Writes are atomic; corrupt files are ignored with a warning, never fatal."""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from .summarizers.base import Summary
from .task_result import UUID_PATTERN

SEEN_DAYS, SUMMARY_DAYS, TASK_DAYS = 30, 30, 7
CACHE_KEY = re.compile(r"^[a-f0-9]{64}$")


def cache_key(
    content_hash: str, provider: str, model: str, depth: str, language: str, prompt_version: str
) -> str:
    """Spec 6.4 key plus `language` (Appendix C 29): the language changes the output."""
    raw = "|".join([content_hash, provider, model, depth, language, prompt_version])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def content_hash(url: str, text: str) -> str:
    return hashlib.sha256(f"{url}\n{text}".encode()).hexdigest()


class CachedSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cached_at: AwareDatetime
    provider: str
    model: str
    summary: Summary


class StateStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.warnings: list[str] = []

    # -- helpers --------------------------------------------------------------------
    def _write(self, rel: str, data: Any) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def _read(self, rel: str) -> Any | None:
        path = self.root / rel
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            self.warnings.append(f"state/{rel}: unreadable, ignored")
            return None

    # -- seen -----------------------------------------------------------------------
    def seen(self) -> dict[str, datetime]:
        raw = self._read("seen.json")
        out: dict[str, datetime] = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                try:
                    out[str(key)] = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                except ValueError:
                    continue
        return out

    def mark_seen(self, ids: list[str], now: datetime) -> None:
        seen = self.seen()
        for item_id in ids:
            seen[item_id] = now
        cutoff = now - timedelta(days=SEEN_DAYS)
        kept = {k: v.isoformat() for k, v in seen.items() if v >= cutoff}
        self._write("seen.json", kept)

    # -- summary cache --------------------------------------------------------------
    def get_summary(self, key: str) -> CachedSummary | None:
        if not CACHE_KEY.match(key):
            return None
        raw = self._read(f"summaries/{key}.json")
        if raw is None:
            return None
        try:
            return CachedSummary.model_validate(raw)
        except ValidationError:
            self.warnings.append(f"state/summaries/{key}.json: invalid, ignored")
            return None

    def put_summary(
        self, key: str, summary: Summary, provider: str, model: str, now: datetime
    ) -> None:
        if not CACHE_KEY.match(key):
            raise ValueError("bad cache key")
        entry = CachedSummary(cached_at=now, provider=provider, model=model, summary=summary)
        self._write(f"summaries/{key}.json", json.loads(entry.model_dump_json()))

    # -- tasks (Tier 3) ---------------------------------------------------------------
    def task_path(self, task_id: str) -> str:
        if not re.fullmatch(UUID_PATTERN, task_id):
            raise ValueError("task id must be a lowercase UUID")
        return f"tasks/{task_id}.json"

    def write_task(self, task_id: str, payload: dict[str, Any]) -> None:
        self._write(self.task_path(task_id), payload)

    def tasks_on(self, day: datetime) -> int:
        folder = self.root / "tasks"
        count = 0
        for path in folder.glob("*.json") if folder.is_dir() else []:
            try:
                finished = json.loads(path.read_text(encoding="utf-8"))["finished_at"]
                when = datetime.fromisoformat(str(finished).replace("Z", "+00:00"))
            except (ValueError, KeyError, OSError, TypeError):
                continue
            if when.astimezone(UTC).date() == day.astimezone(UTC).date():
                count += 1
        return count

    # -- last run + pruning --------------------------------------------------------------
    def write_last_run(self, payload: dict[str, Any]) -> None:
        self._write("last_run.json", payload)

    def prune(self, now: datetime) -> int:
        removed = 0
        for sub, days in (("summaries", SUMMARY_DAYS), ("tasks", TASK_DAYS)):
            folder = self.root / sub
            for path in folder.glob("*.json") if folder.is_dir() else []:
                stamp = "cached_at" if sub == "summaries" else "finished_at"
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))[stamp]
                    when = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                except (ValueError, KeyError, OSError, TypeError):
                    when = datetime.fromtimestamp(path.stat().st_mtime, UTC)
                if when < now - timedelta(days=days):
                    path.unlink()
                    removed += 1
        return removed
