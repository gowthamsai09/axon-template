from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from digest.state import StateStore, cache_key, content_hash
from digest.summarizers.base import Summary
from tests.conftest import NOW

SUMMARY = Summary(
    status="ok", title="t", one_line="x", key_points=[], why_it_matters="", tags=[], entities=[],
    notable_figures=[], source_url="https://example.com/a", injection_flag=False,
)  # fmt: skip
UUID = "123e4567-e89b-12d3-a456-426614174000"


def test_cache_key_depends_on_every_part() -> None:
    base = ("h", "deepseek", "m", "brief", "en", "p-1")
    keys = {cache_key(*base)}
    for i, other in enumerate(["h2", "x", "m2", "detailed", "de", "p-2"]):
        parts = list(base)
        parts[i] = other
        keys.add(cache_key(*parts))
    assert len(keys) == 7
    assert content_hash("u", "a") != content_hash("u2", "a")


def test_seen_roundtrip_and_pruning(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    store.mark_seen(["a" * 16], NOW - timedelta(days=40))
    store.mark_seen(["b" * 16], NOW)  # prunes the 40-day-old entry
    assert set(store.seen()) == {"b" * 16}


def test_corrupt_files_are_ignored_with_a_warning(tmp_path: Path) -> None:
    (tmp_path / "seen.json").write_text("{broken")
    store = StateStore(tmp_path)
    assert store.seen() == {}
    assert any("seen.json" in w for w in store.warnings)


def test_summary_cache_roundtrip_and_invalid_entry(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    key = cache_key("h", "deepseek", "m", "brief", "en", "p-1")
    assert store.get_summary(key) is None
    store.put_summary(key, SUMMARY, "deepseek", "m", NOW)
    hit = store.get_summary(key)
    assert hit is not None and hit.summary == SUMMARY and hit.provider == "deepseek"
    (tmp_path / "summaries" / f"{key}.json").write_text(json.dumps({"nope": 1}))
    assert store.get_summary(key) is None
    assert store.get_summary("../../etc/passwd") is None
    with pytest.raises(ValueError):
        store.put_summary("bad", SUMMARY, "p", "m", NOW)


def test_task_files_count_per_day_and_reject_bad_ids(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    store.write_task(UUID, {"finished_at": NOW.isoformat()})
    store.write_task(UUID.replace("4", "5"), {"finished_at": (NOW - timedelta(days=1)).isoformat()})
    assert store.tasks_on(NOW) == 1
    with pytest.raises(ValueError):
        store.write_task("../evil", {})


def test_prune_by_age(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    old = NOW - timedelta(days=45)
    store.put_summary(cache_key("old", "d", "m", "b", "en", "p"), SUMMARY, "d", "m", old)
    store.put_summary(cache_key("new", "d", "m", "b", "en", "p"), SUMMARY, "d", "m", NOW)
    store.write_task(UUID, {"finished_at": (NOW - timedelta(days=8)).isoformat()})
    assert store.prune(NOW) == 2
    assert len(list((tmp_path / "summaries").glob("*.json"))) == 1


def test_writes_are_atomic_no_tmp_left(tmp_path: Path) -> None:
    StateStore(tmp_path).write_last_run({"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["last_run.json"]
