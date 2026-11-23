"""keepalive.yml (spec 6.7, Appendix C 80): contract checks everywhere, the script under bash with a
fake `gh` where a POSIX shell exists (skipped on Windows, runs on Linux CI)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
PATH = ROOT / ".github" / "workflows" / "keepalive.yml"
TEXT = PATH.read_text(encoding="utf-8")


def load() -> dict[Any, Any]:
    data = yaml.safe_load(TEXT)
    assert isinstance(data, dict)
    return data


def step() -> dict[str, Any]:
    steps: list[dict[str, Any]] = load()["jobs"]["enable"]["steps"]
    assert len(steps) == 1
    return steps[0]


def test_triggers_are_a_monthly_cron_and_manual_only() -> None:
    triggers = load()[True]  # YAML parses the key `on` as True
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    cron = triggers["schedule"]
    assert cron == [{"cron": "0 3 1 * *"}]


def test_least_privilege_and_no_secrets_or_actions() -> None:
    doc = load()
    assert doc["permissions"] == {"actions": "write"}
    assert "secrets." not in TEXT and "uses:" not in TEXT
    job = doc["jobs"]["enable"]
    assert job["timeout-minutes"] <= 5
    assert job["if"] == "github.repository != 'gowthamsai09/axon-template'"
    assert doc["concurrency"]["cancel-in-progress"] is False


def test_the_token_reaches_the_script_through_env_only() -> None:
    s = step()
    assert s["env"] == {"GH_TOKEN": "${{ github.token }}"}
    assert "${{" not in s["run"]
    assert s["run"] == 'gh workflow enable digest.yml --repo "$GITHUB_REPOSITORY"'


def test_it_targets_the_digest_workflow_that_exists() -> None:
    assert (ROOT / ".github" / "workflows" / "digest.yml").is_file()


needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="needs a POSIX bash (runs in CI on ubuntu)",
)


def run_script(tmp: Path, repo: str, gh_exit: int) -> tuple[int, str]:
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "$FAKE_LOG"\nexit ${FAKE_EXIT}\n')
    gh.chmod(0o755)
    log = tmp / "gh.log"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GITHUB_REPOSITORY": repo,
        "GH_TOKEN": "ghs_UNIQUE_keepalive_token_01",
        "FAKE_LOG": str(log),
        "FAKE_EXIT": str(gh_exit),
    }
    proc = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step()["run"]], env=env, capture_output=True, text=True
    )
    assert "ghs_UNIQUE_keepalive_token_01" not in proc.stdout + proc.stderr
    return proc.returncode, log.read_text(encoding="utf-8") if log.exists() else ""


@needs_bash
def test_script_calls_gh_with_exactly_these_arguments(tmp_path: Path) -> None:
    code, log = run_script(tmp_path, "alice/axon-digest", 0)
    assert code == 0
    assert log.split("\n")[:-1] == [
        "workflow",
        "enable",
        "digest.yml",
        "--repo",
        "alice/axon-digest",
    ]


@needs_bash
def test_a_failing_gh_fails_the_job_so_the_owner_sees_it(tmp_path: Path) -> None:
    code, log = run_script(tmp_path, "alice/axon-digest", 1)
    assert code == 1 and log != ""  # the failure came from gh, which was reached


@needs_bash
def test_a_hostile_repository_value_stays_a_single_argument(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    hostile = f"a/b; touch {marker}; echo $(id)"
    code, log = run_script(tmp_path, hostile, 0)
    assert code == 0 and not marker.exists()
    assert log.split("\n")[:-1] == ["workflow", "enable", "digest.yml", "--repo", hostile]
