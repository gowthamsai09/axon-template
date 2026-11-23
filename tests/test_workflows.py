"""Workflow security lint (spec section 10): runs offline in every repo."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from digest.profile import load_profile
from digest.schedule import cron_utc

ROOT = Path(__file__).resolve().parent.parent
WF = ROOT / ".github" / "workflows"
FILES = sorted(WF.glob("*.yml"))
REAL = {
    "ci.yml",
    "digest.yml",
    "register.yml",
    "issue-token.yml",
    "revoke-token.yml",
    "keepalive.yml",
}  # everything else is still a stub (TODO markers required)
SECRETS_OK = {
    "LLM_API_KEY", "LLM_API_KEY_FALLBACK", "SMTP_USER", "SMTP_PASS", "MAIL_TO",
    "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "SLACK_WEBHOOK_URL", "GITHUB_TOKEN",
}  # fmt: skip
UNTRUSTED = re.compile(r"\$\{\{\s*(inputs\.|github\.event|github\.head_ref|vars\.)")


def load(p: Path) -> dict[Any, Any]:
    data = yaml.safe_load(p.read_text())
    assert isinstance(data, dict)
    return data


def steps(doc: dict[Any, Any]) -> list[dict[str, Any]]:
    return [s for job in doc["jobs"].values() for s in job["steps"]]


def test_workflows_exist() -> None:
    assert FILES


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_explicit_permissions_and_triggers(path: Path) -> None:
    doc = load(path)
    assert isinstance(doc.get("permissions"), dict), "top-level permissions required"
    triggers = doc.get(True, doc.get("on"))  # PyYAML parses `on:` as boolean True
    assert triggers and "pull_request_target" not in triggers


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_actions_pinned(path: Path) -> None:
    for s in steps(load(path)):
        uses = s.get("uses")
        if uses and not uses.startswith("./"):
            assert re.search(r"@(?:[0-9a-f]{40}|<pin-to-commit-sha>)$", uses), uses


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_no_untrusted_expressions_in_run(path: Path) -> None:
    for s in steps(load(path)):
        assert not UNTRUSTED.search(s.get("run", "")), s.get("name")


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_only_known_secrets_and_no_deploys(path: Path) -> None:
    text = json.dumps(load(path))  # comments are dropped by the parser
    assert set(re.findall(r"secrets\.([A-Za-z_]+)", text)) <= SECRETS_OK
    assert not re.search(r"wrangler|npm publish|pages deploy", text)


@pytest.mark.parametrize("path", [p for p in FILES if p.name not in REAL], ids=lambda p: p.name)
def test_stub_steps_carry_todo_markers(path: Path) -> None:
    for s in steps(load(path)):
        assert "TODO" in (s.get("name", "") + s.get("run", "")), s.get("name")


# ---- the real digest workflow (spec 6.6, 9.3, 10) -----------------------------------------

DIGEST = WF / "digest.yml"


def test_digest_keeps_task_inputs_and_run_name() -> None:
    doc = load(DIGEST)
    inputs = doc[True]["workflow_dispatch"]["inputs"]
    assert set(inputs) == {"task_id", "params"}
    assert "digest task" in doc["run-name"] and doc["concurrency"]["group"] == "digest"
    assert doc["permissions"] == {"contents": "write"}


def test_digest_passes_untrusted_input_through_env_only() -> None:
    run_step = next(s for s in steps(load(DIGEST)) if s.get("name") == "Run digest")
    assert run_step["env"]["TASK_ID"] == "${{ inputs.task_id }}"
    assert run_step["env"]["TASK_PARAMS"] == "${{ inputs.params }}"
    assert run_step["run"] == "python -m digest.cli run --state-dir state"
    for s in steps(load(DIGEST)):
        assert "${{" not in s.get("run", ""), s.get("name")


def test_digest_checkouts_do_not_persist_credentials_and_state_may_be_missing() -> None:
    checkouts = [s for s in steps(load(DIGEST)) if s.get("uses", "").startswith("actions/checkout")]
    assert len(checkouts) == 2
    assert all(c["with"]["persist-credentials"] is False for c in checkouts)
    assert checkouts[1]["with"]["ref"] == "state" and checkouts[1]["continue-on-error"] is True


def test_digest_failure_notice_and_state_commit_conditions() -> None:
    by_name = {s.get("name"): s for s in steps(load(DIGEST))}
    assert by_name["Failure notice"]["if"] == "failure()"
    assert by_name["Commit state"]["if"] == "always()"
    notice_env = by_name["Failure notice"]["env"]
    assert set(notice_env) == {"SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS", "MAIL_TO"}


def test_digest_never_runs_inside_the_template_repo() -> None:
    job = next(iter(load(DIGEST)["jobs"].values()))
    assert "gowthamsai09/axon-template" in job["if"]


def test_digest_cron_matches_the_shipped_profile() -> None:
    schedule = load(DIGEST)[True]["schedule"][0]["cron"]
    profile = load_profile(ROOT / "profile.yaml")
    assert schedule == cron_utc(profile.delivery, datetime(2026, 10, 20, tzinfo=UTC))
