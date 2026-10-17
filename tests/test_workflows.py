"""Workflow security lint (spec section 10): runs offline in every repo."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows"
FILES = sorted(WF.glob("*.yml"))
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


@pytest.mark.parametrize("path", [p for p in FILES if p.name != "ci.yml"], ids=lambda p: p.name)
def test_stub_steps_carry_todo_markers(path: Path) -> None:
    for s in steps(load(path)):
        assert "TODO" in (s.get("name", "") + s.get("run", "")), s.get("name")
