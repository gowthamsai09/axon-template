"""Contract checks for onboard.yml (Appendix C 101): who may trigger it, what it may touch."""

from __future__ import annotations

import re
from typing import Any

from tests.test_workflows import WF, load, steps

ONBOARD = WF / "onboard.yml"


def job() -> dict[str, Any]:
    jobs = load(ONBOARD)["jobs"]
    assert list(jobs) == ["onboard"]
    return dict(jobs["onboard"])


def by_name() -> dict[str, dict[str, Any]]:
    return {s.get("name", s.get("uses", s.get("run", ""))): s for s in steps(load(ONBOARD))}


def test_triggers_and_permissions_are_minimal() -> None:
    doc = load(ONBOARD)
    assert doc[True] == {"issues": {"types": ["opened", "edited"]}}
    assert doc["permissions"] == {"contents": "write", "issues": "write"}
    assert doc["concurrency"]["cancel-in-progress"] is False


def test_only_the_repository_owner_can_trigger_it() -> None:
    cond = " ".join(str(job()["if"]).split())
    for part in (
        "github.repository != 'gowthamsai09/axon-template'",
        "github.event.issue.user.login == github.repository_owner",
        "github.event.sender.login == github.repository_owner",
        "github.event.issue.user.type == 'User'",
        "github.event.issue.state == 'open'",
        "contains(github.event.issue.labels.*.name, 'onboarding')",
    ):
        assert part in cond, part
    assert cond.count("&&") == 5 and "||" not in cond


def test_issue_body_reaches_code_through_env_only() -> None:
    check = by_name()["Check the form and write profile.yaml"]
    assert check["env"]["ISSUE_BODY"] == "${{ github.event.issue.body }}"
    for step in steps(load(ONBOARD)):
        assert "${{" not in step.get("run", ""), step.get("name")
        assert "issue.title" not in str(step) and "issue.body" not in step.get("run", "")


def test_checkout_does_not_persist_credentials_and_uses_default_branch() -> None:
    checkout = next(s for s in steps(load(ONBOARD)) if s.get("uses", "").startswith("actions/c"))
    assert checkout["with"]["persist-credentials"] is False
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"


def test_only_profile_yaml_is_committed() -> None:
    run = by_name()["Commit profile.yaml"]["run"]
    assert "git add profile.yaml" in run and "git add -A" not in run and "git add ." not in run
    assert "git add .github" not in run
    assert by_name()["Commit profile.yaml"]["if"] == "steps.check.outputs.valid == 'true'"


def test_commit_token_is_masked_and_never_in_the_remote_url() -> None:
    run = by_name()["Commit profile.yaml"]["run"]
    assert "::add-mask::" in run and "http.extraheader" in run
    assert not re.search(r"https://[^ ]*@", run)


def test_reply_validates_the_issue_number_and_closes_only_valid_forms() -> None:
    reply = by_name()["Reply on the Issue"]
    assert reply["env"]["ISSUE_NUMBER"] == "${{ github.event.issue.number }}"
    assert "*[!0-9]*" in reply["run"]
    assert "gh issue close" in reply["run"] and '"$VALID" = "true"' in reply["run"]
    assert "--body-file" in reply["run"]


def test_comment_text_is_never_built_from_the_issue_in_shell() -> None:
    text = ONBOARD.read_text(encoding="utf-8")
    assert "ISSUE_BODY" in text and text.count("ISSUE_BODY") == 1
    assert 'echo "$ISSUE_BODY"' not in text and '--body "' not in text


def test_workflow_files_are_never_touched() -> None:
    text = ONBOARD.read_text(encoding="utf-8")
    assert "workflows/" not in text.split("jobs:")[1]
