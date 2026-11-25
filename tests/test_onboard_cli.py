"""`digest.cli onboard` and the issue form file (Appendix C 101)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from digest import onboarding as o
from digest.cli import EXIT_FAIL, EXIT_OK, main
from digest.profile import TOPIC_SLUGS, load_profile
from tests.conftest import NOW, ROOT
from tests.onboard_helpers import body

FORM = ROOT / ".github" / "ISSUE_TEMPLATE" / "onboarding.yml"


def run(tmp_path: Path, issue: str) -> tuple[int, Path, Path]:
    profile, comment = tmp_path / "profile.yaml", tmp_path / "comment.md"
    code = main(
        ["onboard", "--profile", str(profile), "--comment-file", str(comment)],
        env={"ISSUE_BODY": issue},
        now=NOW,
    )
    return code, profile, comment


def test_valid_issue_writes_profile_and_comment(tmp_path: Path) -> None:
    code, profile, comment = run(tmp_path, body())
    assert code == EXIT_OK
    assert load_profile(profile).delivery.timezone == "Asia/Kolkata"
    text = comment.read_text(encoding="utf-8")
    assert '"0 2 * * *"' in text and "`SMTP_PASS`" in text


def test_invalid_issue_changes_nothing_and_explains(tmp_path: Path) -> None:
    keep = tmp_path / "profile.yaml"
    keep.write_text("keep: me\n", encoding="utf-8")
    code, _, comment = run(tmp_path, body(L_ZONE="Mars/Base"))
    assert code == EXIT_FAIL
    assert keep.read_text(encoding="utf-8") == "keep: me\n"
    text = comment.read_text(encoding="utf-8")
    assert o.L_ZONE in text and "Mars" not in text and "Nothing was changed" in text


def test_missing_body_is_a_clean_failure(tmp_path: Path) -> None:
    code, profile, comment = run(tmp_path, "")
    assert code == EXIT_FAIL and not profile.exists() and comment.exists()


def test_rerun_keeps_agent_access_edits(tmp_path: Path) -> None:
    assert run(tmp_path, body())[0] == EXIT_OK
    path = tmp_path / "profile.yaml"
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["agent_access"]["allowed_modes"] = ["data"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert run(tmp_path, body(L_MAX="5"))[0] == EXIT_OK
    loaded = load_profile(path)
    assert loaded.agent_access.allowed_modes == ["data"] and loaded.delivery.max_items == 5


def test_stdout_and_stderr_never_carry_the_issue_text(tmp_path: Path, capsys: Any) -> None:
    secret = "QX-NOT-A-KEY-4471"
    run(tmp_path, body(L_ENTITIES=[secret], L_ZONE=secret))
    seen = capsys.readouterr()
    assert secret not in seen.out and secret not in seen.err


# ---- the issue form file stays in step with the parser ----------------------------------


def _form() -> list[dict[str, Any]]:
    doc = yaml.safe_load(FORM.read_text(encoding="utf-8"))
    return [b for b in doc["body"] if b["type"] != "markdown"]


def test_form_labels_equal_parser_labels_in_order() -> None:
    assert [b["attributes"]["label"] for b in _form()] == list(o.SPEC)


def test_form_options_equal_parser_options() -> None:
    for block in _form():
        label = block["attributes"]["label"]
        kind, allowed, required = o.SPEC[label]
        if block["type"] == "dropdown":
            assert tuple(block["attributes"]["options"]) == allowed and kind == "choice"
        if block["type"] == "checkboxes":
            labels = tuple(x["label"] for x in block["attributes"]["options"])
            assert labels == allowed and kind == "checks"
        if block["type"] in {"input", "dropdown"}:
            assert block["validations"]["required"] is required


def test_form_topics_are_every_profile_topic() -> None:
    topics = next(b for b in _form() if b["attributes"]["label"] == o.L_TOPICS)
    assert tuple(x["label"] for x in topics["attributes"]["options"]) == TOPIC_SLUGS


def test_form_is_labelled_and_warns_about_keys() -> None:
    doc = yaml.safe_load(FORM.read_text(encoding="utf-8"))
    assert doc["labels"] == ["onboarding"]
    assert "Never enter API keys" in doc["body"][0]["attributes"]["value"]


def test_form_has_no_field_that_could_hold_a_secret_by_name() -> None:
    text = FORM.read_text(encoding="utf-8").lower()
    assert "api key:" not in text and "password:" not in text and "token:" not in text
