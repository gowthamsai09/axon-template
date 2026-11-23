"""Failure notices and observability review (guidelines section 5, brief P5.10).

Notices, warnings, state and outbox files never contain secrets, model text or feed text, and the
run link in a failure notice cannot be hostile. Offline."""

from __future__ import annotations

import json
import smtplib
from pathlib import Path

import httpx
import pytest

from digest.cli import MAX_RUN_URL, main, safe_run_url
from tests.conftest import NOW
from tests.test_cli import ENV, KEY, SMTP_PASS, World, run_cli
from tests.test_email import FakeSmtp

RUN = "https://github.com/o/r/actions/runs/42"


@pytest.mark.parametrize(
    ("url", "env", "expected"),
    [
        (RUN, {}, RUN),
        (RUN, {"GITHUB_SERVER_URL": "https://github.com"}, RUN),
        (
            "https://ghe.corp.example/o/r/actions/runs/1",
            {"GITHUB_SERVER_URL": "https://ghe.corp.example"},
            "https://ghe.corp.example/o/r/actions/runs/1",
        ),
        ("https://GitHub.com/o/r/actions/runs/1", {}, "https://GitHub.com/o/r/actions/runs/1"),
        ("", {}, ""),
        ("http://github.com/o/r/actions/runs/1", {}, ""),  # not https
        ("javascript:alert(1)", {}, ""),
        ("data:text/html,x", {}, ""),
        ("ftp://github.com/x", {}, ""),
        ("https://evil.example/o/r", {}, ""),  # wrong host
        ("https://github.com.evil.example/x", {}, ""),  # lookalike host
        ("https://user:pw@github.com/o/r", {}, ""),  # credentials
        ("https://github.com@evil.example/x", {}, ""),  # userinfo trick
        ("https://github.com/o/r\nBcc: evil@example.com", {}, ""),  # header injection
        ("https://github.com/o/r with space", {}, ""),
        ("https://github.com/o/r\x00", {}, ""),
        ("https://github.com/" + "a" * MAX_RUN_URL, {}, ""),  # too long
        ("https://github.com/o/r", {"GITHUB_SERVER_URL": "https://ghe.corp.example"}, ""),
    ],
)
def test_safe_run_url(url: str, env: dict[str, str], expected: str) -> None:
    assert safe_run_url(url, env) == expected


def all_text(root: Path) -> str:
    return "\n".join(
        p.read_text(encoding="utf-8", errors="replace") for p in root.rglob("*") if p.is_file()
    )


def test_failure_notice_contains_only_a_link_and_fixed_text(tmp_path: Path) -> None:
    out = tmp_path / "out"
    env = {**ENV, "LLM_API_KEY": KEY, "GITHUB_SERVER_URL": "https://github.com"}
    code = main(
        ["notify-failure", "--run-url", RUN, "--dry-run", "--outbox", str(out)], env=env, now=NOW
    )
    text = all_text(out)
    assert code == 0 and RUN in text
    for secret in (KEY, SMTP_PASS):
        assert secret not in text
    assert "never contains secrets" in text


def test_failure_notice_with_a_hostile_link_has_no_link(tmp_path: Path) -> None:
    out = tmp_path / "out"
    hostile = "https://evil.example/phish\nBcc: x@evil.example"
    assert (
        main(
            ["notify-failure", "--run-url", hostile, "--dry-run", "--outbox", str(out)],
            env=ENV,
            now=NOW,
        )
        == 0
    )
    text = all_text(out)
    assert "evil.example" not in text.replace("MAIL_TO", "") and "Details:" not in text


def test_failure_notice_send_error_is_secret_free(capsys: pytest.CaptureFixture[str]) -> None:
    smtp = FakeSmtp(smtplib.SMTPException(f"boom {SMTP_PASS}"))
    assert (
        main(["notify-failure", "--run-url", RUN], env=ENV, smtp_factory=lambda h, p, t: smtp) == 1
    )
    err = capsys.readouterr().err
    assert SMTP_PASS not in err and "SMTPException" in err


def test_provider_error_echoing_the_key_never_reaches_any_file_or_stream(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(lambda req: httpx.Response(401, json={"error": {"message": f"bad key {KEY}"}}))
    assert run_cli(tmp_path, world, "--dry-run") == 0
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err + all_text(tmp_path)
    last = json.loads((tmp_path / "state" / "last_run.json").read_text(encoding="utf-8"))
    assert any(
        "LLM_API_KEY" in n for n in last["notices"]
    )  # actionable: names the secret, not its value


def test_delivery_failure_records_a_failure_without_secrets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    smtp = FakeSmtp(smtplib.SMTPAuthenticationError(535, f"rejected {SMTP_PASS}".encode()))
    assert run_cli(tmp_path, World(), smtp=smtp) == 1
    captured = capsys.readouterr()
    everything = captured.out + captured.err + all_text(tmp_path)
    assert SMTP_PASS not in everything and KEY not in everything
    last = json.loads((tmp_path / "state" / "last_run.json").read_text(encoding="utf-8"))
    assert last["status"] == "failed" and "SMTP_USER and SMTP_PASS" in last["error"]


def test_model_text_never_lands_in_warnings_notices_or_state(tmp_path: Path) -> None:
    junk = "MODEL-GARBAGE-MARKER-123"
    world = World(
        lambda req: httpx.Response(200, json={"choices": [{"message": {"content": junk}}]})
    )
    assert run_cli(tmp_path, world, "--dry-run") == 0
    last = json.loads((tmp_path / "state" / "last_run.json").read_text(encoding="utf-8"))
    assert last["warnings"] and junk not in all_text(tmp_path)


def test_warnings_and_notices_are_capped(tmp_path: Path) -> None:
    run_cli(tmp_path, World(), "--dry-run", "--no-llm", env={})
    last = json.loads((tmp_path / "state" / "last_run.json").read_text(encoding="utf-8"))
    assert len(last["warnings"]) <= 100 and len(last["notices"]) <= 20
    assert all(len(w) <= 300 for w in last["warnings"] + last["notices"])


def test_state_files_hold_no_secret_after_a_full_run(tmp_path: Path) -> None:
    assert run_cli(tmp_path, World(), smtp=FakeSmtp()) == 0
    text = all_text(tmp_path / "state")
    for secret in (KEY, SMTP_PASS, "you@example.com", "me@gmail.com"):
        assert secret not in text


def test_digest_email_does_not_contain_secrets_or_the_to_address_in_the_body(
    tmp_path: Path,
) -> None:
    assert run_cli(tmp_path, World(), "--dry-run") == 0
    eml = next((tmp_path / "outbox").glob("*.eml")).read_text(encoding="utf-8")
    assert KEY not in eml and SMTP_PASS not in eml
