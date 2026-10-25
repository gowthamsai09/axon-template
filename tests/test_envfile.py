from __future__ import annotations

import json
from pathlib import Path

import pytest

from digest.cli import main
from digest.envfile import EnvFileError, merge_env, parse_env_file
from tests.conftest import FEED, NOW
from tests.test_cli import World, write_profile


def test_parse_quotes_comments_export_and_empty(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text(
        '# comment\n\nLLM_API_KEY="sk-abc 123"\nexport SMTP_USER = me@gmail.com\n'
        "MAIL_TO='you@example.com'\nSMTP_PASS=pw # trailing\nSMTP_HOST=\nSMTP_PORT=587\n",
        encoding="utf-8",
    )
    values, warnings = parse_env_file(f)
    assert values == {
        "LLM_API_KEY": "sk-abc 123", "SMTP_USER": "me@gmail.com",
        "MAIL_TO": "you@example.com", "SMTP_PASS": "pw", "SMTP_PORT": "587",
    }  # fmt: skip
    assert warnings == []


def test_unknown_names_are_ignored_with_a_warning_naming_only_the_variable(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("TASK_ID=123\nPATH=/evil\nLLM_API_KEY=k" + "k" * 8)
    values, warnings = parse_env_file(f)
    assert list(values) == ["LLM_API_KEY"] and len(warnings) == 2
    assert "TASK_ID" in warnings[0] and "123" not in warnings[0]


def test_malformed_line_error_never_echoes_the_value(tmp_path: Path) -> None:
    f = tmp_path / ".env"
    f.write_text("LLM_API_KEY sk-secret-value-123\n")
    with pytest.raises(EnvFileError) as info:
        parse_env_file(f)
    assert "line 1" in str(info.value) and "sk-secret" not in str(info.value)
    with pytest.raises(EnvFileError):
        parse_env_file(tmp_path / "missing.env")


def test_real_environment_wins_but_empty_values_do_not() -> None:
    merged = merge_env(
        {"LLM_API_KEY": "from-file", "MAIL_TO": "a@b.c"}, {"LLM_API_KEY": "real", "MAIL_TO": ""}
    )
    assert merged["LLM_API_KEY"] == "real" and merged["MAIL_TO"] == "a@b.c"


def test_cli_loads_dot_env_from_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("FEED_BASE", "LLM_API_KEY", "GITHUB_ACTIONS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"FEED_BASE={FEED}\nSMTP_USER=me@gmail.com\n", encoding="utf-8")
    argv = [
        "run", "--profile", str(write_profile(tmp_path)), "--no-llm", "--dry-run",
        "--state-dir", str(tmp_path / "state"), "--outbox", str(tmp_path / "outbox"),
    ]  # fmt: skip
    assert main(argv, now=NOW, http=World().client) == 0
    err = capsys.readouterr().err
    assert "loaded .env: FEED_BASE, SMTP_USER" in err and "me@gmail.com" not in err
    assert json.loads((tmp_path / "state" / "last_run.json").read_text())["status"] == "ok"


def test_dot_env_is_ignored_inside_github_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("FEED_BASE", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f"FEED_BASE={FEED}\n", encoding="utf-8")
    argv = ["run", "--profile", str(write_profile(tmp_path)), "--state-dir", str(tmp_path / "s")]
    assert main(argv, now=NOW) == 2  # no feed: the file was not read


def test_malformed_dot_env_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    bad = tmp_path / "my.env"
    bad.write_text("not a pair\n")
    argv = ["run", "--env-file", str(bad), "--profile", str(write_profile(tmp_path))]
    assert main(argv, now=NOW) == 2 and "line 1" in capsys.readouterr().err
