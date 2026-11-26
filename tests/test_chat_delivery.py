"""Stage 1 through the CLI: Telegram and Slack dry runs, real sends against fakes, secrets never
printed or stored, and Tier 3 owner_channel delivery on both channels. Offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from digest.cli import main
from digest.delivery.common import Sleep
from digest.delivery.email import SmtpClient
from digest.task_result import TaskResult
from tests.chat_helpers import Script, refusing, routed, slack_ok, tg_error, tg_ok
from tests.conftest import FEED, NOW, ROOT
from tests.test_cli import seen, write_profile
from tests.test_phase4 import TASK, _feed

TOKEN = "123456789:AA" + "FuniqueTokenXyZ_0123456789abcdefg"
CHAT = "-1001234567890"
SECRET = "uNiQuEwEbHoOkSeCrEt123456"
HOOK = f"https://hooks.slack.com/services/T0AAAAAAA/B0BBBBBBB/{SECRET}"
ECHO = "UNIQUE-BODY-ECHO-777"
TG_ENV = {"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": CHAT}
SL_ENV = {"SLACK_WEBHOOK_URL": HOOK}
STEM = "digest-20261020T060000Z"


def no_smtp(host: str, port: int, timeout: float) -> SmtpClient:
    raise AssertionError("SMTP must not be used for a chat channel")


def go(
    tmp_path: Path,
    *,
    channel: str,
    env: dict[str, str],
    http: httpx.Client,
    extra: tuple[str, ...] = (),
    sleeps: list[float] | None = None,
) -> int:
    profile = write_profile(tmp_path, delivery={"channel": channel})
    argv = ["run", "--profile", str(profile), "--feed", str(FEED),
            "--state-dir", str(tmp_path / "state"), "--outbox", str(tmp_path / "outbox"),
            "--no-llm", *extra]  # fmt: skip
    sleep: Sleep = sleeps.append if sleeps is not None else (lambda _s: None)
    return main(
        argv,
        env=env,
        now=NOW,
        http=http,
        smtp_factory=no_smtp,
        resolver=lambda _h: ["93.184.216.34"],
        sleep=sleep,
    )


def last_run(tmp_path: Path) -> dict[str, Any]:
    data = json.loads((tmp_path / "state" / "last_run.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def stored_text(root: Path) -> str:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)


def printed(capsys: pytest.CaptureFixture[str]) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


def assert_no_leak(root: Path, printed: str, *secrets: str) -> None:
    haystack = printed + stored_text(root)
    assert haystack  # the scan ran over something
    for secret in secrets:
        assert secret not in haystack


# ---- dry run ---------------------------------------------------------------------------


def test_telegram_dry_run_writes_numbered_files_and_sends_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(tmp_path, channel="telegram", env={}, http=refusing(), extra=("--dry-run",)) == 0
    files = sorted(p.name for p in (tmp_path / "outbox").iterdir())
    assert files == [f"{STEM}.telegram-01.md"]
    text = (tmp_path / "outbox" / files[0]).read_text(encoding="utf-8")
    assert text.startswith("*Axon digest*") and "\\." in text
    assert seen(tmp_path) == set() and last_run(tmp_path)["delivered"] == "dry-run"
    assert "dry run: wrote" in capsys.readouterr().out


def test_slack_dry_run_writes_numbered_files_and_sends_nothing(tmp_path: Path) -> None:
    assert go(tmp_path, channel="slack", env={}, http=refusing(), extra=("--dry-run",)) == 0
    files = sorted(p.name for p in (tmp_path / "outbox").iterdir())
    assert files == [f"{STEM}.slack-01.txt"]
    text = (tmp_path / "outbox" / files[0]).read_text(encoding="utf-8")
    assert text.startswith("*Axon digest*") and "1. <" in text
    assert seen(tmp_path) == set() and last_run(tmp_path)["delivered"] == "dry-run"


# ---- real sends against fakes ----------------------------------------------------------


def test_telegram_sends_marks_items_seen_and_leaks_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    tg = Script(tg_ok)
    assert go(tmp_path, channel="telegram", env=TG_ENV, http=routed({"api.telegram.org": tg})) == 0
    (req,) = tg.requests
    assert req.url.path == f"/bot{TOKEN}/sendMessage"
    body = tg.bodies()[0]
    assert body["parse_mode"] == "MarkdownV2" and str(body["text"]).startswith("*Axon digest*")
    assert len(seen(tmp_path)) == 2
    assert last_run(tmp_path)["status"] == "ok" and last_run(tmp_path)["delivered"] == "telegram"
    assert_no_leak(tmp_path, printed(capsys), TOKEN, CHAT)


def test_slack_sends_marks_items_seen_and_leaks_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sl = Script(slack_ok)
    assert go(tmp_path, channel="slack", env=SL_ENV, http=routed({"hooks.slack.com": sl})) == 0
    (req,) = sl.requests
    assert str(req.url) == HOOK and set(sl.bodies()[0]) == {"text"}
    assert len(seen(tmp_path)) == 2 and last_run(tmp_path)["delivered"] == "slack"
    assert_no_leak(tmp_path, printed(capsys), HOOK, SECRET)


def test_the_injected_sleep_is_used_for_rate_limit_waits(tmp_path: Path) -> None:
    tg = Script(tg_ok, tg_error(429, "Too Many Requests", retry_after=3))
    sleeps: list[float] = []
    http = routed({"api.telegram.org": tg})
    assert go(tmp_path, channel="telegram", env=TG_ENV, http=http, sleeps=sleeps) == 0
    assert sleeps == [3.0] and len(tg.requests) == 2


# ---- failures --------------------------------------------------------------------------


def test_a_rejected_token_fails_the_run_without_echoing_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rejected = httpx.Response(401, json={"ok": False, "description": f"{ECHO} {TOKEN}"})
    tg = Script(tg_ok, rejected)
    code = go(tmp_path, channel="telegram", env=TG_ENV, http=routed({"api.telegram.org": tg}))
    shown = printed(capsys)
    assert code == 1 and "Telegram rejected TELEGRAM_BOT_TOKEN" in shown
    assert seen(tmp_path) == set()
    last = last_run(tmp_path)
    assert last["status"] == "failed" and "Telegram rejected" in last["error"]
    assert_no_leak(tmp_path, shown, TOKEN, CHAT, ECHO)


def test_a_dead_webhook_fails_the_run_without_echoing_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sl = Script(slack_ok, httpx.Response(404, text=f"no_service {ECHO} {HOOK}"))
    code = go(tmp_path, channel="slack", env=SL_ENV, http=routed({"hooks.slack.com": sl}))
    shown = printed(capsys)
    assert code == 1 and "Slack webhook no longer exists" in shown
    assert seen(tmp_path) == set()
    assert_no_leak(tmp_path, shown, HOOK, SECRET, ECHO)


def test_an_unreachable_service_fails_the_run_without_the_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    tg = Script(tg_ok, *(httpx.ConnectError(f"failed for {url}") for _ in range(3)))
    code = go(tmp_path, channel="telegram", env=TG_ENV, http=routed({"api.telegram.org": tg}))
    shown = printed(capsys)
    assert code == 1 and "Telegram unreachable (ConnectError)" in shown
    assert seen(tmp_path) == set()
    assert_no_leak(tmp_path, shown, TOKEN, CHAT)


@pytest.mark.parametrize(
    ("channel", "names"),
    [("telegram", "TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID"), ("slack", "SLACK_WEBHOOK_URL")],
)
def test_missing_secrets_are_a_config_error_naming_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], channel: str, names: str
) -> None:
    assert go(tmp_path, channel=channel, env={}, http=refusing()) == 2
    assert f"missing secret(s): {names}" in capsys.readouterr().err
    assert seen(tmp_path) == set()


def test_a_malformed_webhook_is_rejected_before_any_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = {"SLACK_WEBHOOK_URL": "https://evil.example/hook-UNIQUE-4242"}
    assert go(tmp_path, channel="slack", env=env, http=refusing()) == 2
    err = capsys.readouterr().err
    assert "hooks.slack.com/services" in err and "hook-UNIQUE-4242" not in err


def test_a_malformed_bot_token_is_rejected_before_any_request(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = {"TELEGRAM_BOT_TOKEN": "UNIQUE-BAD-TOKEN-4242/../x", "TELEGRAM_CHAT_ID": CHAT}
    assert go(tmp_path, channel="telegram", env=env, http=refusing()) == 2
    err = capsys.readouterr().err
    assert "does not look like a bot token" in err and "UNIQUE-BAD-TOKEN-4242" not in err


# ---- Tier 3 owner_channel --------------------------------------------------------------


def t3_profile(root: Path, channel: str, *, flag: bool = True) -> Path:
    data = yaml.safe_load((ROOT / "profile.yaml").read_text(encoding="utf-8"))
    data["interests"] = {
        "topics": ["rag"], "follow_entities": [], "keywords_include": [], "keywords_exclude": [],
    }  # fmt: skip
    data["agent_access"]["allow_owner_channel"] = flag
    data["delivery"]["channel"] = channel
    path = root / "profile.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def t3(
    tmp_path: Path,
    channel: str,
    env: dict[str, str],
    http: httpx.Client,
    *,
    flag: bool = True,
    dry: bool = False,
) -> tuple[int, dict[str, Any]]:
    argv = ["run", "--profile", str(t3_profile(tmp_path, channel, flag=flag)),
            "--feed", str(_feed(tmp_path)), "--state-dir", str(tmp_path / "state"),
            "--outbox", str(tmp_path / "outbox"), "--no-llm"]  # fmt: skip
    if dry:
        argv.append("--dry-run")
    code = main(
        argv,
        env={**env, "TASK_ID": TASK, "TASK_PARAMS": '{"delivery":"owner_channel"}'},
        now=NOW,
        http=http,
        smtp_factory=no_smtp,
        resolver=lambda _h: ["93.184.216.34"],
        sleep=lambda _s: None,
    )
    path = tmp_path / "state" / "tasks" / f"{TASK}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    TaskResult.model_validate(result)  # the file always satisfies the contract
    return code, result


def test_owner_channel_delivers_through_telegram(tmp_path: Path) -> None:
    tg = Script(tg_ok)
    code, result = t3(tmp_path, "telegram", TG_ENV, routed({"api.telegram.org": tg}))
    assert code == 0 and len(tg.requests) == 1 and len(result["items"]) == 1
    assert last_run(tmp_path)["delivered"] == "owner_channel:telegram"
    assert not (tmp_path / "state" / "seen.json").exists()  # Tier 3 never consumes items


def test_owner_channel_delivers_through_slack(tmp_path: Path) -> None:
    sl = Script(slack_ok)
    code, result = t3(tmp_path, "slack", SL_ENV, routed({"hooks.slack.com": sl}))
    assert code == 0 and len(sl.requests) == 1 and len(result["items"]) == 1
    assert last_run(tmp_path)["delivered"] == "owner_channel:slack"


@pytest.mark.parametrize(
    ("channel", "env", "host", "secrets"),
    [
        ("telegram", TG_ENV, "api.telegram.org", (TOKEN, CHAT)),
        ("slack", SL_ENV, "hooks.slack.com", (HOOK, SECRET)),
    ],
)
def test_a_delivery_failure_keeps_the_result_and_leaks_nothing(
    tmp_path: Path,
    channel: str,
    env: dict[str, str],
    host: str,
    secrets: tuple[str, ...],
) -> None:
    failing = httpx.Response(403, text=f"{ECHO} {TOKEN} {HOOK}")
    script = Script(tg_ok if channel == "telegram" else slack_ok, failing)
    code, result = t3(tmp_path, channel, env, routed({host: script}))
    assert code == 0 and result["status"] == "completed" and len(result["items"]) == 1
    assert any("owner_channel delivery failed" in w for w in result["warnings"])
    dump = json.dumps(result) + stored_text(tmp_path / "state")
    for secret in (*secrets, ECHO):
        assert secret not in dump


def test_owner_channel_without_secrets_is_a_warning_not_a_lost_run(tmp_path: Path) -> None:
    code, result = t3(tmp_path, "telegram", {}, refusing())
    assert code == 0 and result["status"] == "completed"
    assert any("missing secret(s): TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID" in w
               for w in result["warnings"])  # fmt: skip


def test_the_flag_still_gates_chat_delivery(tmp_path: Path) -> None:
    code, result = t3(tmp_path, "telegram", TG_ENV, refusing(), flag=False)
    assert code == 0 and any("allow_owner_channel" in w for w in result["warnings"])


def test_owner_channel_dry_run_writes_the_outbox_and_sends_nothing(tmp_path: Path) -> None:
    code, _ = t3(tmp_path, "slack", {}, refusing(), dry=True)
    assert code == 0 and list((tmp_path / "outbox").glob("*.slack-01.txt"))
