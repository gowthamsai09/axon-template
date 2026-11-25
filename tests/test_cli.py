from __future__ import annotations

import json
import smtplib
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import yaml

from digest.cli import main
from digest.profile import Profile, load_profile
from digest.task_result import TaskResult
from tests.conftest import FEED, NOW, ROOT, profile_data, summary_json
from tests.test_email import FakeSmtp

KEY = "sk-live-key-987654321"
SMTP_PASS = "smtp-pass-123456"
ENV = {
    "LLM_API_KEY": KEY, "SMTP_USER": "me@gmail.com", "SMTP_PASS": SMTP_PASS,
    "MAIL_TO": "you@example.com", "SMTP_HOST": "smtp.example.com", "SMTP_PORT": "587",
}  # fmt: skip
UUID = "123e4567-e89b-12d3-a456-426614174000"


class World:
    """Mock internet: DeepSeek answers valid summaries; everything else is 404."""

    def __init__(self, llm: Callable[[httpx.Request], httpx.Response] | None = None) -> None:
        self.llm_calls = 0
        self.llm = llm
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, req: httpx.Request) -> httpx.Response:
        if req.url.host == "api.deepseek.com":
            self.llm_calls += 1
            if self.llm is not None:
                return self.llm(req)
            user = json.loads(req.content)["messages"][1]["content"]
            body = {
                "choices": [{"message": {"content": summary_json(user)}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            }
            return httpx.Response(200, json=body)
        return httpx.Response(404)


def write_profile(tmp: Path, **sections: dict[str, object]) -> Path:
    data = profile_data()
    data["interests"].update(topics=["agents-tooling"], keywords_exclude=[], keywords_include=[])
    data["delivery"].update(timezone="UTC", max_items=2)
    for name, over in sections.items():
        data[name].update(over)
    path = tmp / "profile.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def run_cli(
    tmp: Path, world: World, *extra: str, env: dict[str, str] | None = None,
    smtp: FakeSmtp | None = None, profile: Path | None = None,
) -> int:  # fmt: skip
    argv = ["run", "--profile", str(profile or write_profile(tmp)), "--feed", str(FEED),
            "--state-dir", str(tmp / "state"), "--outbox", str(tmp / "outbox"), *extra]  # fmt: skip
    fake = smtp or FakeSmtp()
    return main(
        argv,
        env=ENV if env is None else env,
        now=NOW,
        http=world.client,
        smtp_factory=lambda h, p, t: fake,
        resolver=lambda h: ["93.184.216.34"],
    )


def seen(tmp: Path) -> set[str]:
    path = tmp / "state" / "seen.json"
    return set(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else set()


def test_validate_profile_exit_codes(tmp_path: Path) -> None:
    assert main(["validate-profile", str(ROOT / "profile.yaml")]) == 0
    bad = tmp_path / "p.yaml"
    bad.write_text("version: 2\n")
    assert main(["validate-profile", str(bad)]) == 1


def test_run_without_a_feed_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        ["run", "--profile", str(ROOT / "profile.yaml"), "--state-dir", str(tmp_path / "s")], env={}
    )
    assert rc == 2 and "FEED_BASE" in capsys.readouterr().err and not (tmp_path / "s").exists()


def test_missing_llm_key_names_the_secret_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = {k: v for k, v in ENV.items() if k != "LLM_API_KEY"}
    assert run_cli(tmp_path, World(), env=env) == 2
    assert "LLM_API_KEY" in capsys.readouterr().err


def test_dry_run_without_llm_writes_the_outbox_and_does_not_consume_items(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World()
    assert run_cli(tmp_path, world, "--dry-run", "--no-llm", env={}) == 0
    names = sorted(p.suffix for p in (tmp_path / "outbox").iterdir())
    assert names == [".eml", ".html"] and world.llm_calls == 0 and seen(tmp_path) == set()
    last = json.loads((tmp_path / "state" / "last_run.json").read_text(encoding="utf-8"))
    assert last["status"] == "ok" and last["delivered"] == "dry-run" and last["items"] == 2
    assert "dry run: wrote" in capsys.readouterr().out


def test_full_run_summarizes_sends_marks_seen_and_never_repeats(tmp_path: Path) -> None:
    world, smtp = World(), FakeSmtp()
    assert run_cli(tmp_path, world, smtp=smtp) == 0
    assert smtp.log == ["starttls", "login:me@gmail.com", "send:you@example.com", "quit"]
    first = seen(tmp_path)
    assert len(first) == 2 and world.llm_calls == 2
    assert run_cli(tmp_path, world, smtp=smtp) == 0
    assert len(seen(tmp_path)) == 4 and len(seen(tmp_path) - first) == 2
    last = json.loads((tmp_path / "state" / "last_run.json").read_text(encoding="utf-8"))
    assert last["usage"]["prompt_tokens"] == 100 and last["delivered"] == "email"
    assert run_cli(tmp_path, world, smtp=smtp) == 0  # everything seen
    assert smtp.log.count("quit") == 2  # the third run sent nothing


def test_cached_summaries_make_a_repeat_dry_run_free(tmp_path: Path) -> None:
    world = World()
    run_cli(tmp_path, world, "--dry-run")
    run_cli(tmp_path, world, "--dry-run")
    assert world.llm_calls == 2  # second dry run reused the cache


def test_rejected_key_still_delivers_extractive_digest_with_a_notice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    world = World(lambda req: httpx.Response(401, json={"error": {"message": f"bad {KEY}"}}))
    assert run_cli(tmp_path, world, "--dry-run") == 0
    out = capsys.readouterr()
    assert "LLM_API_KEY" in out.err and KEY not in out.out + out.err
    assert world.llm_calls == 1  # provider skipped after the first rejection
    eml = next((tmp_path / "outbox").glob("*.eml")).read_text(encoding="utf-8")
    assert "not AI-summarized" in eml or "AI-summarized" in eml


def test_delivery_failure_exits_1_and_keeps_items_unseen(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    smtp = FakeSmtp(smtplib.SMTPAuthenticationError(535, b"nope"))
    assert run_cli(tmp_path, World(), smtp=smtp) == 1
    err = capsys.readouterr().err
    assert "SMTP_USER and SMTP_PASS" in err and SMTP_PASS not in err and seen(tmp_path) == set()
    last = json.loads((tmp_path / "state" / "last_run.json").read_text(encoding="utf-8"))
    assert last["status"] == "failed"


def test_a_chat_channel_without_its_secrets_is_a_config_error(tmp_path: Path) -> None:
    profile = write_profile(tmp_path, delivery={"channel": "telegram"})  # ENV has no bot token
    assert run_cli(tmp_path, World(), profile=profile) == 2 and seen(tmp_path) == set()


def test_missing_smtp_secrets_are_named(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(tmp_path, World(), env={"LLM_API_KEY": KEY}) == 2
    assert "SMTP_USER, SMTP_PASS, MAIL_TO" in capsys.readouterr().err


def test_bad_feed_location_exits_1_with_state_written(tmp_path: Path) -> None:
    argv = ["run", "--profile", str(write_profile(tmp_path)), "--feed", str(tmp_path / "nope"),
            "--state-dir", str(tmp_path / "state"), "--no-llm"]  # fmt: skip
    assert main(argv, env={}, now=NOW, http=World().client) == 1
    assert (
        json.loads((tmp_path / "state" / "last_run.json").read_text(encoding="utf-8"))["status"]
        == "failed"
    )


# ---- Tier 3 ---------------------------------------------------------------------------


def tier3_env(**params: object) -> dict[str, str]:
    return {**ENV, "TASK_ID": UUID, "TASK_PARAMS": json.dumps(params)}


def test_tier3_returns_a_valid_task_result_and_touches_nothing_else(tmp_path: Path) -> None:
    world, smtp = World(), FakeSmtp()
    rc = run_cli(tmp_path, world, env=tier3_env(max_items=2, delivery="owner_channel"), smtp=smtp)
    assert rc == 0 and smtp.log == [] and seen(tmp_path) == set()
    result = TaskResult.model_validate_json(
        (tmp_path / "state" / "tasks" / f"{UUID}.json").read_text(encoding="utf-8")
    )
    assert result.status == "completed" and len(result.items) == 2
    assert all(i.summary is not None for i in result.items) and result.model.provider == "deepseek"
    assert any("owner_channel delivery is not enabled" in w for w in result.warnings)


def test_tier3_ignores_seen_state_and_clamps_max_items(tmp_path: Path) -> None:
    run_cli(tmp_path, World())  # scheduled run marks two items seen
    run_cli(
        tmp_path,
        World(),
        env=tier3_env(max_items=99, evil="x"),
        profile=write_profile(tmp_path, delivery={"max_items": 4}),
    )
    result = TaskResult.model_validate_json(
        (tmp_path / "state" / "tasks" / f"{UUID}.json").read_text(encoding="utf-8")
    )
    assert len(result.items) == 4 and any("unknown params" in w for w in result.warnings)


def test_tier3_refusal_is_recorded_and_exits_0(tmp_path: Path) -> None:
    profile = write_profile(tmp_path, agent_access={"allowed_modes": ["data"]})
    world = World()
    assert run_cli(tmp_path, world, env=tier3_env(), profile=profile) == 0
    result = TaskResult.model_validate_json(
        (tmp_path / "state" / "tasks" / f"{UUID}.json").read_text(encoding="utf-8")
    )
    assert result.status == "failed" and result.items == [] and "owner_policy" in result.warnings[0]
    assert world.llm_calls == 0


def test_tier3_invalid_task_id_exits_2_without_writing(tmp_path: Path) -> None:
    env = {**ENV, "TASK_ID": "../../evil", "TASK_PARAMS": "{}"}
    assert run_cli(tmp_path, World(), env=env) == 2 and not (tmp_path / "state").exists()


def test_tier3_failure_writes_a_failed_result(tmp_path: Path) -> None:
    argv = ["run", "--profile", str(write_profile(tmp_path)), "--feed", str(tmp_path / "nope"),
            "--state-dir", str(tmp_path / "state"), "--no-llm"]  # fmt: skip
    assert main(argv, env=tier3_env(), now=NOW, http=World().client) == 1
    result = TaskResult.model_validate_json(
        (tmp_path / "state" / "tasks" / f"{UUID}.json").read_text(encoding="utf-8")
    )
    assert result.status == "failed"


def test_tier3_config_error_still_leaves_a_failed_result(tmp_path: Path) -> None:
    env = {k: v for k, v in tier3_env().items() if k != "LLM_API_KEY"}
    assert run_cli(tmp_path, World(), env=env) == 2
    path = tmp_path / "state" / "tasks" / f"{UUID}.json"
    result = TaskResult.model_validate_json(path.read_text(encoding="utf-8"))
    assert result.status == "failed" and "LLM_API_KEY" in result.warnings[0]


# ---- notify-failure and init ----------------------------------------------------------


def test_notify_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(["notify-failure", "--run-url", "https://github.com/o/r/actions/runs/1"], env={}) == 0
    )
    assert "no failure notice sent" in capsys.readouterr().out
    smtp = FakeSmtp()
    url = "https://github.com/o/r/actions/runs/1"
    assert (
        main(["notify-failure", "--run-url", url], env=ENV, smtp_factory=lambda h, p, t: smtp) == 0
    )
    assert smtp.log[-2:] == ["send:you@example.com", "quit"]
    out = tmp_path / "out"
    assert (
        main(
            ["notify-failure", "--run-url", "javascript:x", "--dry-run", "--outbox", str(out)],
            env=ENV,
            now=NOW,
        )
        == 0
    )
    text = next(out.glob("*.eml")).read_text(encoding="utf-8")
    assert "javascript" not in text and SMTP_PASS not in text


def answers(*values: str) -> Callable[[str], str]:
    queue = list(values)

    def ask(prompt: str) -> str:
        if not queue:
            raise EOFError
        return queue.pop(0)

    return ask


def test_init_writes_a_valid_profile_and_prints_the_cron(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "profile.yaml"
    ask = answers(
        "rag, agents-tooling", "Qwen", "", "crypto", "",                    # interests, sources
        "email", "weekly", "mon", "07:30", "Asia/Kolkata", "detailed", "8", "en",  # delivery
        "deepseek", "", "",                                                  # summarizer
        "n", "data", "3", "n",                                                    # agent access
        "n",                                                                 # optional extras
    )  # fmt: skip
    assert main(["init", "--out", str(out)], now=NOW, ask=ask) == 0
    profile: Profile = load_profile(out)
    assert profile.delivery.weekday == "mon" and profile.summarizer.model == "deepseek-flash"
    assert (
        profile.agent_access.allow_inbound_agents is False
        and profile.agent_access.allowed_modes == ["data"]
    )
    assert 'cron (UTC): "0 2 * * 1"' in capsys.readouterr().out


def test_init_reasks_an_invalid_section_and_protects_existing_files(tmp_path: Path) -> None:
    out = tmp_path / "profile.yaml"
    ask = answers(
        "made-up-topic", "", "", "",   # interests: invalid topic, so the section is asked again
        "rag", "", "", "",             # interests again
        "",                            # extra RSS
        "", "", "", "", "", "abc", "", # delivery: max items "abc" is invalid
        "", "", "", "", "", "", "",    # delivery again, all defaults
        "", "", "",                    # summarizer defaults
        "", "", "", "",                    # agent access defaults
        "",                            # optional extras: default is no
    )  # fmt: skip
    assert main(["init", "--out", str(out)], now=NOW, ask=ask) == 0
    assert load_profile(out).interests.topics == ["rag"]
    assert main(["init", "--out", str(ROOT / "profile.yaml")], ask=answers()) == 2  # exists
    assert main(["init", "--out", str(tmp_path / "x.yaml")], ask=answers()) == 2  # EOF mid-way
