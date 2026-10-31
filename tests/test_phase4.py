"""Phase 4 (template side): the allow_owner_channel flag, Tier 3 owner_channel delivery, init, the
onboarding prompt, and the three real workflows (contract checks plus their shell scripts run under
bash with a fake curl). Offline: no network, no real SMTP."""

from __future__ import annotations

import json
import os
import re
import shutil
import smtplib
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from digest.cli import main, run_init
from digest.profile import AgentAccess, Profile, load_profile
from digest.task_result import TaskResult

ROOT = Path(__file__).resolve().parent.parent
NOW = datetime(2026, 10, 20, 6, 0, tzinfo=UTC)
TASK = "01a18b17-aff2-7e5b-ac28-6d150dd0950c"
WORKFLOWS = ROOT / ".github" / "workflows"
THREE = ("register.yml", "issue-token.yml", "revoke-token.yml")


# ---- profile flag -----------------------------------------------------------------------


def test_flag_defaults_to_false_so_old_profiles_keep_working() -> None:
    assert AgentAccess().allow_owner_channel is False
    data = yaml.safe_load((ROOT / "profile.yaml").read_text(encoding="utf-8"))
    del data["agent_access"]["allow_owner_channel"]
    assert Profile.model_validate(data).agent_access.allow_owner_channel is False


def test_shipped_profile_sets_it_explicitly_to_false() -> None:
    profile = load_profile(ROOT / "profile.yaml")
    assert profile.agent_access.allow_owner_channel is False


def test_flag_can_be_enabled() -> None:
    assert AgentAccess(allow_owner_channel=True).allow_owner_channel is True


def test_unknown_agent_access_keys_are_still_rejected() -> None:
    with pytest.raises(ValueError):
        AgentAccess.model_validate({"allow_owner_chanel": True})


def test_exported_schema_has_the_flag() -> None:
    schema = Profile.model_json_schema()
    props = schema["$defs"]["AgentAccess"]["properties"]
    assert props["allow_owner_channel"]["default"] is False
    assert props["allow_owner_channel"]["type"] == "boolean"


def test_onboarding_prompt_lists_every_agent_access_field() -> None:
    text = (ROOT / "digest" / "prompts" / "onboarding_system.txt").read_text(encoding="utf-8")
    block = text.split("agent_access:\n", 1)[1].split("\n\n", 1)[0]
    keys = set(re.findall(r"^  (\w+):", block, re.MULTILINE))
    assert keys == set(AgentAccess.model_fields)
    assert "allow_owner_channel" in text.split("INTERVIEW FLOW", 1)[1].split("TOPIC SLUGS", 1)[0]


# ---- init -------------------------------------------------------------------------------


def _scripted(answers: dict[str, str]) -> Callable[[str], str]:
    def ask(prompt: str) -> str:
        for needle, value in answers.items():
            if needle in prompt:
                return value
        return ""

    return ask


@pytest.mark.parametrize(
    ("answer", "expected"), [("", False), ("n", False), ("y", True), ("Yes", True)]
)
def test_init_asks_about_owner_channel(tmp_path: Path, answer: str, expected: bool) -> None:
    out = tmp_path / "profile.yaml"
    code = run_init(_scripted({"also send the digest": answer}), out, force=False, now=NOW)
    assert code == 0
    assert load_profile(out).agent_access.allow_owner_channel is expected


# ---- Tier 3 owner_channel delivery ------------------------------------------------------


class FakeSmtp:
    sent: list[EmailMessage]

    def __init__(self, sent: list[EmailMessage], fail: bool = False) -> None:
        self.sent, self._fail = sent, fail

    def starttls(self) -> object:
        return None

    def login(self, user: str, password: str) -> object:
        return None

    def send_message(self, msg: EmailMessage) -> object:
        if self._fail:
            raise smtplib.SMTPException("boom")
        self.sent.append(msg)
        return None

    def quit(self) -> object:
        return None


def _no_network() -> httpx.Client:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected network call: {request.url}")

    return httpx.Client(transport=httpx.MockTransport(refuse))


def _feed(root: Path, *, empty: bool = False) -> Path:
    feed = root / "feed"
    (feed / "tags").mkdir(parents=True)
    item = {
        "id": "0123456789abcdef",
        "url": "https://example.com/a",
        "canonical_url": "https://example.com/a",
        "title": "A RAG release",
        "source_id": "example-blog",
        "published_at": "2026-10-19T10:00:00Z",
        "snippet": "A snippet about retrieval. " * 9,
        "tags": ["rag"],
        "entities": [],
        "base_score": 0.6,
    }
    shard = {"items": [] if empty else [item]}
    (feed / "tags" / "rag.json").write_text(json.dumps(shard), encoding="utf-8")
    (feed / "latest.json").write_text(json.dumps(shard), encoding="utf-8")
    (feed / "meta.json").write_text(
        json.dumps({"generated_at": "2026-10-20T03:00:00Z"}), encoding="utf-8"
    )
    return feed


def _profile(root: Path, *, flag: bool) -> Path:
    data = yaml.safe_load((ROOT / "profile.yaml").read_text(encoding="utf-8"))
    data["interests"] = {
        "topics": ["rag"],
        "follow_entities": [],
        "keywords_include": [],
        "keywords_exclude": [],
    }
    data["agent_access"]["allow_owner_channel"] = flag
    path = root / "profile.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


ENV = {
    "SMTP_USER": "bot@example.com",
    "SMTP_PASS": "super-secret-pass-123",
    "MAIL_TO": "me@example.com",
    "TASK_ID": TASK,
}


def _run(
    tmp_path: Path,
    *,
    flag: bool,
    params: str,
    fail: bool = False,
    empty: bool = False,
    dry: bool = False,
) -> tuple[int, list[EmailMessage], dict[str, Any]]:
    sent: list[EmailMessage] = []
    argv = [
        "run",
        "--profile",
        str(_profile(tmp_path, flag=flag)),
        "--feed",
        str(_feed(tmp_path, empty=empty)),
        "--state-dir",
        str(tmp_path / "state"),
        "--outbox",
        str(tmp_path / "outbox"),
        "--no-llm",
    ]
    if dry:
        argv.append("--dry-run")
    code = main(
        argv,
        env={**ENV, "TASK_PARAMS": params},
        now=NOW,
        smtp_factory=lambda h, p, t: FakeSmtp(sent, fail),
    )
    result = json.loads((tmp_path / "state" / "tasks" / f"{TASK}.json").read_text(encoding="utf-8"))
    TaskResult.model_validate(result)  # the file always satisfies the contract
    return code, sent, result


def test_flag_false_downgrades_with_a_warning_and_sends_nothing(tmp_path: Path) -> None:
    code, sent, result = _run(tmp_path, flag=False, params='{"delivery":"owner_channel"}')
    assert code == 0 and sent == []
    assert result["status"] == "completed" and len(result["items"]) == 1
    assert any("allow_owner_channel" in w for w in result["warnings"])


def test_flag_true_sends_through_the_owner_channel_and_still_writes_the_result(
    tmp_path: Path,
) -> None:
    code, sent, result = _run(tmp_path, flag=True, params='{"delivery":"owner_channel"}')
    assert code == 0
    assert len(sent) == 1 and sent[0]["To"] == "me@example.com"
    assert result["status"] == "completed" and len(result["items"]) == 1
    last = json.loads((tmp_path / "state" / "last_run.json").read_text(encoding="utf-8"))
    assert last["delivered"] == "owner_channel:email" and last["task_id"] == TASK


def test_tier3_never_marks_items_seen_even_when_delivering(tmp_path: Path) -> None:
    _run(tmp_path, flag=True, params='{"delivery":"owner_channel"}')
    assert not (tmp_path / "state" / "seen.json").exists()


def test_return_only_sends_nothing_even_with_the_flag(tmp_path: Path) -> None:
    code, sent, result = _run(tmp_path, flag=True, params='{"delivery":"return_only"}')
    assert code == 0 and sent == [] and result["warnings"] == []


def test_default_delivery_is_return_only(tmp_path: Path) -> None:
    _, sent, _ = _run(tmp_path, flag=True, params="{}")
    assert sent == []


def test_delivery_failure_keeps_the_result_and_never_leaks_secrets(tmp_path: Path) -> None:
    code, sent, result = _run(tmp_path, flag=True, params='{"delivery":"owner_channel"}', fail=True)
    assert code == 0 and sent == []
    assert result["status"] == "completed" and len(result["items"]) == 1
    assert any("owner_channel delivery failed" in w for w in result["warnings"])
    assert "super-secret-pass-123" not in json.dumps(result)


def test_nothing_to_deliver_adds_a_warning_and_sends_no_email(tmp_path: Path) -> None:
    code, sent, result = _run(
        tmp_path, flag=True, params='{"delivery":"owner_channel"}', empty=True
    )
    assert code == 0 and sent == [] and result["items"] == []
    assert any("nothing to deliver" in w for w in result["warnings"])


def test_dry_run_owner_channel_writes_to_the_outbox_instead(tmp_path: Path) -> None:
    code, sent, _ = _run(tmp_path, flag=True, params='{"delivery":"owner_channel"}', dry=True)
    assert code == 0 and sent == []
    assert list((tmp_path / "outbox").glob("*.eml"))


def test_unsupported_channel_is_a_warning_not_a_lost_run(tmp_path: Path) -> None:
    data = yaml.safe_load(_profile(tmp_path, flag=True).read_text(encoding="utf-8"))
    data["delivery"]["channel"] = "slack"
    (tmp_path / "profile.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    sent: list[EmailMessage] = []
    code = main(
        [
            "run",
            "--profile",
            str(tmp_path / "profile.yaml"),
            "--feed",
            str(_feed(tmp_path)),
            "--state-dir",
            str(tmp_path / "state"),
            "--no-llm",
        ],
        env={**ENV, "TASK_PARAMS": '{"delivery":"owner_channel"}'},
        now=NOW,
        http=_no_network(),
        smtp_factory=lambda h, p, t: FakeSmtp(sent),
    )
    result = json.loads((tmp_path / "state" / "tasks" / f"{TASK}.json").read_text(encoding="utf-8"))
    assert code == 0 and result["status"] == "completed"
    assert any("owner_channel delivery failed" in w for w in result["warnings"])


# ---- workflows: contract ----------------------------------------------------------------


def _load(name: str) -> dict[Any, Any]:
    loaded = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _runs(name: str) -> list[str]:
    jobs = _load(name)["jobs"]
    return [s["run"] for j in jobs.values() for s in j["steps"] if "run" in s]


def test_id_token_write_only_in_the_three_registration_workflows() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        perms = _load(path.name).get("permissions", {})
        has = isinstance(perms, dict) and perms.get("id-token") == "write"
        assert has == (path.name in THREE), path.name


@pytest.mark.parametrize("name", sorted(p.name for p in WORKFLOWS.glob("*.yml")))
def test_no_expression_inside_any_run_script(name: str) -> None:
    for script in _runs(name):
        assert "${{" not in script, name


@pytest.mark.parametrize("name", THREE)
def test_registration_workflows_are_least_privilege_and_guarded(name: str) -> None:
    wf = _load(name)
    assert wf["permissions"] == {"id-token": "write", "contents": "read"}
    trigger = wf[True]  # YAML parses the key `on` as True
    assert (set(trigger) if isinstance(trigger, dict) else {trigger}) == {"workflow_dispatch"}
    job = next(iter(wf["jobs"].values()))
    assert job["if"] == "github.repository != 'gowthamsai09/axon-template'"
    assert job["timeout-minutes"] <= 5
    raw = (WORKFLOWS / name).read_text(encoding="utf-8")
    assert "secrets." not in raw and "uses:" not in raw  # no secrets, no third-party actions to pin
    assert "persist-credentials" not in raw  # there is no checkout at all


def test_untrusted_inputs_reach_scripts_only_through_env() -> None:
    issue = _load("issue-token.yml")["jobs"]["issue"]["steps"][0]
    assert issue["env"]["LABEL"] == "${{ inputs.label }}"
    assert issue["env"]["EXPIRES_IN_DAYS"] == "${{ inputs.expires_in_days }}"
    revoke = _load("revoke-token.yml")["jobs"]["revoke"]["steps"][0]
    assert revoke["env"]["TOKEN_ID"] == "${{ inputs.token_id }}"


def test_token_is_masked_before_it_is_written_and_never_echoed() -> None:
    script = _runs("issue-token.yml")[0]
    assert script.index("::add-mask::$TOKEN") < script.index('>> "$GITHUB_STEP_SUMMARY"')
    for line in script.splitlines():
        if "$TOKEN" in line and "TOKEN_ID" not in line:
            assert not line.strip().startswith("echo ") or "add-mask" in line, line
    assert script.count("GITHUB_STEP_SUMMARY") == 1  # written once


def test_issue_token_never_prints_the_response_body() -> None:
    script = _runs("issue-token.yml")[0]
    assert "cat " not in script and "-o /dev/stdout" not in script
    assert 'rm -f "$RUNNER_TEMP/token.json"' in script


# ---- workflows: the scripts, run under bash with a fake curl ------------------------------

# The workflow scripts target ubuntu runners. They run only where a real POSIX bash and jq exist.
# On Windows `bash` is usually the WSL launcher (UTF-16 output, no distro) and PATH uses `;`, so
# they are skipped there (CI on ubuntu runs them). The contract tests above run everywhere.
needs_shell = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None or shutil.which("jq") is None,
    reason="needs a POSIX bash and jq (runs in CI on ubuntu)",
)

FAKE_CURL = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$FAKE_LOG"
case "$*" in *audience=*) printf '{"value":"aaa.bbb.ccc"}'; exit 0;; esac
out=""; args=("$@")
for ((i=0; i<${#args[@]}; i++)); do [ "${args[$i]}" = "-o" ] && out="${args[$((i+1))]}"; done
if [ -n "$out" ] && [ "$out" != /dev/null ]; then printf '%s' "$FAKE_BODY" > "$out"; fi
printf '%s' "$FAKE_CODE"
"""
TOKEN = "axn_" + "T" * 43


def _exec(
    tmp_path: Path, workflow: str, env: dict[str, str], *, code: str, body: str = "{}"
) -> tuple[subprocess.CompletedProcess[str], str, str]:
    script = _runs(workflow)[0]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(FAKE_CURL, encoding="utf-8")
    curl.chmod(0o755)
    log, summary = tmp_path / "curl.log", tmp_path / "summary.md"
    full = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "RUNNER_TEMP": str(tmp_path),
        "GITHUB_STEP_SUMMARY": str(summary),
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "req-token",
        "ACTIONS_ID_TOKEN_REQUEST_URL": "https://tok.test/idtoken?x=1",
        "FAKE_LOG": str(log),
        "FAKE_BODY": body,
        "FAKE_CODE": code,
        **env,
    }
    proc = subprocess.run(
        ["bash", "-c", script], env=full, capture_output=True, text=True, check=False
    )
    return (
        proc,
        log.read_text(encoding="utf-8") if log.exists() else "",
        summary.read_text(encoding="utf-8") if summary.exists() else "",
    )


ISSUE_BODY = json.dumps(
    {"token": TOKEN, "id": "tk_0123456789abcdef", "expires_at": "2027-01-01T00:00:00Z"}
)


@needs_shell
def test_issue_token_happy_path_writes_the_token_once_to_the_summary_only(tmp_path: Path) -> None:
    env = {"GATEWAY": "https://gw.example/", "LABEL": "my agent", "EXPIRES_IN_DAYS": "30"}
    proc, log, summary = _exec(tmp_path, "issue-token.yml", env, code="201", body=ISSUE_BODY)
    assert proc.returncode == 0, proc.stderr
    assert summary.count(TOKEN) == 1 and "tk_0123456789abcdef" in summary
    assert proc.stderr.count(TOKEN) == 0
    assert (
        proc.stdout.count(TOKEN) == 1 and f"::add-mask::{TOKEN}" in proc.stdout
    )  # only the mask command
    assert (
        "audience=https%3A%2F%2Fgw.example" in log
    )  # trailing slash stripped, audience url-encoded
    assert json.loads(
        next(line for line in log.splitlines() if "/tokens" in line).split("-d ", 1)[1]
    ) == {
        "label": "my agent",
        "expires_in_days": 30,
    }
    assert not (tmp_path / "token.json").exists()


@pytest.mark.parametrize("label", ["", "x" * 65, "bad/label", "a;b", "$(id)", "line\nbreak", "`x`"])
@needs_shell
def test_issue_token_rejects_bad_labels_before_any_request(tmp_path: Path, label: str) -> None:
    env = {"GATEWAY": "https://gw.example", "LABEL": label, "EXPIRES_IN_DAYS": "30"}
    proc, log, _ = _exec(tmp_path, "issue-token.yml", env, code="201", body=ISSUE_BODY)
    assert proc.returncode == 1 and log == ""


@pytest.mark.parametrize("days", ["", "0", "366", "-1", "1.5", "abc", "1000", "30; id"])
@needs_shell
def test_issue_token_rejects_bad_days_before_any_request(tmp_path: Path, days: str) -> None:
    env = {"GATEWAY": "https://gw.example", "LABEL": "ok", "EXPIRES_IN_DAYS": days}
    proc, log, _ = _exec(tmp_path, "issue-token.yml", env, code="201", body=ISSUE_BODY)
    assert proc.returncode == 1 and log == ""


@pytest.mark.parametrize("gateway", ["", "http://gw.example", "ftp://gw.example", "gw.example"])
@pytest.mark.parametrize("workflow", THREE)
@needs_shell
def test_non_https_gateway_is_refused(tmp_path: Path, workflow: str, gateway: str) -> None:
    env = {
        "GATEWAY": gateway,
        "LABEL": "ok",
        "EXPIRES_IN_DAYS": "30",
        "TOKEN_ID": "tk_0123456789abcdef",
    }
    proc, log, _ = _exec(tmp_path, workflow, env, code="200")
    assert proc.returncode == 1 and log == ""


@needs_shell
def test_issue_token_failure_does_not_leak_the_response(tmp_path: Path) -> None:
    env = {"GATEWAY": "https://gw.example", "LABEL": "ok", "EXPIRES_IN_DAYS": "30"}
    proc, _, summary = _exec(
        tmp_path, "issue-token.yml", env, code="403", body='{"error":"SECRET-BODY"}'
    )
    assert proc.returncode == 1 and summary == ""
    assert "SECRET-BODY" not in proc.stdout + proc.stderr and "HTTP 403" in proc.stderr
    assert not (tmp_path / "token.json").exists()


@needs_shell
def test_register_happy_path_and_failure(tmp_path: Path) -> None:
    ok = json.dumps({"repo": "alice/axon-digest", "agent_id": "ag_1", "default_branch": "main"})
    proc, log, _ = _exec(
        tmp_path, "register.yml", {"GATEWAY": "https://gw.example"}, code="200", body=ok
    )
    assert proc.returncode == 0 and "Registered alice/axon-digest as ag_1" in proc.stdout
    assert "-d {}" in log and "/register" in log
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    proc2, _, _ = _exec(
        bad_dir,
        "register.yml",
        {"GATEWAY": "https://gw.example"},
        code="502",
        body="UPSTREAM-DETAIL",
    )
    assert proc2.returncode == 1 and "UPSTREAM-DETAIL" not in proc2.stdout + proc2.stderr
    assert not (bad_dir / "register.json").exists()


@pytest.mark.parametrize(
    "token_id",
    [
        "",
        "tk_zz",
        "tk_0123456789abcde",
        "TK_0123456789abcdef",
        "../x",
        "tk_0123456789abcdef/../x",
        "tk_0123456789abcdef;id",
    ],
)
@needs_shell
def test_revoke_rejects_bad_ids_before_any_request(tmp_path: Path, token_id: str) -> None:
    proc, log, _ = _exec(
        tmp_path,
        "revoke-token.yml",
        {"GATEWAY": "https://gw.example", "TOKEN_ID": token_id},
        code="200",
    )
    assert proc.returncode == 1 and log == ""


@pytest.mark.parametrize(("code", "rc"), [("200", 0), ("404", 1), ("401", 1), ("500", 1)])
@needs_shell
def test_revoke_outcomes(tmp_path: Path, code: str, rc: int) -> None:
    proc, log, _ = _exec(
        tmp_path,
        "revoke-token.yml",
        {"GATEWAY": "https://gw.example", "TOKEN_ID": "tk_0123456789abcdef"},
        code=code,
    )
    assert proc.returncode == rc
    assert "-X DELETE https://gw.example/tokens/tk_0123456789abcdef" in log
