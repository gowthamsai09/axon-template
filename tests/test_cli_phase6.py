"""Phase 6 through the CLI: semantic matching (cache, fail-soft, key hygiene), weekly trends,
thumbs links, delivered.json, weights, and the `feedback` command. Offline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from digest.cli import main
from digest.state import StateStore
from digest.weights import FILE as WEIGHTS_FILE
from digest.weights import load_weights
from tests.conftest import FEED, NOW, summary_json
from tests.stage23_helpers import EMB_MODEL
from tests.test_cli import ENV, KEY, write_profile
from tests.test_email import FakeSmtp
from tests.test_feedback import REPO, TOKEN, GitHub, delivered, raw_issue

EKEY = "AXON-EMBED-KEY-909090"
ECHO = "UNIQUE-BODY-ECHO-6464"
GH_ENV = {"GITHUB_REPOSITORY": "o/r", "GITHUB_SERVER_URL": "https://github.com"}
EMB = {"enabled": True, "provider": "huggingface", "model": EMB_MODEL}


class Net:
    """Mock internet: DeepSeek summaries, Hugging Face embeddings, everything else 404."""

    def __init__(self, *, hf_status: int = 200) -> None:
        self.hf_status = hf_status
        self.hf: list[httpx.Request] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, req: httpx.Request) -> httpx.Response:
        if req.url.host == "api.deepseek.com":
            user = json.loads(req.content)["messages"][1]["content"]
            body = {
                "choices": [{"message": {"content": summary_json(user)}}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            }
            return httpx.Response(200, json=body)
        if req.url.host == "router.huggingface.co":
            self.hf.append(req)
            if self.hf_status != 200:
                return httpx.Response(self.hf_status, text=f"{ECHO} {EKEY}")
            inputs = json.loads(req.content)["inputs"]
            return httpx.Response(200, json=[[1.0, 0.5, 0.25] for _ in inputs])
        return httpx.Response(404)


def cli(
    tmp: Path,
    net: Net,
    *extra: str,
    env: dict[str, str] | None = None,
    channel: str = "rss",
    sections: Mapping[str, Mapping[str, object]] | None = None,
) -> int:
    tmp.mkdir(parents=True, exist_ok=True)
    parts = {k: dict(v) for k, v in (sections or {}).items()}
    parts["delivery"] = {"channel": channel, **parts.get("delivery", {})}
    profile = write_profile(tmp, **parts)
    argv = ["run", "--profile", str(profile), "--feed", str(FEED),
            "--state-dir", str(tmp / "state"), "--outbox", str(tmp / "outbox"), *extra]  # fmt: skip
    return main(
        argv,
        env=ENV if env is None else env,
        now=NOW,
        http=net.client,
        smtp_factory=lambda h, p, t: FakeSmtp(),
        resolver=lambda _h: ["93.184.216.34"],
        sleep=lambda _s: None,
    )


def last(tmp: Path) -> dict[str, Any]:
    data = json.loads((tmp / "state" / "last_run.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def everything(tmp: Path, capsys: pytest.CaptureFixture[str]) -> str:
    out = capsys.readouterr()
    files = [p for p in tmp.rglob("*") if p.is_file()]
    saved = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in files)
    return out.out + out.err + saved


# ---- semantic matching -----------------------------------------------------------------


def test_semantic_matching_embeds_once_then_serves_from_the_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    net, env = Net(), {**ENV, "EMBED_API_KEY": EKEY}
    assert cli(tmp_path, net, "--dry-run", env=env, sections={"embeddings": EMB}) == 0
    assert net.hf and all(r.headers["authorization"] == f"Bearer {EKEY}" for r in net.hf)
    assert all(r.url.host == "router.huggingface.co" for r in net.hf)
    usage = last(tmp_path)["usage"]
    assert usage["embedded"] > 0 and usage.get("embed_cache_hits", 0) == 0
    assert list((tmp_path / "state" / "embeddings").glob("*.json"))
    first_calls = len(net.hf)
    assert cli(tmp_path, net, "--dry-run", env=env, sections={"embeddings": EMB}) == 0
    assert len(net.hf) == first_calls  # nothing new to embed
    assert last(tmp_path)["usage"]["embed_cache_hits"] > 0
    assert EKEY not in everything(tmp_path, capsys)


def test_a_failing_embeddings_service_never_costs_the_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    net, env = Net(hf_status=503), {**ENV, "EMBED_API_KEY": EKEY}
    assert cli(tmp_path, net, env=env, sections={"embeddings": EMB}) == 0
    record = last(tmp_path)
    assert record["status"] == "ok" and record["items"] > 0
    assert any("semantic matching skipped, keywords only" in w for w in record["warnings"])
    text = everything(tmp_path, capsys)
    assert EKEY not in text and ECHO not in text


@pytest.mark.parametrize("status", [401, 402, 404, 429])
def test_embeddings_error_statuses_are_warnings_with_fixed_text(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], status: int
) -> None:
    net, env = Net(hf_status=status), {**ENV, "EMBED_API_KEY": EKEY}
    assert cli(tmp_path, net, "--dry-run", env=env, sections={"embeddings": EMB}) == 0
    warnings = " ".join(last(tmp_path)["warnings"])
    assert f"HTTP {status}" in warnings
    assert EKEY not in everything(tmp_path, capsys)


def test_a_missing_embed_key_turns_matching_off_with_a_clear_warning(tmp_path: Path) -> None:
    net = Net()
    assert cli(tmp_path, net, "--dry-run", sections={"embeddings": EMB}) == 0
    assert net.hf == []
    assert any("EMBED_API_KEY secret is not set" in w for w in last(tmp_path)["warnings"])


def test_no_llm_and_disabled_never_call_the_embeddings_service(tmp_path: Path) -> None:
    net, env = Net(), {**ENV, "EMBED_API_KEY": EKEY}
    assert cli(tmp_path, net, "--dry-run", "--no-llm", env=env, sections={"embeddings": EMB}) == 0
    assert cli(tmp_path / "b", net, "--dry-run", env=env) == 0
    assert net.hf == []


def test_a_private_openai_compatible_endpoint_is_refused(tmp_path: Path) -> None:
    net, env = Net(), {**ENV, "EMBED_API_KEY": EKEY}
    cfg = {
        "enabled": True, "provider": "openai_compatible", "model": "m",
        "base_url": "https://169.254.169.254/v1",
    }  # fmt: skip
    assert cli(tmp_path, net, "--dry-run", env=env, sections={"embeddings": cfg}) == 0
    assert any("rejected" in w for w in last(tmp_path)["warnings"])


# ---- thumbs links and delivered.json ---------------------------------------------------


def html_of(tmp: Path) -> str:
    (path,) = (tmp / "outbox").glob("*.html")
    return path.read_text(encoding="utf-8")


def test_thumbs_links_appear_only_when_feedback_is_on(tmp_path: Path) -> None:
    env = {**ENV, **GH_ENV}
    assert cli(tmp_path / "off", Net(), "--dry-run", "--no-llm", env=env, channel="email") == 0
    assert "issues/new" not in html_of(tmp_path / "off")
    on = {"feedback": {"enabled": True}}
    assert cli(tmp_path / "on", Net(), "--dry-run", "--no-llm", env=env, channel="email",
               sections=on) == 0  # fmt: skip
    assert "https://github.com/o/r/issues/new?title=feedback" in html_of(tmp_path / "on")


def test_a_real_run_records_what_was_delivered_a_dry_run_does_not(tmp_path: Path) -> None:
    assert cli(tmp_path / "dry", Net(), "--dry-run", "--no-llm") == 0
    assert not (tmp_path / "dry" / "state" / "delivered.json").exists()
    assert cli(tmp_path / "real", Net(), "--no-llm") == 0
    rows = json.loads((tmp_path / "real" / "state" / "delivered.json").read_text("utf-8"))
    assert len(rows) == 2 and all(r["tags"] and r["at"] for r in rows.values())
    assert set(rows) == set(StateStore(tmp_path / "real" / "state").seen())


def test_a_damaged_weights_file_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir()
    (state / WEIGHTS_FILE).write_text('{"version": 1, "topics": {"rag": NaN}}', "utf-8")
    on = {"feedback": {"enabled": True}}
    assert cli(tmp_path, Net(), "--dry-run", "--no-llm", sections=on) == 0
    assert any("weights.json" in w for w in last(tmp_path)["warnings"])


# ---- weekly trends ---------------------------------------------------------------------


def test_weekly_digests_carry_a_trends_section(tmp_path: Path) -> None:
    sections: dict[str, dict[str, object]] = {
        "trends": {"enabled": True},
        "delivery": {"frequency": "weekly", "weekday": "mon"},
    }
    assert cli(tmp_path, Net(), "--dry-run", "--no-llm", sections=sections) == 0
    (path,) = (tmp_path / "outbox").glob("*.xml")
    content = path.read_text(encoding="utf-8")
    assert "Trends this week" in content and "agents tooling" in content


def test_daily_digests_and_disabled_trends_have_none(tmp_path: Path) -> None:
    daily = {"trends": {"enabled": True}}
    assert cli(tmp_path / "daily", Net(), "--dry-run", "--no-llm", sections=daily) == 0
    off = {"delivery": {"frequency": "weekly", "weekday": "mon"}}
    assert cli(tmp_path / "off", Net(), "--dry-run", "--no-llm", sections=off) == 0
    for name in ("daily", "off"):
        (path,) = (tmp_path / name / "outbox").glob("*.xml")
        assert "Trends this week" not in path.read_text(encoding="utf-8")


# ---- the feedback command --------------------------------------------------------------


def feedback_cli(
    tmp: Path, gh: GitHub | None, *extra: str, env: dict[str, str] | None = None, on: bool = True
) -> int:
    profile = write_profile(tmp, feedback={"enabled": on})
    argv = ["feedback", "--profile", str(profile), "--state-dir", str(tmp / "state"), *extra]
    environment = {"GITHUB_REPOSITORY": REPO, "GITHUB_TOKEN": TOKEN} if env is None else env
    return main(argv, env=environment, now=NOW, http=gh.client if gh else None)


def test_feedback_is_off_by_default(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gh = GitHub([])
    assert feedback_cli(tmp_path, gh, on=False) == 0
    assert "feedback is off" in capsys.readouterr().out and gh.requests == []


def test_feedback_applies_votes_and_reports_without_echoing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    delivered(tmp_path / "state")
    a = "a" * 16
    gh = GitHub([raw_issue(1, f"item:{a} vote:up"), raw_issue(2, f"{ECHO} {TOKEN}")])
    assert feedback_cli(tmp_path, gh) == 0
    out = capsys.readouterr()
    assert "applied 1 vote(s) from 1 issue(s); 1 ignored" in out.out
    assert TOKEN not in out.out + out.err and ECHO not in out.out + out.err
    weights = load_weights(StateStore(tmp_path / "state"))
    assert weights is not None and weights.topics == {"rag": 0.1} and gh.closed == [1]


def test_feedback_dry_run_changes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    delivered(tmp_path / "state")
    gh = GitHub([raw_issue(1, f"item:{'a' * 16} vote:up")])
    assert feedback_cli(tmp_path, gh, "--dry-run") == 0
    assert "would apply 1 vote(s)" in capsys.readouterr().out
    assert gh.closed == [] and load_weights(StateStore(tmp_path / "state")) is None


def test_feedback_is_weekly_unless_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    delivered(tmp_path / "state")
    assert feedback_cli(tmp_path, GitHub([])) == 0
    capsys.readouterr()
    again = GitHub([raw_issue(1, f"item:{'a' * 16} vote:up")])
    assert feedback_cli(tmp_path, again) == 0
    assert "feedback skipped" in capsys.readouterr().out and again.requests == []
    assert feedback_cli(tmp_path, again, "--force") == 0 and again.closed == [1]


@pytest.mark.parametrize(
    "env",
    [{}, {"GITHUB_REPOSITORY": REPO}, {"GITHUB_TOKEN": TOKEN},
     {"GITHUB_REPOSITORY": "../evil", "GITHUB_TOKEN": TOKEN}],
)  # fmt: skip
def test_feedback_without_a_valid_repository_and_token_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], env: dict[str, str]
) -> None:
    gh = GitHub([])
    assert feedback_cli(tmp_path, gh, env=env) == 2
    err = capsys.readouterr().err
    assert "feedback not configured" in err and TOKEN not in err and gh.requests == []


def test_a_rejected_token_fails_with_a_fixed_message_and_no_echo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    gh = GitHub([])
    gh.fail = lambda _r: httpx.Response(401, text=f"{ECHO} {TOKEN}")
    assert feedback_cli(tmp_path, gh) == 1
    err = capsys.readouterr().err
    assert "feedback failed: GitHub rejected the GITHUB_TOKEN (HTTP 401)" in err
    assert TOKEN not in err and ECHO not in err


def test_a_close_failure_is_a_warning_on_stderr_with_exit_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    delivered(tmp_path / "state")
    gh = GitHub([raw_issue(1, f"item:{'a' * 16} vote:up")], close_fails={1})
    assert feedback_cli(tmp_path, gh) == 0
    out = capsys.readouterr()
    assert "WARNING: could not close issue #1" in out.err
    assert TOKEN not in out.out + out.err and ECHO not in out.out + out.err


def test_the_profile_can_switch_everything_on_and_still_validate(tmp_path: Path) -> None:
    path = write_profile(
        tmp_path, embeddings=EMB, trends={"enabled": True, "llm_labels": True},
        feedback={"enabled": True}, delivery={"frequency": "weekly", "weekday": "mon"},
    )  # fmt: skip
    assert main(["validate-profile", str(path)]) == 0
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["embeddings"]["enabled"] and data["trends"]["llm_labels"]
    assert KEY not in path.read_text(encoding="utf-8")
