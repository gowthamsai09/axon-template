"""Hub corpus/hostile-text replayed against the real onboarding parser, vote reader and chat
renderers (Appendix C 101). Skipped only when the hub copy is absent and REQUIRE_HUB_CORPUS is
not 1 (CI sets it)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from digest import onboarding as o
from digest.cli import DEFAULT_MODELS, EXIT_FAIL, EXIT_OK, SUGGESTED_EMBEDDING_MODEL, main
from digest.delivery.common import units
from digest.delivery.slack import MAX_UNITS as SLACK_UNITS
from digest.delivery.slack import slack_chunks
from digest.delivery.telegram import MAX_UNITS as TELEGRAM_UNITS
from digest.delivery.telegram import telegram_chunks
from digest.feedback import eligible, parse_votes
from digest.pipeline import DigestEntry
from digest.profile import AgentAccess
from digest.render import build_view, render_slack, render_telegram
from digest.summarizers.base import Summary
from tests.conftest import NOW, ROOT, make_profile
from tests.onboard_helpers import body_from_corpus
from tests.test_rank import mk
from tests.test_render import run_of

HUB = Path(os.environ.get("HUB_DIR", str(ROOT.parent / "Axon")))
DIR = HUB / "corpus" / "hostile-text"
REQUIRED = os.environ.get("REQUIRE_HUB_CORPUS") == "1"


def load(name: str) -> list[dict[str, Any]]:
    path = DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


ONBOARDING, FEEDBACK, CHAT = load("onboarding"), load("feedback"), load("chat")
needs_corpus = pytest.mark.skipif(
    not (ONBOARDING and FEEDBACK and CHAT) and not REQUIRED,
    reason="hub corpus not found (set HUB_DIR)",
)


@needs_corpus
def test_the_corpus_was_really_loaded() -> None:
    assert len(ONBOARDING) >= 30 and len(FEEDBACK) >= 25 and len(CHAT) >= 15


# ---- onboarding -------------------------------------------------------------------------


def build(case: dict[str, Any]) -> o.Onboarded:
    return o.build(
        body_from_corpus(case),
        now=NOW,
        default_models=DEFAULT_MODELS,
        default_embedding_model=SUGGESTED_EMBEDDING_MODEL,
    )


@needs_corpus
@pytest.mark.parametrize("case", ONBOARDING, ids=lambda c: c["id"])
def test_onboarding_cases(case: dict[str, Any]) -> None:
    marker = case["marker"]
    if case["expect"] == "rejected":
        with pytest.raises(o.OnboardError) as caught:
            build(case)
        assert marker not in str(caught.value)
        assert marker not in o.comment_problems(caught.value.problems)
        return
    result = build(case)
    assert marker not in o.comment_ok(result)
    assert result.profile.agent_access == AgentAccess()  # nothing in an answer reaches this
    reloaded = yaml.safe_load(result.yaml_text)
    assert reloaded["agent_access"] == AgentAccess().model_dump(mode="json")
    assert set(reloaded) == {
        "version", "interests", "sources", "delivery", "summarizer", "embeddings", "trends",
        "feedback", "agent_access",
    }  # fmt: skip


@needs_corpus
@pytest.mark.parametrize("case", ONBOARDING, ids=lambda c: c["id"])
def test_onboarding_cli_never_prints_or_comments_the_marker(
    case: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile, comment = tmp_path / "profile.yaml", tmp_path / "comment.md"
    code = main(
        ["onboard", "--profile", str(profile), "--comment-file", str(comment)],
        env={"ISSUE_BODY": body_from_corpus(case)},
        now=NOW,
    )
    seen = capsys.readouterr()
    assert code == (EXIT_FAIL if case["expect"] == "rejected" else EXIT_OK)
    assert case["marker"] not in seen.out + seen.err
    assert case["marker"] not in comment.read_text(encoding="utf-8")
    assert profile.exists() is (case["expect"] == "accepted")


# ---- feedback ---------------------------------------------------------------------------


@needs_corpus
@pytest.mark.parametrize("case", FEEDBACK, ids=lambda c: c["id"])
def test_feedback_cases(case: dict[str, Any]) -> None:
    issue = case["issue"]
    counted = eligible(issue) and parse_votes(issue["body"]) is not None
    assert counted is (case["expect"] == "counted")


# ---- chat -------------------------------------------------------------------------------


def hostile_entry(text: str, ident: str = "a" * 16) -> DigestEntry:
    summary = Summary(
        status="ok", title="t", one_line=text[:200], key_points=[text[:240]],
        why_it_matters=text[:300], tags=[], entities=[], notable_figures=[],
        source_url="https://example.com/x", injection_flag=False,
    )  # fmt: skip
    return DigestEntry(
        mk(id_=ident, title=text[:500]), 0.9, summary=summary, provider="deepseek", model="m"
    )


def rendered(text: str) -> tuple[str, str]:
    entries = [hostile_entry(text, f"{i:016x}") for i in range(12)]
    view = build_view(run_of(*entries), make_profile(delivery={"max_items": 12}), NOW)
    return render_slack(view), render_telegram(view)


@needs_corpus
@pytest.mark.parametrize("case", CHAT, ids=lambda c: c["id"])
def test_chat_text_cannot_become_markup(case: dict[str, Any]) -> None:
    slack, telegram = rendered(case["text"])
    for needle in case["slack_forbidden"]:
        assert needle not in slack, needle
    for needle in case["telegram_forbidden"]:
        assert needle not in telegram, needle


@needs_corpus
@pytest.mark.parametrize("case", CHAT, ids=lambda c: c["id"])
def test_chat_chunks_stay_inside_the_limits(case: dict[str, Any]) -> None:
    slack, telegram = rendered(case["text"])
    assert all(units(c) <= SLACK_UNITS for c in slack_chunks(slack))
    assert all(units(c) <= TELEGRAM_UNITS for c in telegram_chunks(telegram))
