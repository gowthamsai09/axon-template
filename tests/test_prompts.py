from __future__ import annotations

import re
from pathlib import Path

from digest.profile import TOPIC_SLUGS

PROMPTS = Path(__file__).resolve().parent.parent / "digest" / "prompts"
KEYS = [
    "status", "title", "one_line", "key_points", "why_it_matters", "tags", "entities",
    "notable_figures", "source_url", "injection_flag",
]  # fmt: skip


def test_prompts_present_and_fence_free() -> None:
    for name in ("summarizer_system.txt", "onboarding_system.txt"):
        text = (PROMPTS / name).read_text()
        assert len(text) > 500 and "```" not in text  # spec 13: copyable verbatim


def test_summarizer_prompt_contract() -> None:
    text = (PROMPTS / "summarizer_system.txt").read_text()
    assert text.count("{{NONCE}}") >= 2
    for key in KEYS:
        assert f'"{key}"' in text


def test_onboarding_slug_list_matches_code() -> None:
    text = (PROMPTS / "onboarding_system.txt").read_text()
    m = re.search(r"TOPIC SLUGS \(use only these\)\n(.+)\n", text)
    assert m is not None
    assert tuple(s.strip() for s in m.group(1).split(",")) == TOPIC_SLUGS


def test_onboarding_prompt_covers_v11_profile_rules() -> None:
    text = (PROMPTS / "onboarding_system.txt").read_text()
    assert "weekday" in text and "never openai_compatible" in text
