"""Stage 23 odds and ends: regressions found while building, prompt invariants, and the
shipped profile.yaml with every Phase 6 section. Offline."""

from __future__ import annotations

import yaml

from digest.delivery.common import number
from digest.profile import Profile, load_profile
from digest.redact import redact, secret_values
from digest.summarizers.prompting import prompt_version
from tests.conftest import ROOT

SUMMARIZER_PROMPT_ID = "p-56a22986b1"  # locked: only a recorded, deliberate change moves it


def test_number_survives_a_json_integer_too_large_for_a_float() -> None:
    assert number(10**400) is None and number(-(10**400)) is None
    assert number(True) is None and number("5") is None and number(None) is None
    assert number(5) == 5.0 and number(2.5) == 2.5


def test_the_summarizer_prompt_is_unchanged_by_phase_6() -> None:
    assert prompt_version() == SUMMARIZER_PROMPT_ID


def test_the_shipped_profile_validates_and_keeps_every_extra_off() -> None:
    profile = load_profile(ROOT / "profile.yaml")
    assert profile.embeddings.enabled is False
    assert profile.trends.enabled is False and profile.trends.llm_labels is False
    assert profile.feedback.enabled is False
    assert profile.summarizer.model == "deepseek-flash"


def test_a_profile_without_the_new_sections_still_loads() -> None:
    data = yaml.safe_load((ROOT / "profile.yaml").read_text(encoding="utf-8"))
    for name in ("embeddings", "trends", "feedback"):
        del data[name]
    profile = Profile.model_validate(data)
    assert profile.embeddings.enabled is False and profile.feedback.enabled is False


def test_every_new_profile_field_is_documented_in_the_onboarding_prompt() -> None:
    text = (ROOT / "digest" / "prompts" / "onboarding_system.txt").read_text(encoding="utf-8")
    for field in ("embeddings:", "trends:", "feedback:", "min_similarity", "max_clusters",
                  "llm_labels", "EMBED_API_KEY", "rss"):  # fmt: skip
        assert field in text


def test_the_embeddings_key_is_a_known_secret_everywhere() -> None:
    value = "AXON-EMBED-SECRET-123456"
    assert value in secret_values({"EMBED_API_KEY": value})
    assert value not in redact(f"failed with {value}", secret_values({"EMBED_API_KEY": value}))
