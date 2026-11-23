"""Phase 5 (template side): secret-echo guard (Appendix C 58), strict injection_flag."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from digest.summarizers.base import SourceItem, Summary, Usage
from digest.summarizers.llm import summarize_with
from digest.summarizers.router import SummarizerRouter, get_summarizer
from digest.summarizers.validation import SummaryValidationError, validate_summary
from tests.conftest import FakeChat, make_profile, summary_json

KEY = "sk-UNIQUE-primary-key-5f4e3d2c1b"
FALLBACK_KEY = "hf_UNIQUE_fallback_key_a1b2c3d4"
URL = "https://example.com/a"
ITEM = SourceItem("a" * 16, "Title", URL, "hn-ai", "d", "Body. " * 10)


def leaking(secret: str) -> Callable[[str], str]:
    return lambda user: summary_json(user, one_line=f"The key is {secret}")


def test_output_echoing_the_key_is_refused_after_one_retry() -> None:
    chat = FakeChat([leaking(KEY), leaking(KEY)])
    with pytest.raises(SummaryValidationError, match="output contains a secret"):
        summarize_with(chat, ITEM, "brief", "en", Usage(), None, (KEY,))
    assert len(chat.calls) == 2  # original plus exactly one retry, never a third


def test_the_retry_reminder_never_repeats_the_secret() -> None:
    chat = FakeChat([leaking(KEY), summary_json])
    summary = summarize_with(chat, ITEM, "brief", "en", Usage(), None, (KEY,))
    assert KEY not in chat.calls[1][1] and "output contains a secret" in chat.calls[1][1]
    assert KEY not in summary.model_dump_json()


def test_a_secret_inside_otherwise_invalid_output_is_caught_first() -> None:
    chat = FakeChat([f"not json but {KEY}", f"still not json {KEY}"])
    with pytest.raises(SummaryValidationError, match="output contains a secret"):
        summarize_with(chat, ITEM, "brief", "en", Usage(), None, (KEY,))


def test_without_forbidden_values_nothing_changes() -> None:
    chat = FakeChat([leaking("sk-whatever-1234567")])
    assert summarize_with(chat, ITEM, "brief", "en", Usage()).status == "ok"


def test_values_shorter_than_six_characters_are_ignored() -> None:
    chat = FakeChat([leaking("abc")])
    assert summarize_with(chat, ITEM, "brief", "en", Usage(), None, ("abc", "")).status == "ok"


@pytest.mark.parametrize("secret", [KEY, FALLBACK_KEY])
def test_router_degrades_the_item_and_leaks_nothing(secret: str) -> None:
    router = SummarizerRouter(
        FakeChat([leaking(secret), leaking(secret)]), forbidden=(KEY, FALLBACK_KEY)
    )
    result = router.summarize(ITEM, "brief", "en")
    assert result.summary is None and result.reason == "invalid model output"
    everything = " ".join([*result.warnings, *router.notices, result.reason])
    assert secret not in everything and "output contains a secret" in everything
    assert (
        router.summarize(ITEM, "brief", "en").summary is not None
    )  # no breaker trip, next item ok


def test_get_summarizer_hands_both_keys_to_the_router() -> None:
    profile = make_profile(
        summarizer={"fallback_provider": "huggingface", "fallback_model": "org/model"}
    )
    env = {"LLM_API_KEY": KEY, "LLM_API_KEY_FALLBACK": FALLBACK_KEY}
    router, _ = get_summarizer(profile, env, http=httpx.Client())
    assert set(router.forbidden) == {KEY, FALLBACK_KEY}
    only_primary, _ = get_summarizer(profile, {"LLM_API_KEY": KEY}, http=httpx.Client())
    assert only_primary.forbidden == (KEY,)


def good(**over: object) -> dict[str, object]:
    base: dict[str, object] = {
        "status": "ok", "title": "T", "one_line": "x", "key_points": [], "why_it_matters": "",
        "tags": [], "entities": [], "notable_figures": [], "source_url": URL,
        "injection_flag": False,
    }  # fmt: skip
    return {**base, **over}


@pytest.mark.parametrize("flag", ["false", "true", 0, 1, None])
def test_injection_flag_must_be_a_real_boolean(flag: object) -> None:
    with pytest.raises(SummaryValidationError, match="injection_flag"):
        validate_summary(good(injection_flag=flag), depth="brief", expected_url=URL, source_text="")


def test_a_real_boolean_still_validates() -> None:
    for flag in (True, False):
        out = validate_summary(
            good(injection_flag=flag), depth="brief", expected_url=URL, source_text=""
        )
        assert isinstance(out, Summary) and out.injection_flag is flag
