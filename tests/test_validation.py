from __future__ import annotations

import json
from typing import Any

import pytest

from digest.profile import Depth
from digest.summarizers.base import Summary
from digest.summarizers.validation import (
    SummaryValidationError,
    parse_json_object,
    validate_summary,
)

URL = "https://example.com/a"
SOURCE = "Qwen3 has 235B parameters and a 128K context window."


def good(**over: Any) -> dict[str, object]:
    data: dict[str, object] = {
        "status": "ok", "title": "t", "one_line": "One line.", "key_points": ["a", "b"],
        "why_it_matters": "w", "tags": ["rag"], "entities": ["Qwen3"],
        "notable_figures": [{"value": "235B", "context": "parameters"}],
        "source_url": URL, "injection_flag": False,
    }  # fmt: skip
    data.update(over)
    return data


def check(raw: dict[str, object], depth: Depth = "brief") -> Summary:
    return validate_summary(raw, depth=depth, expected_url=URL, source_text=SOURCE)


def test_good_summary_passes() -> None:
    assert check(good()).one_line == "One line."


def test_parse_json_variants() -> None:
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 1} Hope it helps') == {"a": 1}
    for bad in ["", "no json here", "[1, 2]", "{broken", "```\n```"]:
        with pytest.raises(SummaryValidationError):
            parse_json_object(bad)


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"extra_key": 1}, "schema violation"),
        ({"status": "maybe"}, "schema violation"),
        ({"tags": ["not-a-slug"]}, "schema violation"),
        ({"tags": ["rag"] * 5}, "schema violation"),
        ({"one_line": "x" * 201}, "schema violation"),
        ({"source_url": "https://evil.example/"}, "source_url differs"),
        ({"key_points": ["a", "b", "c", "d"]}, "too many key_points"),
        ({"key_points": ["x" * 161]}, "too long"),
        ({"one_line": ""}, "one_line is empty"),
        ({"title": "<b>bold</b>"}, "markup"),
        ({"why_it_matters": "see ```code```"}, "markup"),
        ({"key_points": ["[link](https://example.com/a)"]}, "markup"),
        ({"one_line": "Read https://evil.example/x now"}, "URL other than source_url"),
        ({"one_line": "go to www.evil.example"}, "URL other than source_url"),
        ({"notable_figures": [{"value": "999B", "context": "invented"}]}, "does not appear"),
    ],
)
def test_rejections(over: dict[str, object], message: str) -> None:
    with pytest.raises(SummaryValidationError, match=message):
        check(good(**over))


def test_source_url_may_appear_in_text() -> None:
    check(good(one_line=f"See {URL}."))


def test_depth_rules() -> None:
    with pytest.raises(SummaryValidationError, match="depth headlines"):
        check(good(), "headlines")
    check(good(key_points=[]), "headlines")
    check(good(key_points=["x"] * 6), "detailed")
    with pytest.raises(SummaryValidationError):
        check(good(key_points=["x"] * 7), "detailed")


def test_non_ok_status_requires_empty_content() -> None:
    empty = good(title="", one_line="", key_points=[], why_it_matters="", tags=[],
                 entities=[], notable_figures=[], status="insufficient_content")  # fmt: skip
    assert check(empty).status == "insufficient_content"
    with pytest.raises(SummaryValidationError, match="must be empty"):
        check(good(status="suspected_injection"))
    flagged = {**empty, "status": "suspected_injection", "injection_flag": True}
    assert check(flagged).injection_flag is True


def test_errors_never_echo_model_text() -> None:
    secret = "sk-model-said-this-12345"
    for raw in ({"one_line": f"visit https://{secret}.example"}, {"extra": secret}):
        with pytest.raises(SummaryValidationError) as info:
            check(good(**raw))
        assert secret not in str(info.value)
    assert json.dumps(good())  # fixture itself is serialisable
