"""Shared hostile-document corpus (hub corpus/injection, spec 11.3) against the REAL sanitizer,
prompt assembly and output validation. No LLM. Skipped only when the hub copy is absent and
REQUIRE_HUB_CORPUS is not 1 (CI sets it)."""

from __future__ import annotations

import json
import os
import random
import re
import unicodedata
from pathlib import Path
from typing import Any

import pytest

from digest.summarizers.base import SourceItem
from digest.summarizers.prompting import (
    MAX_DOC_CHARS,
    document_for,
    expected_source_url,
    sanitize_text,
    user_message,
)
from digest.summarizers.validation import (
    SummaryValidationError,
    parse_json_object,
    validate_summary,
)

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(os.environ.get("HUB_CORPUS_DIR", str(ROOT.parent / "Axon" / "corpus" / "injection")))
REQUIRED = os.environ.get("REQUIRE_HUB_CORPUS") == "1"
NONCE = "0123456789abcdef0123456789abcdef"
TAGS = re.compile(r"<\s*/?\s*(?:untrusted_document|task)\b", re.IGNORECASE)


def load() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(CORPUS.glob("*.json"))]


CASES = load() if CORPUS.is_dir() else []
BASE = next((c for c in CASES if c["kind"] == "base"), {})
DOCS = [c for c in CASES if c["kind"] == "document"]
OUTS = [c for c in CASES if c["kind"] == "output"]
METAS = [c for c in CASES if c["kind"] == "meta"]
pytestmark = pytest.mark.skipif(
    not CASES and not REQUIRED, reason="hub corpus not found (set HUB_CORPUS_DIR)"
)


def hidden(text: str) -> list[str]:
    return [
        c for c in text if c not in "\n\t" and unicodedata.category(c) in {"Cc", "Cf", "Cs", "Co"}
    ]


def test_the_corpus_was_really_loaded() -> None:
    assert len(DOCS) >= 30 and len(OUTS) >= 6 and BASE and len(METAS) == 1


def item(text: str, title: str = "Corpus case") -> SourceItem:
    return SourceItem(
        "a" * 16, title, "https://example.com/a", "hn-ai", "2026-10-20T00:00:00Z", text
    )


@pytest.mark.parametrize("case", DOCS, ids=lambda c: c["id"])
def test_document_cases(case: dict[str, Any]) -> None:
    raw = case["text"] * case["text_repeat"]
    it = item(raw, case["title"])
    document = document_for(it)
    msg = user_message(it, "brief", "en", NONCE, document)
    expect = case["expect"]
    # invariants for every hostile document
    assert TAGS.search(document) is None
    assert hidden(document) == [] and len(document) <= MAX_DOC_CHARS
    assert msg.count(f'<untrusted_document nonce="{NONCE}">') == 1
    assert msg.count(f'</untrusted_document nonce="{NONCE}">') == 1
    assert msg.count("<task>") == 1 and msg.count("</task>") == 1
    head = msg[: msg.index("<untrusted_document")]
    assert re.findall(r"^source_url: (.*)$", head, re.MULTILINE) == [expected_source_url(it)]
    # per-case expectations
    for needle in expect["forbid_ci"]:
        assert needle.lower() not in document.lower(), needle
    for needle in expect["require"]:
        assert needle in document, needle
        assert needle in msg[msg.index("<untrusted_document") :]  # data, inside the block
    if expect["outcome"] == "sanitized":
        assert document != raw.strip()  # the case really exercised the sanitizer
    if expect["outcome"] == "truncated":
        assert len(raw) > MAX_DOC_CHARS and len(document) == MAX_DOC_CHARS


@pytest.mark.parametrize("case", METAS, ids=lambda c: c["id"])
def test_hostile_task_block_fields_stay_single_line(case: dict[str, Any]) -> None:
    it = SourceItem(
        "a" * 16,
        case["title"],
        case["url"],
        case["source_name"],
        case["published_at"],
        case["text"],
    )
    msg = user_message(it, "brief", "en", NONCE, document_for(it))
    head = msg[: msg.index("\n\n<untrusted_document")]
    lines = head.split("\n")
    assert lines[0] == "<task>" and lines[-1] == "</task>" and len(lines) == 8
    assert head.count("<") == 2 and head.count(">") == 2
    assert all('"' not in ln for ln in lines[4:-1])  # the three fields that carry caller text
    prefixes = [ln.split(":", 1)[0] for ln in lines[1:-1]]
    assert prefixes == [
        "depth",
        "language",
        "allowed_tags",
        "source_url",
        "source_name",
        "published_at",
    ]


def build_output(case: dict[str, Any]) -> dict[str, object]:
    good: dict[str, Any] = dict(BASE["good_summary"])
    if case["raw_text"] is not None:
        return parse_json_object(case["raw_text"].replace("{JSON}", json.dumps(good)))
    for key, value in case["patch"].items():
        if value is None:
            good.pop(key, None)
        else:
            good[key] = value
    return good


@pytest.mark.parametrize("case", OUTS, ids=lambda c: c["id"])
def test_output_cases(case: dict[str, Any]) -> None:
    expect = case["expect"]

    def run() -> object:
        return validate_summary(
            build_output(case), depth=case["depth"], expected_url=BASE["source_url"],
            source_text=BASE["source_text"],
        )  # fmt: skip

    if expect["outcome"] == "accepted":
        assert run() is not None
    else:
        with pytest.raises(SummaryValidationError, match=re.escape(expect["reason"])):
            run()


def test_delimiter_stripping_fuzz() -> None:
    """Random splices of delimiter fragments never leave a delimiter behind (seeded, 3000 runs)."""
    rng = random.Random(20261101)
    frags = ["<", "</", "/", ">", " ", "\n", "untrusted_document", "UNTRUSTED_DOCUMENT", "task",
             "Task", ' nonce="x"', "nonce=", "<task>", "</task>", "</untrusted_document>",
             "\u200b", "\u202e", "ab", "x", "<ta", "sk>", "<untr", "usted_document>"]  # fmt: skip
    for _ in range(3000):
        text = "".join(rng.choice(frags) for _ in range(rng.randint(2, 14)))
        out = sanitize_text(text)
        assert TAGS.search(out) is None, repr(text)
        assert hidden(out) == [], repr(text)
