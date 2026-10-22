"""Prompt assembly for prompt 13.1 and the code-side hygiene it requires: a random nonce per
request, delimiter stripping, control and invisible Unicode removal, truncation, chunking."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import unicodedata
from pathlib import Path

from ..profile import TOPIC_SLUGS, Depth
from .base import SourceItem

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "summarizer_system.txt"
MAX_DOC_CHARS = 24_000
SINGLE_PASS_CHARS = 6_000
MAX_CHUNKS = 4
MERGE_LINE = (
    "merge_mode: true. The document block contains partial summaries of one article. "
    "Produce a single summary of the whole article."
)
RETRY_REMINDER = "Reminder: return valid JSON only, exactly matching the schema. {why}"

_DELIMITER = re.compile(r"<\s*/?\s*(?:untrusted_document|task)\b[^>]*>?", re.IGNORECASE)
_BLANKS = re.compile(r"\n{3,}")
_SPACES = re.compile(r"\s+")


def load_system_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def prompt_version(template: str | None = None) -> str:
    """Derived from the prompt text, so every edit invalidates the summary cache (spec 11.3)."""
    text = template if template is not None else load_system_template()
    return "p-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def new_nonce() -> str:
    return secrets.token_hex(16)


def _strip_hidden(text: str) -> str:
    return "".join(
        ch
        for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cf", "Cs", "Co"}
    )


def sanitize_text(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    text = _strip_hidden(text.replace("\r\n", "\n").replace("\r", "\n"))
    previous = None
    while previous != text:  # nested tricks like "</untr</untrusted_document>usted_document>"
        previous = text
        text = _DELIMITER.sub("", text)
    return _BLANKS.sub("\n\n", text).strip()[:max_chars]


def sanitize_meta(value: str, max_chars: int = 300) -> str:
    cleaned = _SPACES.sub(" ", _strip_hidden(value)).strip()
    return cleaned.replace("<", "").replace(">", "").replace('"', "")[:max_chars]


def system_message(nonce: str, template: str | None = None) -> str:
    return (template if template is not None else load_system_template()).replace(
        "{{NONCE}}", nonce
    )


def user_message(
    item: SourceItem,
    depth: Depth,
    language: str,
    nonce: str,
    document: str,
    *,
    merge: bool = False,
    reminder: str = "",
) -> str:
    task = [
        f"depth: {depth}",
        f"language: {language}",
        f"allowed_tags: {json.dumps(list(TOPIC_SLUGS))}",
        f"source_url: {expected_source_url(item)}",
        f"source_name: {sanitize_meta(item.source_name, 64)}",
        f"published_at: {sanitize_meta(item.published_at, 40)}",
    ]
    if merge:
        task.append(MERGE_LINE)
    parts = [
        "<task>",
        *task,
        "</task>",
        "",
        f'<untrusted_document nonce="{nonce}">',
        document,
        f'</untrusted_document nonce="{nonce}">',
    ]
    if reminder:
        parts += ["", reminder]
    return "\n".join(parts)


def expected_source_url(item: SourceItem) -> str:
    return sanitize_meta(item.url, 2048)


def document_for(item: SourceItem) -> str:
    return sanitize_text(f"{item.title}\n\n{item.text}")


def chunk_text(text: str, size: int = SINGLE_PASS_CHARS, max_chunks: int = MAX_CHUNKS) -> list[str]:
    chunks: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        while len(para) > size:  # hard split of a very long paragraph
            if current:
                chunks.append(current)
                current = ""
            chunks.append(para[:size])
            para = para[size:]
        if current and len(current) + len(para) + 2 > size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks[:max_chunks]
