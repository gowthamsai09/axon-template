"""One-client summarization (prompt 13.1): build the messages, call the model, validate, retry
once with a reminder, and map-reduce long documents (chunk summaries, then a merge pass that
sees only those summaries)."""

from __future__ import annotations

from ..profile import Depth
from .base import ChatClient, SourceItem, Summary, Usage
from .prompting import (
    RETRY_REMINDER,
    SINGLE_PASS_CHARS,
    chunk_text,
    document_for,
    expected_source_url,
    new_nonce,
    system_message,
    user_message,
)
from .validation import SummaryValidationError, parse_json_object, validate_summary


def _one(
    client: ChatClient,
    item: SourceItem,
    depth: Depth,
    language: str,
    document: str,
    source_text: str,
    usage: Usage,
    *,
    merge: bool = False,
) -> Summary:
    reminder = ""
    last: SummaryValidationError | None = None
    for _ in range(2):  # one retry with a reminder, then give up (spec 13.1)
        nonce = new_nonce()
        system = system_message(nonce)
        user = user_message(item, depth, language, nonce, document, merge=merge, reminder=reminder)
        result = client.complete(system, user)
        usage.add(result.usage)
        try:
            return validate_summary(
                parse_json_object(result.text),
                depth=depth,
                expected_url=expected_source_url(item),
                source_text=source_text,
            )
        except SummaryValidationError as exc:
            last = exc
            reminder = RETRY_REMINDER.format(why=f"The previous output was rejected: {exc}.")
    assert last is not None
    raise last


def _merge_document(partials: list[Summary]) -> str:
    lines = []
    for i, part in enumerate(partials, 1):
        points = " ".join(part.key_points)
        lines.append(f"{i}. {part.one_line} {points}".strip())
    return "\n".join(lines)


def summarize_with(
    client: ChatClient, item: SourceItem, depth: Depth, language: str, usage: Usage
) -> Summary:
    """Raises LLMError subclasses (transport) or SummaryValidationError (bad output)."""
    document = document_for(item)
    if len(document) <= SINGLE_PASS_CHARS:
        return _one(client, item, depth, language, document, document, usage)
    partials: list[Summary] = []
    for chunk in chunk_text(document):
        part = _one(client, item, depth, language, chunk, document, usage)
        if part.status == "ok":
            partials.append(part)
    if not partials:  # every chunk was empty or flagged: ask once on the first chunk's verdict
        return _one(client, item, depth, language, chunk_text(document)[0], document, usage)
    merge_doc = _merge_document(partials)
    return _one(client, item, depth, language, merge_doc, document, usage, merge=True)
