"""Strict validation of model output (spec 13.1 code-side requirements). Nothing here trusts
the model: JSON shape, caps, depth rules, URL allow-list, figures grounded in the source."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from ..profile import Depth
from .base import Summary

_URL = re.compile(r"(?:[A-Za-z][A-Za-z0-9+.\-]*://|www\.)[^\s\"'<>]+")
_MARKUP = re.compile(r"```|</?[A-Za-z][^>]*>|\]\(|^\s{0,3}#{1,6}\s|\*\*", re.MULTILINE)
_TRAIL = ".,;:)]}'\""
_DEPTH_LIMITS: dict[str, tuple[int, int]] = {
    "headlines": (0, 0),
    "brief": (3, 160),
    "detailed": (6, 240),
}


class SummaryValidationError(Exception):
    """The model output is rejected. The message is generic: no model text is echoed.

    `repairable` marks the one violation that can be fixed by deleting content (too many or
    too long key points); see `drop_overlong_key_points`."""

    def __init__(self, message: str, *, repairable: bool = False) -> None:
        super().__init__(message)
        self.repairable = repairable


def parse_json_object(text: str) -> dict[str, object]:
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[A-Za-z]*\s*|\s*```$", "", body).strip()
    try:
        data = json.loads(body)
    except ValueError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise SummaryValidationError("output is not JSON") from None
        try:
            data = json.loads(body[start : end + 1])
        except ValueError:
            raise SummaryValidationError("output is not valid JSON") from None
    if not isinstance(data, dict):
        raise SummaryValidationError("output is not a JSON object")
    return data


def _strings(summary: Summary) -> list[str]:
    values = [summary.title, summary.one_line, summary.why_it_matters, *summary.key_points]
    values += summary.entities
    for fig in summary.notable_figures:
        values += [fig.value, fig.context]
    return values


_FIELD_LIMITS = {"title": 300, "one_line": 200, "why_it_matters": 300}


def _length_checks(raw: dict[str, object], depth: Depth) -> None:
    """Specific, number-bearing messages for the length caps: the message is sent back to the
    model on the one retry, so it must say exactly which limit was broken. It contains only
    field names and numbers, never model text."""
    for name, limit in _FIELD_LIMITS.items():
        value = raw.get(name)
        if isinstance(value, str) and len(value) > limit:
            raise SummaryValidationError(f"{name} is too long (limit {limit} characters)")
    points = raw.get("key_points")
    if not isinstance(points, list):
        return
    max_points, max_len = _DEPTH_LIMITS[depth]
    if len(points) > max_points:
        raise SummaryValidationError(
            f"too many key_points for depth {depth} (limit {max_points}, got {len(points)})",
            repairable=True,
        )
    longest = max((len(p) for p in points if isinstance(p, str)), default=0)
    if longest > max_len:
        raise SummaryValidationError(
            f"a key point is too long for depth {depth} "
            f"(limit {max_len} characters, longest is {longest})",
            repairable=True,
        )


def drop_overlong_key_points(raw: dict[str, object], depth: Depth) -> dict[str, object]:
    """Last resort after the retry: delete key points over the length limit and any beyond the
    count limit (first ones kept). Pure deletion, so nothing is invented or edited; the result
    still goes through full validation."""
    max_points, max_len = _DEPTH_LIMITS[depth]
    points = raw.get("key_points")
    if not isinstance(points, list):
        return raw
    kept = [p for p in points if isinstance(p, str) and len(p) <= max_len][:max_points]
    return {**raw, "key_points": kept}


def validate_summary(
    raw: dict[str, object], *, depth: Depth, expected_url: str, source_text: str
) -> Summary:
    _length_checks(raw, depth)
    try:
        summary = Summary.model_validate(raw)
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) or "<root>" for e in exc.errors()})
        raise SummaryValidationError("schema violation in: " + ", ".join(fields[:5])) from None
    if summary.source_url != expected_url:
        raise SummaryValidationError("source_url differs from the task block")
    if summary.status != "ok":
        content = (summary.title, summary.one_line, summary.why_it_matters)
        if any(content) or summary.key_points or summary.tags or summary.notable_figures:
            raise SummaryValidationError("content fields must be empty when status is not ok")
        return summary
    if not summary.one_line.strip():
        raise SummaryValidationError("one_line is empty")
    for text in _strings(summary):
        if _MARKUP.search(text):
            raise SummaryValidationError("markup in a plain-text field")
        for match in _URL.findall(text):
            if match.rstrip(_TRAIL) != expected_url.rstrip(_TRAIL):
                raise SummaryValidationError("output contains a URL other than source_url")
    for fig in summary.notable_figures:
        if fig.value not in source_text:
            raise SummaryValidationError("a notable figure does not appear in the source")
    return summary
