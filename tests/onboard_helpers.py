"""Builds Issue-form bodies the way GitHub renders them (### label, then the answer)."""

from __future__ import annotations

from typing import Any

from digest import onboarding as o
from digest.profile import TOPIC_SLUGS

NONE = o.NO_RESPONSE
VALID: dict[str, Any] = {
    o.L_TOPICS: ["rag", "agents-tooling"],
    o.L_ENTITIES: ["Qwen", "github:ollama/ollama"],
    o.L_INCLUDE: ["quantization"],
    o.L_EXCLUDE: ["crypto"],
    o.L_RSS: [],
    o.L_CHANNEL: "email",
    o.L_FREQUENCY: "daily",
    o.L_WEEKDAY: None,
    o.L_TIME: "07:30",
    o.L_ZONE: "Asia/Kolkata",
    o.L_DEPTH: "brief",
    o.L_MAX: "10",
    o.L_LANG: "en",
    o.L_PROVIDER: "deepseek",
    o.L_MODEL: None,
    o.L_BASE: None,
    o.L_EXTRAS: [],
    o.L_EPROVIDER: None,
    o.L_EMODEL: None,
    o.L_EBASE: None,
    o.L_ACK: [o.ACK],
}
CHECKS = {o.L_TOPICS: TOPIC_SLUGS, o.L_EXTRAS: o.EXTRAS, o.L_ACK: (o.ACK,)}


def section(label: str, value: Any) -> str:
    if label in CHECKS:
        chosen = set(value or [])
        rows = [f"- [{'X' if c in chosen else ' '}] {c}" for c in CHECKS[label]]
        return f"### {label}\n\n" + "\n".join(rows) + "\n\n"
    if isinstance(value, list):
        text = "\n".join(value) if value else NONE
    else:
        text = NONE if value is None else str(value)
    return f"### {label}\n\n{text}\n\n"


def body(**over: Any) -> str:
    """A valid form body; `over` maps a label name (L_*) key to a replacement value."""
    values = dict(VALID)
    for key, value in over.items():
        values[getattr(o, key)] = value
    return "".join(section(label, values[label]) for label in o.SPEC)


def body_from_corpus(case: dict[str, Any]) -> str:
    """A body for a hub corpus case: `set` maps form labels to answers (None clears one)."""
    if "raw" in case:
        return str(case["raw"])
    values = dict(VALID)
    values.update(case["set"])
    text = "".join(section(label, values[label]) for label in o.SPEC)
    extra = case.get("append")
    if extra is not None:
        if "literal" in extra:
            text += str(extra["literal"])
        else:
            text += str(extra["prefix"]) + str(extra["repeat"]) * int(extra["times"])
    return text
