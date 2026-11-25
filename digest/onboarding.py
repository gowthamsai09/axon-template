"""Onboarding issue form to profile.yaml (spec 6.8, Phase 6, Appendix C 101).

The form is `.github/ISSUE_TEMPLATE/onboarding.yml`. GitHub turns it into an Issue body made of
`### <label>` sections. That body is untrusted text: it is cut into sections by the fixed labels
below, every value is checked against a strict shape, and the result goes through the real
Profile model before anything is written. Messages never repeat what the user typed; they name
the form field (our own fixed label) and say it is missing or not valid, so hostile text cannot
travel into the comment the workflow posts."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .profile import TOPIC_SLUGS, Profile, ProfileError, load_profile
from .schedule import cron_utc

MAX_BODY = 20_000
MAX_VALUE = 4_000
MAX_LINES = 50
MAX_LINE = 2_048
NO_RESPONSE = "_No response_"
HEADING = re.compile(r"^### (.+?)[ \t]*$")
CHECK = re.compile(r"^- \[([ xX])\] (.+?)[ \t]*$")
BULLET = re.compile(r"^[-*][ \t]+")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

CHANNELS = ("email", "telegram", "slack", "rss")
FREQUENCIES = ("daily", "weekly")
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
DEPTHS = ("headlines", "brief", "detailed")
PROVIDERS = ("deepseek", "huggingface", "openai_compatible")
EMBED_PROVIDERS = ("huggingface", "openai_compatible")
EXTRAS = (
    "Semantic matching (embeddings)",
    "Weekly trends section",
    "Thumbs up/down feedback links",
)
ACK = "I will not put API keys, tokens or passwords in this form"

# kind: text (one line), choice, list (one per line), checks (checkbox list)
L_TOPICS = "Topics"
L_ENTITIES = "Companies, models or repos to follow"
L_INCLUDE = "Keywords to include"
L_EXCLUDE = "Keywords to exclude"
L_RSS = "Extra RSS feeds"
L_CHANNEL = "Delivery channel"
L_FREQUENCY = "Frequency"
L_WEEKDAY = "Weekday (weekly only)"
L_TIME = "Send time (24h, local)"
L_ZONE = "Timezone (IANA name)"
L_DEPTH = "Depth"
L_MAX = "Maximum items per digest"
L_LANG = "Language code"
L_PROVIDER = "Summarizer provider"
L_MODEL = "Summarizer model"
L_BASE = "Summarizer base URL (openai_compatible only)"
L_EXTRAS = "Optional extras"
L_EPROVIDER = "Embeddings provider"
L_EMODEL = "Embeddings model"
L_EBASE = "Embeddings base URL (openai_compatible only)"
L_ACK = "Secrets"

SPEC: dict[str, tuple[str, tuple[str, ...], bool]] = {
    # label: (kind, allowed values for choice/checks, required)
    L_TOPICS: ("checks", TOPIC_SLUGS, False),
    L_ENTITIES: ("list", (), False),
    L_INCLUDE: ("list", (), False),
    L_EXCLUDE: ("list", (), False),
    L_RSS: ("list", (), False),
    L_CHANNEL: ("choice", CHANNELS, True),
    L_FREQUENCY: ("choice", FREQUENCIES, True),
    L_WEEKDAY: ("choice", WEEKDAYS, False),
    L_TIME: ("text", (), True),
    L_ZONE: ("text", (), True),
    L_DEPTH: ("choice", DEPTHS, True),
    L_MAX: ("text", (), True),
    L_LANG: ("text", (), True),
    L_PROVIDER: ("choice", PROVIDERS, True),
    L_MODEL: ("text", (), False),
    L_BASE: ("text", (), False),
    L_EXTRAS: ("checks", EXTRAS, False),
    L_EPROVIDER: ("choice", EMBED_PROVIDERS, False),
    L_EMODEL: ("text", (), False),
    L_EBASE: ("text", (), False),
    L_ACK: ("checks", (ACK,), True),
}
# Which form field a Profile validation problem belongs to (first matching prefix wins).
LOC_LABELS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("interests", "topics"), L_TOPICS),
    (("interests", "follow_entities"), L_ENTITIES),
    (("interests", "keywords_include"), L_INCLUDE),
    (("interests", "keywords_exclude"), L_EXCLUDE),
    (("interests",), L_INCLUDE),
    (("sources",), L_RSS),
    (("delivery", "channel"), L_CHANNEL),
    (("delivery", "frequency"), L_FREQUENCY),
    (("delivery", "weekday"), L_WEEKDAY),
    (("delivery", "send_time_local"), L_TIME),
    (("delivery", "timezone"), L_ZONE),
    (("delivery", "depth"), L_DEPTH),
    (("delivery", "max_items"), L_MAX),
    (("delivery", "language"), L_LANG),
    (("delivery",), L_FREQUENCY),
    (("summarizer", "provider"), L_PROVIDER),
    (("summarizer", "model"), L_MODEL),
    (("summarizer", "base_url"), L_BASE),
    (("summarizer",), L_BASE),
    (("embeddings", "provider"), L_EPROVIDER),
    (("embeddings", "model"), L_EMODEL),
    (("embeddings", "base_url"), L_EBASE),
    (("embeddings",), L_EBASE),
)
SECRET_FOR_CHANNEL = {
    "email": ("SMTP_USER", "SMTP_PASS", "MAIL_TO"),
    "telegram": ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"),
    "slack": ("SLACK_WEBHOOK_URL",),
    "rss": (),
}


class OnboardError(Exception):
    """The form could not be turned into a valid profile. `problems` are fixed texts."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class Onboarded:
    profile: Profile
    yaml_text: str
    cron: str
    secrets: tuple[str, ...]


Sections = dict[str, list[str]]


def split_sections(body: str) -> tuple[Sections, list[str]]:
    """Known `### label` sections (their lines), and problems such as a repeated section."""
    sections: Sections = {}
    problems: list[str] = []
    current: list[str] | None = None
    for line in body.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        match = HEADING.match(line)
        if match is not None and match.group(1) not in SPEC:
            current = None  # an unknown heading ends the section; its text is ignored
            continue
        if match is not None:
            label = match.group(1)
            if label in sections:
                problems.append(f"{label}: appears more than once")
                current = None
                continue
            current = sections.setdefault(label, [])
            continue
        if current is not None:
            current.append(line)
    return sections, problems


def _clean(lines: list[str]) -> list[str]:
    stripped = [ln.strip() for ln in lines]
    if stripped == [] or [ln for ln in stripped if ln] == [NO_RESPONSE]:
        return []
    return [ln for ln in stripped if ln]


def _text(label: str, lines: list[str], problems: list[str]) -> str:
    if len(lines) > 1 or any(len(ln) > MAX_VALUE or _CTRL.search(ln) for ln in lines):
        problems.append(f"{label}: must be one short line")
        return ""
    return lines[0] if lines else ""


def _items(label: str, lines: list[str], problems: list[str]) -> list[str]:
    out = [BULLET.sub("", ln) for ln in lines]
    if len(out) > MAX_LINES or any(len(v) > MAX_LINE or _CTRL.search(v) for v in out):
        problems.append(f"{label}: too many or too long entries")
        return []
    return out


def _checked(
    label: str, lines: list[str], allowed: tuple[str, ...], problems: list[str]
) -> set[str]:
    chosen: set[str] = set()
    for ln in lines:
        match = CHECK.match(ln)
        if match is None or match.group(2) not in allowed:
            problems.append(f"{label}: not a listed option")
            return set()
        if match.group(1) != " ":
            chosen.add(match.group(2))
    return chosen


def read_form(body: str) -> tuple[dict[str, Any], list[str]]:
    """Typed values by label, and fixed-text problems. Nothing here echoes the body."""
    if len(body) > MAX_BODY:
        return {}, ["The form is too large"]
    sections, problems = split_sections(body)
    values: dict[str, Any] = {}
    for label, (kind, allowed, required) in SPEC.items():
        lines = _clean(sections.get(label, []))
        if label not in sections or (not lines and required):
            if required:
                problems.append(f"{label}: is required")
            continue
        if kind == "text":
            values[label] = _text(label, lines, problems)
        elif kind == "choice":
            text = _text(label, lines, problems)
            if text and text not in allowed:
                problems.append(f"{label}: not a listed option")
            elif text:
                values[label] = text
        elif kind == "list":
            values[label] = _items(label, lines, problems)
        else:
            values[label] = _checked(label, lines, allowed, problems)
    if ACK not in values.get(L_ACK, set()):
        problems.append(f"{L_ACK}: tick the box to confirm no keys are in this form")
    return values, problems


def _problem_label(loc: tuple[Any, ...]) -> str:
    path = tuple(str(p) for p in loc)
    for prefix, label in LOC_LABELS:
        if path[: len(prefix)] == prefix:
            return label
    return "The form"


def profile_data(
    values: Mapping[str, Any],
    *,
    default_models: Mapping[str, str],
    default_embedding_model: str,
    agent_access: Mapping[str, Any] | None,
) -> dict[str, Any]:
    extras: set[str] = values.get(L_EXTRAS, set())
    provider = values.get(L_PROVIDER, "")
    weekly = values.get(L_FREQUENCY) == "weekly"
    data: dict[str, Any] = {
        "version": 1,
        "interests": {
            "topics": [t for t in TOPIC_SLUGS if t in values.get(L_TOPICS, set())],
            "follow_entities": values.get(L_ENTITIES, []),
            "keywords_include": values.get(L_INCLUDE, []),
            "keywords_exclude": values.get(L_EXCLUDE, []),
        },
        "sources": {"use_shared_feed": True, "extra_rss": values.get(L_RSS, [])},
        "delivery": {
            "channel": values.get(L_CHANNEL, ""),
            "frequency": values.get(L_FREQUENCY, ""),
            "send_time_local": values.get(L_TIME, ""),
            "timezone": values.get(L_ZONE, ""),
            "depth": values.get(L_DEPTH, ""),
            "max_items": _int(values.get(L_MAX, "")),
            "language": values.get(L_LANG, ""),
        },
        "summarizer": {
            "provider": provider,
            "model": values.get(L_MODEL) or default_models.get(provider, ""),
            "base_url": values.get(L_BASE, ""),
        },
        "agent_access": dict(agent_access or {}),
    }
    if weekly:
        data["delivery"]["weekday"] = values.get(L_WEEKDAY, "")
    if "Semantic matching (embeddings)" in extras:
        data["embeddings"] = {
            "enabled": True,
            "provider": values.get(L_EPROVIDER) or "huggingface",
            "model": values.get(L_EMODEL) or default_embedding_model,
            "base_url": values.get(L_EBASE, ""),
        }
    data["trends"] = {"enabled": "Weekly trends section" in extras}
    data["feedback"] = {"enabled": "Thumbs up/down feedback links" in extras}
    return data


def _int(text: object) -> object:
    """An ASCII-digit integer, else the text itself (so the model reports it)."""
    if isinstance(text, str) and re.fullmatch(r"[0-9]{1,3}", text):
        return int(text)
    return text


def existing_agent_access(path: Path) -> dict[str, Any] | None:
    """The agent_access section of a valid profile already in the repo, so a re-run of the
    form never silently resets what the owner changed there."""
    try:
        current = load_profile(path)
    except (ProfileError, OSError):
        return None
    return current.agent_access.model_dump(mode="json")


def build(
    body: str,
    *,
    now: datetime,
    default_models: Mapping[str, str],
    default_embedding_model: str,
    agent_access: Mapping[str, Any] | None = None,
) -> Onboarded:
    """Validate the issue body and return the profile. Raises OnboardError (fixed texts)."""
    values, problems = read_form(body)
    if problems:
        raise OnboardError(problems)
    data = profile_data(
        values,
        default_models=default_models,
        default_embedding_model=default_embedding_model,
        agent_access=agent_access,
    )
    try:
        profile = Profile.model_validate(data)
    except ValidationError as exc:
        seen: list[str] = []
        for err in exc.errors():
            label = _problem_label(tuple(err["loc"]))
            if label not in seen:
                seen.append(label)
        raise OnboardError([f"{label}: is missing or not valid" for label in seen]) from None
    header = "# Axon profile. Secrets go in GitHub Settings > Secrets, never in this file.\n"
    text = header + yaml.safe_dump(
        profile.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True
    )
    needed = ["LLM_API_KEY", *SECRET_FOR_CHANNEL[profile.delivery.channel]]
    if profile.embeddings.enabled:
        needed.append("EMBED_API_KEY")
    return Onboarded(profile, text, cron_utc(profile.delivery, now), tuple(needed))


def comment_ok(result: Onboarded) -> str:
    """The reply for a valid form: fixed text plus values the model already validated."""
    secrets = "\n".join(f"   - `{name}`" for name in result.secrets)
    return (
        "Axon onboarding: your answers are valid and `profile.yaml` was updated.\n\n"
        "**Next steps**\n"
        "1. In `.github/workflows/digest.yml`, set the `cron:` line under `schedule:` to "
        f'`"{result.cron}"` (UTC). A workflow cannot edit workflow files, so this one is yours.\n'
        "2. Add these secrets in Settings > Secrets and variables > Actions "
        "(names only here; never post a key in an Issue):\n"
        f"{secrets}\n"
        "3. Set the repository variable `FEED_BASE`, then run the `digest` workflow once.\n\n"
        "This Issue is closed. Edit `profile.yaml` directly for later changes.\n"
    )


def comment_problems(problems: list[str]) -> str:
    lines = "\n".join(f"- {p}" for p in problems)
    return (
        "Axon onboarding: the form could not be turned into a profile. "
        "Nothing was changed. Edit this Issue and fix:\n\n"
        f"{lines}\n\nSaving the edit runs the check again.\n"
    )


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
