"""profile.yaml model (spec 6.1) with the validation rules from 6.1 and the onboarding
prompt (13.2). Fail fast with readable errors; no LLM is involved."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal, Self, get_args
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Keep in sync with axon-feed models/common.py; hub CI checks enum parity across schemas.
TopicSlug = Literal[
    "llm-releases",
    "open-source-llms",
    "small-local-models",
    "multimodal",
    "agents-tooling",
    "rag",
    "fine-tuning-training",
    "research-papers",
    "mlops-infra",
    "ai-hardware",
    "evals-benchmarks",
    "safety-alignment",
    "ai-policy-regulation",
    "ai-products-apps",
    "datasets",
    "robotics",
]
TOPIC_SLUGS: tuple[str, ...] = get_args(TopicSlug)

Provider = Literal["deepseek", "huggingface", "openai_compatible"]
FallbackProvider = Literal["", "deepseek", "huggingface"]  # openai_compatible: see Spec deltas
Channel = Literal["email", "telegram", "slack", "rss"]
Frequency = Literal["daily", "weekly"]
Depth = Literal["headlines", "brief", "detailed"]
Mode = Literal["data", "summary", "digest"]
Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

Term = Annotated[str, Field(min_length=1, max_length=64)]
_GITHUB_ENTITY = re.compile(r"^github:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _unique(values: list[str], what: str) -> list[str]:
    folded = [v.casefold() for v in values]
    if len(set(folded)) != len(folded):
        raise ValueError(f"duplicate {what}")
    return values


def _no_control(values: list[str]) -> list[str]:
    if any(_CTRL.search(v) for v in values):
        raise ValueError("control characters are not allowed")
    return values


class Interests(Strict):
    topics: list[TopicSlug] = Field(default_factory=list, max_length=16)
    follow_entities: list[Term] = Field(default_factory=list, max_length=50)
    keywords_include: list[Term] = Field(default_factory=list, max_length=50)
    keywords_exclude: list[Term] = Field(default_factory=list, max_length=50)

    @field_validator("follow_entities")
    @classmethod
    def _entities(cls, v: list[str]) -> list[str]:
        for e in v:
            if e.startswith("github:") and not _GITHUB_ENTITY.match(e):
                raise ValueError(f"{e!r}: use github:owner/repo")
        return _no_control(_unique(v, "entities"))

    @field_validator("keywords_include", "keywords_exclude")
    @classmethod
    def _keywords(cls, v: list[str]) -> list[str]:
        return _no_control(_unique(v, "keywords"))

    @field_validator("topics")
    @classmethod
    def _topics(cls, v: list[str]) -> list[str]:
        return _unique(v, "topics")

    @model_validator(mode="after")
    def _no_overlap(self) -> Self:
        both = {k.casefold() for k in self.keywords_include} & {
            k.casefold() for k in self.keywords_exclude
        }
        if both:
            raise ValueError(f"keywords both included and excluded: {sorted(both)}")
        return self


class Sources(Strict):
    use_shared_feed: bool = True
    extra_rss: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("extra_rss")
    @classmethod
    def _urls(cls, v: list[str]) -> list[str]:
        for url in v:
            parts = urlsplit(url)
            if len(url) > 2048 or _CTRL.search(url) or " " in url:
                raise ValueError(f"bad URL {url!r}")
            if parts.scheme not in {"http", "https"} or not parts.hostname:
                raise ValueError(f"{url!r}: must be an http(s) URL")
            if parts.username or parts.password:
                raise ValueError(f"{url!r}: credentials in URLs are not allowed")
        return _unique(v, "extra_rss URLs")  # runtime SSRF guard still applies (spec 10.1)


class Delivery(Strict):
    channel: Channel
    frequency: Frequency
    weekday: Weekday | None = None  # not in spec 6.1; required when frequency is weekly
    send_time_local: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    timezone: str
    depth: Depth
    max_items: int = Field(ge=1, le=25)
    language: str = Field(pattern=r"^[a-z]{2,3}(-[A-Z]{2})?$")

    @field_validator("timezone")
    @classmethod
    def _iana(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
            raise ValueError(f"{v!r} is not an IANA timezone name (e.g. Asia/Kolkata)") from exc
        return v

    @model_validator(mode="after")
    def _weekly(self) -> Self:
        if self.frequency == "weekly" and self.weekday is None:
            raise ValueError("weekday is required when frequency is weekly")
        if self.frequency == "daily" and self.weekday is not None:
            raise ValueError("weekday is only valid when frequency is weekly")
        return self


class Summarizer(Strict):
    provider: Provider
    model: str = Field(min_length=1, max_length=100)
    fallback_provider: FallbackProvider = ""
    fallback_model: str = Field("", max_length=100)
    base_url: str = Field("", max_length=2048)

    @model_validator(mode="after")
    def _rules(self) -> Self:
        if self.provider == "openai_compatible":
            if not self.base_url:
                raise ValueError("base_url is required for openai_compatible")
            parts = urlsplit(self.base_url)
            if parts.scheme != "https" or not parts.hostname:
                raise ValueError("base_url must start with https://")
            if parts.username or parts.password:
                raise ValueError("base_url must not contain credentials")
        elif self.base_url:
            raise ValueError("base_url is only allowed for openai_compatible")
        if bool(self.fallback_provider) != bool(self.fallback_model):
            raise ValueError("fallback_provider and fallback_model must be set together")
        return self


def _all_modes() -> list[Mode]:
    return ["data", "summary", "digest"]


class AgentAccess(Strict):
    allow_inbound_agents: bool = True
    allowed_modes: list[Mode] = Field(default_factory=_all_modes)
    max_tier3_runs_per_day: int = Field(5, ge=1, le=20)
    # Phase 4 (Appendix C 62): lets an authorized agent's run ALSO deliver the digest through the
    # owner's configured channel. Off by default; older profile.yaml files keep working.
    allow_owner_channel: bool = False

    @field_validator("allowed_modes")
    @classmethod
    def _modes(cls, v: list[str]) -> list[str]:
        return _unique(v, "modes")


class Profile(Strict):
    version: Literal[1]
    interests: Interests
    sources: Sources
    delivery: Delivery
    summarizer: Summarizer
    agent_access: AgentAccess


class ProfileError(Exception):
    """Readable validation failure."""


def load_profile(path: Path) -> Profile:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ProfileError(f"{path}: invalid YAML: {exc}") from exc
    try:
        return Profile.model_validate(data)
    except ValidationError as exc:
        lines = [f"{path}: {exc.error_count()} problem(s) in profile.yaml"]
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"]) or "<root>"
            lines.append(f"  - {loc}: {err['msg']}")
        raise ProfileError("\n".join(lines)) from exc
