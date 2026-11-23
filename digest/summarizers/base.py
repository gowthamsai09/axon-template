"""Shared types for the summarizers (spec 6.4, A.2): the validated Summary, usage counters,
the error taxonomy that the router maps to behaviour, and the protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StrictBool

from ..profile import Depth, TopicSlug

Status = Literal["ok", "insufficient_content", "suspected_injection"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NotableFigure(Strict):
    value: str = Field(max_length=200)
    context: str = Field(max_length=200)


class Summary(Strict):
    """Output of the summarizer (spec A.2 and 13.1). Caps are the spec's plus Appendix C 7."""

    status: Status
    title: str = Field(max_length=300)
    one_line: str = Field(max_length=200)
    key_points: list[Annotated[str, Field(max_length=240)]] = Field(max_length=6)
    why_it_matters: str = Field(max_length=300)
    tags: list[TopicSlug] = Field(max_length=4)
    entities: list[Annotated[str, Field(max_length=100)]] = Field(max_length=8)
    notable_figures: list[NotableFigure] = Field(max_length=10)
    source_url: str = Field(max_length=2048)
    injection_flag: StrictBool  # a string such as "false" is invalid output, not coerced


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, other: Usage) -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.calls += other.calls


@dataclass(frozen=True)
class ChatResult:
    text: str
    usage: Usage


class LLMError(Exception):
    """Base class; messages never contain keys or request headers."""


class AuthError(LLMError):
    """401/403: the key is bad. Stop using this provider."""


class QuotaError(LLMError):
    """402: balance exhausted. Stop using this provider."""


class RateLimitError(LLMError):
    """429 after backoff."""


class TransientError(LLMError):
    """5xx or network failure after retries."""


class BadResponseError(LLMError):
    """The provider answered but the answer is unusable (truncated, wrong shape)."""


class ChatClient(Protocol):
    provider: str
    model: str

    def complete(self, system: str, user: str) -> ChatResult: ...


@dataclass(frozen=True)
class SourceItem:
    id: str
    title: str
    url: str
    source_name: str
    published_at: str
    text: str


@dataclass
class SummarizeResult:
    summary: Summary | None  # None: use the extractive snippet
    provider: str
    model: str
    usage: Usage = field(default_factory=Usage)
    warnings: list[str] = field(default_factory=list)
    reason: str = ""  # why the item fell back to the extractive snippet


class Summarizer(Protocol):
    def summarize(self, item: SourceItem, depth: Depth, language: str) -> SummarizeResult: ...
