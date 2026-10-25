"""Summarizer router (spec 6.4): primary provider, per-run circuit breaker, optional fallback
provider, last resort the extractive snippet. Behaviour by error:
  401/403, 402 -> stop using that provider for the run and raise an actionable notice
  429, 5xx, timeouts -> (already retried with backoff) count toward the breaker, try fallback
  invalid output after the retry -> extractive snippet for that item (not a provider fault)"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

import httpx

from ..profile import Depth, Profile
from ..safe_http import Resolver, UnsafeUrlError, check_public_url, system_resolver
from .base import (
    AuthError,
    ChatClient,
    LLMError,
    QuotaError,
    SourceItem,
    SummarizeResult,
    Usage,
)
from .deepseek import deprecation_warning, make_deepseek
from .huggingface import make_huggingface
from .llm import summarize_with
from .openai_compat import OpenAICompatClient
from .validation import SummaryValidationError

BREAKER_THRESHOLD = 3


class ConfigError(Exception):
    """Missing or unusable configuration; the message names secrets by name only."""


@dataclass
class Breaker:
    threshold: int = BREAKER_THRESHOLD
    failures: int = 0
    tripped: bool = False

    def success(self) -> None:
        self.failures = 0

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.tripped = True

    def trip(self) -> None:
        self.tripped = True


@dataclass
class SummarizerRouter:
    primary: ChatClient
    fallback: ChatClient | None = None
    threshold: int = BREAKER_THRESHOLD
    notices: list[str] = field(default_factory=list)
    _breakers: dict[int, Breaker] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        for client in self.clients:
            self._breakers[id(client)] = Breaker(self.threshold)

    @property
    def clients(self) -> list[ChatClient]:
        return [c for c in (self.primary, self.fallback) if c is not None]

    @property
    def identities(self) -> list[tuple[str, str]]:
        return [(c.provider, c.model) for c in self.clients]

    def _notice(self, text: str) -> None:
        if text not in self.notices:
            self.notices.append(text)

    def summarize(self, item: SourceItem, depth: Depth, language: str) -> SummarizeResult:
        usage = Usage()
        warnings: list[str] = []
        for client in self.clients:
            breaker = self._breakers[id(client)]
            if breaker.tripped:
                continue
            try:
                summary = summarize_with(client, item, depth, language, usage, warnings)
            except AuthError:
                breaker.trip()
                self._notice(
                    f"{client.provider}: the API key was rejected. Check the "
                    f"{self._secret_name(client)} secret; this provider is skipped for this run."
                )
                continue
            except QuotaError:
                breaker.trip()
                self._notice(
                    f"{client.provider}: balance exhausted. Top up the account or raise the "
                    "spending limit; this provider is skipped for this run."
                )
                continue
            except LLMError as exc:
                breaker.failure()
                warnings.append(str(exc))
                if breaker.tripped:
                    self._notice(
                        f"{client.provider}: {breaker.failures} consecutive failures; skipped "
                        "for the rest of this run."
                    )
                continue
            except SummaryValidationError as exc:
                warnings.append(f"{client.provider}: summary rejected ({exc})")
                return SummarizeResult(
                    None, client.provider, client.model, usage, warnings, "invalid model output"
                )
            breaker.success()
            return SummarizeResult(summary, client.provider, client.model, usage, warnings)
        first = self.primary
        return SummarizeResult(
            None, first.provider, first.model, usage, warnings, "no summarizer available"
        )

    def _secret_name(self, client: ChatClient) -> str:
        return "LLM_API_KEY" if client is self.primary else "LLM_API_KEY_FALLBACK"


def _build(
    provider: str,
    model: str,
    base_url: str,
    key: str,
    http: httpx.Client,
    resolver: Resolver,
) -> ChatClient:
    if provider == "deepseek":
        return make_deepseek(key, model, http)
    if provider == "huggingface":
        return make_huggingface(key, model, http)
    try:
        check_public_url(base_url, resolver)
    except UnsafeUrlError as exc:
        raise ConfigError(f"summarizer.base_url rejected: {exc}") from exc
    return OpenAICompatClient(
        provider="openai_compatible", base_url=base_url, api_key=key, model=model, http=http
    )


def get_summarizer(
    profile: Profile,
    env: Mapping[str, str] | None = None,
    *,
    http: httpx.Client,
    resolver: Resolver = system_resolver,
) -> tuple[SummarizerRouter, list[str]]:
    """Returns the router and configuration warnings. Raises ConfigError (no secret values)."""
    env = os.environ if env is None else env
    cfg = profile.summarizer
    warnings: list[str] = []
    key = env.get("LLM_API_KEY", "")
    if not key:
        raise ConfigError("the LLM_API_KEY secret is not set (Settings > Secrets > Actions)")
    primary = _build(cfg.provider, cfg.model, cfg.base_url, key, http, resolver)
    if cfg.provider == "deepseek" and (w := deprecation_warning(cfg.model)):
        warnings.append(w)
    fallback: ChatClient | None = None
    if cfg.fallback_provider:
        fkey = env.get("LLM_API_KEY_FALLBACK", "")
        if fkey:
            fallback = _build(cfg.fallback_provider, cfg.fallback_model, "", fkey, http, resolver)
            if cfg.fallback_provider == "deepseek" and (
                w := deprecation_warning(cfg.fallback_model)
            ):
                warnings.append(w)
        else:
            warnings.append("fallback provider configured but LLM_API_KEY_FALLBACK is not set")
    return SummarizerRouter(primary, fallback), warnings
