"""OpenAI-compatible chat completions client (spec 6.4): used directly for custom endpoints and
as the transport behind the DeepSeek and Hugging Face adapters. One request shape, one error
mapping: 401/403 AuthError, 402 QuotaError, 429 backoff then RateLimitError, 5xx and network
errors retried with a cap then TransientError. The key lives only in the Authorization header
of the request; it is never logged, stored, or put in an exception message."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import Any

import httpx

from ..redact import redact
from .base import (
    AuthError,
    BadResponseError,
    ChatResult,
    QuotaError,
    RateLimitError,
    TransientError,
    Usage,
)

MAX_RETRY_AFTER = 30.0


class OpenAICompatClient:
    def __init__(
        self,
        *,
        provider: str,
        base_url: str,
        api_key: str,
        model: str,
        http: httpx.Client,
        json_mode: bool = True,
        extra_body: dict[str, Any] | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.2,
        max_attempts: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self.provider = provider
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self._http = http
        self._json_mode = json_mode
        self._extra = dict(extra_body or {})
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._rng = rng

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatClient(provider={self.provider!r}, model={self.model!r})"

    def _payload(self, system: str, user: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "stream": False,
        }
        if self._json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self._extra)
        return body

    def _detail(self, resp: httpx.Response) -> str:
        try:
            err = resp.json().get("error")
            message = err.get("message") if isinstance(err, dict) else err
        except (ValueError, AttributeError):
            message = None
        return redact(str(message or "")[:200], [self._key])

    def _backoff(self, attempt: int, resp: httpx.Response | None) -> None:
        delay = min(8.0, 2.0**attempt) * (0.5 + self._rng())
        retry_after = resp.headers.get("retry-after") if resp is not None else None
        if retry_after and retry_after.isdigit():
            delay = max(delay, min(float(retry_after), MAX_RETRY_AFTER))
        self._sleep(delay)

    def complete(self, system: str, user: str) -> ChatResult:
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}
        last: Exception | None = None
        resp: httpx.Response | None = None
        attempt = 0
        json_retry_used = False
        while attempt < self._max_attempts:
            resp = None
            try:
                resp = self._http.post(self._url, json=self._payload(system, user), headers=headers)
            except httpx.HTTPError as exc:
                last = TransientError(f"{self.provider}: network error ({type(exc).__name__})")
            if resp is not None:
                code = resp.status_code
                if code == 200:
                    return self._parse(resp)
                if code in {401, 403}:
                    raise AuthError(f"{self.provider}: HTTP {code}, API key rejected")
                if code == 402:
                    raise QuotaError(f"{self.provider}: HTTP 402, balance exhausted")
                if code in {400, 422} and self._json_mode and not json_retry_used:
                    json_retry_used = True  # some models/providers reject JSON mode
                    self._json_mode = False
                    continue
                if code == 429:
                    last = RateLimitError(f"{self.provider}: HTTP 429, rate limited")
                elif code >= 500:
                    last = TransientError(f"{self.provider}: HTTP {code}")
                else:
                    detail = self._detail(resp)
                    raise BadResponseError(f"{self.provider}: HTTP {code} {detail}".strip())
            attempt += 1
            if attempt < self._max_attempts:
                self._backoff(attempt, resp)
        assert last is not None
        raise last

    def _parse(self, resp: httpx.Response) -> ChatResult:
        try:
            data = resp.json()
            choice = data["choices"][0]
            text = choice["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            raise BadResponseError(f"{self.provider}: unexpected response shape") from None
        if not isinstance(text, str) or not text.strip():
            raise BadResponseError(f"{self.provider}: empty completion")
        if choice.get("finish_reason") == "length":
            raise BadResponseError(f"{self.provider}: output truncated (max_tokens)")
        raw_usage = data.get("usage") if isinstance(data, dict) else None
        usage = Usage(calls=1)
        if isinstance(raw_usage, dict):
            usage.prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
            usage.completion_tokens = int(raw_usage.get("completion_tokens") or 0)
        return ChatResult(text, usage)
