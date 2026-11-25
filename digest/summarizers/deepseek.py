"""DeepSeek adapter (spec Appendix B item 8, verified 2026-10-21; model name updated in
Appendix C 94, checked 2026-11-25 against the official docs).

Base URL https://api.deepseek.com (OpenAI format). The model to use is deepseek-flash; the
docs say deepseek-v4-flash is still accepted but is served by the newer Flash model, so the
profile default now names deepseek-flash directly. The legacy ids deepseek-chat and
deepseek-reasoner were retired on 2026-07-24. JSON mode needs response_format json_object plus
the word "json" in the prompt (prompt 13.1 has it) and a sufficient max_tokens. The V4 family
thinks by default, so summaries send thinking disabled (first-push check for deepseek-flash)."""

from __future__ import annotations

import httpx

from .openai_compat import OpenAICompatClient

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
LEGACY_MODELS = {"deepseek-chat", "deepseek-reasoner"}


def deprecation_warning(model: str) -> str | None:
    if model in LEGACY_MODELS:
        return (
            f"summarizer.model {model!r} was retired by DeepSeek on 2026-07-24; use deepseek-flash"
        )
    return None


def make_deepseek(api_key: str, model: str, http: httpx.Client) -> OpenAICompatClient:
    v4 = model.startswith(("deepseek-v4", "deepseek-flash"))
    extra = {"thinking": {"type": "disabled"}} if v4 else None
    return OpenAICompatClient(
        provider="deepseek",
        base_url=DEEPSEEK_BASE_URL,
        api_key=api_key,
        model=model,
        http=http,
        json_mode=True,
        extra_body=extra,
    )
