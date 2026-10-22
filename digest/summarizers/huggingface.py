"""Hugging Face Inference Providers adapter (spec Appendix B item 9, verified 2026-10-21).

OpenAI-compatible router at https://router.huggingface.co/v1, authenticated with an HF token as
a Bearer key. Model ids are "org/name" with an optional ":provider" suffix (for example
openai/gpt-oss-120b:cerebras). Which models accept response_format json_object varies by
provider, so the shared client retries once without it on HTTP 400/422; output validation
(13.1) never depends on JSON mode."""

from __future__ import annotations

import httpx

from .openai_compat import OpenAICompatClient

HF_BASE_URL = "https://router.huggingface.co/v1"


def make_huggingface(api_key: str, model: str, http: httpx.Client) -> OpenAICompatClient:
    return OpenAICompatClient(
        provider="huggingface",
        base_url=HF_BASE_URL,
        api_key=api_key,
        model=model,
        http=http,
        json_mode=True,
    )
