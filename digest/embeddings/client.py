"""HTTP embedders (Phase 6, Appendix C 96) for the two providers a profile can name.

huggingface        POST https://router.huggingface.co/hf-inference/models/<model>/pipeline/
                   feature-extraction with {"inputs": [...]}; the answer is a list of vectors.
                   The host is fixed and the model id is validated; the exact path is a
                   first-push check (the task page documents the payload, not the URL).
openai_compatible  POST <base_url>/embeddings with {"model", "input": [...]}; the answer is
                   {"data": [{"index", "embedding"}]}. The URL passes the SSRF guard.

The key travels only in the Authorization header. Messages are fixed texts: no response body,
exception text or URL is ever repeated, because any of them could carry a secret."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from ..delivery.common import Sleep, post_with_retry
from ..delivery.email import DeliveryError
from ..profile import Profile
from ..safe_http import Resolver, UnsafeUrlError, check_public_url, system_resolver
from .base import Embedder, EmbeddingError, normalize

BATCH_SIZE = 16
MAX_RESPONSE_BYTES = 8_000_000
HF_URL = "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"
HF_URL_RE = re.compile(
    r"https://router\.huggingface\.co/hf-inference/models/"
    r"(?:[A-Za-z0-9][A-Za-z0-9._-]{0,63}/)?[A-Za-z0-9][A-Za-z0-9._-]{0,63}"
    r"/pipeline/feature-extraction"
)
Wire = Literal["hf", "openai"]


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after", "")
    return float(int(raw)) if raw.isascii() and raw.isdigit() and len(raw) <= 4 else None


def status_error(code: int) -> EmbeddingError:
    if code in {401, 403}:
        return EmbeddingError(f"embeddings: the EMBED_API_KEY was rejected (HTTP {code})")
    if code == 402:
        return EmbeddingError("embeddings: balance or free credits exhausted (HTTP 402)")
    if code == 404:
        return EmbeddingError("embeddings: model or endpoint not found (HTTP 404)")
    if code in {400, 422}:
        return EmbeddingError(f"embeddings: the request was rejected (HTTP {code})")
    if code == 429:
        return EmbeddingError("embeddings: rate limited (HTTP 429)")
    if code >= 500:
        return EmbeddingError(f"embeddings: the service is unavailable (HTTP {code})")
    return EmbeddingError(f"embeddings: unexpected HTTP {code}")


_SHAPE = "embeddings: unexpected answer shape"


def _rows_openai(data: Any, count: int) -> list[list[object]]:
    rows = data.get("data") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != count:
        raise EmbeddingError(_SHAPE)
    ordered: list[list[object] | None] = [None] * count
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            raise EmbeddingError(_SHAPE)
        index = row.get("index", position)
        vector = row.get("embedding")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
            raise EmbeddingError(_SHAPE)
        if ordered[index] is not None or not isinstance(vector, list):
            raise EmbeddingError(_SHAPE)
        ordered[index] = vector
    return [row for row in ordered if row is not None]


def _rows_hf(data: Any, count: int) -> list[list[object]]:
    if not isinstance(data, list) or len(data) != count:
        raise EmbeddingError(_SHAPE)
    rows: list[list[object]] = []
    for row in data:
        # a list of lists per text means token-level output, which is not a sentence vector
        if not isinstance(row, list) or any(isinstance(v, list) for v in row):
            raise EmbeddingError(_SHAPE)
        rows.append(row)
    return rows


class HttpEmbedder:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        url: str,
        wire: Wire,
        api_key: str,
        http: httpx.Client,
        batch_size: int = BATCH_SIZE,
        sleep: Sleep = time.sleep,
    ) -> None:
        self.provider = provider
        self.model = model
        self._url = url
        self._wire: Wire = wire
        self._key = api_key
        self._http = http
        self._batch = max(1, batch_size)
        self._sleep = sleep

    def __repr__(self) -> str:  # never show the key or the URL
        return f"HttpEmbedder(provider={self.provider!r}, model={self.model!r})"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch):
            vectors.extend(self._call(texts[start : start + self._batch]))
        if len(vectors) != len(texts):
            raise EmbeddingError(_SHAPE)
        if len({len(v) for v in vectors}) > 1:
            raise EmbeddingError("embeddings: vectors of different sizes in one answer")
        return vectors

    def _call(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, object] = (
            {"inputs": batch} if self._wire == "hf" else {"model": self.model, "input": batch}
        )
        headers = {"Authorization": f"Bearer {self._key}"}
        try:
            resp = post_with_retry(
                self._http,
                self._url,
                payload,
                service="embeddings",
                retry_after=_retry_after,
                sleep=self._sleep,
                headers=headers,
            )
        except DeliveryError as exc:
            raise EmbeddingError(str(exc)) from None
        if resp.status_code != 200:
            raise status_error(resp.status_code)
        if len(resp.content) > MAX_RESPONSE_BYTES:
            raise EmbeddingError("embeddings: the answer is too large")
        try:
            data = resp.json()
        except ValueError:
            raise EmbeddingError("embeddings: the answer is not JSON") from None
        rows = _rows_hf(data, len(batch)) if self._wire == "hf" else _rows_openai(data, len(batch))
        return [normalize(row) for row in rows]


def get_embedder(
    profile: Profile,
    env: Mapping[str, str],
    *,
    http: httpx.Client,
    resolver: Resolver = system_resolver,
    sleep: Sleep = time.sleep,
) -> tuple[Embedder | None, list[str]]:
    """The embedder for the profile, or None plus a warning. Never raises: semantic matching is
    an optional extra, so a missing key or a bad URL turns it off for the run, nothing more."""
    cfg = profile.embeddings
    if not cfg.enabled:
        return None, []
    key = env.get("EMBED_API_KEY", "")
    if not key:
        return None, ["semantic matching is off: the EMBED_API_KEY secret is not set"]
    if cfg.provider == "huggingface":
        url = HF_URL.format(model=cfg.model)
        if HF_URL_RE.fullmatch(url) is None:
            return None, ["semantic matching is off: embeddings.model is not usable"]
        wire: Wire = "hf"
    else:
        parts = urlsplit(cfg.base_url)
        if parts.query or parts.fragment:
            return None, ["semantic matching is off: embeddings.base_url must have no query"]
        try:
            check_public_url(cfg.base_url, resolver)
        except UnsafeUrlError as exc:
            return None, [f"semantic matching is off: embeddings.base_url rejected ({exc})"]
        url = cfg.base_url.rstrip("/") + "/embeddings"
        wire = "openai"
    embedder = HttpEmbedder(
        provider=cfg.provider, model=cfg.model, url=url, wire=wire, api_key=key, http=http,
        sleep=sleep,
    )  # fmt: skip
    return embedder, []
