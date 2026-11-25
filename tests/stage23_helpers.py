"""Shared fakes for the Stage 23 tests: a fake embedder, an in-memory feed, item builders."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from digest.embeddings.base import EmbeddingError
from digest.feed_client import FeedError, FeedItem
from digest.profile import Profile
from tests.conftest import NOW, make_profile

KEYWORDS = ("agent", "rag", "quantiz", "crypto", "policy", "robot")
EMB_MODEL = "org/emb-model"


class FakeEmbedder:
    """Deterministic embedder: counts keyword hits per text, plus a small base so no text has
    a zero vector. Records every call; can be told to fail."""

    def __init__(
        self,
        *,
        provider: str = "huggingface",
        model: str = EMB_MODEL,
        fail: Exception | None = None,
        short: bool = False,
    ) -> None:
        self.provider = provider
        self.model = model
        self.fail = fail
        self.short = short
        self.calls: list[list[str]] = []

    @property
    def texts_sent(self) -> int:
        return sum(len(c) for c in self.calls)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if self.fail is not None:
            raise self.fail
        vectors = [[0.05 + t.casefold().count(k) for k in KEYWORDS] for t in texts]
        return vectors[:-1] if self.short and vectors else vectors


class BrokenEmbedder:
    provider = "huggingface"
    model = EMB_MODEL

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise EmbeddingError("embeddings: the service is unavailable (HTTP 503)")


class MemoryFeed:
    """A FeedReader over a dict of relative path -> bytes."""

    def __init__(self, files: dict[str, bytes], *, fail: bool = False) -> None:
        self.files = files
        self.fail = fail
        self.reads: list[str] = []

    def read_bytes(self, rel: str) -> bytes | None:
        self.reads.append(rel)
        if self.fail:
            raise FeedError(f"{rel}: network error (ConnectError)")
        return self.files.get(rel)


def item_json(
    n: int,
    title: str,
    *,
    tags: list[str] | None = None,
    entities: list[str] | None = None,
    hours_ago: float = 10.0,
    base: float = 0.5,
    snippet: str = "",
    url: str | None = None,
    now: datetime = NOW,
) -> dict[str, Any]:
    link = url or f"https://example.com/post/{n}"
    when = now - timedelta(hours=hours_ago)
    return {
        "id": f"{n:016x}", "url": link, "canonical_url": link, "title": title,
        "source_id": "src", "published_at": when.isoformat().replace("+00:00", "Z"),
        "snippet": snippet, "tags": tags or [], "entities": entities or [], "base_score": base,
    }  # fmt: skip


def shard_bytes(items: list[dict[str, Any]]) -> bytes:
    return json.dumps({"schema_version": 1, "items": items}).encode("utf-8")


def feed_item(n: int, title: str, **over: Any) -> FeedItem:
    return FeedItem.model_validate(item_json(n, title, **over))


def embed_profile(**over: Any) -> Profile:
    """A profile with embeddings on (huggingface, a valid model)."""
    section: dict[str, Any] = {"enabled": True, "provider": "huggingface", "model": EMB_MODEL}
    section.update(over)
    return make_profile(
        embeddings=section,
        interests={
            "topics": ["agents-tooling"],
            "follow_entities": [],
            "keywords_include": [],
            "keywords_exclude": [],
        },  # fmt: skip
        delivery={"timezone": "UTC"},
    )
