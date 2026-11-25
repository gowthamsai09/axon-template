"""Semantic scores for ranking (Phase 6, Appendix C 96).

An item's score is its best cosine similarity to any of the user's interest phrases, rescaled so
`min_similarity` maps to 0 and 1.0 maps to 1. Interest phrases and item texts are embedded once
and cached in state, keyed by provider, model and text hash, so a rerun costs nothing. Cost is
bounded: at most `max_items` candidates and 40 interest phrases per run, 1500 characters each."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime

from ..feed_client import FeedItem
from ..profile import Embeddings, Interests
from ..state import CachedVector, StateStore
from .base import Embedder, EmbeddingError, cosine, pack, quantize, unpack

TEXT_VERSION = "e1"  # bump when the text sent for an item changes (it is part of every key)
MAX_TEXT_CHARS = 1500
MAX_INTEREST_TEXTS = 40
MAX_INTEREST_CHARS = 120
_WS = re.compile(r"\s+")
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass
class SemanticResult:
    scores: dict[str, float] = field(default_factory=dict)
    embedded: int = 0  # texts sent to the provider
    cached: int = 0  # texts answered from state
    warnings: list[str] = field(default_factory=list)


def clean_text(text: str, limit: int) -> str:
    """One line, no control characters, at most `limit` characters."""
    return _WS.sub(" ", _CTRL.sub(" ", text)).strip()[:limit]


def interest_texts(interests: Interests) -> list[str]:
    raw = [slug.replace("-", " ") for slug in interests.topics]
    raw += [e.removeprefix("github:") for e in interests.follow_entities]
    raw += interests.keywords_include
    out: list[str] = []
    seen: set[str] = set()
    for value in raw:
        text = clean_text(value, MAX_INTEREST_CHARS)
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
    return out[:MAX_INTEREST_TEXTS]


def item_text(item: FeedItem) -> str:
    return clean_text(f"{item.title}. {item.snippet}", MAX_TEXT_CHARS)


def vector_key(provider: str, model: str, text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    raw = "|".join([TEXT_VERSION, provider, model, digest])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def vectors_for(
    texts: list[str], embedder: Embedder, state: StateStore, now: datetime
) -> tuple[list[list[float]], int, int]:
    """(vectors in order, texts embedded, texts served from the cache). Raises EmbeddingError."""
    keys = [vector_key(embedder.provider, embedder.model, t) for t in texts]
    found: dict[str, list[float]] = {}
    for key in dict.fromkeys(keys):
        hit = state.get_vector(key)
        if hit is None or hit.provider != embedder.provider or hit.model != embedder.model:
            continue
        vec = unpack(hit.v, hit.dim)
        if vec is not None:
            found[key] = vec
    cached = sum(1 for key in keys if key in found)
    missing = [key for key in dict.fromkeys(keys) if key not in found]
    if missing:
        first = {key: texts[keys.index(key)] for key in missing}
        fresh = embedder.embed([first[key] for key in missing])
        if len(fresh) != len(missing):
            raise EmbeddingError("embeddings: unexpected number of vectors")
        for key, raw in zip(missing, fresh, strict=True):
            vec = quantize(raw)
            found[key] = vec
            entry = CachedVector(
                cached_at=now, provider=embedder.provider, model=embedder.model,
                dim=len(vec), v=pack(vec),
            )  # fmt: skip
            state.put_vector(key, entry)
    return [found[key] for key in keys], len(missing), cached


def semantic_scores(
    items: list[FeedItem],
    interests: Interests,
    cfg: Embeddings,
    embedder: Embedder,
    state: StateStore,
    now: datetime,
) -> SemanticResult:
    """Scores (0..1) for the top `cfg.max_items` candidates; others simply get no entry."""
    result = SemanticResult()
    phrases = interest_texts(interests)
    if not phrases:
        result.warnings.append("semantic matching skipped: the profile has no interests yet")
        return result
    pool = sorted(items, key=lambda i: (-i.base_score, -i.published_at.timestamp(), i.id))
    pool = pool[: cfg.max_items]
    if not pool:
        return result
    vectors, result.embedded, result.cached = vectors_for(
        phrases + [item_text(i) for i in pool], embedder, state, now
    )
    wanted = vectors[: len(phrases)]
    floor = cfg.min_similarity
    for item, vec in zip(pool, vectors[len(phrases) :], strict=True):
        best = max(cosine(vec, w) for w in wanted)
        result.scores[item.id] = round(max(0.0, (best - floor) / (1.0 - floor)), 4)
    return result
