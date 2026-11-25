"""Personal scoring (spec 5.3, extended in Phase 6, Appendix C 96 and 98):

user_score = base + 0.50*interest_match + w*semantic + 0.20*feedback - 0.50*exclusion_penalty

interest_match (0..1) = 0.5*topics + 0.3*entities + 0.2*keywords. Matching is case-insensitive
and whole-token (hyphens, spaces and underscores are interchangeable), like the feed taxonomy.
semantic (0..1) and w (profile embeddings.weight) apply only when embeddings are enabled and
worked; feedback is the clamped sum of the per-topic weights of the item's tags (-1..1). Both
extra terms are 0 when their feature is off, so the Phase 5 scores are unchanged."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

from .feed_client import FeedItem
from .profile import Interests, Profile

W_TOPICS, W_ENTITIES, W_KEYWORDS = 0.5, 0.3, 0.2
MATCH_WEIGHT, EXCLUDE_WEIGHT = 0.50, 0.50
EXTRA_RSS_MATCH = 0.5  # items from the user's own extra_rss feeds always qualify
FEEDBACK_SCALE = 0.20  # the most a topic vote history can move one item's score


@lru_cache(maxsize=512)
def _phrase(term: str) -> re.Pattern[str] | None:
    tokens = [t for t in re.split(r"[\s_\-]+", term.strip().casefold()) if t]
    if not tokens:
        return None
    body = r"[\s_\-]+".join(re.escape(t) for t in tokens)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


def _has(text: str, term: str) -> bool:
    pattern = _phrase(term)
    return pattern is not None and pattern.search(text) is not None


def item_text(item: FeedItem) -> str:
    return " \n".join(p for p in (item.title, item.snippet, item.excerpt or "") if p)


def has_interests(interests: Interests) -> bool:
    """Only positive interests filter; a profile with just exclusions keeps everything."""
    return bool(interests.topics or interests.follow_entities or interests.keywords_include)


def _entity_hit(item: FeedItem, entity: str) -> bool:
    if entity.startswith("github:"):
        repo = entity.removeprefix("github:").casefold()
        return repo in item.canonical_url.casefold() or _has(item.title, repo)
    folded = entity.casefold()
    return any(e.casefold() == folded for e in item.entities) or _has(item.title, entity)


def interest_match(item: FeedItem, interests: Interests) -> float:
    topics = 0.0
    if interests.topics:
        hits = len(set(item.tags) & set(interests.topics))
        topics = min(1.0, hits / min(2, len(interests.topics)))
    entities = 1.0 if any(_entity_hit(item, e) for e in interests.follow_entities) else 0.0
    keywords = 0.0
    if interests.keywords_include:
        text = item_text(item)
        keywords = min(1.0, sum(_has(text, k) for k in interests.keywords_include) / 2)
    return W_TOPICS * topics + W_ENTITIES * entities + W_KEYWORDS * keywords


def exclusion_penalty(item: FeedItem, interests: Interests) -> float:
    text = item_text(item)
    return 1.0 if any(_has(text, k) for k in interests.keywords_exclude) else 0.0


def feedback_adjustment(item: FeedItem, weights: Mapping[str, float] | None) -> float:
    """Bounded nudge from per-topic vote weights: at most +-FEEDBACK_SCALE."""
    if not weights:
        return 0.0
    total = sum(weights.get(tag, 0.0) for tag in set(item.tags))
    return FEEDBACK_SCALE * max(-1.0, min(1.0, total))


@dataclass(frozen=True)
class Scored:
    item: FeedItem
    score: float
    match: float
    penalty: float
    semantic: float = 0.0
    feedback: float = 0.0


def rank(
    items: list[FeedItem],
    profile: Profile,
    *,
    seen: set[str],
    limit: int,
    semantic: Mapping[str, float] | None = None,
    weights: Mapping[str, float] | None = None,
) -> list[Scored]:
    interests = profile.interests
    filtering = has_interests(interests)
    emb = profile.embeddings
    w_semantic = emb.weight if emb.enabled else 0.0
    out: list[Scored] = []
    for item in items:
        if item.id in seen:
            continue
        match = interest_match(item, interests)
        if item.from_extra_rss:
            match = max(match, EXTRA_RSS_MATCH)
        sem = max(0.0, min(1.0, semantic.get(item.id, 0.0))) if semantic and emb.enabled else 0.0
        if filtering and match <= 0 and sem <= 0:
            continue
        penalty = exclusion_penalty(item, interests)
        nudge = feedback_adjustment(item, weights)
        score = (
            item.base_score + MATCH_WEIGHT * match + w_semantic * sem + nudge
            - EXCLUDE_WEIGHT * penalty
        )  # fmt: skip
        out.append(Scored(item, score, match, penalty, sem, nudge))
    out.sort(key=lambda s: (-s.score, -s.item.published_at.timestamp(), s.item.id))
    return out[:limit]
