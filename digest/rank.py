"""Personal scoring (spec 5.3): user_score = base + 0.50*interest_match - 0.50*exclusion_penalty.

interest_match (0..1) = 0.5*topics + 0.3*entities + 0.2*keywords. Matching is case-insensitive
and whole-token (hyphens, spaces and underscores are interchangeable), like the feed taxonomy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from .feed_client import FeedItem
from .profile import Interests, Profile

W_TOPICS, W_ENTITIES, W_KEYWORDS = 0.5, 0.3, 0.2
MATCH_WEIGHT, EXCLUDE_WEIGHT = 0.50, 0.50
EXTRA_RSS_MATCH = 0.5  # items from the user's own extra_rss feeds always qualify


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


@dataclass(frozen=True)
class Scored:
    item: FeedItem
    score: float
    match: float
    penalty: float


def rank(items: list[FeedItem], profile: Profile, *, seen: set[str], limit: int) -> list[Scored]:
    interests = profile.interests
    filtering = has_interests(interests)
    out: list[Scored] = []
    for item in items:
        if item.id in seen:
            continue
        match = interest_match(item, interests)
        if item.from_extra_rss:
            match = max(match, EXTRA_RSS_MATCH)
        if filtering and match <= 0:
            continue
        penalty = exclusion_penalty(item, interests)
        score = item.base_score + MATCH_WEIGHT * match - EXCLUDE_WEIGHT * penalty
        out.append(Scored(item, score, match, penalty))
    out.sort(key=lambda s: (-s.score, -s.item.published_at.timestamp(), s.item.id))
    return out[:limit]
