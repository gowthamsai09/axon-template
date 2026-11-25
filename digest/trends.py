"""Weekly trends (Phase 6, Appendix C 97): what the shared feed was busy with this week.

Deterministic by default, with no model involved:
  1. read the last seven days of the shared feed (every tag shard plus latest.json);
  2. group near-duplicate stories (same canonical URL, or titles that share most of their
     words; with embeddings enabled, also items whose vectors are nearly identical);
  3. cluster the stories: first by entity (the most-covered names), then by tag for what is
     left; a cluster needs at least three distinct stories;
  4. rank clusters by stories plus half the stories from the last 72 hours.

Optional labels (profile trends.llm_labels, off by default) send each cluster's headlines
through the same summarizer path as articles, so the same injection rules, output validation
and cache apply, and the model that wrote a label is named next to it. Nothing here can fail a
run: callers turn any exception into a warning."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal

from .embeddings.base import Embedder, EmbeddingError, cosine
from .embeddings.matcher import item_text, vectors_for
from .feed_client import FeedItem, FeedReader, load_week
from .profile import Profile
from .state import StateStore, cache_key, content_hash
from .summarizers.base import SourceItem, Summary, Usage
from .summarizers.prompting import document_for, prompt_version
from .summarizers.router import SummarizerRouter

MIN_STORIES = 3
RECENT_HOURS = 72
MAX_ITEMS = 600  # items considered per run (the best by base_score)
MAX_EMBED_ITEMS = 100  # items embedded for near-duplicate detection
MAX_TITLE = 200
HEADLINES = 3
DUP_JACCARD = 0.6
DUP_COSINE = 0.92
MIN_ENTITY_LEN = 3
_WORDS = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "the", "and", "for", "with", "from", "that", "this", "its", "into", "over",
        "after", "about", "what", "how", "why", "who", "new", "not", "are", "was",
        "were", "has", "have", "had", "can", "will", "you", "your", "our", "their",
        "they", "them", "but", "all", "one", "two", "via", "per",
    ]
)  # fmt: skip


@dataclass(frozen=True)
class Headline:
    title: str
    url: str


@dataclass
class Trend:
    label: str
    kind: Literal["entity", "tag"]
    stories: int
    recent: int
    headlines: list[Headline] = field(default_factory=list)
    blurb: str = ""  # the optional AI label (validated one-liner)
    model: str = ""  # provider/model that wrote the label


def words(title: str) -> frozenset[str]:
    return frozenset(w for w in _WORDS.findall(title.casefold()) if len(w) > 2 and w not in _STOP)


def _ts(item: FeedItem) -> float:
    return item.published_at.timestamp()


def _order(items: Sequence[FeedItem]) -> list[FeedItem]:
    return sorted(items, key=lambda i: (-i.base_score, -_ts(i), i.id))


class _Union:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


def group_duplicates(
    items: Sequence[FeedItem], extra_pairs: Sequence[tuple[int, int]] = ()
) -> list[list[FeedItem]]:
    """Stories: items grouped when they are the same page or share most title words.
    `items` should already be in rank order; each group keeps that order."""
    sets = [words(i.title) for i in items]
    forest = _Union(len(items))
    by_url: dict[str, int] = {}
    for index, item in enumerate(items):
        first = by_url.setdefault(item.canonical_url, index)
        if first != index:
            forest.union(first, index)
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            shared = len(sets[a] & sets[b])
            if shared >= 3 and shared / len(sets[a] | sets[b]) >= DUP_JACCARD:
                forest.union(a, b)
    for a, b in extra_pairs:
        forest.union(a, b)
    groups: dict[int, list[FeedItem]] = {}
    for index, item in enumerate(items):
        groups.setdefault(forest.find(index), []).append(item)
    return [groups[root] for root in sorted(groups)]


def _clip_title(title: str) -> str:
    return " ".join(title.split())[:MAX_TITLE]


def _cluster(
    stories: list[list[FeedItem]],
    members: dict[str, list[int]],
    names: dict[str, str],
    kind: Literal["entity", "tag"],
    taken: set[int],
    now: datetime,
) -> list[Trend]:
    cutoff = now - timedelta(hours=RECENT_HOURS)
    out: list[Trend] = []
    for key in sorted(members, key=lambda k: (-len(members[k]), k)):
        free = [s for s in members[key] if s not in taken]
        if len(free) < MIN_STORIES:
            continue
        taken.update(free)
        reps = sorted((stories[s][0] for s in free), key=lambda i: (-i.base_score, -_ts(i), i.id))
        recent = sum(1 for s in free if max(i.published_at for i in stories[s]) >= cutoff)
        heads = [Headline(_clip_title(i.title), i.canonical_url) for i in reps[:HEADLINES]]
        out.append(Trend(names[key], kind, len(free), recent, heads))
    return out


def cluster_stories(stories: list[list[FeedItem]], now: datetime, max_clusters: int) -> list[Trend]:
    """Entity clusters first (most-covered name wins a story), then tag clusters."""
    by_entity: dict[str, list[int]] = {}
    spelled: dict[str, Counter[str]] = {}
    by_tag: dict[str, list[int]] = {}
    for index, group in enumerate(stories):
        entity_keys: set[str] = set()
        tag_keys: set[str] = set()
        for item in group:
            for name in item.entities:
                key = name.strip().casefold()
                if len(key) >= MIN_ENTITY_LEN:
                    entity_keys.add(key)
                    spelled.setdefault(key, Counter())[name.strip()] += 1
            tag_keys.update(item.tags)
        for key in entity_keys:
            by_entity.setdefault(key, []).append(index)
        for key in tag_keys:
            by_tag.setdefault(key, []).append(index)
    shown = {k: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for k, c in spelled.items()}
    taken: set[int] = set()
    trends = _cluster(stories, by_entity, shown, "entity", taken, now)
    tag_names = {t: t.replace("-", " ") for t in by_tag}
    trends += _cluster(stories, by_tag, tag_names, "tag", taken, now)
    trends.sort(key=lambda t: (-(t.stories + 0.5 * t.recent), -t.stories, t.label.casefold()))
    return trends[:max_clusters]


def _embedding_pairs(
    pool: Sequence[FeedItem], embedder: Embedder, state: StateStore, now: datetime
) -> list[tuple[int, int]]:
    head = list(pool[:MAX_EMBED_ITEMS])
    vectors, _, _ = vectors_for([item_text(i) for i in head], embedder, state, now)
    return [
        (a, b)
        for a in range(len(head))
        for b in range(a + 1, len(head))
        if cosine(vectors[a], vectors[b]) >= DUP_COSINE
    ]


def build_trends(
    items: Sequence[FeedItem],
    now: datetime,
    *,
    max_clusters: int,
    embedder: Embedder | None = None,
    state: StateStore | None = None,
) -> tuple[list[Trend], list[str]]:
    warnings: list[str] = []
    pool = _order(items)[:MAX_ITEMS]
    pairs: list[tuple[int, int]] = []
    if embedder is not None and state is not None:
        try:
            pairs = _embedding_pairs(pool, embedder, state, now)
        except EmbeddingError as exc:
            warnings.append(f"trends: near-duplicate detection used titles only ({exc})")
    stories = group_duplicates(pool, pairs)
    return cluster_stories(stories, now, max_clusters), warnings


def _source_for(trend: Trend, now: datetime) -> SourceItem | None:
    first = next((h for h in trend.headlines if h.url.startswith(("https://", "http://"))), None)
    if first is None:
        return None
    ident = hashlib.sha256(f"{trend.kind}:{trend.label}".encode()).hexdigest()[:16]
    return SourceItem(
        id=ident,
        title=trend.label,
        url=first.url,
        source_name="weekly trends",
        published_at=now.isoformat(),
        text="\n".join(f"- {h.title}" for h in trend.headlines),
    )


def label_trends(
    trends: list[Trend],
    router: SummarizerRouter,
    state: StateStore,
    now: datetime,
    language: str,
    usage: Usage,
) -> list[str]:
    """Add a validated one-line label to each trend. Returns warnings."""
    warnings: list[str] = []
    version = prompt_version()
    for trend in trends:
        source = _source_for(trend, now)
        if source is None:
            continue
        chash = content_hash(source.url, document_for(source))
        keys = [
            (p, m, cache_key(chash, p, m, "headlines", language, version))
            for p, m in router.identities
        ]
        hit = next((c for _, _, k in keys if (c := state.get_summary(k)) is not None), None)
        summary: Summary | None
        if hit is not None:
            summary, provider, model = hit.summary, hit.provider, hit.model
        else:
            result = router.summarize(source, "headlines", language)
            usage.add(result.usage)
            warnings.extend(result.warnings)
            summary, provider, model = result.summary, result.provider, result.model
            if summary is not None:
                key = cache_key(chash, provider, model, "headlines", language, version)
                state.put_summary(key, summary, provider, model, now)
        if summary is not None and summary.status == "ok" and not summary.injection_flag:
            trend.blurb, trend.model = summary.one_line, f"{provider}/{model}"
    return warnings


def collect_trends(
    profile: Profile,
    reader: FeedReader | None,
    state: StateStore,
    now: datetime,
    *,
    router: SummarizerRouter | None,
    embedder: Embedder | None,
    usage: Usage,
) -> tuple[list[Trend], list[str]]:
    """The trends section for a weekly digest, or [] with a warning. Never raises for a bad
    feed or a bad model answer."""
    cfg = profile.trends
    if reader is None:
        return [], ["trends skipped: the shared feed is turned off in this profile"]
    snap = load_week(reader, now)
    trends, warnings = build_trends(
        snap.items, now, max_clusters=cfg.max_clusters, embedder=embedder, state=state
    )
    warnings = [*snap.warnings, *warnings]
    if cfg.llm_labels and trends and router is not None:
        warnings.extend(label_trends(trends, router, state, now, profile.delivery.language, usage))
    return trends, warnings
