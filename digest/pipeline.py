"""The run flow (spec 6.3 steps 1 to 8): feed -> drop seen -> rank -> text -> summarize
(cache, router, extractive fallback). Delivery and state writes happen in the CLI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .embeddings.base import Embedder, EmbeddingError
from .embeddings.matcher import semantic_scores
from .extract import TextEnricher
from .feed_client import FeedError, FeedItem, FeedReader, fetch_extra_rss, load_feed
from .profile import Profile
from .rank import rank
from .safe_http import SafeFetcher
from .state import StateStore, cache_key, content_hash
from .summarizers.base import SourceItem, Summary, Usage
from .summarizers.prompting import document_for, prompt_version
from .summarizers.router import SummarizerRouter
from .trends import Trend, collect_trends

STALE_FEED_HOURS = 30
_NO_PREVIEW = "(no preview available)"


@dataclass
class DigestEntry:
    item: FeedItem
    score: float
    summary: Summary | None = None
    fallback_snippet: str | None = None
    provider: str = ""
    model: str = ""
    cached: bool = False
    note: str = ""


@dataclass
class DigestRun:
    entries: list[DigestEntry] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cache_hits: int = 0
    candidates: int = 0
    trends: list[Trend] = field(default_factory=list)
    embedded: int = 0  # texts sent to the embeddings provider this run
    embed_cache_hits: int = 0


def extractive_snippet(item: FeedItem, text: str) -> str:
    raw = " ".join((item.snippet or text).split())
    return (raw[:297] + "...") if len(raw) > 300 else (raw or _NO_PREVIEW)


def run_digest(
    profile: Profile,
    *,
    reader: FeedReader | None,
    fetcher: SafeFetcher | None,
    enricher: TextEnricher,
    router: SummarizerRouter | None,
    state: StateStore,
    now: datetime,
    max_items: int | None = None,
    use_seen: bool = True,
    embedder: Embedder | None = None,
    weights: Mapping[str, float] | None = None,
    with_trends: bool = True,
) -> DigestRun:
    run = DigestRun()
    items: list[FeedItem] = []
    if profile.sources.use_shared_feed:
        if reader is None:
            raise ValueError("a feed reader is required when use_shared_feed is true")
        snap = load_feed(reader, profile.interests)
        items.extend(snap.items)
        run.warnings.extend(snap.warnings)
        if snap.generated_at and now - snap.generated_at > timedelta(hours=STALE_FEED_HOURS):
            hours = int((now - snap.generated_at).total_seconds() // 3600)
            run.warnings.append(f"the shared feed is {hours} hours old (collector may be stalled)")
    if profile.sources.extra_rss:
        if fetcher is None:
            run.warnings.append("extra_rss ignored: no fetcher available")
        else:
            extra, extra_warnings = fetch_extra_rss(profile.sources.extra_rss, fetcher, now)
            items.extend(i for i in extra if i.id not in {x.id for x in items})
            run.warnings.extend(extra_warnings)
    run.candidates = len(items)
    limit = max_items or profile.delivery.max_items
    seen = set(state.seen()) if use_seen else set()
    semantic = _semantic(run, profile, items, seen, embedder, state, now)
    votes = weights if profile.feedback.enabled else None
    top = rank(items, profile, seen=seen, limit=limit, semantic=semantic, weights=votes)
    depth, language = profile.delivery.depth, profile.delivery.language
    version = prompt_version()
    for scored in top:
        item = scored.item
        text = enricher.text_for(item)
        run.warnings.extend(text.warnings)
        entry = DigestEntry(item, scored.score)
        source = SourceItem(
            id=item.id,
            title=item.title,
            url=item.canonical_url,
            source_name=item.source_id,
            published_at=item.published_at.isoformat(),
            text=text.text,
        )
        fallback = extractive_snippet(item, text.text)
        if router is None:
            entry.fallback_snippet = fallback
            run.entries.append(entry)
            continue
        chash = content_hash(item.canonical_url, document_for(source))
        keys = [
            (p, m, cache_key(chash, p, m, depth, language, version)) for p, m in router.identities
        ]
        result_summary: Summary | None
        hit = next((c for _, _, k in keys if (c := state.get_summary(k)) is not None), None)
        if hit is not None:
            run.cache_hits += 1
            result_summary, entry.provider, entry.model, entry.cached = (
                hit.summary, hit.provider, hit.model, True,
            )  # fmt: skip
        else:
            result = router.summarize(source, depth, language)
            run.usage.add(result.usage)
            run.warnings.extend(result.warnings)
            result_summary, entry.provider, entry.model = (
                result.summary, result.provider, result.model,
            )  # fmt: skip
            if result_summary is not None:
                state.put_summary(
                    cache_key(chash, result.provider, result.model, depth, language, version),
                    result_summary, result.provider, result.model, now,
                )  # fmt: skip
        _apply(entry, result_summary, fallback, run)
        run.entries.append(entry)
    if with_trends and profile.trends.enabled and profile.delivery.frequency == "weekly":
        _trends(run, profile, reader, state, now, router, embedder)
    run.notices.extend(router.notices if router is not None else [])
    run.warnings.extend(state.warnings)
    return run


def _semantic(
    run: DigestRun,
    profile: Profile,
    items: list[FeedItem],
    seen: set[str],
    embedder: Embedder | None,
    state: StateStore,
    now: datetime,
) -> dict[str, float] | None:
    """Semantic scores, or None (keyword matching only) with a warning on any failure."""
    if embedder is None or not profile.embeddings.enabled:
        return None
    pool = [i for i in items if i.id not in seen]
    try:
        result = semantic_scores(pool, profile.interests, profile.embeddings, embedder, state, now)
    except EmbeddingError as exc:
        run.warnings.append(f"semantic matching skipped, keywords only: {exc}")
        return None
    except Exception as exc:  # an optional extra must never cost the user their digest
        name = type(exc).__name__
        run.warnings.append(f"semantic matching skipped, keywords only ({name})")
        return None
    run.warnings.extend(result.warnings)
    run.embedded, run.embed_cache_hits = result.embedded, result.cached
    return result.scores


def _trends(
    run: DigestRun,
    profile: Profile,
    reader: FeedReader | None,
    state: StateStore,
    now: datetime,
    router: SummarizerRouter | None,
    embedder: Embedder | None,
) -> None:
    """Fill run.trends for a weekly digest. Any failure is a warning, never a failed run."""
    try:
        run.trends, extra = collect_trends(
            profile, reader, state, now, router=router,
            embedder=embedder if profile.embeddings.enabled else None, usage=run.usage,
        )  # fmt: skip
    except FeedError as exc:
        run.warnings.append(f"trends skipped: {exc}")
        return
    except Exception as exc:
        run.warnings.append(f"trends skipped ({type(exc).__name__})")
        return
    run.warnings.extend(extra)


def _apply(entry: DigestEntry, summary: Summary | None, fallback: str, run: DigestRun) -> None:
    if summary is None:
        entry.fallback_snippet = fallback
        entry.note = "not AI-summarized (extractive snippet)"
    elif summary.status == "suspected_injection":
        entry.fallback_snippet = fallback
        entry.note = "source text contained instructions aimed at AI systems; not summarized"
        run.warnings.append(f"{entry.item.id}: suspected prompt injection in the source")
    elif summary.status == "insufficient_content":
        entry.fallback_snippet = fallback
        entry.note = "too little content to summarize"
    else:
        entry.summary = summary
        if summary.injection_flag:
            entry.note = "the source contained text aimed at AI systems (ignored)"
            run.warnings.append(f"{entry.item.id}: injection flag set by the summarizer")
