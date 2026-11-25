"""Phase 6 inside run_digest: semantic scores, fail-soft embeddings, weights gating and weekly
trends. Items are built by hand so every order and score is known. Offline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from digest.embeddings.base import Embedder, EmbeddingError
from digest.extract import TextEnricher
from digest.feed_client import FeedError
from digest.pipeline import DigestRun, run_digest
from digest.profile import Profile
from digest.rank import rank
from digest.safe_http import SafeFetcher
from digest.state import StateStore
from tests.conftest import NOW, make_profile
from tests.stage23_helpers import (
    BrokenEmbedder,
    FakeEmbedder,
    MemoryFeed,
    item_json,
    shard_bytes,
)
from tests.test_rank import mk

LEAK = "AXON-PIPE-LEAK-5252"
LOW, HIGH = "01" * 8, "02" * 8


def profile(
    *, topics: tuple[str, ...] = ("agents-tooling",), frequency: str = "daily", **sections: Any
) -> Profile:
    day = {"weekday": "mon"} if frequency == "weekly" else {}
    return make_profile(
        interests={
            "topics": list(topics), "follow_entities": [],
            "keywords_include": [], "keywords_exclude": [],
        },
        delivery={"timezone": "UTC", "frequency": frequency, **day},
        **sections,
    )  # fmt: skip


def two_items() -> list[dict[str, Any]]:
    a = item_json(1, "crypto policy rules", tags=["agents-tooling"])
    b = item_json(2, "agent agent loops", tags=["agents-tooling"])
    return [{**a, "id": LOW}, {**b, "id": HIGH}]


def feed_of(items: list[dict[str, Any]]) -> MemoryFeed:
    return MemoryFeed({"latest.json": shard_bytes(items)})


def go(
    prof: Profile,
    reader: MemoryFeed,
    tmp: Path,
    *,
    embedder: Embedder | None = None,
    weights: dict[str, float] | None = None,
    with_trends: bool = True,
) -> DigestRun:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    fetcher = SafeFetcher(client, resolver=lambda _h: ["93.184.216.34"])
    return run_digest(
        prof, reader=reader, fetcher=fetcher, enricher=TextEnricher(reader, fetcher),
        router=None, state=StateStore(tmp), now=NOW, embedder=embedder, weights=weights,
        with_trends=with_trends,
    )  # fmt: skip


def order(run: DigestRun) -> list[str]:
    return [e.item.id for e in run.entries]


EMB = {"enabled": True, "provider": "huggingface", "model": "org/emb-model", "min_similarity": 0.3}


# ---- semantic scores in the run --------------------------------------------------------


def test_without_an_embedder_equal_matches_fall_back_to_id_order(tmp_path: Path) -> None:
    run = go(profile(), feed_of(two_items()), tmp_path)
    assert order(run) == [LOW, HIGH] and run.embedded == 0


def test_semantic_matching_reorders_equal_keyword_matches(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    run = go(profile(embeddings=EMB), feed_of(two_items()), tmp_path, embedder=emb)
    assert order(run) == [HIGH, LOW]
    assert run.embedded == 3 and run.embed_cache_hits == 0 and emb.texts_sent == 3


def test_a_fresh_run_and_a_cached_run_give_identical_scores(tmp_path: Path) -> None:
    prof, feed = profile(embeddings=EMB), feed_of(two_items())
    first = go(prof, feed, tmp_path, embedder=FakeEmbedder())
    emb = FakeEmbedder()
    second = go(prof, feed, tmp_path, embedder=emb)
    assert [e.score for e in first.entries] == [e.score for e in second.entries]
    assert emb.calls == [] and second.embedded == 0 and second.embed_cache_hits == 3


def test_an_embedder_is_ignored_when_the_profile_has_it_switched_off(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    run = go(profile(), feed_of(two_items()), tmp_path, embedder=emb)
    assert order(run) == [LOW, HIGH] and emb.calls == []


def test_a_semantic_match_can_surface_an_item_keywords_missed(tmp_path: Path) -> None:
    quiet = item_json(3, "agent agent agent", tags=["robotics"])
    prof = profile(embeddings=EMB)
    plain = go(prof, feed_of([quiet]), tmp_path / "a")
    assert plain.entries == []
    run = go(prof, feed_of([quiet]), tmp_path / "b", embedder=FakeEmbedder())
    assert order(run) == [quiet["id"]]


def test_no_interests_means_no_embedding_call(tmp_path: Path) -> None:
    prof = profile(topics=(), embeddings=EMB)
    emb = FakeEmbedder()
    run = go(prof, feed_of(two_items()), tmp_path, embedder=emb)
    assert emb.calls == [] and any("no interests" in w for w in run.warnings)


# ---- fail soft -------------------------------------------------------------------------


def test_an_embeddings_error_falls_back_to_keywords_with_a_warning(tmp_path: Path) -> None:
    run = go(profile(embeddings=EMB), feed_of(two_items()), tmp_path, embedder=BrokenEmbedder())
    assert order(run) == [LOW, HIGH]
    (warning,) = [w for w in run.warnings if "semantic matching skipped" in w]
    assert "keywords only" in warning and "HTTP 503" in warning


def test_an_unexpected_embedder_crash_is_a_warning_that_names_only_the_type(
    tmp_path: Path,
) -> None:
    emb = FakeEmbedder(fail=RuntimeError(f"boom {LEAK}"))
    run = go(profile(embeddings=EMB), feed_of(two_items()), tmp_path, embedder=emb)
    assert order(run) == [LOW, HIGH]
    text = " ".join(run.warnings)
    assert "semantic matching skipped, keywords only (RuntimeError)" in text and LEAK not in text


def test_a_short_answer_is_an_error_not_a_crash(tmp_path: Path) -> None:
    emb = FakeEmbedder(short=True)
    run = go(profile(embeddings=EMB), feed_of(two_items()), tmp_path, embedder=emb)
    assert order(run) == [LOW, HIGH]
    assert any("semantic matching skipped" in w for w in run.warnings)


def test_a_failed_embedding_caches_nothing(tmp_path: Path) -> None:
    emb = FakeEmbedder(fail=EmbeddingError("embeddings: rate limited (HTTP 429)"))
    go(profile(embeddings=EMB), feed_of(two_items()), tmp_path, embedder=emb)
    assert not (tmp_path / "embeddings").exists()


# ---- rank: the semantic term -----------------------------------------------------------


def test_the_semantic_term_adds_weight_times_score() -> None:
    prof = profile(embeddings={**EMB, "weight": 0.5})
    item = mk(tags=["agents-tooling"], base_score=0.4)
    (plain,) = rank([item], prof, seen=set(), limit=3)
    (boosted,) = rank([item], prof, seen=set(), limit=3, semantic={item.id: 0.8})
    assert boosted.score == pytest.approx(plain.score + 0.5 * 0.8)
    assert boosted.semantic == 0.8 and plain.semantic == 0.0


def test_semantic_scores_are_clamped_and_ignored_when_embeddings_are_off() -> None:
    on = profile(embeddings={**EMB, "weight": 1.0})
    off = profile()
    item = mk(tags=["agents-tooling"])
    assert rank([item], on, seen=set(), limit=3, semantic={item.id: 7.0})[0].semantic == 1.0
    assert rank([item], on, seen=set(), limit=3, semantic={item.id: -3.0})[0].semantic == 0.0
    assert rank([item], off, seen=set(), limit=3, semantic={item.id: 1.0})[0].semantic == 0.0


def test_a_semantic_only_match_passes_the_interest_filter() -> None:
    on = profile(embeddings=EMB)
    item = mk(tags=["robotics"])
    assert rank([item], on, seen=set(), limit=3) == []
    assert len(rank([item], on, seen=set(), limit=3, semantic={item.id: 0.4})) == 1


# ---- feedback weights ------------------------------------------------------------------


def tied_feed() -> MemoryFeed:
    a = {**item_json(1, "alpha story words", tags=["agents-tooling"]), "id": LOW}
    b = {**item_json(2, "bravo story terms", tags=["rag"]), "id": HIGH}
    return feed_of([a, b])


def test_weights_apply_only_when_feedback_is_enabled(tmp_path: Path) -> None:
    topics = ("agents-tooling", "rag")
    votes = {"rag": 1.0}
    off = go(profile(topics=topics), tied_feed(), tmp_path / "a", weights=votes)
    on = go(
        profile(topics=topics, feedback={"enabled": True}),
        tied_feed(),
        tmp_path / "b",
        weights=votes,
    )
    assert order(off) == [LOW, HIGH] and order(on) == [HIGH, LOW]


def test_a_downvoted_topic_is_demoted_not_removed(tmp_path: Path) -> None:
    topics = ("agents-tooling", "rag")
    prof = profile(topics=topics, feedback={"enabled": True})
    run = go(prof, tied_feed(), tmp_path, weights={"agents-tooling": -1.0})
    assert order(run) == [HIGH, LOW]


# ---- weekly trends ---------------------------------------------------------------------


def week_items() -> list[dict[str, Any]]:
    titles = ["Quartz scheduler ships checkpoints", "Marble tokenizer gains streaming",
              "Falcon gateway adds budget alerts", "Cobalt profiler visualizes memory"]  # fmt: skip
    return [
        item_json(n, t, tags=["agents-tooling"], hours_ago=20 + n) for n, t in enumerate(titles)
    ]


WEEKLY: dict[str, Any] = {"trends": {"enabled": True, "max_clusters": 3}}


def test_a_weekly_digest_gets_trends(tmp_path: Path) -> None:
    run = go(profile(frequency="weekly", **WEEKLY), feed_of(week_items()), tmp_path)
    assert [(t.label, t.kind, t.stories) for t in run.trends] == [("agents tooling", "tag", 4)]
    assert run.usage.calls == 0  # no model unless llm_labels is on


def test_trends_need_weekly_enabled_and_the_flag(tmp_path: Path) -> None:
    feed = feed_of(week_items())
    assert go(profile(frequency="daily", **WEEKLY), feed, tmp_path / "a").trends == []
    assert go(profile(frequency="weekly"), feed, tmp_path / "b").trends == []
    skipped = go(profile(frequency="weekly", **WEEKLY), feed, tmp_path / "c", with_trends=False)
    assert skipped.trends == []


def test_with_trends_false_never_reads_the_week_shards(tmp_path: Path) -> None:
    feed = feed_of(week_items())
    go(profile(frequency="weekly", **WEEKLY), feed, tmp_path, with_trends=False)
    assert "tags/rag.json" not in feed.reads  # only the user's own topics were read


class FlakyFeed(MemoryFeed):
    def __init__(self, files: dict[str, bytes], error: Exception) -> None:
        super().__init__(files)
        self.error = error

    def read_bytes(self, rel: str) -> bytes | None:
        if rel == "tags/rag.json":
            raise self.error
        return super().read_bytes(rel)


def test_a_trends_feed_error_is_a_warning_and_the_digest_survives(tmp_path: Path) -> None:
    feed = FlakyFeed(feed_of(week_items()).files, FeedError("tags/rag.json: network error"))
    run = go(profile(frequency="weekly", **WEEKLY), feed, tmp_path)
    assert len(run.entries) == 4 and run.trends == []
    assert any(w.startswith("trends skipped: tags/rag.json") for w in run.warnings)


def test_an_unexpected_trends_crash_names_only_the_type(tmp_path: Path) -> None:
    feed = FlakyFeed(feed_of(week_items()).files, RuntimeError(f"boom {LEAK}"))
    run = go(profile(frequency="weekly", **WEEKLY), feed, tmp_path)
    assert len(run.entries) == 4 and run.trends == []
    text = " ".join(run.warnings)
    assert "trends skipped (RuntimeError)" in text and LEAK not in text


def test_trends_use_embeddings_for_duplicates_only_when_enabled(tmp_path: Path) -> None:
    items = [item_json(n, f"agent topic {w}", tags=["agents-tooling"]) for n, w in
             enumerate("abc", start=1)]  # fmt: skip
    prof = profile(frequency="weekly", embeddings=EMB, **WEEKLY)
    emb = FakeEmbedder()
    run = go(prof, feed_of(items), tmp_path / "a", embedder=emb)
    assert run.trends == []  # the three look like one story to the embedder
    off = go(profile(frequency="weekly", **WEEKLY), feed_of(items), tmp_path / "b", embedder=emb)
    assert [t.stories for t in off.trends] == [3]
