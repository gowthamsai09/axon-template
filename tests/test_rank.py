from __future__ import annotations

from datetime import UTC, datetime

from digest.feed_client import FeedItem, LocalFeed, load_feed
from digest.profile import Interests, Profile
from digest.rank import exclusion_penalty, has_interests, interest_match, rank
from tests.conftest import make_profile


def prof(**over: object) -> Profile:
    base: dict[str, object] = {"topics": [], "follow_entities": [], "keywords_include": [],
                               "keywords_exclude": []}  # fmt: skip
    base.update(over)
    return make_profile(interests=base)


def mk(id_: str = "a" * 16, **over: object) -> FeedItem:
    data: dict[str, object] = {
        "id": id_, "url": "https://example.com/x", "canonical_url": "https://example.com/x",
        "title": "A title", "source_id": "src", "published_at": datetime(2026, 10, 20, tzinfo=UTC),
        "snippet": "", "tags": [], "entities": [], "base_score": 0.5,
    }  # fmt: skip
    data.update(over)
    return FeedItem.model_validate(data)


def test_topic_match_scales_with_hits() -> None:
    interests = Interests(topics=["rag", "agents-tooling"])
    assert interest_match(mk(tags=["rag"]), interests) == 0.25
    assert interest_match(mk(tags=["rag", "agents-tooling"]), interests) == 0.5
    assert interest_match(mk(tags=["robotics"]), interests) == 0


def test_entity_and_keyword_match_whole_token() -> None:
    interests = Interests(
        follow_entities=["Qwen", "github:ollama/ollama"], keywords_include=["local inference"]
    )
    assert interest_match(mk(entities=["qwen"]), interests) == 0.3
    assert interest_match(mk(title="Notes on Qwen3"), interests) == 0  # no substring matches
    assert (
        interest_match(
            mk(canonical_url="https://github.com/ollama/ollama/releases/tag/v1"), interests
        )
        == 0.3
    )
    assert (
        interest_match(mk(snippet="fast local-inference engine"), interests) == 0.1
    )  # hyphen == space


def test_exclusion_penalty_phrase() -> None:
    interests = Interests(keywords_exclude=["funding round"])
    assert exclusion_penalty(mk(title="Startup closes funding round"), interests) == 1.0
    assert exclusion_penalty(mk(title="Funding rounds explained"), interests) == 0.0


def test_has_interests_only_positive_terms_filter() -> None:
    assert not has_interests(Interests())
    assert not has_interests(Interests(keywords_exclude=["x"]))
    assert has_interests(Interests(topics=["rag"]))


def test_rank_filters_seen_and_non_matching_and_sorts() -> None:
    profile = prof(topics=["rag"], keywords_exclude=[])
    a = mk("a" * 16, tags=["rag"], base_score=0.5)
    b = mk("b" * 16, tags=["rag"], base_score=0.7)
    c = mk("c" * 16, tags=[], base_score=0.99)  # no match: dropped
    d = mk("d" * 16, tags=["rag"], base_score=0.9)
    out = rank([a, b, c, d], profile, seen={"d" * 16}, limit=10)
    assert [s.item.id for s in out] == ["b" * 16, "a" * 16]
    assert out[0].score == 0.7 + 0.5 * 0.5
    assert len(rank([a, b], profile, seen=set(), limit=1)) == 1


def test_exclusion_lowers_score_and_extra_rss_always_qualifies() -> None:
    profile = prof(topics=["rag"], keywords_exclude=["crypto"])
    plain = mk("a" * 16, tags=["rag"], base_score=0.5)
    bad = mk("b" * 16, tags=["rag"], title="Crypto rag", base_score=0.5)
    own = mk("c" * 16, from_extra_rss=True, base_score=0.4)
    out = {s.item.id: s for s in rank([plain, bad, own], profile, seen=set(), limit=10)}
    assert out["b" * 16].score < out["a" * 16].score
    assert "c" * 16 in out and out["c" * 16].match == 0.5


def test_ties_break_by_recency_then_id() -> None:
    profile = prof(topics=[], keywords_exclude=[])
    old = mk("b" * 16, published_at=datetime(2026, 10, 1, tzinfo=UTC))
    new = mk("c" * 16, published_at=datetime(2026, 10, 2, tzinfo=UTC))
    tie = mk("a" * 16, published_at=datetime(2026, 10, 2, tzinfo=UTC))
    assert [s.item.id for s in rank([old, new, tie], profile, seen=set(), limit=3)] == [
        "a" * 16, "c" * 16, "b" * 16,
    ]  # fmt: skip


def test_rank_on_real_fixture(feed_dir: object) -> None:
    from tests.conftest import FEED

    profile = make_profile(interests={"topics": ["agents-tooling"], "follow_entities": [],
                                      "keywords_include": [], "keywords_exclude": []})  # fmt: skip
    snap = load_feed(LocalFeed(FEED), profile.interests)
    top = rank(snap.items, profile, seen=set(), limit=10)
    assert top[0].item.id == "aa4ccde86d8a3d2e"
    assert all("agents-tooling" in s.item.tags for s in top)  # HN noise never qualifies
