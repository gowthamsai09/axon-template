"""P6.4 embeddings: vector helpers, the HTTP client for both providers, get_embedder (SSRF,
URL and key rules), profile validation, and the matcher with its cache. Offline."""

from __future__ import annotations

import json
import math
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from digest.embeddings.base import (
    MAX_DIM,
    EmbeddingError,
    cosine,
    normalize,
    pack,
    quantize,
    unpack,
)
from digest.embeddings.client import (
    HF_URL,
    HF_URL_RE,
    HttpEmbedder,
    get_embedder,
    status_error,
)
from digest.embeddings.matcher import (
    MAX_INTEREST_TEXTS,
    MAX_TEXT_CHARS,
    clean_text,
    interest_texts,
    item_text,
    semantic_scores,
    vector_key,
    vectors_for,
)
from digest.profile import Embeddings, Interests
from digest.state import CachedVector, StateStore
from tests.chat_helpers import Script
from tests.conftest import NOW, make_profile
from tests.stage23_helpers import EMB_MODEL, FakeEmbedder, embed_profile, feed_item

KEY = "AXON-EMBED-KEY-424242"
OK_HOST = "93.184.216.34"
BASE = "https://api.example.com/v1"


def hf_ok(rows: list[list[float]]) -> httpx.Response:
    return httpx.Response(200, json=rows)


def oa_ok(rows: list[list[float]]) -> httpx.Response:
    return httpx.Response(
        200, json={"data": [{"index": i, "embedding": r} for i, r in enumerate(rows)]}
    )


def embedder(
    script: Script, *, wire: str = "hf", batch: int = 16, sleeps: list[float] | None = None
) -> HttpEmbedder:
    url = HF_URL.format(model=EMB_MODEL) if wire == "hf" else BASE + "/embeddings"
    return HttpEmbedder(
        provider="huggingface" if wire == "hf" else "openai_compatible",
        model=EMB_MODEL, url=url, wire="hf" if wire == "hf" else "openai",
        api_key=KEY, http=script.client, batch_size=batch,
        sleep=(sleeps.append if sleeps is not None else (lambda _s: None)),
    )  # fmt: skip


# ---- vector helpers --------------------------------------------------------------------


def test_normalize_gives_unit_length() -> None:
    unit = normalize([3, 4])
    assert unit == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(v * v for v in unit)) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "bad",
    [
        [], [0, 0], [1, "x"], [1, None], [True, 2], [1, float("nan")], [float("inf"), 1],
        [10**400, 1], [1] * (MAX_DIM + 1),
    ],
)  # fmt: skip
def test_normalize_rejects_unusable_vectors(bad: list[Any]) -> None:
    with pytest.raises(EmbeddingError):
        normalize(bad)


def test_pack_unpack_round_trip_and_quantize_matches_a_cached_copy() -> None:
    unit = normalize([0.3, -1.2, 5.0, 0.01])
    restored = unpack(pack(unit), 4)
    assert restored is not None and restored == pytest.approx(unit, abs=1e-3)
    assert quantize([0.3, -1.2, 5.0, 0.01]) == restored  # fresh and cached agree exactly


@pytest.mark.parametrize("data", ["!!!", "AAAA", "", "é"])
def test_unpack_rejects_garbage(data: str) -> None:
    assert unpack(data, 4) is None


def test_unpack_rejects_non_finite_values() -> None:
    assert unpack(pack([float("inf"), 0.0]), 2) is None


def test_cosine_edges() -> None:
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 0], [1, 0, 0]) == 0.0 and cosine([], []) == 0.0
    assert cosine([0, 0], [1, 0]) == 0.0
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)


# ---- status mapping --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "needle"),
    [
        (401, "EMBED_API_KEY was rejected"), (403, "rejected"), (402, "credits exhausted"),
        (404, "not found"), (400, "request was rejected"), (422, "rejected"),
        (429, "rate limited"), (500, "unavailable"), (503, "unavailable"), (302, "unexpected"),
    ],
)  # fmt: skip
def test_status_messages_are_fixed_and_mention_the_code(code: int, needle: str) -> None:
    message = str(status_error(code))
    assert needle in message and str(code) in message


# ---- the HTTP client -------------------------------------------------------------------


def test_hf_request_shape_and_key_in_header_only() -> None:
    script = Script(lambda: hf_ok([[1.0, 0.0], [0.0, 1.0]]))
    vectors = embedder(script).embed(["a", "b"])
    (req,) = script.requests
    assert vectors == [pytest.approx([1.0, 0.0]), pytest.approx([0.0, 1.0])]
    assert req.url.host == "router.huggingface.co" and req.method == "POST"
    assert HF_URL_RE.fullmatch(str(req.url)) is not None
    assert req.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(req.content) == {"inputs": ["a", "b"]}
    assert KEY not in str(req.url) and KEY not in req.content.decode()


def test_openai_request_shape_and_index_order() -> None:
    body = {"data": [{"index": 1, "embedding": [0, 2]}, {"index": 0, "embedding": [3, 0]}]}
    script = Script(lambda: httpx.Response(200, json=body))
    vectors = embedder(script, wire="openai").embed(["a", "b"])
    (req,) = script.requests
    assert str(req.url) == BASE + "/embeddings"
    assert json.loads(req.content) == {"model": EMB_MODEL, "input": ["a", "b"]}
    assert vectors[0] == pytest.approx([1.0, 0.0]) and vectors[1] == pytest.approx([0.0, 1.0])


def test_texts_are_sent_in_batches_in_order() -> None:
    two, one = hf_ok([[1.0, 1.0]] * 2), hf_ok([[1.0, 1.0]])
    script = Script(lambda: two, two, two, one)
    vectors = embedder(script, batch=2).embed(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert [b["inputs"] for b in script.bodies()] == [["a", "b"], ["c", "d"], ["e"]]


def test_no_texts_means_no_request() -> None:
    script = Script(lambda: hf_ok([]))
    assert embedder(script).embed([]) == [] and script.requests == []


@pytest.mark.parametrize(
    "answer",
    [
        [[1.0, 0.0]],  # one row for two texts
        [[[1.0], [2.0]], [[1.0], [2.0]]],  # token-level output
        [[1.0, "x"], [1.0, 0.0]],
        [[], [1.0]],
        [[0.0, 0.0], [1.0, 0.0]],  # zero vector
        {"error": "x"},
        "text",
        [[1.0, 0.0], [1.0]],  # different sizes
    ],
)
def test_hf_unusable_answers_raise_a_fixed_error(answer: Any) -> None:
    script = Script(lambda: httpx.Response(200, json=answer))
    with pytest.raises(EmbeddingError) as err:
        embedder(script).embed(["a", "b"])
    assert str(err.value).startswith("embeddings:")


@pytest.mark.parametrize(
    "body",
    [
        {"data": [{"index": 0, "embedding": [1, 0]}]},
        {"data": [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [1, 0]}]},
        {"data": [{"index": 5, "embedding": [1, 0]}, {"index": 0, "embedding": [1, 0]}]},
        {"data": [{"index": True, "embedding": [1, 0]}, {"index": 1, "embedding": [1, 0]}]},
        {"data": [{"index": 0, "embedding": "x"}, {"index": 1, "embedding": [1, 0]}]},
        {"data": [7, 8]},
        {"data": "x"},
        [],
    ],
)
def test_openai_unusable_answers_raise_a_fixed_error(body: Any) -> None:
    script = Script(lambda: httpx.Response(200, json=body))
    with pytest.raises(EmbeddingError):
        embedder(script, wire="openai").embed(["a", "b"])


def test_a_non_json_answer_is_reported_without_echoing_it() -> None:
    marker = "UNIQUE-BODY-ECHO-5151"
    script = Script(lambda: httpx.Response(200, text=f"<html>{marker} {KEY}</html>"))
    with pytest.raises(EmbeddingError) as err:
        embedder(script).embed(["a"])
    assert marker not in str(err.value) and KEY not in str(err.value)


def test_an_oversized_answer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("digest.embeddings.client.MAX_RESPONSE_BYTES", 10)
    script = Script(lambda: hf_ok([[1.0, 0.0]]))
    with pytest.raises(EmbeddingError, match="too large"):
        embedder(script).embed(["a"])


@pytest.mark.parametrize("code", [400, 401, 402, 403, 404, 422])
def test_error_statuses_never_echo_the_body_or_the_key(code: int) -> None:
    marker = "UNIQUE-BODY-ECHO-8282"
    script = Script(lambda: httpx.Response(code, text=f"{marker} Bearer {KEY}"))
    with pytest.raises(EmbeddingError) as err:
        embedder(script).embed(["a"])
    assert str(code) in str(err.value)
    assert marker not in str(err.value) and KEY not in str(err.value)
    assert len(script.requests) == 1  # a client error is not retried


def test_429_and_5xx_are_retried_then_succeed() -> None:
    sleeps: list[float] = []
    busy = httpx.Response(429, headers={"retry-after": "3"})
    script = Script(lambda: hf_ok([[1.0, 0.0]]), busy, httpx.Response(503))
    assert len(embedder(script, sleeps=sleeps).embed(["a"])) == 1
    assert len(script.requests) == 3 and sleeps[0] == 3.0


def test_retries_stop_and_report_the_last_status() -> None:
    script = Script(lambda: httpx.Response(503), httpx.Response(503), httpx.Response(503))
    with pytest.raises(EmbeddingError, match="HTTP 503"):
        embedder(script).embed(["a"])
    assert len(script.requests) == 3


@pytest.mark.parametrize("hint", ["١٢", "-1", "1.5", "99999", "abc", ""])
def test_a_bad_retry_after_falls_back_to_backoff(hint: str) -> None:
    sleeps: list[float] = []
    script = Script(
        lambda: hf_ok([[1.0, 0.0]]), httpx.Response(429, headers={b"retry-after": hint.encode()})
    )
    embedder(script, sleeps=sleeps).embed(["a"])
    assert sleeps == [1.0]  # 2 ** 0, never the hint


def test_an_unreachable_service_gives_a_fixed_message() -> None:
    boom = httpx.ConnectError(f"cannot reach {KEY}")
    script = Script(lambda: hf_ok([[1.0, 0.0]]), boom, boom, boom)
    with pytest.raises(EmbeddingError) as err:
        embedder(script).embed(["a"])
    assert KEY not in str(err.value) and "unreachable" in str(err.value)


def test_redirects_are_not_followed() -> None:
    moved = httpx.Response(302, headers={"location": "https://evil.example/steal"})
    script = Script(lambda: hf_ok([[1.0, 0.0]]), moved)
    with pytest.raises(EmbeddingError, match="HTTP 302"):
        embedder(script).embed(["a"])
    assert [r.url.host for r in script.requests] == ["router.huggingface.co"]


def test_the_embedder_never_shows_its_key_or_url() -> None:
    text = repr(embedder(Script(lambda: hf_ok([]))))
    assert KEY not in text and "router.huggingface.co" not in text and EMB_MODEL in text


# ---- get_embedder ----------------------------------------------------------------------


def make(profile_over: dict[str, Any], env: dict[str, str] | None = None) -> Any:
    profile = make_profile(embeddings=profile_over)
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    return get_embedder(
        profile, {"EMBED_API_KEY": KEY} if env is None else env,
        http=http, resolver=lambda _h: [OK_HOST],
    )  # fmt: skip


def test_disabled_gives_nothing_and_no_warning() -> None:
    assert make({}) == (None, [])


def test_a_missing_key_turns_it_off_with_a_clear_warning() -> None:
    emb, warnings = make({"enabled": True, "model": EMB_MODEL}, env={})
    assert emb is None and warnings == [
        "semantic matching is off: the EMBED_API_KEY secret is not set"
    ]


def test_huggingface_builds_an_embedder_on_the_fixed_host() -> None:
    emb, warnings = make({"enabled": True, "model": EMB_MODEL})
    assert warnings == [] and emb is not None
    assert (emb.provider, emb.model) == ("huggingface", EMB_MODEL)
    assert "router.huggingface.co" in emb._url and KEY not in repr(emb)


def test_openai_compatible_builds_with_a_public_host() -> None:
    cfg = {"enabled": True, "provider": "openai_compatible", "model": "m", "base_url": BASE}
    emb, warnings = make(cfg)
    assert warnings == [] and emb is not None and emb._url == BASE + "/embeddings"
    emb2, _ = make({**cfg, "base_url": BASE + "/"})
    assert emb2 is not None and emb2._url == BASE + "/embeddings"


@pytest.mark.parametrize("host", ["127.0.0.1", "10.0.0.5", "169.254.169.254", "192.168.1.1"])
def test_openai_compatible_refuses_private_addresses(host: str) -> None:
    cfg = {
        "enabled": True, "provider": "openai_compatible", "model": "m",
        "base_url": f"https://{host}/v1",
    }  # fmt: skip
    emb, warnings = make(cfg)
    assert emb is None and warnings and "rejected" in warnings[0] and KEY not in warnings[0]


def test_openai_compatible_refuses_a_name_that_resolves_privately() -> None:
    profile = make_profile(
        embeddings={
            "enabled": True, "provider": "openai_compatible", "model": "m", "base_url": BASE,
        }
    )  # fmt: skip
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    emb, warnings = get_embedder(
        profile, {"EMBED_API_KEY": KEY}, http=http, resolver=lambda _h: ["10.1.2.3"]
    )
    assert emb is None and "non-public" in warnings[0]


def test_the_hf_model_must_fit_the_url_pattern() -> None:
    # the profile already rejects odd models; HF_URL_RE is the second, independent check
    for model in ("a/b", "model", "org/name.v2-x_1"):
        assert HF_URL_RE.fullmatch(HF_URL.format(model=model)) is not None
    for model in ("a/b/c", "../x", "a b", "a?x=1", "a#f", "a/../b", "@evil", "a\n/b", ""):
        assert HF_URL_RE.fullmatch(HF_URL.format(model=model)) is None


def test_get_embedder_checks_the_model_again_even_for_a_loosely_built_profile() -> None:
    loose = Embeddings.model_construct(
        enabled=True, provider="huggingface", model="a/../b", base_url="",
        weight=0.25, min_similarity=0.35, max_items=50,
    )  # fmt: skip
    profile = make_profile().model_copy(update={"embeddings": loose})
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    emb, warnings = get_embedder(profile, {"EMBED_API_KEY": KEY}, http=http)
    assert emb is None and warnings == ["semantic matching is off: embeddings.model is not usable"]


# ---- profile validation ----------------------------------------------------------------


def valid(**over: Any) -> Embeddings:
    base: dict[str, Any] = {"enabled": True, "model": EMB_MODEL}
    return Embeddings.model_validate({**base, **over})


def test_defaults_are_off_and_safe() -> None:
    cfg = Embeddings()
    assert (cfg.enabled, cfg.provider) == (False, "huggingface")
    assert (cfg.weight, cfg.max_items) == (0.25, 50)
    assert make_profile().embeddings.enabled is False
    assert make_profile().trends.enabled is False and make_profile().feedback.enabled is False


@pytest.mark.parametrize(
    "over",
    [
        {"model": ""}, {"model": "a b"}, {"model": "a/b/c"}, {"model": "../x"},
        {"model": "x" * 101}, {"provider": "deepseek"}, {"weight": 1.5}, {"weight": -0.1},
        {"weight": float("nan")}, {"min_similarity": 1.0}, {"min_similarity": -0.1},
        {"min_similarity": float("inf")}, {"max_items": 0}, {"max_items": 201},
        {"base_url": BASE}, {"surprise": 1},
        {"provider": "openai_compatible", "base_url": ""},
        {"provider": "openai_compatible", "base_url": "http://api.example.com/v1"},
        {"provider": "openai_compatible", "base_url": "https://u:p@api.example.com/v1"},
        {"provider": "openai_compatible", "base_url": BASE + "?k=1"},
        {"provider": "openai_compatible", "base_url": BASE + "#f"},
        {"provider": "openai_compatible", "base_url": BASE, "model": "a//b"},
        {"provider": "openai_compatible", "base_url": BASE, "model": "a/../b"},
    ],
)  # fmt: skip
def test_bad_embedding_settings_are_rejected(over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        valid(**over)


def test_good_openai_compatible_settings_pass() -> None:
    cfg = valid(provider="openai_compatible", base_url=BASE, model="text-embedding-3-small")
    assert cfg.base_url == BASE


def test_disabled_embeddings_need_no_model() -> None:
    assert Embeddings(enabled=False).model == ""


def test_the_trends_and_feedback_sections_validate() -> None:
    make_profile(trends={"enabled": True, "llm_labels": True, "max_clusters": 10})
    with pytest.raises(ValidationError):
        make_profile(trends={"enabled": False, "llm_labels": True})
    with pytest.raises(ValidationError):
        make_profile(trends={"enabled": True, "max_clusters": 11})
    with pytest.raises(ValidationError):
        make_profile(feedback={"enabled": True, "surprise": 1})


# ---- matcher: text helpers -------------------------------------------------------------


def test_clean_text_makes_one_short_line() -> None:
    assert clean_text("a\x00b\n\n c\t\x1bd", 100) == "a b c d"
    assert clean_text("x" * 50, 10) == "x" * 10


def test_item_text_is_capped() -> None:
    item = feed_item(1, "Title", snippet="s" * 5000)
    text = item_text(item)
    assert text.startswith("Title. ") and len(text) == MAX_TEXT_CHARS


def test_interest_texts_are_deduplicated_cleaned_and_capped() -> None:
    interests = Interests(
        topics=["agents-tooling", "rag"], follow_entities=["github:langchain-ai/langgraph"],
        keywords_include=["RAG", "tool calling"], keywords_exclude=["nope"],
    )  # fmt: skip
    assert interest_texts(interests) == [
        "agents tooling",
        "rag",
        "langchain-ai/langgraph",
        "tool calling",
    ]
    many = Interests(topics=[], follow_entities=[], keywords_include=[f"k{i}" for i in range(50)])
    assert len(interest_texts(many)) == MAX_INTEREST_TEXTS


def test_vector_keys_depend_on_provider_model_and_text() -> None:
    base = vector_key("p", "m", "t")
    assert len(base) == 64 and base == vector_key("p", "m", "t")
    assert len({base, vector_key("q", "m", "t"), vector_key("p", "n", "t"),
                vector_key("p", "m", "u")}) == 4  # fmt: skip


# ---- matcher: cache and scores ---------------------------------------------------------


def test_vectors_are_cached_and_a_rerun_costs_nothing(tmp_path: Path) -> None:
    state, emb = StateStore(tmp_path), FakeEmbedder()
    first, embedded, cached = vectors_for(["agent a", "rag b"], emb, state, NOW)
    assert (embedded, cached) == (2, 0)
    again, embedded2, cached2 = vectors_for(["agent a", "rag b"], emb, StateStore(tmp_path), NOW)
    assert (embedded2, cached2) == (0, 2) and again == first  # identical, not just close
    assert emb.texts_sent == 2


def test_duplicate_texts_are_sent_once(tmp_path: Path) -> None:
    emb = FakeEmbedder()
    vectors, embedded, _ = vectors_for(["same", "same", "other"], emb, StateStore(tmp_path), NOW)
    assert embedded == 2 and emb.texts_sent == 2 and vectors[0] == vectors[1]


def test_a_changed_model_or_provider_does_not_reuse_vectors(tmp_path: Path) -> None:
    state = StateStore(tmp_path)
    vectors_for(["agent"], FakeEmbedder(), state, NOW)
    other = FakeEmbedder(model="org/other")
    assert vectors_for(["agent"], other, state, NOW)[1] == 1
    third = FakeEmbedder(provider="openai_compatible")
    assert vectors_for(["agent"], third, state, NOW)[1] == 1


def test_a_corrupt_cache_entry_is_re_embedded(tmp_path: Path) -> None:
    state, emb = StateStore(tmp_path), FakeEmbedder()
    vectors_for(["agent"], emb, state, NOW)
    (path,) = (tmp_path / "embeddings").glob("*.json")
    path.write_text("{broken", encoding="utf-8")
    fresh = StateStore(tmp_path)
    assert vectors_for(["agent"], emb, fresh, NOW)[1] == 1


def test_a_cache_entry_with_a_wrong_size_is_re_embedded(tmp_path: Path) -> None:
    state, emb = StateStore(tmp_path), FakeEmbedder()
    key = vector_key(emb.provider, emb.model, "agent")
    state.put_vector(key, CachedVector(
        cached_at=NOW, provider=emb.provider, model=emb.model, dim=3, v=pack([1.0, 0.0]),
    ))  # fmt: skip
    assert vectors_for(["agent"], emb, state, NOW)[1] == 1


def test_an_embedder_that_returns_too_few_vectors_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingError):
        vectors_for(["a", "b"], FakeEmbedder(short=True), StateStore(tmp_path), NOW)


def test_an_embedder_that_fails_caches_nothing(tmp_path: Path) -> None:
    emb = FakeEmbedder(fail=EmbeddingError("embeddings: rate limited (HTTP 429)"))
    with pytest.raises(EmbeddingError):
        vectors_for(["a"], emb, StateStore(tmp_path), NOW)
    assert not (tmp_path / "embeddings").exists()


def test_old_cached_vectors_are_pruned_after_seven_days(tmp_path: Path) -> None:
    state, emb = StateStore(tmp_path), FakeEmbedder()
    vectors_for(["agent"], emb, state, NOW - timedelta(days=8))
    vectors_for(["rag"], emb, state, NOW - timedelta(days=2))
    assert state.prune(NOW) == 1
    assert len(list((tmp_path / "embeddings").glob("*.json"))) == 1


def cfg(**over: Any) -> Embeddings:
    return valid(**over)


def test_related_items_score_higher_than_unrelated_ones(tmp_path: Path) -> None:
    interests = Interests(topics=["agents-tooling"], follow_entities=[], keywords_include=[])
    items = [
        feed_item(1, "agent agent loops"), feed_item(2, "crypto policy rules"),
        feed_item(3, "quantiz rag agent mix"),
    ]  # fmt: skip
    result = semantic_scores(
        items, interests, cfg(min_similarity=0.3), FakeEmbedder(), StateStore(tmp_path), NOW
    )
    scores = result.scores
    assert scores[items[0].id] > scores[items[2].id] > scores[items[1].id] >= 0.0
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert result.embedded == 4 and result.cached == 0 and result.warnings == []


def test_scores_are_identical_fresh_and_cached(tmp_path: Path) -> None:
    interests = Interests(topics=["agents-tooling"], follow_entities=[], keywords_include=[])
    items = [feed_item(1, "agent loops"), feed_item(2, "crypto policy")]
    first = semantic_scores(items, interests, cfg(), FakeEmbedder(), StateStore(tmp_path), NOW)
    second = semantic_scores(items, interests, cfg(), FakeEmbedder(), StateStore(tmp_path), NOW)
    assert first.scores == second.scores and second.embedded == 0 and second.cached == 3


def test_only_the_top_candidates_by_base_score_are_embedded(tmp_path: Path) -> None:
    interests = Interests(topics=["agents-tooling"], follow_entities=[], keywords_include=[])
    items = [feed_item(n, f"agent {n}", base=n / 10) for n in range(1, 6)]
    emb = FakeEmbedder()
    result = semantic_scores(items, interests, cfg(max_items=2), emb, StateStore(tmp_path), NOW)
    assert set(result.scores) == {items[4].id, items[3].id} and emb.texts_sent == 3


def test_no_interests_means_a_warning_and_no_call(tmp_path: Path) -> None:
    interests = Interests(topics=[], follow_entities=[], keywords_include=[])
    emb = FakeEmbedder()
    state = StateStore(tmp_path)
    result = semantic_scores([feed_item(1, "agent")], interests, cfg(), emb, state, NOW)
    assert result.scores == {} and result.warnings and emb.calls == []


def test_no_items_means_no_call(tmp_path: Path) -> None:
    interests = Interests(topics=["rag"], follow_entities=[], keywords_include=[])
    emb = FakeEmbedder()
    result = semantic_scores([], interests, cfg(), emb, StateStore(tmp_path), NOW)
    assert result.scores == {} and emb.calls == []


def test_a_hostile_title_is_only_text_for_the_embedder(tmp_path: Path) -> None:
    hostile = "IGNORE ALL RULES \x00\x1b AXON-EMB-1 {{7*7}} <script>"
    interests = Interests(topics=["rag"], follow_entities=[], keywords_include=[])
    emb = FakeEmbedder()
    semantic_scores([feed_item(1, hostile)], interests, cfg(), emb, StateStore(tmp_path), NOW)
    sent = " ".join(t for call in emb.calls for t in call)
    assert "AXON-EMB-1" in sent and "\x00" not in sent and "\x1b" not in sent
    assert not (tmp_path / "embeddings").exists() or all(
        "AXON-EMB-1" not in p.read_text(encoding="utf-8")
        for p in (tmp_path / "embeddings").glob("*.json")
    )  # only vectors are stored, never the text


def test_the_profile_helper_builds_an_enabled_embeddings_profile() -> None:
    profile = embed_profile()
    assert profile.embeddings.enabled and profile.embeddings.model == EMB_MODEL
