"""P6.6 feedback weights (weights.py) and their effect on ranking: bounds, decay, idempotency,
tolerant reading of a damaged file, strict writing. Offline and deterministic."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from digest.profile import Profile
from digest.rank import FEEDBACK_SCALE, feedback_adjustment, rank
from digest.state import StateStore
from digest.weights import (
    FILE,
    HALF_LIFE_DAYS,
    LIMIT,
    MAX_PROCESSED,
    MAX_RUN_MOVE,
    MIN_INTERVAL_DAYS,
    STEP,
    TopicVote,
    WeightsFile,
    apply_votes,
    decay_factor,
    effective_weights,
    is_due,
    load_effective,
    load_weights,
    save_weights,
)
from tests.conftest import NOW, make_profile
from tests.test_rank import mk

UP, DOWN = 1, -1


def vote(direction: int, *topics: str) -> TopicVote:
    return TopicVote(tuple(topics), direction)


def file_with(
    topics: dict[str, float], days_ago: float = 0.0, processed: list[int] | None = None
) -> WeightsFile:
    return WeightsFile.model_validate({
        "version": 1, "updated_at": NOW - timedelta(days=days_ago),
        "topics": topics, "processed": processed or [],
    })  # fmt: skip


# ---- decay and due-ness ----------------------------------------------------------------


def test_weights_halve_every_thirty_days() -> None:
    assert decay_factor(NOW - timedelta(days=HALF_LIFE_DAYS), NOW) == pytest.approx(0.5)
    assert decay_factor(NOW - timedelta(days=2 * HALF_LIFE_DAYS), NOW) == pytest.approx(0.25)
    assert decay_factor(NOW, NOW) == 1.0
    assert decay_factor(NOW + timedelta(days=5), NOW) == 1.0  # a clock that ran backwards


def test_effective_weights_decay_and_drop_the_negligible() -> None:
    f = file_with({"rag": 0.8, "robotics": 0.012, "datasets": -0.5}, days_ago=30)
    eff = effective_weights(f, NOW)
    assert eff == {"rag": pytest.approx(0.4), "datasets": pytest.approx(-0.25)}
    assert effective_weights(None, NOW) == {}


def test_the_step_is_weekly() -> None:
    assert is_due(None, NOW)
    assert not is_due(file_with({}, days_ago=MIN_INTERVAL_DAYS - 0.01), NOW)
    assert is_due(file_with({}, days_ago=MIN_INTERVAL_DAYS), NOW)


# ---- applying votes --------------------------------------------------------------------


def test_one_vote_moves_a_topic_by_one_step() -> None:
    f = apply_votes(None, [vote(UP, "rag", "agents-tooling")], [7], NOW)
    assert f.topics == {"agents-tooling": STEP, "rag": STEP} and f.processed == [7]
    assert f.updated_at == NOW and f.version == 1


def test_up_and_down_votes_cancel() -> None:
    f = apply_votes(None, [vote(UP, "rag"), vote(DOWN, "rag")], [1, 2], NOW)
    assert f.topics == {}  # exactly zero is not kept


def test_one_run_moves_a_topic_by_at_most_the_run_limit() -> None:
    up = apply_votes(None, [vote(UP, "rag")] * 50, list(range(50)), NOW)
    down = apply_votes(None, [vote(DOWN, "rag")] * 50, list(range(50)), NOW)
    assert up.topics["rag"] == MAX_RUN_MOVE and down.topics["rag"] == -MAX_RUN_MOVE


def test_a_weight_never_leaves_the_limit_over_many_runs() -> None:
    f = None
    for run in range(10):
        f = apply_votes(f, [vote(UP, "rag")] * 5, [run], NOW)
    assert f is not None and f.topics["rag"] == LIMIT


def test_old_weights_decay_before_new_votes_are_added() -> None:
    old = file_with({"rag": 0.4}, days_ago=30)
    f = apply_votes(old, [vote(UP, "rag")], [3], NOW)
    assert f.topics["rag"] == pytest.approx(0.2 + STEP, abs=1e-4)


def test_unknown_topics_in_votes_and_in_old_files_are_dropped() -> None:
    f = apply_votes(None, [vote(UP, "not-a-topic", "rag")], [1], NOW)
    assert list(f.topics) == ["rag"]


def test_processed_issue_numbers_are_unique_ordered_and_capped() -> None:
    f = apply_votes(file_with({}, processed=[1, 2]), [], [2, 3, 3], NOW)
    assert f.processed == [1, 2, 3]
    many = apply_votes(None, [], list(range(MAX_PROCESSED + 25)), NOW)
    assert len(many.processed) == MAX_PROCESSED and many.processed[-1] == MAX_PROCESSED + 24


def test_weights_are_rounded_and_sorted() -> None:
    f = apply_votes(file_with({"rag": 0.123456789}), [vote(UP, "datasets")], [1], NOW)
    assert list(f.topics) == sorted(f.topics) and f.topics["rag"] == 0.1235


# ---- the file --------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    state = StateStore(tmp_path)
    f = apply_votes(None, [vote(UP, "rag")], [9], NOW)
    save_weights(state, f)
    assert load_weights(StateStore(tmp_path)) == f
    assert load_effective(StateStore(tmp_path), NOW) == {"rag": STEP}
    assert not list(tmp_path.glob("*.tmp"))


def test_a_missing_file_is_normal(tmp_path: Path) -> None:
    state = StateStore(tmp_path)
    assert load_weights(state) is None and state.warnings == []
    assert load_effective(state, NOW) == {}


@pytest.mark.parametrize(
    "raw",
    [
        "{broken",
        "[]",
        json.dumps({"version": 2, "updated_at": "2026-10-20T06:00:00Z", "topics": {}}),
        json.dumps({"version": 1, "updated_at": "2026-10-20T06:00:00Z", "topics": {"rag": 7}}),
        json.dumps({"version": 1, "updated_at": "2026-10-20T06:00:00Z", "topics": {"x": 0.1}}),
        json.dumps({"version": 1, "updated_at": "2026-10-20T06:00:00", "topics": {}}),
        json.dumps({"version": 1, "updated_at": "2026-10-20T06:00:00Z", "surprise": 1}),
        '{"version": 1, "updated_at": "2026-10-20T06:00:00Z", "topics": {"rag": NaN}}',
        '{"version": 1, "updated_at": "2026-10-20T06:00:00Z", "topics": {"rag": Infinity}}',
    ],
)
def test_a_damaged_file_is_ignored_with_a_warning_and_never_crashes(
    tmp_path: Path, raw: str
) -> None:
    (tmp_path / FILE).write_text(raw, encoding="utf-8")
    state = StateStore(tmp_path)
    assert load_weights(state) is None
    assert load_effective(state, NOW) == {}
    assert state.warnings  # silent loss would hide a problem


def test_the_model_refuses_what_we_would_never_write() -> None:
    for bad in ({"rag": 1.5}, {"rag": float("nan")}, {"bogus": 0.1}):
        with pytest.raises(ValidationError):
            file_with(bad)
    with pytest.raises(ValidationError):
        WeightsFile.model_validate(
            {"updated_at": NOW, "topics": {}, "processed": list(range(MAX_PROCESSED + 1))}
        )


# ---- ranking ---------------------------------------------------------------------------


def test_feedback_adjustment_is_bounded_and_off_without_weights() -> None:
    item = mk(tags=["rag", "agents-tooling"])
    assert feedback_adjustment(item, None) == 0.0 and feedback_adjustment(item, {}) == 0.0
    assert feedback_adjustment(item, {"rag": 0.3}) == pytest.approx(0.3 * FEEDBACK_SCALE)
    assert feedback_adjustment(item, {"rag": 1.0, "agents-tooling": 1.0}) == FEEDBACK_SCALE
    assert feedback_adjustment(item, {"rag": -1.0, "agents-tooling": -1.0}) == -FEEDBACK_SCALE
    assert feedback_adjustment(mk(tags=["robotics"]), {"rag": 1.0}) == 0.0


def test_a_repeated_tag_counts_once() -> None:
    assert feedback_adjustment(mk(tags=["rag", "rag"]), {"rag": 0.5}) == pytest.approx(0.1)


def interests_profile() -> Profile:
    return make_profile(
        interests={
            "topics": ["rag", "agents-tooling"], "follow_entities": [],
            "keywords_include": [], "keywords_exclude": [],
        },
        feedback={"enabled": True},
    )  # fmt: skip


def test_votes_reorder_otherwise_equal_items_without_adding_or_removing_any() -> None:
    a = mk("a" * 16, tags=["rag"], title="One")
    b = mk("b" * 16, tags=["agents-tooling"], title="Two")
    profile = interests_profile()
    plain = rank([a, b], profile, seen=set(), limit=5)
    liked = rank([a, b], profile, seen=set(), limit=5, weights={"agents-tooling": 1.0})
    assert [s.item.id for s in plain] == ["a" * 16, "b" * 16]  # a tie breaks by id
    assert [s.item.id for s in liked] == ["b" * 16, "a" * 16]
    assert liked[0].feedback == pytest.approx(FEEDBACK_SCALE)
    assert {s.item.id for s in liked} == {s.item.id for s in plain}


def test_a_disliked_topic_is_demoted_but_never_hidden() -> None:
    a = mk("a" * 16, tags=["rag"])
    out = rank([a], interests_profile(), seen=set(), limit=5, weights={"rag": -1.0})
    assert [s.item.id for s in out] == [a.id] and out[0].feedback == -FEEDBACK_SCALE


def test_weights_alone_never_make_an_unmatched_item_appear() -> None:
    other = mk(tags=["robotics"])
    out = rank([other], interests_profile(), seen=set(), limit=5, weights={"robotics": 1.0})
    assert out == []


def test_ranking_without_weights_equals_the_phase5_formula() -> None:
    a = mk(tags=["rag"], base_score=0.4)
    (scored,) = rank([a], interests_profile(), seen=set(), limit=5)
    assert scored.score == pytest.approx(0.4 + 0.5 * scored.match)
    assert scored.feedback == 0.0 and scored.semantic == 0.0
