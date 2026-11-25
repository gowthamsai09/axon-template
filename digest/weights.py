"""Per-topic feedback weights (spec 6.9, Phase 6, Appendix C 98): state/weights.json.

Bounded: every weight stays within -1..1, and one run moves a topic by at most 0.3.
Decaying: weights halve every 30 days, applied from the elapsed time whenever they are read.
Idempotent: the numbers of the issues already counted are kept (the last 500), so a vote is
never counted twice even when closing its issue failed."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from .profile import TOPIC_SLUGS, TopicSlug
from .state import StateStore

FILE = "weights.json"
STEP = 0.10  # one vote on one topic
MAX_RUN_MOVE = 0.30  # most a single run can move one topic
LIMIT = 1.0
HALF_LIFE_DAYS = 30.0
MIN_KEEP = 0.01  # weights that decayed below this are dropped
MAX_PROCESSED = 500
MIN_INTERVAL_DAYS = 6  # the feedback step is weekly

Weight = Annotated[float, Field(ge=-LIMIT, le=LIMIT, allow_inf_nan=False)]


class WeightsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    updated_at: AwareDatetime
    topics: dict[TopicSlug, Weight] = Field(default_factory=dict)
    processed: list[int] = Field(default_factory=list, max_length=MAX_PROCESSED)


@dataclass(frozen=True)
class TopicVote:
    """One accepted vote: the topics of the item and the direction (+1 up, -1 down)."""

    topics: tuple[str, ...]
    direction: int


def decay_factor(since: datetime, now: datetime) -> float:
    days = max(0.0, (now - since).total_seconds() / 86400)
    return float(0.5 ** (days / HALF_LIFE_DAYS))


def load_weights(state: StateStore) -> WeightsFile | None:
    raw = state.read_json(FILE)
    if raw is None:
        return None
    try:
        return WeightsFile.model_validate(raw)
    except ValidationError:
        state.warnings.append(f"state/{FILE}: invalid, ignored")
        return None


def effective_weights(file: WeightsFile | None, now: datetime) -> dict[str, float]:
    """The weights as they stand `now`, decayed from the last update."""
    if file is None:
        return {}
    factor = decay_factor(file.updated_at, now)
    return {t: w * factor for t, w in file.topics.items() if abs(w * factor) >= MIN_KEEP}


def is_due(file: WeightsFile | None, now: datetime) -> bool:
    return file is None or now - file.updated_at >= timedelta(days=MIN_INTERVAL_DAYS)


def apply_votes(
    file: WeightsFile | None, votes: Iterable[TopicVote], issues: Iterable[int], now: datetime
) -> WeightsFile:
    """Decay what is there, add the votes (bounded), remember the issues."""
    current = effective_weights(file, now)
    moves: dict[str, float] = {}
    for vote in votes:
        for topic in vote.topics:
            if topic in TOPIC_SLUGS:  # delivered.json is tolerant; weights.json is strict
                moves[topic] = moves.get(topic, 0.0) + vote.direction * STEP
    merged: dict[str, float] = {t: w for t, w in current.items() if t in TOPIC_SLUGS}
    for topic, move in moves.items():
        bounded = max(-MAX_RUN_MOVE, min(MAX_RUN_MOVE, move))
        merged[topic] = max(-LIMIT, min(LIMIT, merged.get(topic, 0.0) + bounded))
    kept = {t: round(w, 4) for t, w in merged.items() if abs(w) >= MIN_KEEP and math.isfinite(w)}
    done = [*(file.processed if file else []), *issues]
    return WeightsFile.model_validate(
        {
            "version": 1,
            "updated_at": now,
            "topics": dict(sorted(kept.items())),
            "processed": list(dict.fromkeys(done))[-MAX_PROCESSED:],
        }
    )


def save_weights(state: StateStore, file: WeightsFile) -> None:
    state.write_json(FILE, file.model_dump(mode="json"))


def load_effective(state: StateStore, now: datetime) -> Mapping[str, float]:
    return effective_weights(load_weights(state), now)
