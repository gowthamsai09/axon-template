"""state/tasks/<task_id>.json (spec 9.3b) plus an extractive-fallback marker (6.3 step 7)."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .profile import Provider, TopicSlug

UUID_PATTERN = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskItemSummary(Strict):
    one_line: str = Field(max_length=200)
    key_points: list[Annotated[str, Field(max_length=240)]] = Field(max_length=6)
    why_it_matters: str = Field(max_length=300)
    tags: list[TopicSlug] = Field(max_length=4)


class TaskItem(Strict):
    id: str = Field(pattern=r"^[a-f0-9]{16}$")
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=8, max_length=2048)
    summary: TaskItemSummary | None = None
    fallback_snippet: str | None = Field(None, max_length=300)  # not AI-summarized

    @model_validator(mode="after")
    def _exactly_one(self) -> Self:
        if (self.summary is None) == (self.fallback_snippet is None):
            raise ValueError("provide exactly one of summary or fallback_snippet")
        return self


class ModelInfo(Strict):
    provider: Provider
    model: str = Field(min_length=1, max_length=100)


class TaskResult(Strict):
    task_id: str = Field(pattern=UUID_PATTERN)
    status: Literal["completed", "failed"]
    finished_at: AwareDatetime
    items: list[TaskItem] = Field(max_length=25)
    model: ModelInfo
    warnings: list[Annotated[str, Field(max_length=500)]] = Field(max_length=100)
