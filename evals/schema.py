"""Case format and loader.

A case is one source document plus what a good summary must show.

`recorded.response` is a stored model answer used by the offline `replay` mode. The seed cases
carry a REFERENCE answer written by hand from the source text: replay therefore proves that the
harness, the scorers and the real validation work, not that any model is good. The
human-checked set of 30 to 50 cases is the maintainer's to add (see evals/README.md)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

CASES_DIR = Path(__file__).resolve().parent / "cases"
Status = Literal["ok", "insufficient_content", "suspected_injection"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Strict):
    id: str = Field(pattern=r"^[a-f0-9]{16}$")
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(min_length=8, max_length=2048)
    source_name: str = Field(min_length=1, max_length=64)
    published_at: str = Field(min_length=10, max_length=40)
    text: str = Field(max_length=24_000)


class Expect(Strict):
    status: Status
    required_terms: list[str] = Field(default_factory=list, max_length=20)
    forbidden_terms: list[str] = Field(default_factory=list, max_length=20)
    min_coverage: float = Field(1.0, ge=0, le=1)
    injection_flag: bool | None = None  # None: not checked


class Recorded(Strict):
    kind: Literal["reference"]
    note: str = Field(min_length=1, max_length=300)
    response: dict[str, Any]


class EvalCase(Strict):
    id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=64)
    synthetic: bool = False  # True: written for the harness, not taken from the real feed
    depth: Literal["headlines", "brief", "detailed"] = "brief"
    language: str = Field("en", pattern=r"^[a-z]{2,3}(-[A-Z]{2})?$")
    source: Source
    expect: Expect
    recorded: Recorded


class CaseError(Exception):
    """A case file is unreadable or invalid. The message names files, never their content."""


def load_cases(directory: Path = CASES_DIR) -> list[EvalCase]:
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    cases: list[EvalCase] = []
    for path in files:
        try:
            case = EvalCase.model_validate(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, ValidationError):
            raise CaseError(f"{path.name}: not a valid evaluation case") from None
        if case.id != path.stem:
            raise CaseError(f"{path.name}: the id must equal the file name")
        cases.append(case)
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise CaseError("duplicate case ids")
    return cases


def term_pattern(term: str) -> re.Pattern[str]:
    """Whole-token, case-insensitive match (so 'sea' never matches 'research')."""
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
