"""Run the evaluation (spec 11.3, Appendix C 85).

  python -m evals.run replay [--cases DIR] [--json]
      Offline. Scores the stored answers of every case through the REAL validation and
      scorers. Used in CI.
  python -m evals.run live --profile profile.yaml [--cases DIR] [--limit N]
                           [--min-pass-rate 0.9] [--out FILE]
      Uses YOUR model through the real router and YOUR key (LLM_API_KEY in the environment).
      Costs tokens.

Exit codes: 0 passed, 1 below the required pass rate, 2 configuration problem. The report
holds case ids, check results and token counts: never the key, never model text."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from digest.profile import ProfileError, load_profile
from digest.summarizers.base import SourceItem, Usage
from digest.summarizers.prompting import document_for, expected_source_url
from digest.summarizers.router import ConfigError, SummarizerRouter, get_summarizer
from digest.summarizers.validation import SummaryValidationError, validate_summary
from evals.schema import CASES_DIR, CaseError, EvalCase, load_cases
from evals.scorers import CaseScore, score_case


@dataclass
class Report:
    mode: str
    scores: list[CaseScore] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    model: str = ""

    @property
    def pass_rate(self) -> float:
        return sum(s.passed for s in self.scores) / len(self.scores) if self.scores else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "model": self.model, "cases": len(self.scores),
            "pass_rate": round(self.pass_rate, 4),
            "usage": {"prompt_tokens": self.usage.prompt_tokens,
                      "completion_tokens": self.usage.completion_tokens, "calls": self.usage.calls},
            "results": [
                {"id": s.case_id, "passed": s.passed, "checks": s.checks,
                 "coverage": s.coverage, "problems": s.problems}
                for s in self.scores
            ],
        }  # fmt: skip

    def text(self) -> str:
        lines = [f"{'PASS' if s.passed else 'FAIL'}  {s.case_id}" for s in self.scores]
        for s in self.scores:
            lines += [f"      {s.case_id}: {p}" for p in s.problems]
        lines.append(
            f"{self.mode}: {sum(s.passed for s in self.scores)}/{len(self.scores)} passed "
            f"({self.pass_rate:.0%})"
        )
        return "\n".join(lines)


def item_of(case: EvalCase) -> SourceItem:
    s = case.source
    return SourceItem(s.id, s.title, s.url, s.source_name, s.published_at, s.text)


def run_replay(cases: list[EvalCase]) -> Report:
    report = Report("replay", model="recorded reference answers")
    for case in cases:
        item = item_of(case)
        document = document_for(item)
        try:
            summary = validate_summary(
                case.recorded.response, depth=case.depth,
                expected_url=expected_source_url(item), source_text=document,
            )  # fmt: skip
            report.scores.append(score_case(case, summary, source_text=document))
        except SummaryValidationError as exc:
            report.scores.append(score_case(case, None, source_text=document, reason=str(exc)))
    return report


def run_live(cases: list[EvalCase], router: SummarizerRouter) -> Report:
    first = router.primary
    report = Report("live", model=f"{first.provider}/{first.model}")
    for case in cases:
        item = item_of(case)
        result = router.summarize(item, case.depth, case.language)
        report.usage.add(result.usage)
        report.scores.append(
            score_case(case, result.summary, source_text=document_for(item), reason=result.reason)
        )
    return report


def main(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    router: SummarizerRouter | None = None,
) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("replay", "live"):
        p = sub.add_parser(name)
        p.add_argument("--cases", type=Path, default=CASES_DIR)
        p.add_argument("--json", action="store_true", help="print the report as JSON")
        p.add_argument("--out", type=Path, default=None, help="also write the JSON report here")
    live = sub.choices["live"]
    live.add_argument("--profile", type=Path, default=Path("profile.yaml"))
    live.add_argument("--limit", type=int, default=0, help="score only the first N cases (0 = all)")
    live.add_argument("--min-pass-rate", type=float, default=0.9)
    args = ap.parse_args(argv)
    try:
        cases = load_cases(args.cases)
    except CaseError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not cases:
        print(f"no evaluation cases found in {args.cases}", file=sys.stderr)
        return 2
    required = 1.0
    if args.mode == "replay":
        report = run_replay(cases)
    else:
        required = args.min_pass_rate
        if args.limit > 0:
            cases = cases[: args.limit]
        environment = os.environ if env is None else env
        client: httpx.Client | None = None
        try:
            if router is None:
                profile = load_profile(args.profile)
                client = httpx.Client(timeout=60.0)
                router, _warnings = get_summarizer(profile, environment, http=client)
            report = run_live(cases, router)
        except (ProfileError, ConfigError, OSError) as exc:
            print(f"configuration problem: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2
        finally:
            if client is not None:
                client.close()
    payload = json.dumps(report.to_dict(), indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload if args.json else report.text())
    return 0 if report.pass_rate >= required else 1


if __name__ == "__main__":
    sys.exit(main())
