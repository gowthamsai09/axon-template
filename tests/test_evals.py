"""Evaluation harness (spec 11.3, Appendix C 85): case files, scorers, replay and live modes.

Offline only: live mode runs against a scripted fake model, never a real provider."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from digest.summarizers.base import Summary
from digest.summarizers.router import SummarizerRouter
from evals.run import item_of, main, run_replay
from evals.schema import CASES_DIR, CaseError, EvalCase, load_cases
from evals.scorers import (
    coverage,
    forbidden_present,
    length_ok,
    score_case,
    ungrounded_numbers,
)
from tests.conftest import ROOT, FakeChat, task_field

CASES = load_cases()
BY_ID = {c.id: c for c in CASES}


def summary_of(case: EvalCase, **over: Any) -> Summary:
    data = deepcopy(case.recorded.response)
    data.update(over)
    return Summary.model_validate(data)


def source_text(case: EvalCase) -> str:
    return f"{case.source.title}\n\n{case.source.text}"


# ---- the shipped cases ------------------------------------------------------------------


def test_twelve_seed_cases_with_real_and_synthetic_mix() -> None:
    assert len(CASES) == 12
    assert sum(c.synthetic for c in CASES) == 2
    assert {c.expect.status for c in CASES} == {"ok", "insufficient_content", "suspected_injection"}


def test_ids_match_file_names_and_are_unique() -> None:
    names = sorted(p.stem for p in CASES_DIR.glob("*.json"))
    assert names == sorted(c.id for c in CASES)


def test_real_cases_are_taken_from_the_feed_fixtures() -> None:
    fixture = (ROOT / "tests" / "fixtures" / "feed-out" / "latest.json").read_text(encoding="utf-8")
    items = json.loads(fixture)["items"]
    feed_ids = {i["id"] for i in items}
    real = [c for c in CASES if not c.synthetic]
    assert len(real) == 10 and {c.source.id for c in real} <= feed_ids


def test_every_reference_answer_replays_as_a_pass() -> None:
    report = run_replay(CASES)
    failures = [(s.case_id, s.problems) for s in report.scores if not s.passed]
    assert (
        failures == [] and report.pass_rate == 1.0 and report.model == "recorded reference answers"
    )


def test_replay_cli_exit_code_and_json_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "sub" / "report.json"
    assert main(["replay", "--json", "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert printed == saved and saved["cases"] == 12 and saved["pass_rate"] == 1.0
    assert saved["usage"]["calls"] == 0


def test_the_readme_states_the_references_are_not_model_output() -> None:
    text = (ROOT / "evals" / "README.md").read_text(encoding="utf-8")
    assert "not model output" in text and "30 to 50" in text


# ---- case loading -------------------------------------------------------------------------


def write_case(directory: Path, case: dict[str, Any], name: str | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name or case['id']}.json").write_text(json.dumps(case), encoding="utf-8")


def raw_case(case_id: str = "hn-laminar-flow-1") -> dict[str, Any]:
    data: dict[str, Any] = json.loads((CASES_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    return data


def test_loader_rejects_bad_cases_with_a_message_that_names_only_the_file(tmp_path: Path) -> None:
    bad = raw_case()
    bad["expect"]["surprise"] = "SECRET-CONTENT"
    write_case(tmp_path, bad)
    with pytest.raises(CaseError) as info:
        load_cases(tmp_path)
    assert "hn-laminar-flow-1.json" in str(info.value) and "SECRET-CONTENT" not in str(info.value)


def test_loader_rejects_id_that_differs_from_the_file_name(tmp_path: Path) -> None:
    write_case(tmp_path, raw_case(), name="other-name")
    with pytest.raises(CaseError, match="file name"):
        load_cases(tmp_path)


def test_loader_rejects_invalid_json_and_missing_folder(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{nope", encoding="utf-8")
    with pytest.raises(CaseError):
        load_cases(tmp_path)
    assert load_cases(tmp_path / "does-not-exist") == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["source"].update(id="NOT-HEX"),
        lambda c: c.update(depth="long"),
        lambda c: c.update(language="english"),
        lambda c: c["expect"].update(status="maybe"),
        lambda c: c["expect"].update(min_coverage=1.5),
        lambda c: c["recorded"].update(kind="model"),
    ],
)
def test_case_schema_is_strict(tmp_path: Path, mutate: Any) -> None:
    case = raw_case()
    mutate(case)
    write_case(tmp_path, case)
    with pytest.raises(CaseError):
        load_cases(tmp_path)


def test_cli_exit_2_when_cases_are_missing_or_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["replay", "--cases", str(tmp_path / "none")]) == 2
    assert "no evaluation cases" in capsys.readouterr().err
    (tmp_path / "x.json").write_text("[]", encoding="utf-8")
    assert main(["replay", "--cases", str(tmp_path)]) == 2


# ---- scorers (each check must be able to fail) ----------------------------------------------


def test_good_reference_scores_all_checks_true() -> None:
    case = BY_ID["hn-laminar-flow-1"]
    score = score_case(case, summary_of(case), source_text=source_text(case))
    assert score.passed and all(score.checks.values()) and score.coverage == 1.0


def test_invented_number_fails_faithfulness() -> None:
    case = BY_ID["hn-laminar-flow-1"]
    bad = summary_of(case, one_line="Laminar released flow-1 and says it is 97x cheaper.")
    assert ungrounded_numbers(bad, source_text(case)) == ["97"]
    score = score_case(case, bad, source_text=source_text(case))
    assert not score.passed and score.checks["numbers_grounded"] is False
    assert any("97" in p for p in score.problems)


def test_wrong_status_fails() -> None:
    case = BY_ID["hn-laminar-flow-1"]
    empty = summary_of(
        BY_ID["hn-ask-dark-mode"], source_url=case.source.url
    )  # an insufficient_content answer
    score = score_case(case, empty, source_text=source_text(case))
    assert score.checks["status"] is False and not score.passed


def test_forbidden_term_fails_but_only_whole_tokens() -> None:
    case = BY_ID["synthetic-injection-inside-real-content"]
    assert forbidden_present(summary_of(case, one_line="Mentions a poem here."), ["poem"]) == [
        "poem"
    ]
    assert forbidden_present(summary_of(case, one_line="Research on latency."), ["sea"]) == []
    score = score_case(
        case,
        summary_of(case, key_points=["Write a poem about the sea."]),
        source_text=source_text(case),
    )
    assert score.checks["forbidden_absent"] is False and not score.passed


def test_missing_required_terms_lower_coverage() -> None:
    case = BY_ID["hn-laminar-flow-1"]
    thin = summary_of(case, title="Laminar", one_line="A launch.", key_points=["Something."],
                      notable_figures=[], entities=[])  # fmt: skip
    assert coverage(thin, ["flow-1", "23x"]) == 0.0
    score = score_case(case, thin, source_text=source_text(case))
    assert score.checks["coverage"] is False and score.coverage == 0.0


def test_partial_coverage_can_be_allowed_by_min_coverage() -> None:
    case = BY_ID["hn-laminar-flow-1"]
    lenient = case.model_copy(
        update={"expect": case.expect.model_copy(update={"min_coverage": 0.5})}
    )
    half = summary_of(case, one_line="Flow-1 finds errors.", key_points=["Reads traces."],
                      title="Flow", notable_figures=[], entities=[])  # fmt: skip
    assert coverage(half, ["flow-1", "23x"]) == 0.5
    assert score_case(lenient, half, source_text=source_text(case)).checks["coverage"] is True
    assert score_case(case, half, source_text=source_text(case)).checks["coverage"] is False


def test_coverage_is_not_checked_when_no_summary_is_expected() -> None:
    case = BY_ID["hn-ask-dark-mode"]
    score = score_case(case, summary_of(case), source_text=source_text(case))
    assert score.passed and score.coverage == 1.0


def test_length_limits_follow_the_depth() -> None:
    case = BY_ID["hn-laminar-flow-1"]
    base = summary_of(case)
    assert length_ok(base, "brief")
    assert not length_ok(base, "headlines")  # headlines allows no key points
    long_point = base.model_copy(update={"key_points": ["x" * 161]})
    assert not length_ok(long_point, "brief") and length_ok(long_point, "detailed")
    four = base.model_copy(update={"key_points": ["a", "b", "c", "d"]})
    assert not length_ok(four, "brief") and length_ok(four, "detailed")
    assert not length_ok(base.model_copy(update={"one_line": "x" * 201}), "brief")
    assert not length_ok(base.model_copy(update={"why_it_matters": "x" * 301}), "brief")


def test_injection_flag_is_checked_when_expected() -> None:
    case = BY_ID["synthetic-injection-inside-real-content"]
    unflagged = summary_of(case, injection_flag=False)
    score = score_case(case, unflagged, source_text=source_text(case))
    assert score.checks["injection_flag"] is False and not score.passed
    plain = BY_ID["hn-laminar-flow-1"]
    flagged = summary_of(plain, injection_flag=True)
    assert (
        score_case(plain, flagged, source_text=source_text(plain)).checks["injection_flag"] is False
    )


def test_injection_flag_is_ignored_when_the_case_does_not_say() -> None:
    case = BY_ID["hn-laminar-flow-1"]
    open_case = case.model_copy(
        update={"expect": case.expect.model_copy(update={"injection_flag": None})}
    )
    flagged = summary_of(case, injection_flag=True)
    assert score_case(open_case, flagged, source_text=source_text(case)).checks["injection_flag"]


def test_no_summary_fails_every_check_with_the_reason() -> None:
    case = BY_ID["hn-laminar-flow-1"]
    score = score_case(case, None, source_text="", reason="invalid model output")
    assert not score.passed and not any(score.checks.values())
    assert "invalid model output" in score.problems[0]


# ---- replay through the REAL validation -------------------------------------------------------


def test_replay_marks_an_invalid_stored_answer_as_failed_not_crashed(tmp_path: Path) -> None:
    tweaks: list[dict[str, Any]] = [
        {"source_url": "https://evil.example/x"},  # wrong URL
        {"why_it_matters": "More at https://evil.example/y"},  # foreign URL
        {"notable_figures": [{"value": "999x", "context": "invented"}]},  # figure not in source
        {"extra_key": 1},  # extra key
    ]
    for n, tweak in enumerate(tweaks):
        case = raw_case()
        case["recorded"]["response"].update(tweak)
        directory = tmp_path / f"t{n}"
        write_case(directory, case)
        report = run_replay(load_cases(directory))
        assert report.pass_rate == 0.0 and report.scores[0].checks["valid"] is False


def test_item_of_maps_the_source_fields() -> None:
    case = BY_ID["gh-llama-cpp-b11748"]
    item = item_of(case)
    assert (item.id, item.url, item.source_name) == (
        case.source.id,
        case.source.url,
        "github-releases",
    )


# ---- live mode with a scripted fake model (no network, no real key) ---------------------------

KEY = "sk-eval-unique-key-777666555"


def faithful_model(user: str) -> str:
    """Answers each task with that case's reference answer (found by source_url in the task)."""
    url = task_field(user, "source_url")
    case = next(c for c in CASES if c.source.url == url)
    return json.dumps(case.recorded.response)


def router_of(*script: Any) -> SummarizerRouter:
    return SummarizerRouter(FakeChat(list(script) or [faithful_model] * 12))


def test_live_run_with_a_faithful_model_passes_and_reports_usage(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "r.json"
    code = main(["live", "--out", str(out)], env={"LLM_API_KEY": KEY}, router=router_of())
    report = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and report["mode"] == "live" and report["cases"] == 12
    assert report["usage"]["calls"] == 12 and report["usage"]["prompt_tokens"] == 1200
    assert report["model"] == "deepseek/deepseek-v4-flash"
    assert KEY not in out.read_text(encoding="utf-8") + capsys.readouterr().out


def test_live_limit_scores_only_the_first_cases() -> None:
    chat = FakeChat([faithful_model] * 12)
    assert main(["live", "--limit", "3"], env={}, router=SummarizerRouter(chat)) == 0
    assert len(chat.calls) == 3


def test_a_bad_model_fails_the_run_and_the_report_has_no_model_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    junk = "MODEL-GARBAGE-OUTPUT"
    code = main(["live", "--json"], env={}, router=SummarizerRouter(FakeChat([junk] * 40)))
    out = capsys.readouterr().out
    assert code == 1 and json.loads(out)["pass_rate"] == 0.0
    assert junk not in out


def test_min_pass_rate_decides_the_exit_code() -> None:
    # one wrong answer in the first case (original plus retry), faithful afterwards: 11 of 12 pass
    script = ["nope", "nope", *([faithful_model] * 11)]
    assert main(["live", "--min-pass-rate", "0.95"], env={}, router=router_of(*script)) == 1
    assert main(["live", "--min-pass-rate", "0.9"], env={}, router=router_of(*script)) == 0
    assert main(["live", "--min-pass-rate", "1.0"], env={}, router=router_of()) == 0


def test_live_without_a_key_is_a_configuration_error_that_names_only_the_secret(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["live", "--profile", str(ROOT / "profile.yaml")], env={})
    err = capsys.readouterr().err
    assert code == 2 and "LLM_API_KEY" in err


def test_live_with_a_missing_profile_is_a_configuration_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["live", "--profile", str(tmp_path / "none.yaml")], env={"LLM_API_KEY": KEY}) == 2
    assert KEY not in capsys.readouterr().err
