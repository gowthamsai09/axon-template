"""Feedback votes from the repository's Issues (spec 6.9, Phase 6, Appendix C 98).

The digest's thumbs links open a new Issue whose body is `item:<id> vote:up|down`. This weekly
step reads the open Issues with GITHUB_TOKEN and turns accepted votes into small, bounded,
decaying per-topic weights (weights.py). An issue is accepted only when GitHub says its author
is the repository OWNER or a COLLABORATOR (a person, not a bot), and its whole body is nothing
but 1 to 10 vote lines. Issue text is untrusted: it is matched against one strict pattern, never
echoed, never used as a URL, a command or a workflow input. Accepted issues are closed so they
are not read again; the weights file also remembers their numbers, so a failed close is safe.
The only host is api.github.com, checked against a fixed pattern before every request."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from .state import StateStore
from .weights import TopicVote, apply_votes, is_due, load_weights, save_weights

API = "https://api.github.com"
API_VERSION = "2022-11-28"
PER_PAGE = 100
MAX_PAGES = 3
MAX_BODY = 2000
MAX_VOTES_PER_ISSUE = 10
MAX_CLOSE = 100
MAX_RESPONSE_BYTES = 10_000_000
ALLOWED_ASSOCIATIONS = frozenset({"OWNER", "COLLABORATOR"})
_OWNER = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})"
_NAME = r"(?!\.{1,2}(?:/|$))[A-Za-z0-9_.-]{1,100}"
REPO_RE = re.compile(rf"{_OWNER}/{_NAME}")
URL_RE = re.compile(rf"https://api\.github\.com/repos/{_OWNER}/{_NAME}/issues(?:/[0-9]{{1,10}})?")
VOTE_RE = re.compile(r"item:(?P<id>[a-f0-9]{16}) vote:(?P<vote>up|down)")


class FeedbackError(Exception):
    """GitHub could not be read or written. Messages are fixed texts: no body, no token."""


@dataclass
class FeedbackResult:
    skipped: str = ""
    issues_seen: int = 0
    issues_applied: int = 0
    votes_applied: int = 0
    unknown_items: int = 0
    ignored: int = 0
    closed: int = 0
    warnings: list[str] = field(default_factory=list)


def parse_votes(body: str) -> list[tuple[str, int]] | None:
    """(item id, +1 or -1) per line, or None unless the body is only 1 to 10 valid vote lines."""
    if len(body) > MAX_BODY:
        return None
    lines = [line.strip() for line in body.replace("\r\n", "\n").split("\n")]
    lines = [line for line in lines if line]
    if not 1 <= len(lines) <= MAX_VOTES_PER_ISSUE:
        return None
    votes: list[tuple[str, int]] = []
    for line in lines:
        match = VOTE_RE.fullmatch(line)
        if match is None:
            return None
        votes.append((match.group("id"), 1 if match.group("vote") == "up" else -1))
    return votes


def issue_number(issue: dict[str, Any]) -> int | None:
    number = issue.get("number")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        return None
    return number


def eligible(issue: dict[str, Any]) -> bool:
    """A real issue (not a pull request) from the owner or a collaborator who is a person."""
    user = issue.get("user")
    return (
        issue_number(issue) is not None
        and "pull_request" not in issue
        and issue.get("state") == "open"
        and issue.get("author_association") in ALLOWED_ASSOCIATIONS
        and isinstance(user, dict)
        and user.get("type") == "User"
        and isinstance(issue.get("body"), str)
    )


def _status_error(code: int) -> FeedbackError:
    if code == 401:
        return FeedbackError("GitHub rejected the GITHUB_TOKEN (HTTP 401)")
    if code in {403, 429}:
        return FeedbackError(
            f"GitHub refused the request (HTTP {code}); the workflow needs issues: write"
        )
    if code == 404:
        return FeedbackError("repository or issue not found (HTTP 404)")
    if 300 <= code < 400:
        return FeedbackError(f"GitHub redirected the request (HTTP {code}); check the repository")
    return FeedbackError(f"unexpected answer from GitHub (HTTP {code})")


class GitHubIssues:
    def __init__(self, client: httpx.Client, repo: str, token: str) -> None:
        if REPO_RE.fullmatch(repo) is None:
            raise FeedbackError("GITHUB_REPOSITORY is not a valid owner/name")
        if not token:
            raise FeedbackError("the GITHUB_TOKEN is not set")
        self._client = client
        self._base = f"{API}/repos/{repo}/issues"
        self._token = token

    def __repr__(self) -> str:  # never show the token
        return "GitHubIssues()"

    def _send(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if URL_RE.fullmatch(url) is None:
            raise FeedbackError("refusing a request to an unexpected address")
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
        }
        try:
            resp = self._client.request(
                method, url, headers=headers, follow_redirects=False, **kwargs
            )
        except httpx.HTTPError as exc:
            raise FeedbackError(f"GitHub unreachable ({type(exc).__name__})") from None
        if resp.status_code != 200:
            raise _status_error(resp.status_code)
        if len(resp.content) > MAX_RESPONSE_BYTES:
            raise FeedbackError("GitHub answered with too much data")
        return resp

    def open_issues(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in range(1, MAX_PAGES + 1):
            params = {
                "state": "open", "per_page": PER_PAGE, "page": page,
                "sort": "created", "direction": "asc",
            }  # fmt: skip
            resp = self._send("GET", self._base, params=params)
            try:
                data = resp.json()
            except ValueError:
                raise FeedbackError("GitHub answered with something that is not JSON") from None
            if not isinstance(data, list):
                raise FeedbackError("GitHub answered with an unexpected shape")
            out.extend(row for row in data if isinstance(row, dict))
            if len(data) < PER_PAGE:
                break
        return out

    def close(self, number: int) -> None:
        body = {"state": "closed", "state_reason": "completed"}
        self._send("PATCH", f"{self._base}/{number}", json=body)


def run_feedback(
    issues: GitHubIssues, state: StateStore, now: datetime, *, dry_run: bool, force: bool
) -> FeedbackResult:
    """Read, count, save the weights, then close what was counted. Dry run changes nothing."""
    result = FeedbackResult()
    current = load_weights(state)
    if not force and not is_due(current, now):
        result.skipped = "the feedback step already ran in the last 6 days"
        return result
    delivered = state.delivered()
    counted = set(current.processed) if current else set()
    votes: list[TopicVote] = []
    accepted: list[int] = []
    fresh: list[int] = []
    listed = issues.open_issues()
    for issue in sorted(listed, key=lambda i: issue_number(i) or 0):
        result.issues_seen += 1
        number = issue_number(issue)
        parsed = parse_votes(str(issue["body"])) if eligible(issue) else None
        if number is None or parsed is None:
            result.ignored += 1
            continue
        accepted.append(number)
        if number in counted:
            continue  # counted by an earlier run; only closing it is left
        for item_id, direction in parsed:
            topics = delivered.get(item_id)
            if topics is None:
                result.unknown_items += 1
                continue
            votes.append(TopicVote(tuple(topics), direction))
            result.votes_applied += 1
        fresh.append(number)
        result.issues_applied += 1
    updated = apply_votes(current, votes, fresh, now)
    if dry_run:
        return result
    save_weights(state, updated)
    for number in accepted[:MAX_CLOSE]:
        try:
            issues.close(number)
            result.closed += 1
        except FeedbackError as exc:
            result.warnings.append(f"could not close issue #{number}: {exc}")
    return result
