"""Mutation sanity (guidelines 3 and 13): break ONE security rule at a time, expect a failure.

For each mutation the repo is copied to a temp folder, one string is replaced, and the tests
for that rule run there. KILLED means a test failed (good). SURVIVED means every test passed:
the rule has no test that notices it, so write one. Your working tree is never touched.

    python scripts/mutation_sanity.py            run all mutations (baseline first)
    python scripts/mutation_sanity.py --check    only verify each pattern matches once
    python scripts/mutation_sanity.py --only tg-url-guard
    python scripts/mutation_sanity.py --prefix fb-        every mutation whose name starts so
Exit code 0 when every mutation was killed, 1 otherwise."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAT = (
    "tests/test_chat_common.py",
    "tests/test_telegram.py",
    "tests/test_slack.py",
    "tests/test_chat_delivery.py",
)
RSS_T = ("tests/test_rss.py",)
EMB_T = ("tests/test_embeddings.py", "tests/test_pipeline_phase6.py")
TREND_T = ("tests/test_trends.py", "tests/test_pipeline_phase6.py")
FB_T = ("tests/test_feedback.py", "tests/test_weights.py", "tests/test_cli_phase6.py")
ONB_T = ("tests/test_onboarding.py", "tests/test_onboard_cli.py")
ONW_T = ("tests/test_onboard_workflow.py",)
PIPE_T = ("tests/test_pipeline_phase6.py", "tests/test_cli_phase6.py", "tests/test_weights.py")
IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache", ".venv", "venv",
    "*.pyc", "outbox",
)  # fmt: skip
TG = "digest/delivery/telegram.py"
SL = "digest/delivery/slack.py"
CM = "digest/delivery/common.py"
RD = "digest/render/__init__.py"
CLI = "digest/cli.py"
UNREACH = 'unreachable ({type(exc).__name__})")'
UTF16 = 'return len(text.encode("utf-16-le", errors="surrogatepass")) // 2'
LOOKALIKE = 'str.maketrans({"*": "\\u2217", "~": "\\u223c", "`": "\'"})'
URL_UNSAFE = 'r"[\\x00-\\x20\\x7f<>|]"'
SPACES = 'if re.search(r"[\\s\\x00-\\x1f\\x7f]", url):'
TG_DRY = 'if dry_run:\n            return _write_parts(outbox, stem, "telegram", ".md", parts)'
SL_DRY = 'if dry_run:\n            return _write_parts(outbox, stem, "slack", ".txt", parts)'
ECHO = "raise DeliveryError(resp.text)"
RSS = "digest/delivery/rss.py"
EMC = "digest/embeddings/client.py"
EMB = "digest/embeddings/base.py"
EMM = "digest/embeddings/matcher.py"
PIPE = "digest/pipeline.py"
TRN = "digest/trends.py"
FBK = "digest/feedback.py"
WGT = "digest/weights.py"
ONB = "digest/onboarding.py"
ONW = ".github/workflows/onboard.yml"
RNK = "digest/rank.py"
PRF = "digest/profile.py"
RSS_DRY = (
    'if dry_run:\n            return _write_parts(outbox, stem, "rss", ".xml", [xml.rstrip("\\n")])'
)
HTTPS_EMB = (
    'if parts.scheme != "https" or not parts.hostname:\n'
    '                raise ValueError("embeddings base_url must start with https://")'
)
FB_ASSOC = 'frozenset({"OWNER", "COLLABORATOR"})'
SEM_GATE = "if semantic and emb.enabled else 0.0"
TREND_EXC = (
    "except Exception as exc:\n"
    '        run.warnings.append(f"trends skipped ({type(exc).__name__})")'
)
SEM_EXC = "except Exception as exc:  # an optional extra must never cost the user their digest"
SEM_MSG = 'f"semantic matching skipped, keywords only ({name})"'
HTTP_ONLY = 'if h.url.startswith(("https://", "http://"))'
LABEL_OK = 'summary.status == "ok" and not summary.injection_flag'
FB_DRY = "if dry_run:\n        return result"
HOOK_GUARD = "if WEBHOOK_RE.fullmatch(cfg.webhook_url) is None:"
TEXT_ONLY = '{"text": clean(text)}'
SEND = "send_telegram(parts, telegram_config_from_env(env), client=client"


@dataclass(frozen=True)
class Mutation:
    name: str
    path: str
    old: str
    new: str
    tests: tuple[str, ...] = CHAT


def off(code: str) -> str:
    return code.replace("if dry_run:", "if False:", 1)


MUTATIONS = (
    Mutation("tg-url-guard", TG, "if URL_RE.fullmatch(url) is None:", "if False:"),
    Mutation("tg-token-shape", TG, "if TOKEN_RE.fullmatch(token) is None:", "if False:"),
    Mutation("tg-ascii-digits", TG, 'r"[0-9]{3,20}:', 'r"\\d{3,20}:'),
    Mutation("tg-chat-shape", TG, "if CHAT_RE.fullmatch(chat) is None:", "if False:"),
    Mutation("tg-error-echo", TG, "raise _error_for(resp.status_code)", ECHO),
    Mutation("no-redirects", CM, "follow_redirects=False", "follow_redirects=True"),
    Mutation("unreachable-echo", CM, UNREACH, 'unreachable ({exc})")'),
    Mutation("unreachable-chain", CM, UNREACH + " from None", UNREACH + " from exc"),
    Mutation("chunk-limit", CM, "units(piece) > limit:", "units(piece) > limit * 100:"),
    Mutation("oversize-line", CM, 'line if units(line) <= limit else placeholder + "\\n"', "line"),
    Mutation("wait-cap", CM, "return min(MAX_WAIT, max(0.0, hint))", "return hint"),
    Mutation("utf16-units", CM, UTF16, "return len(text)"),
    Mutation("slack-shape", SL, "if WEBHOOK_RE.fullmatch(url) is None:", "if False:"),
    Mutation("slack-send-guard", SL, HOOK_GUARD, "if False:"),
    Mutation("slack-text-only", SL, TEXT_ONLY, TEXT_ONLY.replace("}", ', "mrkdwn": True}')),
    Mutation("slack-ascii-retry", SL, "raw.isascii() and raw.isdigit()", "raw.isdigit()"),
    Mutation("slack-lt", RD, '.replace("<", "&lt;")', '.replace("<", "<")'),
    Mutation("slack-gt", RD, '.replace(">", "&gt;")', '.replace(">", ">")'),
    Mutation("slack-amp", RD, 'flat.replace("&", "&amp;")', "flat"),
    Mutation("slack-flatten", RD, 'flat = " ".join(value.split())', "flat = value"),
    Mutation("slack-lookalike", RD, LOOKALIKE, "str.maketrans({})"),
    Mutation("slack-url-pipe", RD, URL_UNSAFE, URL_UNSAFE.replace("|", "")),
    Mutation("slack-label-pipe", RD, '.replace("|", "\\u00a6")', '.replace("|", "|")'),
    Mutation("http-url-whitespace", RD, SPACES, "if False:"),
    Mutation("cli-telegram-dry-run", CLI, TG_DRY, off(TG_DRY)),
    Mutation("cli-slack-dry-run", CLI, SL_DRY, off(SL_DRY)),
    Mutation("cli-injected-sleep", CLI, SEND + ", sleep=sleep)", SEND + ")"),
    # ---- Phase 6 (Stage 23) ----------------------------------------------------------
    Mutation("rss-illegal-chars", RSS, 'return _XML_ILLEGAL.sub("", value)', "return value", RSS_T),
    Mutation(
        "rss-text-type",
        RSS,
        'add(node, "content", entry.content, type="text")',
        'add(node, "content", entry.content, type="html")',
        RSS_T,
    ),
    Mutation("rss-stable-id", RSS, "','.join(sorted(item_ids))", "','.join(item_ids)", RSS_T),
    Mutation("rss-entry-cap", RSS, "return merged[:MAX_ENTRIES]", "return merged", RSS_T),
    Mutation(
        "rss-content-cap",
        RSS,
        "content=_clip(text, MAX_CONTENT, _TRUNCATED),",
        "content=text,",
        RSS_T,
    ),
    Mutation("rss-dry-run", CLI, RSS_DRY, off(RSS_DRY), RSS_T),
    Mutation(
        "rss-self-link-host", CLI, 'if repo and server == "https://github.com"', "if repo", RSS_T
    ),
    Mutation(
        "rss-repo-guard", CLI, 'repo = repo if _REPO.match(repo) else ""', "repo = repo", RSS_T
    ),
    Mutation("emb-hf-url-guard", EMC, "if HF_URL_RE.fullmatch(url) is None:", "if False:", EMB_T),
    Mutation("emb-ssrf", EMC, "check_public_url(cfg.base_url, resolver)", "None", EMB_T),
    Mutation(
        "emb-key-header",
        EMC,
        'headers = {"Authorization": f"Bearer {self._key}"}',
        "headers = {}",
        EMB_T,
    ),
    Mutation(
        "emb-retry-ascii",
        EMC,
        "raw.isascii() and raw.isdigit() and len(raw) <= 4",
        "raw.isdigit() and len(raw) <= 4",
        EMB_T,
    ),
    Mutation(
        "emb-status-echo",
        EMC,
        "raise status_error(resp.status_code)",
        "raise EmbeddingError(resp.text)",
        EMB_T,
    ),
    Mutation("emb-no-redirect", CM, "follow_redirects=False", "follow_redirects=True", EMB_T),
    Mutation(
        "emb-dim-cap", EMB, "if not 1 <= len(vec) <= MAX_DIM:", "if not 1 <= len(vec):", EMB_T
    ),
    Mutation(
        "emb-bool-number",
        EMB,
        "if isinstance(v, bool) or not isinstance(v, int | float):",
        "if not isinstance(v, int | float):",
        EMB_T,
    ),
    Mutation(
        "emb-key-has-model",
        EMM,
        "[TEXT_VERSION, provider, model, digest]",
        "[TEXT_VERSION, provider, digest]",
        EMB_T,
    ),
    Mutation("emb-item-cap", EMM, "pool = pool[: cfg.max_items]", "pool = pool", EMB_T),
    Mutation("emb-quantized", EMM, "vec = quantize(raw)", "vec = normalize(raw)", EMB_T),
    Mutation(
        "emb-profile-https",
        PRF,
        HTTPS_EMB,
        HTTPS_EMB.replace('parts.scheme != "https" or ', ""),
        EMB_T,
    ),
    Mutation(
        "emb-failsoft", PIPE, SEM_EXC, SEM_EXC.replace("Exception", "ZeroDivisionError"), EMB_T
    ),
    Mutation("emb-no-error-text", PIPE, SEM_MSG, SEM_MSG.replace("{name}", "{exc}"), EMB_T),
    Mutation("emb-rank-gate", RNK, SEM_GATE, "if semantic else 0.0", PIPE_T),
    Mutation("trend-min-stories", TRN, "MIN_STORIES = 3", "MIN_STORIES = 1", TREND_T),
    Mutation("trend-same-page", TRN, "if first != index:", "if False:", TREND_T),
    Mutation("trend-http-only", TRN, HTTP_ONLY, "if True", TREND_T),
    Mutation("trend-flagged-label", TRN, LABEL_OK, 'summary.status == "ok"', TREND_T),
    Mutation(
        "trend-failsoft",
        PIPE,
        TREND_EXC,
        TREND_EXC.replace("Exception", "ZeroDivisionError"),
        TREND_T,
    ),
    Mutation(
        "fb-association", FBK, FB_ASSOC, FB_ASSOC.replace("}", ', "CONTRIBUTOR", "NONE"}'), FB_T
    ),
    Mutation(
        "fb-person-only",
        FBK,
        'and user.get("type") == "User"',
        'and user.get("type") is not None',
        FB_T,
    ),
    Mutation("fb-not-a-pr", FBK, 'and "pull_request" not in issue', "and True", FB_T),
    Mutation(
        "fb-vote-fullmatch",
        FBK,
        "match = VOTE_RE.fullmatch(line)",
        "match = VOTE_RE.search(line)",
        FB_T,
    ),
    Mutation("fb-body-limit", FBK, "if len(body) > MAX_BODY:", "if False:", FB_T),
    Mutation(
        "fb-vote-cap",
        FBK,
        "if not 1 <= len(lines) <= MAX_VOTES_PER_ISSUE:",
        "if not 1 <= len(lines):",
        FB_T,
    ),
    Mutation("fb-url-guard", FBK, "if URL_RE.fullmatch(url) is None:", "if False:", FB_T),
    Mutation("fb-repo-guard", FBK, "if REPO_RE.fullmatch(repo) is None:", "if False:", FB_T),
    Mutation(
        "fb-no-redirect",
        FBK,
        "follow_redirects=False, **kwargs",
        "follow_redirects=True, **kwargs",
        FB_T,
    ),
    Mutation(
        "fb-error-echo",
        FBK,
        "raise _status_error(resp.status_code)",
        "raise FeedbackError(resp.text)",
        FB_T,
    ),
    Mutation(
        "fb-token-repr",
        FBK,
        'return "GitHubIssues()"',
        'return f"GitHubIssues({self._token})"',
        FB_T,
    ),
    Mutation(
        "fb-idempotent",
        FBK,
        "if number in counted:\n            continue",
        "if False:\n            continue",
        FB_T,
    ),
    Mutation(
        "fb-weekly-gate", FBK, "if not force and not is_due(current, now):", "if False:", FB_T
    ),
    Mutation("fb-dry-run", FBK, FB_DRY, FB_DRY.replace("if dry_run:", "if False:"), FB_T),
    Mutation(
        "fb-run-bound",
        WGT,
        "bounded = max(-MAX_RUN_MOVE, min(MAX_RUN_MOVE, move))",
        "bounded = move",
        FB_T,
    ),
    Mutation("fb-decay", WGT, "return float(0.5 ** (days / HALF_LIFE_DAYS))", "return 1.0", FB_T),
    Mutation(
        "fb-weight-limit",
        WGT,
        "merged[topic] = max(-LIMIT, min(LIMIT, merged.get(topic, 0.0) + bounded))",
        "merged[topic] = merged.get(topic, 0.0) + bounded",
        FB_T,
    ),
    Mutation(
        "fb-topic-filter", WGT, "if topic in TOPIC_SLUGS:  # delivered.json", "if True:  #", FB_T
    ),
    Mutation("fb-rank-bound", RNK, "max(-1.0, min(1.0, total))", "total", FB_T),
    Mutation(
        "fb-gate",
        PIPE,
        "votes = weights if profile.feedback.enabled else None",
        "votes = weights",
        PIPE_T,
    ),
    Mutation(
        "fb-thumbs-gate",
        CLI,
        'links = _repo_url(env) if profile.feedback.enabled else ""',
        "links = _repo_url(env)",
        PIPE_T,
    ),
    Mutation("onb-size-cap", ONB, "if len(body) > MAX_BODY:", "if False:", ONB_T),
    Mutation("onb-ack", ONB, "if ACK not in values.get(L_ACK, set()):", "if False:", ONB_T),
    Mutation(
        "onb-checkbox-allowlist",
        ONB,
        "if match is None or match.group(2) not in allowed:",
        "if match is None:",
        ONB_T,
    ),
    Mutation("onb-choice-allowlist", ONB, "if text and text not in allowed:", "if False:", ONB_T),
    Mutation(
        "onb-repeat-section",
        ONB,
        "            if label in sections:\n",
        "            if False:\n",
        ONB_T,
    ),
    Mutation(
        "onb-unknown-heading",
        ONB,
        "current = None  # an unknown heading ends the section; its text is ignored",
        "pass",
        ONB_T,
    ),
    Mutation("onb-control-chars", ONB, "or _CTRL.search(ln) for ln in lines):", "):", ONB_T),
    Mutation(
        "onb-no-echo",
        ONB,
        'raise OnboardError([f"{label}: is missing or not valid" for label in seen]) from None',
        'raise OnboardError([f"{label}: {exc}" for label in seen]) from None',
        ONB_T,
    ),
    Mutation(
        "onb-keep-agent-access",
        ONB,
        'return current.agent_access.model_dump(mode="json")',
        "return None",
        ONB_T,
    ),
    Mutation("onb-ascii-digits", ONB, 'r"[0-9]{1,3}"', 'r"\\d{1,3}"', ONB_T),
    Mutation(
        "onw-owner-author",
        ONW,
        "github.event.issue.user.login == github.repository_owner &&",
        "true &&",
        ONW_T,
    ),
    Mutation(
        "onw-owner-sender",
        ONW,
        "github.event.sender.login == github.repository_owner &&",
        "true &&",
        ONW_T,
    ),
    Mutation(
        "onw-template-guard",
        ONW,
        "github.repository != 'gowthamsai09/axon-template' &&",
        "true &&",
        ONW_T,
    ),
    Mutation(
        "onw-env-only",
        ONW,
        "          ISSUE_BODY: ${{ github.event.issue.body }}   # untrusted: env only\n",
        "          ISSUE_BODY: ${{ github.event.issue.body }}\n"
        "          X: ${{ github.event.issue.title }}\n",
        ONW_T,
    ),
    Mutation("onw-commit-scope", ONW, "git add profile.yaml", "git add -A", ONW_T),
    Mutation(
        "onw-persist-creds",
        ONW,
        "          persist-credentials: false\n      - uses: actions/setup-python",
        "          persist-credentials: true\n      - uses: actions/setup-python",
        ONW_T,
    ),
    Mutation("onw-issue-number", ONW, "*[!0-9]*) echo", "*) echo", ONW_T),
    Mutation(
        "onw-permissions",
        ONW,
        "  issues: write     #",
        "  pull-requests: write\n  issues: write     #",
        ONW_T,
    ),
)


def pytest_run(cwd: Path, timeout: int, tests: tuple[str, ...]) -> int:
    cmd = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests]
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout).returncode


def run_one(mutation: Mutation, timeout: int) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "repo"
        shutil.copytree(ROOT, work, ignore=IGNORE)
        target = work / mutation.path
        text = target.read_text(encoding="utf-8")
        if text.count(mutation.old) != 1:
            return "BROKEN (pattern not found exactly once)"
        target.write_text(text.replace(mutation.old, mutation.new, 1), encoding="utf-8")
        code = pytest_run(work, timeout, mutation.tests)
    if code == 1:
        return "KILLED"
    return "SURVIVED" if code == 0 else f"ERROR (pytest exit code {code})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="only check the patterns")
    parser.add_argument("--only", help="run one mutation by name")
    parser.add_argument("--prefix", help="run every mutation whose name starts with this")
    parser.add_argument("--timeout", type=int, default=600, help="seconds per pytest run")
    args = parser.parse_args(argv)
    chosen = [
        m for m in MUTATIONS
        if args.only in (None, m.name) and m.name.startswith(args.prefix or "")
    ]  # fmt: skip
    if not chosen:
        print(f"no mutation named {args.only or args.prefix!r}", file=sys.stderr)
        return 2
    if args.check:
        bad = [m.name for m in chosen
               if (ROOT / m.path).read_text(encoding="utf-8").count(m.old) != 1]  # fmt: skip
        print(f"{len(chosen) - len(bad)} of {len(chosen)} patterns match exactly once", end="")
        print(f"; BROKEN: {', '.join(bad)}" if bad else "")
        return 1 if bad else 0
    print("baseline (unmodified code) ...", flush=True)
    wanted = tuple(dict.fromkeys(t for m in chosen for t in m.tests))
    if pytest_run(ROOT, args.timeout, wanted) != 0:
        print("these tests fail without any mutation: fix them first", file=sys.stderr)
        return 2
    results = {m.name: run_one(m, args.timeout) for m in chosen}
    for name, verdict in results.items():
        print(f"{verdict:<10} {name}" if " " not in verdict else f"{verdict}  {name}")
    survivors = [n for n, v in results.items() if v != "KILLED"]
    print(f"\n{len(results) - len(survivors)} killed, {len(survivors)} not killed")
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
