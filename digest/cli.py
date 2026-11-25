"""CLI entry point (spec 4.2, 6.3, 6.8): validate-profile, run, init, notify-failure, feedback.

Exit codes: 0 ok (including "nothing new" and a handled Tier 3 refusal), 1 runtime failure
(feed, delivery, unexpected error), 2 configuration or usage error (profile, secrets, feed
location). Secrets only come from the environment and are redacted from everything printed."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
import yaml
from pydantic import BaseModel, ValidationError

from .delivery.common import Sleep
from .delivery.email import (
    DeliveryConfigError,
    DeliveryError,
    SmtpFactory,
    build_message,
    default_factory,
    send_email,
    smtp_config_from_env,
    write_outbox,
)
from .delivery.rss import FEED_FILE, build_feed, store_feed
from .delivery.slack import send_slack, slack_chunks, slack_config_from_env
from .delivery.telegram import send_telegram, telegram_chunks, telegram_config_from_env
from .embeddings.base import Embedder
from .embeddings.client import get_embedder
from .envfile import EnvFileError, merge_env, parse_env_file
from .extract import TextEnricher
from .feed_client import make_reader
from .feedback import FeedbackError, GitHubIssues, run_feedback
from .onboarding import (
    OnboardError,
    comment_ok,
    comment_problems,
    existing_agent_access,
    write_atomic,
)
from .onboarding import build as build_onboarding
from .pipeline import DigestRun, run_digest
from .profile import (
    TOPIC_SLUGS,
    AgentAccess,
    Delivery,
    Embeddings,
    Interests,
    Profile,
    ProfileError,
    Sources,
    Trends,
    load_profile,
)
from .profile import Summarizer as SummarizerCfg
from .redact import redact, secret_values
from .render import build_view, render_email, render_slack, render_telegram
from .safe_http import USER_AGENT, Resolver, SafeFetcher, system_resolver
from .schedule import cron_utc
from .state import StateStore
from .summarizers.router import ConfigError, SummarizerRouter, get_summarizer
from .task_result import TaskItemSummary
from .tier3 import TaskError, TaskRequest, build_result, check_policy, parse_task
from .weights import load_effective

EXIT_OK, EXIT_FAIL, EXIT_CONFIG = 0, 1, 2
_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
DEFAULT_MODELS = {"deepseek": "deepseek-flash", "huggingface": "openai/gpt-oss-120b"}
SUGGESTED_EMBEDDING_MODEL = "thenlper/gte-large"  # listed as recommended on the HF task page


def _repo_url(env: Mapping[str, str]) -> str:
    server, repo = env.get("GITHUB_SERVER_URL", ""), env.get("GITHUB_REPOSITORY", "")
    return f"{server}/{repo}" if server.startswith("https://") and _REPO.match(repo) else ""


# ---- run ------------------------------------------------------------------------------


def _task_entries(
    run: DigestRun,
) -> list[tuple[str, str, str, TaskItemSummary | None, str | None]]:
    out: list[tuple[str, str, str, TaskItemSummary | None, str | None]] = []
    for e in run.entries:
        s = e.summary
        summary = (
            TaskItemSummary(
                one_line=s.one_line,
                key_points=list(s.key_points),
                why_it_matters=s.why_it_matters,
                tags=list(s.tags),
            )
            if s is not None
            else None
        )
        out.append((e.item.id, e.item.title, e.item.canonical_url, summary, e.fallback_snippet))
    return out


def _main_model(run: DigestRun) -> tuple[str, str] | None:
    used = Counter((e.provider, e.model) for e in run.entries if e.summary is not None)
    return used.most_common(1)[0][0] if used else None


def _write_task(
    state: StateStore,
    task_id: str,
    profile: Profile,
    run: DigestRun | None,
    *,
    status: str,
    now: datetime,
    warnings: list[str],
) -> None:
    result = build_result(
        task_id,
        profile,
        _task_entries(run) if run else [],
        status=status,
        now=now,
        warnings=warnings,
        model=_main_model(run) if run else None,
    )
    state.write_task(task_id, result.model_dump(mode="json"))


def _write_parts(outbox: Path, stem: str, label: str, suffix: str, parts: list[str]) -> str:
    """Dry run for a chat channel: one file per message that would be sent."""
    outbox.mkdir(parents=True, exist_ok=True)
    for number, text in enumerate(parts, 1):
        path = outbox / f"{stem}.{label}-{number:02d}{suffix}"
        path.write_text(text + "\n", encoding="utf-8")
        print(f"dry run: wrote {path}")
    return "dry-run"


def _deliver(
    profile: Profile,
    run: DigestRun,
    env: Mapping[str, str],
    now: datetime,
    *,
    dry_run: bool,
    outbox: Path,
    smtp_factory: SmtpFactory,
    client: httpx.Client,
    sleep: Sleep,
    state: StateStore,
) -> str:
    links = _repo_url(env) if profile.feedback.enabled else ""  # thumbs only when counted
    view = build_view(run, profile, now, links)
    channel = profile.delivery.channel
    stem = f"digest-{now.strftime('%Y%m%dT%H%M%SZ')}"
    if channel == "email":
        subject, text, html = render_email(view)
        if dry_run:
            msg = build_message(
                subject, text, html,
                sender=env.get("SMTP_USER") or "axon@localhost",
                to=env.get("MAIL_TO") or "you@example.com",
            )  # fmt: skip
            for path in write_outbox(msg, html, outbox, stem):
                print(f"dry run: wrote {path}")
            return "dry-run"
        cfg = smtp_config_from_env(env)
        msg = build_message(subject, text, html, sender=cfg.user, to=cfg.to)
        send_email(msg, cfg, factory=smtp_factory, env=env)
        return "email"
    if channel == "telegram":
        parts = telegram_chunks(render_telegram(view))
        if dry_run:
            return _write_parts(outbox, stem, "telegram", ".md", parts)
        send_telegram(parts, telegram_config_from_env(env), client=client, sleep=sleep)
        return "telegram"
    if channel == "slack":
        parts = slack_chunks(render_slack(view))
        if dry_run:
            return _write_parts(outbox, stem, "slack", ".txt", parts)
        send_slack(parts, slack_config_from_env(env), client=client, sleep=sleep)
        return "slack"
    if channel == "rss":
        subject, text, _html = render_email(view)
        repo = env.get("GITHUB_REPOSITORY", "")
        repo = repo if _REPO.match(repo) else ""
        server = env.get("GITHUB_SERVER_URL", "")
        self_url = (
            f"https://raw.githubusercontent.com/{repo}/state/{FEED_FILE}"
            if repo and server == "https://github.com"
            else ""
        )
        xml, entries, notes = build_feed(
            state, subject, text, [e.item.id for e in run.entries], now,
            repo=repo, repo_url=_repo_url(env), self_url=self_url,
        )  # fmt: skip
        run.warnings.extend(notes)
        if dry_run:
            return _write_parts(outbox, stem, "rss", ".xml", [xml.rstrip("\n")])
        store_feed(state, entries, xml)
        return "rss"
    raise DeliveryConfigError(f"{channel} delivery is not supported")


def _run(
    args: argparse.Namespace,
    env: Mapping[str, str],
    now: datetime,
    http: httpx.Client | None,
    smtp_factory: SmtpFactory,
    resolver: Resolver,
    sleep: Sleep,
) -> int:
    secrets = secret_values(env)
    try:
        profile = load_profile(args.profile)
    except (ProfileError, OSError) as exc:
        print(redact(str(exc), secrets), file=sys.stderr)
        return EXIT_CONFIG
    task: TaskRequest | None = None
    if env.get("TASK_ID"):
        try:
            task = parse_task(env["TASK_ID"], env.get("TASK_PARAMS", "{}"))
        except TaskError as exc:
            print(f"invalid Tier 3 request: {exc}", file=sys.stderr)
            return EXIT_CONFIG
    state = StateStore(args.state_dir)
    if task is not None:
        reason = check_policy(profile, state, now)
        if reason:
            _write_task(state, task.task_id, profile, None, status="failed", now=now,
                        warnings=[reason, *task.warnings])  # fmt: skip
            print(f"Tier 3 run refused: {reason}")
            return EXIT_OK
    feed = args.feed or env.get("FEED_BASE", "")
    if profile.sources.use_shared_feed and not feed:
        print("no feed configured: pass --feed or set FEED_BASE", file=sys.stderr)
        return EXIT_CONFIG
    owns = http is None
    client = http or httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT})
    try:
        return _execute(
            args, env, now, client, smtp_factory, resolver, sleep, profile, task, state, feed
        )
    finally:
        if owns:
            client.close()


def _execute(
    args: argparse.Namespace,
    env: Mapping[str, str],
    now: datetime,
    client: httpx.Client,
    smtp_factory: SmtpFactory,
    resolver: Resolver,
    sleep: Sleep,
    profile: Profile,
    task: TaskRequest | None,
    state: StateStore,
    feed: str,
) -> int:
    secrets = secret_values(env)
    config_warnings: list[str] = []
    run: DigestRun | None = None
    try:
        reader = make_reader(feed, client) if profile.sources.use_shared_feed else None
        fetcher = SafeFetcher(client, resolver=resolver)
        router: SummarizerRouter | None = None
        embedder: Embedder | None = None
        if not args.no_llm:
            router, config_warnings = get_summarizer(profile, env, http=client, resolver=resolver)
            embedder, embed_warnings = get_embedder(
                profile, env, http=client, resolver=resolver, sleep=sleep
            )
            config_warnings = config_warnings + embed_warnings
        run = run_digest(
            profile,
            reader=reader,
            fetcher=fetcher,
            enricher=TextEnricher(reader, fetcher),
            router=router,
            state=state,
            now=now,
            max_items=task.max_items if task else None,
            use_seen=task is None,  # Tier 3 reads the feed fresh and never consumes items
            embedder=embedder,
            weights=load_effective(state, now) if profile.feedback.enabled else None,
            with_trends=task is None,
        )
        run.warnings = config_warnings + run.warnings
        delivered = "none"
        if task is not None:
            run.warnings.extend(task.warnings)
            delivered = "return_only"
            if task.delivery == "owner_channel":
                if not profile.agent_access.allow_owner_channel:
                    run.warnings.append(
                        "owner_channel delivery is not enabled in this profile "
                        "(agent_access.allow_owner_channel); returned only"
                    )
                elif not run.entries:
                    run.warnings.append("owner_channel: nothing to deliver")
                else:
                    try:
                        sent = _deliver(
                            profile, run, env, now,
                            dry_run=args.dry_run, outbox=args.outbox, smtp_factory=smtp_factory,
                            client=client, sleep=sleep, state=state,
                        )  # fmt: skip
                        delivered = f"owner_channel:{sent}"
                    except (
                        DeliveryError
                    ) as exc:  # the result file is still written (never lose the run)
                        run.warnings.append(
                            f"owner_channel delivery failed: {redact(str(exc), secrets)}"
                        )
            _write_task(state, task.task_id, profile, run, status="completed", now=now,
                        warnings=[*run.notices, *run.warnings])  # fmt: skip
        elif not run.entries:
            print("nothing new for this profile; no digest sent")
        else:
            delivered = _deliver(
                profile, run, env, now,
                dry_run=args.dry_run, outbox=args.outbox, smtp_factory=smtp_factory,
                client=client, sleep=sleep, state=state,
            )  # fmt: skip
            if not args.dry_run:
                ids = [e.item.id for e in run.entries]
                state.record_delivered({e.item.id: list(e.item.tags) for e in run.entries}, now)
                state.mark_seen(ids, now)
        state.prune(now)
        _last_run(state, env, now, task, run, delivered, "ok")
        print(f"done: {len(run.entries)} item(s), delivered={delivered}")
        for notice in run.notices:
            print(f"NOTICE: {redact(notice, secrets)}", file=sys.stderr)
        return EXIT_OK
    except (ConfigError, DeliveryConfigError) as exc:
        message = redact(str(exc), secrets)
        print(message, file=sys.stderr)
        if task is not None:  # the gateway reads this file; never leave it with nothing
            _write_task(
                state, task.task_id, profile, None, status="failed", now=now, warnings=[message]
            )
        return EXIT_CONFIG
    except Exception as exc:  # last resort: record the failure, never leak secrets
        message = redact(f"{type(exc).__name__}: {exc}", secrets)
        print(f"run failed: {message}", file=sys.stderr)
        if task is not None:
            _write_task(state, task.task_id, profile, None, status="failed", now=now,
                        warnings=[message])  # fmt: skip
        _last_run(state, env, now, task, run, "none", "failed", message)
        return EXIT_FAIL


def _last_run(
    state: StateStore,
    env: Mapping[str, str],
    now: datetime,
    task: TaskRequest | None,
    run: DigestRun | None,
    delivered: str,
    status: str,
    error: str = "",
) -> None:
    payload: dict[str, Any] = {
        "run_id": env.get("GITHUB_RUN_ID") or uuid.uuid4().hex,
        "task_id": task.task_id if task else None,
        "finished_at": now.isoformat(),
        "status": status,
        "delivered": delivered,
        "error": error[:500],
        "items": len(run.entries) if run else 0,
        "candidates": run.candidates if run else 0,
        "usage": (
            {
                "prompt_tokens": run.usage.prompt_tokens,
                "completion_tokens": run.usage.completion_tokens,
                "calls": run.usage.calls,
                "cache_hits": run.cache_hits,
            }
            if run
            else {}
        ),
        "warnings": [w[:300] for w in (run.warnings if run else [])[:100]],
        "notices": [n[:300] for n in (run.notices if run else [])[:20]],
    }
    if run is not None and (run.embedded or run.embed_cache_hits):
        payload["usage"]["embedded"] = run.embedded
        payload["usage"]["embed_cache_hits"] = run.embed_cache_hits
    state.write_last_run(payload)


# ---- notify-failure -------------------------------------------------------------------


MAX_RUN_URL = 300


def safe_run_url(url: str, env: Mapping[str, str]) -> str:
    """The run link in a failure notice: https only, no credentials, no whitespace or control
    characters, at most 300 characters, and the host must be the repository host
    (GITHUB_SERVER_URL, default github.com). Anything else is dropped: the notice then has no
    link, never a hostile one."""
    if not url or len(url) > MAX_RUN_URL or re.search(r"[\s\x00-\x1f\x7f]", url):
        return ""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        return ""
    server = urlsplit(env.get("GITHUB_SERVER_URL", "") or "https://github.com").hostname
    return url if parts.hostname.lower() == (server or "github.com").lower() else ""


def _notify(
    args: argparse.Namespace, env: Mapping[str, str], now: datetime, smtp_factory: SmtpFactory
) -> int:
    try:
        cfg = smtp_config_from_env(env)
    except DeliveryConfigError as exc:
        print(f"no failure notice sent: {exc}")
        return EXIT_OK
    url = safe_run_url(args.run_url or "", env)
    lines = ["Your Axon digest run failed."]
    if url:
        lines.append(f"Details: {url}")
    lines.append("This notice never contains secrets; open the run log for the error.")
    msg = build_message(
        "Axon digest failed", "\n".join(lines) + "\n", None, sender=cfg.user, to=cfg.to
    )
    try:
        if args.dry_run:
            for path in write_outbox(msg, None, args.outbox, f"failure-{now:%Y%m%dT%H%M%SZ}"):
                print(f"dry run: wrote {path}")
        else:
            send_email(msg, cfg, factory=smtp_factory, env=env)
    except DeliveryError as exc:
        print(redact(str(exc), secret_values(env)), file=sys.stderr)
        return EXIT_FAIL
    return EXIT_OK


# ---- feedback -------------------------------------------------------------------------


def _feedback(
    args: argparse.Namespace, env: Mapping[str, str], now: datetime, http: httpx.Client | None
) -> int:
    secrets = secret_values(env)
    try:
        profile = load_profile(args.profile)
    except (ProfileError, OSError) as exc:
        print(redact(str(exc), secrets), file=sys.stderr)
        return EXIT_CONFIG
    if not profile.feedback.enabled:
        print("feedback is off in this profile (set feedback.enabled: true)")
        return EXIT_OK
    owns = http is None
    client = http or httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT})
    state = StateStore(args.state_dir)
    try:
        try:
            repo, token = env.get("GITHUB_REPOSITORY", ""), env.get("GITHUB_TOKEN", "")
            issues = GitHubIssues(client, repo, token)
        except FeedbackError as exc:
            print(f"feedback not configured: {redact(str(exc), secrets)}", file=sys.stderr)
            return EXIT_CONFIG
        try:
            result = run_feedback(issues, state, now, dry_run=args.dry_run, force=args.force)
        except FeedbackError as exc:
            print(f"feedback failed: {redact(str(exc), secrets)}", file=sys.stderr)
            return EXIT_FAIL
    finally:
        if owns:
            client.close()
    if result.skipped:
        print(f"feedback skipped: {result.skipped}")
        return EXIT_OK
    verb = "would apply" if args.dry_run else "applied"
    print(
        f"feedback: {verb} {result.votes_applied} vote(s) from {result.issues_applied} issue(s); "
        f"{result.ignored} ignored, {result.unknown_items} for unknown items, "
        f"{result.closed} closed"
    )
    for note in [*result.warnings, *state.warnings]:
        print(f"WARNING: {redact(note, secrets)}", file=sys.stderr)
    return EXIT_OK


# ---- init -----------------------------------------------------------------------------

Ask = Callable[[str], str]


def _ask(ask: Ask, label: str, default: str = "") -> str:
    answer = ask(f"{label}" + (f" [{default}]" if default else "") + ": ").strip()
    return answer or default


def _num(value: str) -> int | str:
    """A number if it looks like one; otherwise the raw text, which validation then rejects."""
    return int(value) if value.lstrip("-").isdigit() else value


def _float(value: str) -> float | str:
    """A number if it looks like one; otherwise the raw text, which validation then rejects."""
    try:
        return float(value)
    except ValueError:
        return value


def _yes(ask: Ask, label: str, default: str = "n") -> bool:
    return _ask(ask, f"{label} (y/n)", default).lower().startswith("y")


def _csv(value: str) -> list[str]:
    return [p.strip() for p in value.split(",") if p.strip()]


def _valid(model: type[BaseModel], collect: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    while True:
        data = collect()
        try:
            model.model_validate(data)
            return data
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"]) or "<section>"
                print(f"  problem: {loc}: {err['msg']}. Let's redo this section.")


def run_init(ask: Ask, out: Path, *, force: bool, now: datetime) -> int:
    if out.exists() and not force:
        print(f"{out} exists; use --force to overwrite", file=sys.stderr)
        return EXIT_CONFIG
    print("Axon setup. Press Enter to accept a default. Never type API keys here.")
    print("Topics: " + ", ".join(TOPIC_SLUGS))

    def interests() -> dict[str, Any]:
        topics = _ask(ask, "Topics (comma separated)", "open-source-llms, agents-tooling")
        return {
            "topics": _csv(topics),
            "follow_entities": _csv(_ask(ask, "Entities to follow (names or github:owner/repo)")),
            "keywords_include": _csv(_ask(ask, "Keywords to include")),
            "keywords_exclude": _csv(_ask(ask, "Keywords to exclude")),
        }

    def sources() -> dict[str, Any]:
        return {
            "use_shared_feed": True,
            "extra_rss": _csv(_ask(ask, "Extra RSS feed URLs (comma separated)")),
        }

    def delivery() -> dict[str, Any]:
        data: dict[str, Any] = {
            "channel": _ask(ask, "Channel (email|telegram|slack|rss)", "email"),
            "frequency": _ask(ask, "Frequency (daily|weekly)", "daily"),
        }
        if data["frequency"] == "weekly":
            data["weekday"] = _ask(ask, "Weekday (mon..sun)", "mon")
        data.update(
            send_time_local=_ask(ask, "Local send time (HH:MM)", "07:30"),
            timezone=_ask(ask, "Timezone (IANA name)", "UTC"),
            depth=_ask(ask, "Depth (headlines|brief|detailed)", "brief"),
            max_items=_num(_ask(ask, "Max items (1-25)", "10")),
            language=_ask(ask, "Language code", "en"),
        )
        return data

    def summarizer() -> dict[str, Any]:
        provider = _ask(
            ask, "Summarizer provider (deepseek|huggingface|openai_compatible)", "deepseek"
        )
        data: dict[str, Any] = {
            "provider": provider,
            "model": _ask(ask, "Model", DEFAULT_MODELS.get(provider, "")),
            "fallback_provider": _ask(
                ask, "Fallback provider (deepseek|huggingface, empty = none)"
            ),
            "fallback_model": "",
            "base_url": "",
        }
        if data["fallback_provider"]:
            data["fallback_model"] = _ask(ask, "Fallback model")
        if provider == "openai_compatible":
            data["base_url"] = _ask(ask, "Base URL (https://...)")
        return data

    def agents() -> dict[str, Any]:
        allow = _ask(ask, "Allow other agents to call your digest? (y/n)", "y").lower()
        return {
            "allow_inbound_agents": allow.startswith("y"),
            "allowed_modes": _csv(
                _ask(ask, "Allowed modes (data, summary, digest)", "data, summary, digest")
            ),
            "max_tier3_runs_per_day": _num(_ask(ask, "Max digest runs per day (1-20)", "5")),
            "allow_owner_channel": _ask(
                ask, "Let agent runs also send the digest to you? (y/n)", "n"
            )
            .lower()
            .startswith("y"),
        }

    def embeddings() -> dict[str, Any]:
        if not _yes(ask, "Match items by meaning too (embeddings; needs an EMBED_API_KEY secret)"):
            return {"enabled": False}
        provider = _ask(ask, "Embeddings provider (huggingface|openai_compatible)", "huggingface")
        data: dict[str, Any] = {
            "enabled": True,
            "provider": provider,
            "model": _ask(ask, "Embeddings model", SUGGESTED_EMBEDDING_MODEL),
            "base_url": "",
            "weight": _float(_ask(ask, "Weight in the score (0 to 1)", "0.25")),
        }
        if provider == "openai_compatible":
            data["base_url"] = _ask(ask, "Embeddings base URL (https://...)")
        return data

    def trends() -> dict[str, Any]:
        if not _yes(ask, "Add a weekly Trends section (weekly digests only)"):
            return {"enabled": False}
        return {"enabled": True, "llm_labels": _yes(ask, "Add a one-line AI label to each trend")}

    def feedback() -> dict[str, Any]:
        return {"enabled": _yes(ask, "Let thumbs up/down votes adjust your topic weights")}

    data: dict[str, Any] = {
        "version": 1,
        "interests": _valid(Interests, interests),
        "sources": _valid(Sources, sources),
        "delivery": _valid(Delivery, delivery),
        "summarizer": _valid(SummarizerCfg, summarizer),
        "agent_access": _valid(AgentAccess, agents),
    }
    if _yes(ask, "Set up the optional extras (semantic matching, trends, feedback)?"):
        data["embeddings"] = _valid(Embeddings, embeddings)
        data["trends"] = _valid(Trends, trends)
        data["feedback"] = feedback()
    try:
        profile = Profile.model_validate(data)
    except ValidationError as exc:
        print(f"profile is not valid: {exc.error_count()} problem(s)", file=sys.stderr)
        return EXIT_CONFIG
    header = "# Axon profile. Secrets go in GitHub Settings > Secrets, never in this file.\n"
    out.write_text(
        header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(f"wrote {out}")
    if profile.embeddings.enabled:
        print("Add the EMBED_API_KEY secret (Settings > Secrets > Actions) for semantic matching.")
    if profile.trends.enabled and profile.delivery.frequency != "weekly":
        print("Note: the Trends section only appears in weekly digests.")
    print(
        f'cron (UTC): "{cron_utc(profile.delivery, now)}"  (update the schedule line in digest.yml)'
    )
    return EXIT_OK


def _onboard(args: argparse.Namespace, env: Mapping[str, str], now: datetime) -> int:
    """Issue form to profile.yaml. The Issue body arrives through the environment only."""
    body = env.get("ISSUE_BODY", "")
    comment: Path = args.comment_file
    try:
        result = build_onboarding(
            body,
            now=now,
            default_models=DEFAULT_MODELS,
            default_embedding_model=SUGGESTED_EMBEDDING_MODEL,
            agent_access=existing_agent_access(args.profile),
        )
    except OnboardError as exc:
        write_atomic(comment, comment_problems(exc.problems))
        print("onboarding: the form is not valid; nothing was changed", file=sys.stderr)
        return EXIT_FAIL
    write_atomic(args.profile, result.yaml_text)
    write_atomic(comment, comment_ok(result))
    print(f"onboarding: wrote {args.profile}")
    return EXIT_OK


# ---- entry point ----------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="digest.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate-profile", help="validate profile.yaml")
    v.add_argument("path", nargs="?", type=Path, default=Path("profile.yaml"))
    r = sub.add_parser("run", help="build and deliver a digest")
    r.add_argument("--state-dir", type=Path, default=Path("state"))
    r.add_argument("--profile", type=Path, default=Path("profile.yaml"))
    r.add_argument("--feed", default="", help="feed URL or folder (default: $FEED_BASE)")
    r.add_argument("--dry-run", action="store_true", help="write to --outbox, send nothing")
    r.add_argument("--outbox", type=Path, default=Path("outbox"))
    r.add_argument("--no-llm", action="store_true", help="extractive snippets only, no key needed")
    r.add_argument("--env-file", type=Path, default=None, help="KEY=VALUE file (default: ./.env)")
    i = sub.add_parser("init", help="interactive profile.yaml creation")
    i.add_argument("--out", type=Path, default=Path("profile.yaml"))
    i.add_argument("--force", action="store_true")
    f = sub.add_parser("feedback", help="turn feedback Issues into topic weights")
    f.add_argument("--state-dir", type=Path, default=Path("state"))
    f.add_argument("--profile", type=Path, default=Path("profile.yaml"))
    f.add_argument("--dry-run", action="store_true", help="read and report; change nothing")
    f.add_argument("--force", action="store_true", help="run even if it ran in the last 6 days")
    f.add_argument("--env-file", type=Path, default=None, help="KEY=VALUE file (default: ./.env)")
    o = sub.add_parser("onboard", help="onboarding Issue form (env ISSUE_BODY) to profile")
    o.add_argument("--profile", type=Path, default=Path("profile.yaml"))
    o.add_argument("--comment-file", type=Path, default=Path("onboard-comment.md"))
    n = sub.add_parser("notify-failure", help="email a short failure notice")
    n.add_argument("--run-url", default="")
    n.add_argument("--dry-run", action="store_true")
    n.add_argument("--outbox", type=Path, default=Path("outbox"))
    n.add_argument("--env-file", type=Path, default=None, help="KEY=VALUE file (default: ./.env)")
    return ap


def _environment(
    explicit: Mapping[str, str] | None, env_file: Path | None
) -> tuple[Mapping[str, str], list[str]]:
    """The process environment, plus an optional local .env (never inside GitHub Actions).
    A caller-supplied mapping (tests) is used as is."""
    if explicit is not None:
        return explicit, []
    real = dict(os.environ)
    path = env_file
    if path is None:
        default = Path(".env")
        if real.get("GITHUB_ACTIONS") == "true" or not default.is_file():
            return real, []
        path = default
    values, warnings = parse_env_file(path)
    names = ", ".join(sorted(values)) or "no variables"
    return merge_env(values, real), [f"loaded {path}: {names}", *warnings]


def main(
    argv: list[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    now: datetime | None = None,
    http: httpx.Client | None = None,
    smtp_factory: SmtpFactory = default_factory,
    resolver: Resolver = system_resolver,
    ask: Ask = input,
    sleep: Sleep = time.sleep,
) -> int:
    args = _parser().parse_args(argv)
    moment = now or datetime.now(UTC)
    environment: Mapping[str, str] = {}
    if args.cmd in {"run", "notify-failure", "feedback"}:
        try:
            environment, notes = _environment(env, args.env_file)
        except EnvFileError as exc:
            print(exc, file=sys.stderr)
            return EXIT_CONFIG
        for note in notes:
            print(note, file=sys.stderr)
    if args.cmd == "validate-profile":
        try:
            load_profile(args.path)
        except ProfileError as exc:
            print(exc, file=sys.stderr)
            return EXIT_FAIL
        print(f"{args.path}: OK")
        return EXIT_OK
    if args.cmd == "run":
        return _run(args, environment, moment, http, smtp_factory, resolver, sleep)
    if args.cmd == "feedback":
        return _feedback(args, environment, moment, http)
    if args.cmd == "onboard":
        return _onboard(args, env if env is not None else dict(os.environ), moment)
    if args.cmd == "init":
        try:
            return run_init(ask, args.out, force=args.force, now=moment)
        except (EOFError, KeyboardInterrupt):
            print("\ninit cancelled", file=sys.stderr)
            return EXIT_CONFIG
    return _notify(args, environment, moment, smtp_factory)


if __name__ == "__main__":
    sys.exit(main())
