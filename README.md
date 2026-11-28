# axon-template

[![ci](https://github.com/gowthamsai09/axon-template/actions/workflows/ci.yml/badge.svg)](https://github.com/gowthamsai09/axon-template/actions/workflows/ci.yml)

**Plane 2 of [Axon](https://github.com/gowthamsai09/Axon): your personal digest pipeline.** Create your own
**private** repository from this template, edit `profile.yaml`, add your secrets, and a scheduled GitHub
Actions workflow reads the shared feed, ranks it by your interests, summarises the top items with **your own**
LLM key and delivers the digest by email, Telegram, Slack or an Atom file.

![Axon overview](docs/img/overview.png)

## Quick start

1. **Use this template** > Create a new repository > **Private** (for example `axon-digest`).
2. Set your interests: Issues > New issue > **Onboarding** (create the label `onboarding` first), or
   `python -m digest.cli init`, or edit `profile.yaml`.
3. Add secrets and a variable (Settings > Secrets and variables > Actions):

| Name | Type | For |
|---|---|---|
| `LLM_API_KEY` | secret | Summariser key (DeepSeek, Hugging Face or OpenAI-compatible) |
| `SMTP_USER`, `SMTP_PASS`, `MAIL_TO` | secrets | Email |
| `SMTP_HOST`, `SMTP_PORT` | variables | Email server |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | secrets | Telegram |
| `SLACK_WEBHOOK_URL` | secret | Slack |
| `FEED_BASE` | variable | `https://gowthamsai09.github.io/axon-feed/feed` |
| `EMBED_API_KEY`, `LLM_API_KEY_FALLBACK` | secrets | Optional |

4. Run **Actions > digest > Run workflow** once, then let the cron run (default `0 2 * * *` UTC = 07:30 IST).

Never put a key in `profile.yaml`, an Issue or the onboarding form. Full guide in the hub:
[getting-started](https://github.com/gowthamsai09/Axon/blob/main/docs/getting-started.md).

## What it does

Load feed > drop what you already received > rank by your profile > fetch text for the top N > summarise with
a router (primary, circuit breaker, fallback, extractive snippet) > validate > deliver > remember.
Every summary links to its source, names the model, and is checked (only the source URL, figures present in the source).
Details: [ARCHITECTURE.md](ARCHITECTURE.md).

Channels: `email`, `telegram`, `slack`, `rss`. Optional, off by default: semantic matching (`embeddings`),
weekly `trends`, thumbs-up/down `feedback`. Optional agent access (`agent_access`) lets agents start your digest
through the Axon gateway with a token you issue and can revoke.

## Develop

```powershell
pip install -r requirements.txt -r requirements-dev.txt
python -m digest.cli validate-profile
python -m digest.cli run --profile profile.yaml --feed ..\feed-out --state-dir .local\state --outbox .local\outbox --dry-run
python -m evals.run replay
ruff format .; ruff check .; mypy .; pytest
```

Tests that read shared prompts and corpora look for the hub in a sibling folder named `Axon`; set `HUB_DIR`
if yours has another name. Tests read the frozen `tests/fixtures/profile_base.yaml`, not your `profile.yaml`.

## Safety

Keys come from GitHub secrets and are redacted from logs and state. Fetching is SSRF-guarded. Untrusted text
never becomes a command, a workflow input or a URL to fetch without validation. Workflows have least-privilege
`permissions:`, pinned actions, `env:`-only inputs and a guard that stops them running in the template repo itself.

## Status and limits

**Verified live:** email, Telegram, Slack, the onboarding form, feedback links, the Atom channel, keepalive, and
Tier 3 over A2A. **Mocked only:** weekly trends, embeddings, `evals live`, the failure notice.
A repo created from a template is a snapshot: later template fixes do not reach it, so copy changed files by hand.
The onboarding form needs a personal-account repository.

## Contributing, security, licence

[CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md). Apache-2.0, Copyright 2026 gowthamsai09.
