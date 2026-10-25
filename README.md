# axon-template

GitHub **template repository** for the per-user pipeline of [Axon](https://github.com/gowthamsai09/Axon)
(Plane 2). Click **Use this template** (not Fork) and make your copy **private**. Spec: sections 4.2, 6, 9.

**Phase 2 status:** the digest pipeline works end to end: shared feed -> rank -> text -> summarize
(DeepSeek, Hugging Face or any OpenAI-compatible endpoint, with fallback and an extractive last
resort) -> email, with a state branch, a summary cache and Tier 3 task results. Telegram, Slack and
RSS delivery arrive in Phase 6; the onboarding issue form and gateway registration in Phases 2b and 4.

## Set up your copy

1. Edit `profile.yaml` (or run `python -m digest.cli init`; it prints the UTC cron line for `digest.yml`).
2. **Settings -> Secrets and variables -> Actions**
   - Secrets: `LLM_API_KEY` (use a dedicated key with a spending limit), `SMTP_USER`, `SMTP_PASS`, `MAIL_TO`;
     optional `LLM_API_KEY_FALLBACK`.
   - Variables: `FEED_BASE` (the published feed, `https://<org>.github.io/axon-feed/feed`); optional
     `SMTP_HOST` and `SMTP_PORT` (default `smtp.gmail.com`, `465`; use port 587 for STARTTLS providers).
3. Run the **digest** workflow once by hand (Actions -> digest -> Run workflow).

## Run it locally (Windows, PowerShell)

```powershell
conda activate axon
pip install -r requirements.txt -r requirements-dev.txt

# No key, no email, no network except the feed folder: extractive snippets into .local\outbox
python -m digest.cli run --profile profile.yaml `
  --feed C:\Users\DGY3KOR\Desktop\Axon\Axon\feed-out `
  --state-dir .local\state --outbox .local\outbox --dry-run --no-llm

# With a real summarizer: either set the key for this session only ...
$env:LLM_API_KEY = Read-Host "LLM key"
# ... or put it in a git-ignored .env (see below)
python -m digest.cli run --profile profile.yaml --feed C:\Users\DGY3KOR\Desktop\Axon\Axon\feed-out `
  --state-dir .local\state --outbox .local\outbox --dry-run
```

`--dry-run` writes an `.eml` and an `.html` preview to the outbox, sends nothing and does not mark items
as seen (summaries are still cached, so repeat runs are free). Behind a TLS-inspecting proxy set
`$env:SSL_CERT_FILE` to your corporate CA bundle. Other commands: `validate-profile`, `init`, `notify-failure`.

### Local `.env` (optional, local runs only)

Copy `.env.example` to `.env` in the repo root and fill in the values (it is git-ignored). `run` and
`notify-failure` read `./.env` automatically (or the file given with `--env-file`) and print the variable
*names* they loaded, never the values. Real environment variables win over the file; empty values count as
unset; the file is ignored inside GitHub Actions. Only the variables the pipeline reads are accepted.
In GitHub, use Secrets and Variables instead.

Secrets (LLM key, SMTP, bot tokens) go in **Settings -> Secrets and variables -> Actions**, never in files.
