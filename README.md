# axon-template

Plane 2 of [Axon](https://github.com/gowthamsai09/Axon): the per-user digest pipeline. Create your own **private** repo
with "Use this template", edit `profile.yaml`, add your secrets, and a scheduled workflow summarizes the shared feed
with your own LLM key and e-mails you. Spec: sections 4.2 and 6.

**Status:** Phases 2, 4 and 5 built and tested locally. Real e-mail, real Actions runs and a live provider call are
first-push checks. Telegram, Slack and RSS delivery and the onboarding form are Phase 6.

```powershell
pip install -r requirements.txt -r requirements-dev.txt
python -m digest.cli init                     # interactive profile.yaml
python -m digest.cli run --profile profile.yaml --feed ..\feed-out --state-dir .local\state --outbox .local\outbox --dry-run
python -m evals.run replay                    # offline check of the evaluation harness
ruff format .; ruff check .; mypy .; pytest
```
Secrets (GitHub Settings, never in files): `LLM_API_KEY`, `SMTP_USER`, `SMTP_PASS`, `MAIL_TO`, optional
`LLM_API_KEY_FALLBACK`. Variables: `FEED_BASE`, optional `SMTP_HOST`, `SMTP_PORT`, `GATEWAY_URL` (Tier 3).

**Safety:** summaries come from untrusted text and are validated (shape, caps, only the source URL, figures present in the
source); output containing your key is rejected; fetching is SSRF-guarded; the failure e-mail carries only a validated run link.
`keepalive.yml` re-enables the digest workflow monthly. `evals/` scores the summarizer on your own model
(`python -m evals.run live`, costs tokens); its 12 seed cases use hand-written reference answers, and the human-checked set
of 30 to 50 cases is yours to complete (`evals/README.md`).
