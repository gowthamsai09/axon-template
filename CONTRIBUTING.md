# Contributing to axon-template

`docs/axon-architecture.md` in the hub repo (https://github.com/gowthamsai09/Axon) is the single source of
truth. If code and spec disagree, open an issue or PR that changes the spec first; do not
deviate silently.

## Setup and checks

```bash
python3.12 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
ruff check . && ruff format --check . && mypy . && pytest
```

## Rules

- Keep files small and typed; add tests for every model or rule.
- No secrets anywhere. Untrusted input reaches code through `env:` only, never `${{ }}` inside `run:`.
- Pin third-party GitHub Actions to commit SHAs (see the hub's `docs/pinning-actions.md`).
- Open a PR against `main`; CI must be green.
