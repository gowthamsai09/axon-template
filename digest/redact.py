"""Keep secrets out of logs, warnings and error messages (spec 1, 10.1)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

SECRET_ENV_NAMES = (
    "LLM_API_KEY",
    "LLM_API_KEY_FALLBACK",
    "SMTP_PASS",
    "SMTP_USER",
    "MAIL_TO",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "SLACK_WEBHOOK_URL",
    "GITHUB_TOKEN",
    "GH_TOKEN",
)
_MIN_SECRET_LEN = 6


def secret_values(env: Mapping[str, str]) -> list[str]:
    """Values of the known secret variables that are set (longest first)."""
    found = {env[n] for n in SECRET_ENV_NAMES if len(env.get(n, "")) >= _MIN_SECRET_LEN}
    return sorted(found, key=len, reverse=True)


def redact(text: str, secrets: Iterable[str]) -> str:
    for s in secrets:
        if len(s) >= _MIN_SECRET_LEN:
            text = text.replace(s, "***")
    return text
