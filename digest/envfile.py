"""Optional `.env` file for LOCAL runs (never used in GitHub Actions, where Secrets and
Variables supply the environment). Plain KEY=VALUE lines; only the names this pipeline reads are
accepted; empty values count as unset; real environment variables always win. Errors and
warnings name the variable or line number, never a value."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

ALLOWED = frozenset(
    {
        "LLM_API_KEY", "LLM_API_KEY_FALLBACK", "SMTP_USER", "SMTP_PASS", "MAIL_TO",
        "SMTP_HOST", "SMTP_PORT", "FEED_BASE", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
        "SLACK_WEBHOOK_URL",
    }
)  # fmt: skip
_LINE = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


class EnvFileError(Exception):
    """The file cannot be read or has a malformed line (message contains no values)."""


def parse_env_file(path: Path) -> tuple[dict[str, str], list[str]]:
    """Returns (variables, warnings)."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise EnvFileError(f"cannot read {path}: {type(exc).__name__}") from None
    values: dict[str, str] = {}
    warnings: list[str] = []
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if match is None:
            raise EnvFileError(f"{path}: line {number} is not KEY=VALUE")
        name, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:  # trailing comment on an unquoted value
            value = value.split(" #", 1)[0].rstrip()
        if name not in ALLOWED:
            warnings.append(f"{path}: ignored {name} (not a variable this pipeline reads)")
        elif value:
            values[name] = value
    return values, warnings


def merge_env(file_values: Mapping[str, str], real: Mapping[str, str]) -> dict[str, str]:
    """File values fill gaps; anything set (non-empty) in the real environment wins."""
    merged = dict(file_values)
    merged.update({k: v for k, v in real.items() if v or k not in merged})
    return merged
