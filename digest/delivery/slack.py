"""Slack delivery (spec 6.3 step 9, Appendix C 89): an incoming webhook, nothing else.

The webhook URL is a secret (anyone holding it can post). It must match the documented shape
https://hooks.slack.com/services/T.../B.../... exactly, is never printed or stored, and no part
of Slack's answer is repeated. The payload carries only `text`: the webhook documentation lists
no other plain-text field. Redirects are not followed."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import httpx

from .common import GAP_SECONDS, Sleep, chunk_text, clean, number, post_with_retry
from .email import DeliveryConfigError, DeliveryError

WEBHOOK_RE = re.compile(
    r"https://hooks\.slack\.com/services/T[A-Z0-9]{2,20}/B[A-Z0-9]{2,20}/[A-Za-z0-9]{16,64}"
)
MAX_UNITS = 3500  # Slack recommends at most 4,000 characters of text per message
_ITEM_START = re.compile(r"(?m)^(?=\d+\. )")


@dataclass(frozen=True)
class SlackConfig:
    webhook_url: str = field(repr=False)


def slack_config_from_env(env: Mapping[str, str]) -> SlackConfig:
    if not env.get("SLACK_WEBHOOK_URL"):
        raise DeliveryConfigError("missing secret(s): SLACK_WEBHOOK_URL")
    url = env["SLACK_WEBHOOK_URL"].strip()
    if WEBHOOK_RE.fullmatch(url) is None:
        raise DeliveryConfigError(
            "SLACK_WEBHOOK_URL must be a https://hooks.slack.com/services/... webhook URL"
        )
    return SlackConfig(url)


def slack_chunks(rendered: str) -> list[str]:
    return chunk_text(rendered, MAX_UNITS, _ITEM_START, "...")


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after", "")
    ok = raw.isascii() and raw.isdigit() and len(raw) <= 4  # int() rejects e.g. "\u00b2"
    return number(int(raw)) if ok else None


def _error_for(code: int) -> DeliveryError:
    if code == 404:
        return DeliveryError(
            "the Slack webhook no longer exists (HTTP 404): create a new one "
            "and update SLACK_WEBHOOK_URL"
        )
    if code in {401, 403}:
        return DeliveryError(f"Slack rejected the webhook (HTTP {code}): check SLACK_WEBHOOK_URL")
    if code == 410:
        return DeliveryError("the Slack channel was archived (HTTP 410): pick another channel")
    if code == 400:
        return DeliveryError("Slack refused the message (HTTP 400)")
    if code == 429:
        return DeliveryError("Slack is rate limiting this webhook (HTTP 429)")
    return DeliveryError(f"Slack error (HTTP {code})")


def _send_one(cfg: SlackConfig, text: str, client: httpx.Client, sleep: Sleep) -> None:
    payload: dict[str, object] = {"text": clean(text)}
    resp = post_with_retry(
        client, cfg.webhook_url, payload, service="Slack", retry_after=_retry_after, sleep=sleep
    )
    if resp.status_code != 200:
        raise _error_for(resp.status_code)
    if resp.text.strip() != "ok":
        raise DeliveryError("Slack answered with an unexpected response")


def send_slack(
    chunks: Sequence[str], cfg: SlackConfig, *, client: httpx.Client, sleep: Sleep = time.sleep
) -> int:
    """Send each chunk as one message. Returns how many were sent."""
    if WEBHOOK_RE.fullmatch(cfg.webhook_url) is None:
        raise DeliveryError("refusing to send: the URL is not a Slack webhook")
    for index, text in enumerate(chunks):
        if index:
            sleep(GAP_SECONDS)
        try:
            _send_one(cfg, text, client, sleep)
        except DeliveryError as exc:
            if index:
                raise DeliveryError(f"{exc} (the first {index} message(s) were sent)") from None
            raise
    return len(chunks)
