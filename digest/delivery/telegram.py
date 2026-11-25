"""Telegram delivery (spec 6.3 step 9, Appendix C 88): the Bot API through ONE fixed host.

The bot token sits in the URL path, so it must never reach a log, an error, a state file or an
outbox file. Errors name the secret (TELEGRAM_BOT_TOKEN), never its value, and nothing from
the response is repeated. The token and chat id are validated before they are used, the whole
URL is matched against one anchored pattern right before sending, and redirects are not
followed."""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import httpx

from .common import (
    GAP_SECONDS,
    Sleep,
    chunk_text,
    clean,
    json_object,
    number,
    post_with_retry,
)
from .email import DeliveryConfigError, DeliveryError

API_HOST = "api.telegram.org"
_TOKEN = r"[0-9]{3,20}:[A-Za-z0-9_-]{20,100}"
TOKEN_RE = re.compile(_TOKEN)
URL_RE = re.compile(rf"https://api\.telegram\.org/bot{_TOKEN}/sendMessage")
CHAT_RE = re.compile(r"-?[0-9]{1,20}|@[A-Za-z][A-Za-z0-9_]{4,31}")
MAX_UNITS = 4000  # the API allows 4096 after entity parsing; ours still has its escapes
_ITEM_START = re.compile(r"(?m)^(?=\d+\\\. )")
_ESCAPED = re.compile(r"\\(.)", re.DOTALL)
_OVERSIZE = "\\.\\.\\."
_PREVIEW_OFF: dict[str, object] = {"is_disabled": True}


@dataclass(frozen=True)
class TelegramConfig:
    token: str = field(repr=False)
    chat_id: str = field(repr=False)


def telegram_config_from_env(env: Mapping[str, str]) -> TelegramConfig:
    missing = [n for n in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") if not env.get(n)]
    if missing:
        raise DeliveryConfigError("missing secret(s): " + ", ".join(missing))
    token, chat = env["TELEGRAM_BOT_TOKEN"].strip(), env["TELEGRAM_CHAT_ID"].strip()
    if TOKEN_RE.fullmatch(token) is None:
        raise DeliveryConfigError("TELEGRAM_BOT_TOKEN does not look like a bot token")
    if CHAT_RE.fullmatch(chat) is None:
        raise DeliveryConfigError("TELEGRAM_CHAT_ID must be a number or @channelname")
    return TelegramConfig(token, chat)


def telegram_chunks(rendered: str) -> list[str]:
    """Messages for one digest: split on item boundaries, each within the length limit."""
    return chunk_text(rendered, MAX_UNITS, _ITEM_START, _OVERSIZE)


def plain_text(markdown: str) -> str:
    """MarkdownV2 without its escapes: the fallback when Telegram cannot parse a message."""
    return _ESCAPED.sub(r"\1", markdown)


def _chat_id(cfg: TelegramConfig) -> int | str:
    return int(cfg.chat_id) if re.fullmatch(r"-?[0-9]+", cfg.chat_id) else cfg.chat_id


def _is_parse_error(resp: httpx.Response) -> bool:
    desc = json_object(resp).get("description")
    return isinstance(desc, str) and "parse entities" in desc.lower()


def _retry_after(resp: httpx.Response) -> float | None:
    params = json_object(resp).get("parameters")
    return number(params.get("retry_after")) if isinstance(params, dict) else None


def _error_for(code: int) -> DeliveryError:
    if code in {401, 404}:
        return DeliveryError(
            f"Telegram rejected TELEGRAM_BOT_TOKEN (HTTP {code}): check the secret"
        )
    if code == 403:
        return DeliveryError(
            "the bot may not post to this chat (HTTP 403): start the bot or add it "
            "to the chat, and check TELEGRAM_CHAT_ID"
        )
    if code == 400:
        return DeliveryError(
            "Telegram refused the message or chat (HTTP 400): check TELEGRAM_CHAT_ID"
        )
    if code == 429:
        return DeliveryError("Telegram is rate limiting this bot (HTTP 429)")
    return DeliveryError(f"Telegram error (HTTP {code})")


def _send_one(url: str, cfg: TelegramConfig, text: str, client: httpx.Client, sleep: Sleep) -> None:
    body = clean(text)
    payload: dict[str, object] = {
        "chat_id": _chat_id(cfg),
        "text": body,
        "parse_mode": "MarkdownV2",
        "link_preview_options": _PREVIEW_OFF,
    }

    def post() -> httpx.Response:
        return post_with_retry(
            client, url, payload, service="Telegram", retry_after=_retry_after, sleep=sleep
        )

    resp = post()
    if resp.status_code == 400 and _is_parse_error(resp):
        payload = {"chat_id": _chat_id(cfg), "text": plain_text(body),
                   "link_preview_options": _PREVIEW_OFF}  # fmt: skip
        resp = post()  # one plain-text try: a message that cannot be formatted still arrives
    if resp.status_code != 200:
        raise _error_for(resp.status_code)
    if json_object(resp).get("ok") is not True:
        raise DeliveryError("Telegram answered with an unexpected response")


def send_telegram(
    chunks: Sequence[str],
    cfg: TelegramConfig,
    *,
    client: httpx.Client,
    sleep: Sleep = time.sleep,
) -> int:
    """Send each chunk as one message. Returns how many were sent."""
    url = f"https://{API_HOST}/bot{cfg.token}/sendMessage"
    if URL_RE.fullmatch(url) is None:
        raise DeliveryError("refusing to send: the URL is not the Telegram API")
    for index, text in enumerate(chunks):
        if index:
            sleep(GAP_SECONDS)
        try:
            _send_one(url, cfg, text, client, sleep)
        except DeliveryError as exc:
            if index:
                raise DeliveryError(f"{exc} (the first {index} message(s) were sent)") from None
            raise
    return len(chunks)
