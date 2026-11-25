"""Shared pieces of the chat channels (Telegram, Slack): sizes, chunking, retries.

Nothing here knows a token or a webhook URL. Error messages are fixed strings plus the
service name; no response body or exception text is ever repeated, because the request URL
(which carries a secret) can appear in both."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

import httpx

from .email import DeliveryError

Sleep = Callable[[float], None]
RetryAfter = Callable[[httpx.Response], float | None]

MAX_ATTEMPTS = 3
MAX_WAIT = 30.0  # longest single wait, whatever the server asks for
GAP_SECONDS = 1.0  # pause between messages to one chat (both services: about one per second)


def units(text: str) -> int:
    """Length in UTF-16 code units: the conservative way chat services count characters."""
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def clean(text: str) -> str:
    """Replace what UTF-8 cannot carry (lone surrogates) so a JSON body can always be built."""
    return text.encode("utf-8", errors="replace").decode("utf-8")


def _fit(segment: str, limit: int, placeholder: str) -> list[str]:
    if units(segment) <= limit:
        return [segment]
    return [
        line if units(line) <= limit else placeholder + "\n"
        for line in segment.splitlines(keepends=True)
    ]


def chunk_text(text: str, limit: int, boundary: re.Pattern[str], placeholder: str) -> list[str]:
    """Split `text` on item boundaries (a zero-width `boundary` pattern), packing greedily.

    A segment over the limit falls back to its lines; a line over the limit is replaced by
    `placeholder`, never cut (a cut could break an escape or a link). Every chunk is
    stripped, non-empty and at most `limit` UTF-16 units."""
    chunks: list[str] = []
    current = ""
    for segment in (s for s in boundary.split(text) if s):
        for piece in _fit(segment, limit, placeholder):
            if current and units(current) + units(piece) > limit:
                chunks.append(current)
                current = piece
            else:
                current += piece
    if current:
        chunks.append(current)
    return [c for c in (chunk.strip() for chunk in chunks) if c]


def wait_for(hint: float | None, attempt: int) -> float:
    """Seconds before the next attempt: the server's hint (capped) or exponential backoff."""
    if hint is not None:
        return min(MAX_WAIT, max(0.0, hint))
    return min(8.0, 2.0 ** (attempt - 1))


def json_object(resp: httpx.Response) -> dict[str, object]:
    """The response body as a JSON object, or {} when it is not one. Never raises."""
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def number(value: object) -> float | None:
    """A real number from untrusted JSON (booleans are not numbers), else None."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return float(value)
    except OverflowError:  # a JSON integer too large for a float
        return None


def post_with_retry(
    client: httpx.Client,
    url: str,
    payload: dict[str, object],
    *,
    service: str,
    retry_after: RetryAfter,
    sleep: Sleep,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """POST JSON without following redirects; retry transport errors, 429 and 5xx.

    Returns the last response (any status) so the caller maps it to a message. Raises
    DeliveryError only when the service could not be reached at all."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.post(url, json=payload, headers=headers, follow_redirects=False)
        except httpx.HTTPError as exc:
            if attempt >= MAX_ATTEMPTS:
                raise DeliveryError(f"{service} unreachable ({type(exc).__name__})") from None
            sleep(wait_for(None, attempt))
            continue
        code = resp.status_code
        if (code == 429 or code >= 500) and attempt < MAX_ATTEMPTS:
            sleep(wait_for(retry_after(resp) if code == 429 else None, attempt))
            continue
        return resp
