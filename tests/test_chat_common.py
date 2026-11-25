"""Shared chat helpers: UTF-16 sizes, chunking, wait rules and the retrying POST (offline)."""

from __future__ import annotations

import json
import re

import httpx
import pytest

from digest.delivery.common import (
    MAX_ATTEMPTS,
    MAX_WAIT,
    chunk_text,
    clean,
    json_object,
    number,
    post_with_retry,
    units,
    wait_for,
)
from digest.delivery.email import DeliveryError
from tests.chat_helpers import Script, slack_ok

BOUNDARY = re.compile(r"(?m)^(?=\d+\. )")
SECRET = "UNIQUE-URL-SECRET-4711"
URL = f"https://chat.example/{SECRET}/send"


def items(count: int, size: int) -> str:
    return "head\n" + "".join(f"\n{i}. " + "x" * size + "\n" for i in range(1, count + 1))


# ---- sizes and cleaning ----------------------------------------------------------------


def test_units_counts_utf16_code_units() -> None:
    assert units("abc") == 3 and units("\U0001f600") == 2 and units("é") == 1


def test_clean_replaces_lone_surrogates_and_keeps_text() -> None:
    assert clean("a\ud800b") == "a?b"
    assert clean("plain é \U0001f600") == "plain é \U0001f600"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5, 5.0), (2.5, 2.5), (True, None), (False, None), ("5", None), (None, None), ([1], None)],
)
def test_number_accepts_only_real_numbers(value: object, expected: float | None) -> None:
    assert number(value) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [(b'{"a": 1}', {"a": 1}), (b"[1]", {}), (b"not json", {}), (b"\xff\xfe\x00", {}), (b"", {})],
)
def test_json_object_never_raises(content: bytes, expected: dict[str, object]) -> None:
    assert json_object(httpx.Response(200, content=content)) == expected


@pytest.mark.parametrize(
    ("hint", "attempt", "expected"),
    [
        (7.0, 1, 7.0),
        (9999.0, 1, MAX_WAIT),
        (-5.0, 1, 0.0),
        (None, 1, 1.0),
        (None, 2, 2.0),
        (None, 3, 4.0),
        (None, 9, 8.0),
    ],
)
def test_wait_follows_the_hint_within_the_cap_else_backs_off(
    hint: float | None, attempt: int, expected: float
) -> None:
    assert wait_for(hint, attempt) == expected


# ---- chunking --------------------------------------------------------------------------


def test_chunks_respect_the_limit_and_keep_every_item_once() -> None:
    chunks = chunk_text(items(10, 40), 120, BOUNDARY, "...")
    assert len(chunks) > 1
    assert all(c and c == c.strip() and units(c) <= 120 for c in chunks)
    joined = "\n".join(chunks)
    for i in range(1, 11):
        assert len(re.findall(rf"(?m)^{i}\. ", joined)) == 1


def test_a_digest_that_fits_stays_one_message() -> None:
    assert len(chunk_text(items(4, 30), 10_000, BOUNDARY, "...")) == 1


def test_an_oversized_line_is_replaced_never_cut() -> None:
    text = "head\n\n1. " + "y" * 300 + "\n\n2. short\n"
    chunks = chunk_text(text, 100, BOUNDARY, "...")
    joined = "\n".join(chunks)
    assert "y" not in joined and "..." in joined and "2. short" in joined
    assert all(units(c) <= 100 for c in chunks)


def test_a_long_segment_falls_back_to_its_lines() -> None:
    body = "1. title\n" + "\n".join(f"line {i}" for i in range(30)) + "\n"
    chunks = chunk_text(body, 60, BOUNDARY, "...")
    assert len(chunks) > 1 and all(units(c) <= 60 for c in chunks)
    assert "".join(chunks).count("line") == 30


def test_the_limit_counts_utf16_units_not_code_points() -> None:
    emoji = "\U0001f600" * 10  # 10 code points, 20 UTF-16 units
    chunks = chunk_text(f"1. {emoji}\n\n2. {emoji}\n", 30, BOUNDARY, "...")
    assert len(chunks) == 2 and all(units(c) <= 30 for c in chunks)


@pytest.mark.parametrize("text", ["", "   \n\n  "])
def test_blank_text_makes_no_chunks(text: str) -> None:
    assert chunk_text(text, 100, BOUNDARY, "...") == []


# ---- post_with_retry -------------------------------------------------------------------


def call(script: Script, sleeps: list[float], hint: float | None = None) -> httpx.Response:
    return post_with_retry(
        script.client,
        URL,
        {"text": "t"},
        service="Svc",
        retry_after=lambda _resp: hint,
        sleep=sleeps.append,
    )


def test_a_success_is_returned_with_the_json_payload() -> None:
    script = Script(slack_ok)
    sleeps: list[float] = []
    assert call(script, sleeps).status_code == 200
    (req,) = script.requests
    assert json.loads(req.content) == {"text": "t"} and sleeps == []
    assert req.headers["content-type"] == "application/json"


def test_5xx_is_retried_with_backoff_until_it_succeeds() -> None:
    script = Script(slack_ok, httpx.Response(500), httpx.Response(503))
    sleeps: list[float] = []
    assert call(script, sleeps, hint=9.0).status_code == 200
    assert len(script.requests) == 3 and sleeps == [1.0, 2.0]  # the hint is for 429 only


def test_429_waits_for_the_hint() -> None:
    script = Script(slack_ok, httpx.Response(429))
    sleeps: list[float] = []
    assert call(script, sleeps, hint=7.0).status_code == 200
    assert sleeps == [7.0] and len(script.requests) == 2


def test_attempts_are_capped_and_the_last_response_is_returned() -> None:
    script = Script(slack_ok, *(httpx.Response(500) for _ in range(5)))
    sleeps: list[float] = []
    assert call(script, sleeps).status_code == 500
    assert len(script.requests) == MAX_ATTEMPTS and sleeps == [1.0, 2.0]


def test_client_errors_are_not_retried() -> None:
    script = Script(slack_ok, httpx.Response(400))
    sleeps: list[float] = []
    assert call(script, sleeps).status_code == 400
    assert len(script.requests) == 1 and sleeps == []


def test_a_transport_error_then_success_is_retried() -> None:
    script = Script(slack_ok, httpx.ConnectError("down"))
    sleeps: list[float] = []
    assert call(script, sleeps).status_code == 200 and sleeps == [1.0]


def test_unreachable_names_the_error_class_only_and_drops_the_chain() -> None:
    errors = (httpx.ConnectError(f"cannot connect to {URL}") for _ in range(MAX_ATTEMPTS))
    script = Script(slack_ok, *errors)
    sleeps: list[float] = []
    with pytest.raises(DeliveryError) as info:
        call(script, sleeps)
    assert str(info.value) == "Svc unreachable (ConnectError)"
    assert SECRET not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__
    assert len(script.requests) == MAX_ATTEMPTS and sleeps == [1.0, 2.0]


def test_redirects_are_not_followed() -> None:
    redirect = httpx.Response(302, headers={"location": "https://evil.example/steal"})
    script = Script(slack_ok, redirect)
    sleeps: list[float] = []
    assert call(script, sleeps).status_code == 302
    assert len(script.requests) == 1  # a followed redirect would have made a second request
