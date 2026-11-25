"""Telegram delivery (P6.1): config checks, the fixed URL, retries, error mapping, redaction.

Everything runs against a scripted fake of the Bot API; nothing touches the network."""

from __future__ import annotations

import re

import httpx
import pytest

from digest.delivery.common import GAP_SECONDS, MAX_ATTEMPTS, MAX_WAIT, units
from digest.delivery.email import DeliveryConfigError, DeliveryError
from digest.delivery.telegram import (
    MAX_UNITS,
    TelegramConfig,
    plain_text,
    send_telegram,
    telegram_chunks,
    telegram_config_from_env,
)
from digest.render import build_view, render_telegram
from tests.chat_helpers import Script, tg_error, tg_ok
from tests.conftest import NOW, make_profile
from tests.test_render import entry, run_of

TOKEN = "123456789:AAFuniqueTokenXyZ_0123456789abcdefg"
CHAT = "-1001234567890"
ECHO = "UNIQUE-BODY-ECHO-777"
CFG = TelegramConfig(TOKEN, CHAT)
PARSE_ERROR = "Bad Request: can't parse entities: Character '.' is reserved"


def send(
    script: Script, chunks: tuple[str, ...] = ("hello",), cfg: TelegramConfig = CFG
) -> tuple[int, list[float]]:
    sleeps: list[float] = []
    sent = send_telegram(chunks, cfg, client=script.client, sleep=sleeps.append)
    return sent, sleeps


def server(*steps: httpx.Response | Exception) -> Script:
    return Script(tg_ok, *steps)


# ---- configuration ---------------------------------------------------------------------


def test_missing_secrets_are_named_without_values() -> None:
    with pytest.raises(DeliveryConfigError) as both:
        telegram_config_from_env({})
    assert "TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID" in str(both.value)
    with pytest.raises(DeliveryConfigError) as one:
        telegram_config_from_env({"TELEGRAM_BOT_TOKEN": TOKEN})
    message = str(one.value)
    assert "TELEGRAM_CHAT_ID" in message and "TELEGRAM_BOT_TOKEN" not in message
    assert TOKEN not in message


@pytest.mark.parametrize(
    "token",
    [
        "abc",
        "12:AAFuniqueTokenXyZ_0123456789abcdefg",
        "123456789:short",
        f"{TOKEN}/../x",
        f"{TOKEN}?x=1",
        f"{TOKEN}#frag",
        f"{TOKEN} x",
        f"{TOKEN}\nTELEGRAM_CHAT_ID",
        "123456789:AAFuniqueTokenXyZ_0123456789abcdefg@evil.example",
        "１２３４５６７８９:AAFuniqueTokenXyZ_0123456789abcdefg",
    ],
)
def test_a_malformed_token_is_refused_by_shape_without_echo(token: str) -> None:
    with pytest.raises(DeliveryConfigError, match="does not look like a bot token") as info:
        telegram_config_from_env({"TELEGRAM_BOT_TOKEN": token, "TELEGRAM_CHAT_ID": CHAT})
    assert "AAFunique" not in str(info.value) and CHAT not in str(info.value)


@pytest.mark.parametrize(
    "chat", ["abc", "12 34", "@ab", "1;2", "@1abcdef", "-", "--5", "１２", "12\n34", "9" * 21]
)
def test_a_malformed_chat_id_is_refused(chat: str) -> None:
    with pytest.raises(DeliveryConfigError, match="must be a number or @channelname"):
        telegram_config_from_env({"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": chat})


@pytest.mark.parametrize("chat", ["-1001234567890", "42", "@my_channel"])
def test_valid_chat_ids_pass(chat: str) -> None:
    cfg = telegram_config_from_env({"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": chat})
    assert cfg.chat_id == chat


def test_values_are_trimmed_and_the_repr_hides_them() -> None:
    cfg = telegram_config_from_env(
        {"TELEGRAM_BOT_TOKEN": f" {TOKEN}\n", "TELEGRAM_CHAT_ID": f"{CHAT} "}
    )
    assert cfg == CFG
    assert TOKEN not in repr(cfg) and CHAT not in repr(cfg)


# ---- the request -----------------------------------------------------------------------


def test_the_request_goes_to_the_fixed_host_with_the_expected_payload() -> None:
    script = server()
    sent, sleeps = send(script, ("*hi* \\- there",))
    assert sent == 1 and sleeps == []
    (req,) = script.requests
    assert req.method == "POST" and req.url.scheme == "https"
    assert req.url.host == "api.telegram.org" and req.url.port is None
    assert req.url.path == f"/bot{TOKEN}/sendMessage" and not req.url.query
    assert req.headers["content-type"] == "application/json"
    assert script.bodies() == [
        {
            "chat_id": -1001234567890,
            "text": "*hi* \\- there",
            "parse_mode": "MarkdownV2",
            "link_preview_options": {"is_disabled": True},
        }
    ]


def test_a_channel_name_stays_a_string_and_a_number_becomes_an_integer() -> None:
    named, plain = server(), server()
    send(named, cfg=TelegramConfig(TOKEN, "@my_channel"))
    send(plain, cfg=TelegramConfig(TOKEN, "42"))
    assert named.bodies()[0]["chat_id"] == "@my_channel"
    assert plain.bodies()[0]["chat_id"] == 42


def test_several_messages_go_out_in_order_with_a_pause_between() -> None:
    script = server()
    sent, sleeps = send(script, ("one", "two", "three"))
    assert sent == 3 and sleeps == [GAP_SECONDS, GAP_SECONDS]
    assert [b["text"] for b in script.bodies()] == ["one", "two", "three"]


@pytest.mark.parametrize(
    ("value", "wait"),
    [(7, 7.0), (9999, MAX_WAIT), (-5, 0.0), ("soon", 1.0), (True, 1.0), ([3], 1.0), (None, 1.0)],
)
def test_429_waits_as_told_but_never_more_than_the_cap(value: object, wait: float) -> None:
    script = server(tg_error(429, "Too Many Requests: retry after 7", retry_after=value))
    sent, sleeps = send(script)
    assert sent == 1 and sleeps == [wait] and len(script.requests) == 2


def test_a_429_without_parameters_backs_off() -> None:
    script = server(tg_error(429, "Too Many Requests"))
    assert send(script)[1] == [1.0]


def test_a_persistent_429_ends_in_a_rate_limit_error() -> None:
    script = server(*(tg_error(429, retry_after=1) for _ in range(5)))
    with pytest.raises(DeliveryError, match=r"rate limiting this bot \(HTTP 429\)"):
        send(script)
    assert len(script.requests) == MAX_ATTEMPTS


def test_a_server_error_is_retried_then_succeeds() -> None:
    script = server(tg_error(502, "Bad Gateway"))
    sent, sleeps = send(script)
    assert sent == 1 and sleeps == [1.0] and len(script.requests) == 2


def test_a_persistent_server_error_gives_up() -> None:
    script = server(*(tg_error(500) for _ in range(5)))
    with pytest.raises(DeliveryError, match=r"Telegram error \(HTTP 500\)"):
        send(script)
    assert len(script.requests) == MAX_ATTEMPTS


# ---- errors never echo anything --------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        (401, "rejected TELEGRAM_BOT_TOKEN"),
        (404, "rejected TELEGRAM_BOT_TOKEN"),
        (403, "may not post to this chat"),
        (400, "refused the message or chat"),
        (418, "Telegram error (HTTP 418)"),
        (302, "Telegram error (HTTP 302)"),
    ],
)
def test_a_status_maps_to_a_safe_message(code: int, fragment: str) -> None:
    body = {"ok": False, "description": f"{ECHO} {TOKEN} {CHAT}"}
    script = server(httpx.Response(code, json=body))
    with pytest.raises(DeliveryError) as info:
        send(script)
    message = str(info.value)
    assert fragment in message and len(script.requests) == 1
    for secret in (ECHO, TOKEN, CHAT, "api.telegram.org"):
        assert secret not in message


def test_a_redirect_is_not_followed() -> None:
    redirect = httpx.Response(302, headers={"location": "https://evil.example/steal"})
    script = server(redirect)
    with pytest.raises(DeliveryError, match=r"HTTP 302"):
        send(script)
    assert [r.url.host for r in script.requests] == ["api.telegram.org"]


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(200, json={"ok": False}),
        httpx.Response(200, text="<html>proxy page</html>"),
        httpx.Response(200, json=[1, 2]),
    ],
)
def test_an_unexpected_success_body_is_an_error(answer: httpx.Response) -> None:
    with pytest.raises(DeliveryError, match="unexpected response"):
        send(server(answer))


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_unreachable_names_the_error_class_only(error: type[httpx.TransportError]) -> None:
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    script = server(*(error(f"failed for {url}") for _ in range(MAX_ATTEMPTS)))
    with pytest.raises(DeliveryError) as info:
        send(script)
    assert str(info.value) == f"Telegram unreachable ({error.__name__})"
    assert TOKEN not in str(info.value) and info.value.__cause__ is None
    assert len(script.requests) == MAX_ATTEMPTS


def test_a_failure_after_some_messages_says_how_many_were_sent() -> None:
    script = server(tg_ok(), tg_error(403))
    with pytest.raises(DeliveryError) as info:
        send(script, ("one", "two", "three"))
    assert "the first 1 message(s) were sent" in str(info.value) and len(script.requests) == 2
    first = server(tg_error(403))
    with pytest.raises(DeliveryError) as info2:
        send(first, ("one", "two"))
    assert "were sent" not in str(info2.value)


# ---- markup that Telegram cannot parse -------------------------------------------------


def test_unparseable_markup_is_resent_once_as_plain_text() -> None:
    script = server(tg_error(400, PARSE_ERROR))
    sent, _ = send(script, ("a \\- b \\(c\\) [t](https://e.com/x)",))
    first, second = script.bodies()
    assert sent == 1 and first["parse_mode"] == "MarkdownV2" and "parse_mode" not in second
    assert second["text"] == "a - b (c) [t](https://e.com/x)"
    assert second["link_preview_options"] == {"is_disabled": True}
    assert second["chat_id"] == first["chat_id"]


def test_a_second_parse_failure_is_an_error_not_a_loop() -> None:
    script = server(tg_error(400, PARSE_ERROR), tg_error(400, PARSE_ERROR))
    with pytest.raises(DeliveryError, match="refused the message or chat"):
        send(script)
    assert len(script.requests) == 2


def test_other_400_errors_do_not_trigger_the_plain_text_retry() -> None:
    script = server(tg_error(400, "Bad Request: chat not found"))
    with pytest.raises(DeliveryError):
        send(script)
    assert len(script.requests) == 1


def test_plain_text_removes_escapes_only() -> None:
    assert plain_text("a\\.b\\(c\\)\\\\d \\- x") == "a.b(c)\\d - x"
    assert plain_text("no escapes *here*") == "no escapes *here*"


# ---- the URL is checked right before sending -------------------------------------------


@pytest.mark.parametrize(
    "token",
    [
        f"{TOKEN}/../../evil",
        f"{TOKEN}?x=1",
        f"{TOKEN}#frag",
        "x@evil.example/",
        f"{TOKEN} space",
        "１２３４５６７８９:AAFuniqueTokenXyZ_0123456789abcdefg",
    ],
)
def test_a_config_built_by_hand_cannot_escape_the_fixed_url(token: str) -> None:
    script = server()
    with pytest.raises(DeliveryError, match="refusing to send"):
        send_telegram(["x"], TelegramConfig(token, CHAT), client=script.client)
    assert script.requests == []


# ---- chunking a real digest ------------------------------------------------------------


def test_a_long_digest_is_split_on_item_boundaries_within_the_limit() -> None:
    entries = [entry(title=f"Item {i} " + "word " * 40) for i in range(1, 31)]
    text = render_telegram(build_view(run_of(*entries), make_profile(), NOW))
    chunks = telegram_chunks(text)
    assert len(chunks) >= 3 and all(units(c) <= MAX_UNITS for c in chunks)
    joined = "\n".join(chunks)
    for i in range(1, 31):
        assert len(re.findall(rf"(?m)^{i}\\\. ", joined)) == 1
    assert chunks[0].startswith("*Axon digest*")
    assert all(re.match(r"\d+\\\. ", c) for c in chunks[1:])


def test_a_short_digest_is_one_message() -> None:
    text = render_telegram(build_view(run_of(entry()), make_profile(), NOW))
    assert telegram_chunks(text) == [text.strip()]
