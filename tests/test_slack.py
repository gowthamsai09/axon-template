"""Slack delivery (P6.2): webhook shape, payload, errors, mrkdwn escaping of hostile text.

Offline: a scripted fake of the incoming webhook; nothing touches the network."""

from __future__ import annotations

import re

import httpx
import pytest

from digest.delivery.common import GAP_SECONDS, MAX_ATTEMPTS, MAX_WAIT, units
from digest.delivery.email import DeliveryConfigError, DeliveryError
from digest.delivery.slack import (
    MAX_UNITS,
    SlackConfig,
    send_slack,
    slack_chunks,
    slack_config_from_env,
)
from digest.pipeline import DigestEntry
from digest.render import build_view, http_url, render_slack
from tests.chat_helpers import Script, slack_ok
from tests.conftest import NOW, make_profile
from tests.test_rank import mk
from tests.test_render import entry, run_of

SECRET = "uNiQuEwEbHoOkSeCrEt123456"
HOOK = f"https://hooks.slack.com/services/T0AAAAAAA/B0BBBBBBB/{SECRET}"
CFG = SlackConfig(HOOK)
ECHO = "UNIQUE-BODY-ECHO-777"


def send(script: Script, chunks: tuple[str, ...] = ("hello",)) -> tuple[int, list[float]]:
    sleeps: list[float] = []
    return send_slack(chunks, CFG, client=script.client, sleep=sleeps.append), sleeps


def server(*steps: httpx.Response | Exception) -> Script:
    return Script(slack_ok, *steps)


def view_text(*entries: DigestEntry) -> str:
    return render_slack(build_view(run_of(*entries), make_profile(), NOW))


# ---- configuration ---------------------------------------------------------------------


def test_a_missing_webhook_is_named_without_a_value() -> None:
    with pytest.raises(DeliveryConfigError) as info:
        slack_config_from_env({})
    assert "missing secret(s): SLACK_WEBHOOK_URL" in str(info.value)


@pytest.mark.parametrize(
    "url",
    [
        HOOK.replace("https://", "http://"),
        HOOK.replace("hooks.slack.com", "hooks.slack.com.evil.example"),
        HOOK.replace("hooks.slack.com", "evil.example"),
        HOOK.replace("hooks.slack.com", "hooks.slack.com:444"),
        HOOK.replace("https://", "https://hooks.slack.com@evil.example/"),
        HOOK.replace("hooks.slack.com", "hooks.slack-gov.com"),
        f"{HOOK}?x=1",
        f"{HOOK}#frag",
        f"{HOOK}/extra",
        f"{HOOK}\nhttps://evil.example/steal",
        f"{HOOK} extra",
        HOOK.replace(SECRET, "short"),
        HOOK.replace("services", "other"),
        "hooks.slack.com/services/T0AAAAAAA/B0BBBBBBB/" + SECRET,
        "not a url",
    ],
)
def test_a_url_that_is_not_a_slack_webhook_is_refused_without_echo(url: str) -> None:
    with pytest.raises(DeliveryConfigError, match=r"hooks\.slack\.com/services") as info:
        slack_config_from_env({"SLACK_WEBHOOK_URL": url})
    assert SECRET not in str(info.value) and "evil.example" not in str(info.value)


def test_the_webhook_is_trimmed_and_the_repr_hides_it() -> None:
    cfg = slack_config_from_env({"SLACK_WEBHOOK_URL": f"  {HOOK}\n"})
    assert cfg == CFG and HOOK not in repr(cfg) and SECRET not in repr(cfg)


# ---- the request -----------------------------------------------------------------------


def test_the_payload_has_only_a_text_field() -> None:
    script = server()
    sent, sleeps = send(script, ("*hi* there",))
    assert sent == 1 and sleeps == []
    (req,) = script.requests
    assert req.method == "POST" and req.url.host == "hooks.slack.com"
    assert str(req.url) == HOOK and req.headers["content-type"] == "application/json"
    assert script.bodies() == [{"text": "*hi* there"}]  # no other field is documented


def test_several_messages_go_out_in_order_with_a_pause_between() -> None:
    script = server()
    sent, sleeps = send(script, ("one", "two", "three"))
    assert sent == 3 and sleeps == [GAP_SECONDS, GAP_SECONDS]
    assert [b["text"] for b in script.bodies()] == ["one", "two", "three"]


@pytest.mark.parametrize(
    ("header", "wait"),
    [
        ("7", 7.0),
        ("9999", MAX_WAIT),
        ("0", 0.0),
        ("soon", 1.0),
        ("-5", 1.0),
        ("12345", 1.0),
        ("", 1.0),
    ],
)
def test_429_honours_retry_after_within_the_cap(header: str, wait: float) -> None:
    script = server(httpx.Response(429, headers={"retry-after": header}, text="rate_limited"))
    sent, sleeps = send(script)
    assert sent == 1 and sleeps == [wait] and len(script.requests) == 2


def test_a_non_ascii_digit_in_retry_after_is_ignored_not_a_crash() -> None:
    answer = httpx.Response(429, headers=[(b"retry-after", b"\xb2")], text="rate_limited")
    sent, sleeps = send(server(answer))
    assert sent == 1 and sleeps == [1.0]


def test_a_persistent_429_ends_in_a_rate_limit_error() -> None:
    script = server(*(httpx.Response(429, text="rate_limited") for _ in range(5)))
    with pytest.raises(DeliveryError, match=r"rate limiting this webhook \(HTTP 429\)"):
        send(script)
    assert len(script.requests) == MAX_ATTEMPTS


def test_a_server_error_is_retried_then_succeeds() -> None:
    sent, sleeps = send(server(httpx.Response(500, text="oops")))
    assert sent == 1 and sleeps == [1.0]


# ---- errors never echo anything --------------------------------------------------------


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        (404, "no longer exists"),
        (403, "rejected the webhook"),
        (401, "rejected the webhook"),
        (410, "channel was archived"),
        (400, "refused the message"),
        (418, "Slack error (HTTP 418)"),
        (302, "Slack error (HTTP 302)"),
    ],
)
def test_a_status_maps_to_a_safe_message(code: int, fragment: str) -> None:
    script = server(httpx.Response(code, text=f"{ECHO} {HOOK}"))
    with pytest.raises(DeliveryError) as info:
        send(script)
    message = str(info.value)
    assert fragment in message and len(script.requests) == 1
    for secret in (ECHO, HOOK, SECRET, "hooks.slack.com"):
        assert secret not in message


def test_a_redirect_is_not_followed() -> None:
    redirect = httpx.Response(302, headers={"location": "https://evil.example/steal"})
    script = server(redirect)
    with pytest.raises(DeliveryError, match="HTTP 302"):
        send(script)
    assert [r.url.host for r in script.requests] == ["hooks.slack.com"]


@pytest.mark.parametrize("body", ["no_text", "invalid_payload", "", "<html>proxy</html>"])
def test_a_200_that_is_not_ok_is_an_error(body: str) -> None:
    with pytest.raises(DeliveryError, match="unexpected response"):
        send(server(httpx.Response(200, text=body)))


@pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ReadTimeout])
def test_unreachable_names_the_error_class_only(error: type[httpx.TransportError]) -> None:
    script = server(*(error(f"failed for {HOOK}") for _ in range(MAX_ATTEMPTS)))
    with pytest.raises(DeliveryError) as info:
        send(script)
    assert str(info.value) == f"Slack unreachable ({error.__name__})"
    assert SECRET not in str(info.value) and info.value.__cause__ is None


def test_a_failure_after_some_messages_says_how_many_were_sent() -> None:
    script = server(httpx.Response(200, text="ok"), httpx.Response(404, text="no_service"))
    with pytest.raises(DeliveryError) as info:
        send(script, ("one", "two", "three"))
    assert "the first 1 message(s) were sent" in str(info.value)


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/services/T0AAAAAAA/B0BBBBBBB/" + SECRET,
        HOOK + "?x=1",
        "http://hooks.slack.com/services/T0AAAAAAA/B0BBBBBBB/" + SECRET,
        "https://hooks.slack.com/other",
    ],
)
def test_a_config_built_by_hand_cannot_point_anywhere_else(url: str) -> None:
    script = server()
    with pytest.raises(DeliveryError, match="refusing to send"):
        send_slack(["x"], SlackConfig(url), client=script.client)
    assert script.requests == []


# ---- rendering hostile text ------------------------------------------------------------


HOSTILE = "<!channel> <@U123> <https://evil.example|click> *b* ~s~ `c` a|b & <!here>"


def links(text: str) -> list[str]:
    return re.findall(r"<[^>\n]*>", text)


def test_hostile_text_cannot_form_mentions_or_links() -> None:
    out = view_text(entry(title=HOSTILE))
    found = links(out)
    assert len(found) == 1 and out.count("<") == out.count(">") == 1
    assert found[0].startswith("<https://example.com/x|")
    for forbidden in ("<!channel>", "<!here>", "<@U123>", "<https://evil.example"):
        assert forbidden not in out
    assert "&lt;!channel&gt;" in out and "&amp;" in out
    assert "*b*" not in out and "~s~" not in out and "`c`" not in out


def test_a_pipe_in_a_title_cannot_end_the_link_label() -> None:
    out = view_text(entry(title="Foo | Bar"))
    (link,) = links(out)
    assert link.count("|") == 1 and "Foo ¦ Bar" in link


def test_newlines_cannot_start_a_quote_or_a_new_item() -> None:
    e = entry(
        summary=None,
        fallback_snippet="line\n> quote <!here>\n2. fake item",
        note="n\n> q",
        title="a\n> b\r\n@here",
    )
    out = view_text(e)
    assert not any(line.startswith(">") for line in out.splitlines())
    assert "&lt;!here&gt;" in out and "<!here>" not in out
    assert out.count("\n2. ") == 0 and "a &gt; b @here" in out


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://example.com/a|b>c<d&e", "<https://example.com/a%7Cb%3Ec%3Cd&amp;e|"),
        ("https://example.com/a'b", "<https://example.com/a'b|"),
    ],
)
def test_link_targets_are_encoded(url: str, expected: str) -> None:
    e = entry()
    e.item = mk(canonical_url=url)
    assert expected in view_text(e)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/a b",
        "https://example.com/a\nb",
        "https://example.com/a\tb",
        "https://example.com/a\x00b",
        "https://exam ple.com/",
        "javascript:alert(1)",
    ],
)
def test_an_unsafe_url_is_dropped_so_only_the_title_remains(url: str) -> None:
    assert http_url(url) == ""
    e = entry()
    e.item = mk(canonical_url=url, title="A <b>title")
    out = view_text(e)
    assert "<" not in out and "1. A &lt;b&gt;title" in out


def test_an_item_without_a_summary_shows_its_snippet_and_note() -> None:
    e = entry(summary=None, fallback_snippet="Just a snippet.", note="extractive")
    out = view_text(e)
    assert "Just a snippet." in out and "_extractive_" in out
    assert "No AI summaries in this digest." in out


def test_the_layout_of_one_ai_item() -> None:
    out = view_text(entry())
    lines = out.splitlines()
    assert lines[0] == "*Axon digest* - Tue Oct 20, 2026"
    assert ":warning: a notice" in out
    assert "1. <https://example.com/x|A &lt;b&gt;title&lt;/b&gt; &amp; more>" in out
    assert "Short one_line." in out and "• Point &lt;1&gt;" in out and "• Point 2" in out
    assert "AI-generated by deepseek/m" in out


# ---- chunking --------------------------------------------------------------------------


def test_a_long_digest_is_split_on_item_boundaries_within_the_limit() -> None:
    entries = [entry(title=f"Item {i} " + "word " * 40) for i in range(1, 41)]
    chunks = slack_chunks(view_text(*entries))
    assert len(chunks) >= 3 and all(units(c) <= MAX_UNITS for c in chunks)
    joined = "\n".join(chunks)
    for i in range(1, 41):
        assert len(re.findall(rf"(?m)^{i}\. ", joined)) == 1
    assert chunks[0].startswith("*Axon digest*") and all(re.match(r"\d+\. ", c) for c in chunks[1:])


def test_an_item_too_long_for_one_message_is_replaced_never_cut() -> None:
    e = entry()
    e.item = mk(canonical_url="https://example.com/" + "|" * 2000)  # 6000 characters encoded
    chunks = slack_chunks(view_text(e, entry(title="Second")))
    joined = "\n".join(chunks)
    assert "%7C" not in joined and "..." in joined and "Second" in joined
    assert all(units(c) <= MAX_UNITS for c in chunks)
