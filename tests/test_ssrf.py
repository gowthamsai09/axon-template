"""SSRF suite for the template's fetching guard (spec 10.1, Appendix C 84).

The shared URL corpus comes from the hub (HUB_DIR, default ../Axon); the behaviour tests below it
always run, with no network."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from digest.safe_http import FetchError, SafeFetcher, UnsafeUrlError, check_public_url

ROOT = Path(__file__).resolve().parent.parent
HUB = Path(os.environ.get("HUB_DIR", str(ROOT.parent / "Axon")))
CORPUS = HUB / "corpus" / "ssrf" / "urls.json"
REQUIRED = os.environ.get("REQUIRE_HUB_CORPUS") == "1"
CASES: list[dict[str, Any]] = (
    json.loads(CORPUS.read_text(encoding="utf-8")) if CORPUS.is_file() else []
)
needs_corpus = pytest.mark.skipif(
    not CASES and not REQUIRED, reason="hub corpus not found (set HUB_DIR)"
)
PUBLIC = "93.184.216.34"


def resolver_for(dns: dict[str, list[str]]) -> Callable[[str], list[str]]:
    return lambda host: list(dns.get(host.rstrip(".").lower(), []))


def never_resolve(host: str) -> list[str]:
    raise AssertionError(f"the resolver was consulted for {host!r}")


@needs_corpus
def test_the_corpus_was_really_loaded() -> None:
    assert len(CASES) >= 80
    assert {c["expect"] for c in CASES} == {"allowed", "blocked"}


@needs_corpus
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_the_guard_follows_the_corpus(case: dict[str, Any]) -> None:
    resolve = resolver_for(case["dns"])
    if case["expect"] == "allowed":
        check_public_url(case["url"], resolve)
    else:
        with pytest.raises(UnsafeUrlError):
            check_public_url(case["url"], resolve)


# ---- refusals that must not depend on DNS -------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://2130706433/", "http://0x7f000001/", "http://0x7f.0.0.1/", "http://017700000001/",
        "http://127.1/", "http://0/", "http://１２７.０.０.１/", "http://127.0.0.1./",
    ],
)  # fmt: skip
def test_numeric_host_forms_are_refused_without_consulting_dns(url: str) -> None:
    with pytest.raises(UnsafeUrlError, match="non-canonical IP address form"):
        check_public_url(url, never_resolve)


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://user:pw@example.com/", "credentials"), ("http://:@example.com/", "credentials"),
        ("http://example.com@127.0.0.1/", "credentials"),
        ("http://example.com:99999/", "invalid port"),
        ("http://example.com:abc/", "invalid port"), ("http://exa mple.com/", "invalid host name"),
        ("http://%31%32%37.0.0.1/", "invalid host name"), ("http://[::1/", "not a valid URL"),
        ("ftp://example.com/", "only http"), ("", "only http"),
    ],
)  # fmt: skip
def test_refusals_name_the_reason_and_never_reach_the_resolver(url: str, message: str) -> None:
    with pytest.raises(UnsafeUrlError, match=message):
        check_public_url(url, never_resolve)


def test_a_name_that_does_not_resolve_is_refused() -> None:
    with pytest.raises(UnsafeUrlError, match="did not resolve"):
        check_public_url("https://nxdomain.example/", resolver_for({}))


# ---- redirects, rebinding, size and time --------------------------------------------------


class Web:
    """A tiny scripted web: `redirects` maps a URL to its Location; anything else answers 200."""

    def __init__(self, redirects: dict[str, str] | None = None) -> None:
        self.redirects = redirects or {}
        self.seen: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(str(request.url))
        target = self.redirects.get(str(request.url))
        if target is not None:
            return httpx.Response(302, headers={"location": target})
        return httpx.Response(200, text="final", headers={"content-type": "text/plain"})

    def fetcher(self, dns: dict[str, list[str]] | None = None, **kw: Any) -> SafeFetcher:
        known = dns or {}
        client = httpx.Client(transport=httpx.MockTransport(self))
        return SafeFetcher(client, resolver=lambda host: known.get(host, [PUBLIC]), **kw)


def test_a_public_redirect_chain_is_followed_and_relative_locations_resolve() -> None:
    web = Web({"http://a.example/1": "http://b.example/2", "http://b.example/2": "/final"})
    result = web.fetcher().get("http://a.example/1")
    assert result.body == b"final" and result.url == "http://b.example/final"
    assert web.seen == ["http://a.example/1", "http://b.example/2", "http://b.example/final"]


def test_a_chain_that_ends_on_a_private_host_is_stopped_before_that_request() -> None:
    web = Web(
        {"http://a.example/1": "http://b.example/2", "http://b.example/2": "http://c.example/3"}
    )
    with pytest.raises(FetchError, match="blocked"):
        web.fetcher({"c.example": ["10.0.0.5"]}).get("http://a.example/1")
    assert web.seen == ["http://a.example/1", "http://b.example/2"]


@pytest.mark.parametrize(
    "target",
    [
        "ftp://example.com/x", "file:///etc/passwd", "javascript:alert(1)", "gopher://example.com/",
        "http://user:pw@example.com/", "//evil.example@127.0.0.1/", "http://2130706433/",
        "http://0x7f.1/", "http://[::ffff:127.0.0.1]/", "http://example.com:99999/",
        "http://169.254.169.254/latest/meta-data", "http://[::1/",
    ],
)  # fmt: skip
def test_a_redirect_to_anything_unsafe_is_refused_and_never_requested(target: str) -> None:
    web = Web({"http://a.example/start": target})
    with pytest.raises(FetchError):
        web.fetcher().get("http://a.example/start")
    assert web.seen == ["http://a.example/start"]


def test_every_hop_is_resolved_again_so_a_rebinding_answer_is_caught() -> None:
    answers: dict[str, int] = {}

    def rebinding(host: str) -> list[str]:
        answers[host] = answers.get(host, 0) + 1
        return [PUBLIC] if answers[host] == 1 else ["10.0.0.5"]

    web = Web({"http://a.example/1": "http://a.example/2"})
    fetcher = SafeFetcher(httpx.Client(transport=httpx.MockTransport(web)), resolver=rebinding)
    with pytest.raises(FetchError, match="blocked"):
        fetcher.get("http://a.example/1")
    assert answers == {"a.example": 2} and web.seen == ["http://a.example/1"]


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_a_response_that_trickles_in_is_cut_off_by_the_total_deadline() -> None:
    clock = Clock()

    def trickle() -> Iterator[bytes]:
        for _ in range(10):
            clock.now += 10.0
            yield b"x"

    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=trickle()))
    )
    fetcher = SafeFetcher(client, resolver=lambda h: [PUBLIC], max_seconds=30.0, clock=clock)
    with pytest.raises(FetchError, match="too slow"):
        fetcher.get("http://a.example/slow")


def test_a_fast_response_is_not_cut_off() -> None:
    clock = Clock()
    client = httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"ok"))
    )
    fetcher = SafeFetcher(client, resolver=lambda h: [PUBLIC], max_seconds=30.0, clock=clock)
    assert fetcher.get("http://a.example/fast").body == b"ok"


def test_a_long_redirect_chain_is_cut_off_by_the_total_deadline() -> None:
    clock = Clock()
    redirects = {f"http://h{i}.example/": f"http://h{i + 1}.example/" for i in range(5)}

    def handler(request: httpx.Request) -> httpx.Response:
        clock.now += 20.0
        target = redirects.get(str(request.url))
        return httpx.Response(302, headers={"location": target}) if target else httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = SafeFetcher(client, resolver=lambda h: [PUBLIC], max_seconds=30.0, clock=clock)
    with pytest.raises(FetchError, match="too slow"):
        fetcher.get("http://h0.example/")


def test_a_url_the_http_library_rejects_becomes_a_fetch_error_not_a_crash() -> None:
    class Broken(httpx.Client):
        def stream(self, *args: Any, **kwargs: Any) -> Any:
            raise httpx.InvalidURL("bad")

    fetcher = SafeFetcher(Broken(), resolver=lambda h: [PUBLIC])
    with pytest.raises(FetchError, match="InvalidURL"):
        fetcher.get("http://a.example/x")


def test_an_invalid_port_never_reaches_the_network() -> None:
    web = Web()
    with pytest.raises(FetchError, match="blocked: invalid port"):
        web.fetcher().get("http://example.com:99999/")
    assert web.seen == []
