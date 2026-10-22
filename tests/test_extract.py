from __future__ import annotations

from collections.abc import Callable

import httpx

from digest.extract import NO_FETCH_HOSTS, TextEnricher
from digest.feed_client import FeedItem, LocalFeed
from digest.safe_http import SafeFetcher
from tests.conftest import FEED
from tests.test_rank import mk

Handler = Callable[[httpx.Request], httpx.Response]
PARA = "<p>This is a long enough paragraph about local inference engines and quantization.</p>"
ARTICLE = "<html><body><article><h1>Heading</h1>" + PARA * 12 + "</article></body></html>"


def fetcher(handler: Handler) -> tuple[SafeFetcher, list[str]]:
    seen: list[str] = []

    def record(req: httpx.Request) -> httpx.Response:
        seen.append(str(req.url))
        return handler(req)

    client = httpx.Client(transport=httpx.MockTransport(record))
    return SafeFetcher(client, resolver=lambda h: ["93.184.216.34"]), seen


def site(req: httpx.Request) -> httpx.Response:
    if req.url.path == "/robots.txt":
        return httpx.Response(404)
    return httpx.Response(200, text=ARTICLE, headers={"content-type": "text/html"})


def thin(**over: object) -> FeedItem:
    return mk("e" * 16, canonical_url="https://example.com/post", snippet="short", **over)


def test_feed_excerpt_is_preferred_and_needs_no_page_fetch() -> None:
    f, seen = fetcher(site)
    item = mk("6a23445afa1896da", snippet="short")
    result = TextEnricher(LocalFeed(FEED), f).text_for(item)
    assert result.source == "excerpt" and "Flow-1" in result.text and seen == []


def test_long_snippet_is_enough() -> None:
    f, seen = fetcher(site)
    result = TextEnricher(LocalFeed(FEED), f).text_for(mk(snippet="x" * 300))
    assert result.source == "snippet" and seen == []


def test_thin_item_fetches_the_page_after_robots() -> None:
    f, seen = fetcher(site)
    result = TextEnricher(None, f).text_for(thin())
    assert result.source == "page" and "local inference engines" in result.text
    assert seen[0] == "https://example.com/robots.txt"


def test_robots_disallow_skips_the_fetch() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /")
        return httpx.Response(200, text=ARTICLE)

    f, seen = fetcher(handler)
    result = TextEnricher(None, f).text_for(thin())
    assert result.source == "snippet" and any("robots.txt" in w for w in result.warnings)
    assert "https://example.com/post" not in seen


def test_unreachable_robots_counts_as_disallowed() -> None:
    f, _ = fetcher(lambda req: httpx.Response(500))
    result = TextEnricher(None, f).text_for(thin())
    assert result.source == "snippet" and result.warnings


def test_video_and_social_hosts_are_never_fetched() -> None:
    f, seen = fetcher(site)
    enricher = TextEnricher(None, f)
    for host in ("www.youtube.com", "x.com", "old.reddit.com"):
        enricher.text_for(mk("f" * 16, canonical_url=f"https://{host}/a", snippet="s"))
    assert seen == [] and "youtube.com" in NO_FETCH_HOSTS


def test_failed_fetch_keeps_the_snippet_and_budget_is_enforced() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(404) if req.url.path == "/robots.txt" else httpx.Response(500)

    f, _ = fetcher(handler)
    enricher = TextEnricher(None, f, max_page_fetches=1)
    first = enricher.text_for(thin())
    assert first.source == "snippet" and any("page fetch failed" in w for w in first.warnings)
    second = enricher.text_for(mk("a" * 16, canonical_url="https://example.com/other", snippet="s"))
    assert second.warnings == []  # budget exhausted: no second attempt
