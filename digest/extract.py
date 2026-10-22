"""Text for the top items (spec 6.3 step 5): the feed's excerpt first (items/<id>.json, no
network beyond the feed), then, only if that is thin, the page itself through the SSRF guard
and robots.txt. Video and social hosts are never fetched (same rule as Plane 1, item 25)."""

from __future__ import annotations

import urllib.robotparser
from dataclasses import dataclass, field
from urllib.parse import urlsplit

import trafilatura

from .feed_client import FeedItem, FeedReader, read_excerpt
from .safe_http import USER_AGENT, FetchError, SafeFetcher

THIN_CHARS = 200
MAX_TEXT_CHARS = 20_000
NO_FETCH_HOSTS = (
    "youtube.com", "youtu.be", "x.com", "twitter.com", "vimeo.com",
    "facebook.com", "instagram.com", "tiktok.com", "linkedin.com", "reddit.com",
)  # fmt: skip


def _no_fetch(url: str) -> bool:
    host = (urlsplit(url).hostname or "").casefold()
    return any(host == h or host.endswith("." + h) for h in NO_FETCH_HOSTS)


class RobotsCache:
    def __init__(self, fetcher: SafeFetcher) -> None:
        self._fetcher = fetcher
        self._cache: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _load(self, origin: str) -> urllib.robotparser.RobotFileParser | None:
        parser = urllib.robotparser.RobotFileParser()
        try:
            body = self._fetcher.get(origin + "/robots.txt").body.decode("utf-8", "replace")
        except FetchError as exc:
            if exc.status in {404, 410}:  # no robots.txt: everything allowed
                parser.parse([])
                return parser
            return None  # unreachable or blocked: treat as disallow
        parser.parse(body.splitlines())
        return parser

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._cache:
            self._cache[origin] = self._load(origin)
        parser = self._cache[origin]
        return parser is not None and parser.can_fetch(USER_AGENT, url)


@dataclass
class ItemText:
    text: str
    source: str  # "excerpt" | "snippet" | "page"
    warnings: list[str] = field(default_factory=list)


class TextEnricher:
    def __init__(
        self,
        reader: FeedReader | None,
        fetcher: SafeFetcher | None,
        *,
        max_page_fetches: int = 10,
    ) -> None:
        self._reader = reader
        self._fetcher = fetcher
        self._robots = RobotsCache(fetcher) if fetcher is not None else None
        self._budget = max_page_fetches

    def text_for(self, item: FeedItem) -> ItemText:
        text, source = item.snippet, "snippet"
        excerpt = item.excerpt
        if excerpt is None and self._reader is not None and not item.from_extra_rss:
            excerpt = read_excerpt(self._reader, item.id)
        if excerpt:
            text, source = excerpt, "excerpt"
        result = ItemText(text, source)
        if len(text) >= THIN_CHARS or self._fetcher is None or self._robots is None:
            return result
        if _no_fetch(item.canonical_url) or self._budget <= 0:
            return result
        if not self._robots.allowed(item.canonical_url):
            result.warnings.append(f"{item.id}: page fetch skipped (robots.txt)")
            return result
        self._budget -= 1
        try:
            page = self._fetcher.get(item.canonical_url)
            html = page.body.decode("utf-8", "replace")
            extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
        except FetchError as exc:
            result.warnings.append(f"{item.id}: page fetch failed ({exc})")
            return result
        if extracted and len(extracted) > len(text):
            return ItemText(extracted[:MAX_TEXT_CHARS], "page", result.warnings)
        return result
