"""Read the shared feed (spec 5.4, 6.3 steps 2 and 5) from FEED_BASE (https) or a local folder.

The template does NOT copy the feed-item schema. `FeedItem` is a tolerant consumer view: it
reads only the fields the pipeline uses and ignores everything else, so Plane 1 can add fields
without breaking Plane 2."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import struct_time
from typing import Any, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import httpx
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from .profile import TOPIC_SLUGS, Interests
from .safe_http import FetchError, SafeFetcher

SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
ITEM_ID = re.compile(r"^[a-f0-9]{16}$")
MAX_SHARD_BYTES = 5_000_000
EXTRA_RSS_SOURCE = "extra-rss"
_TRACKING = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid|ref|ref_src)$", re.I)
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


class FeedError(Exception):
    """The shared feed could not be read."""


class Signals(BaseModel):
    model_config = ConfigDict(extra="ignore")
    stars: int = 0
    points: int = 0
    comments: int = 0


class FeedItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(pattern=r"^[a-f0-9]{16}$")
    url: str = Field(min_length=8, max_length=2048)
    canonical_url: str = Field(min_length=8, max_length=2048)
    title: str = Field(min_length=1, max_length=500)
    source_id: str = Field(min_length=1, max_length=64)
    published_at: AwareDatetime
    snippet: str = ""
    excerpt: str | None = None
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    signals: Signals = Field(default_factory=Signals)
    base_score: float = Field(0.0, ge=0, le=1)
    from_extra_rss: bool = False  # set by this client only; never read from the feed


class FeedReader(Protocol):
    def read_bytes(self, rel: str) -> bytes | None:
        """Bytes of a feed-relative path, or None if it does not exist."""


def _safe_rel(rel: str) -> str:
    parts = rel.split("/")
    if len(parts) > 2 or not all(parts) or any(p in {".", ".."} for p in parts):
        raise FeedError(f"bad feed path {rel!r}")
    return rel


class LocalFeed:
    def __init__(self, root: Path) -> None:
        self._root = root

    def read_bytes(self, rel: str) -> bytes | None:
        path = self._root / _safe_rel(rel)
        if not path.is_file():
            return None
        if path.stat().st_size > MAX_SHARD_BYTES:
            raise FeedError(f"{rel}: file too large")
        return path.read_bytes()


class HttpFeed:
    def __init__(self, base: str, client: httpx.Client) -> None:
        self._base = base.rstrip("/")
        self._client = client

    def read_bytes(self, rel: str) -> bytes | None:
        url = f"{self._base}/{_safe_rel(rel)}"
        try:
            resp = self._client.get(url, follow_redirects=True)
        except httpx.HTTPError as exc:
            raise FeedError(f"{rel}: network error ({type(exc).__name__})") from exc
        if resp.status_code in {404, 410}:
            return None
        if resp.status_code != 200:
            raise FeedError(f"{rel}: HTTP {resp.status_code}")
        if len(resp.content) > MAX_SHARD_BYTES:
            raise FeedError(f"{rel}: response too large")
        return resp.content


def make_reader(source: str, client: httpx.Client) -> FeedReader:
    if source.startswith(("http://", "https://")):
        return HttpFeed(source, client)
    root = Path(source)
    if not root.is_dir():
        raise FeedError(f"feed folder not found: {source}")
    return LocalFeed(root)


def entity_slug(name: str) -> str | None:
    """Shard file name for an entity ('llama.cpp' -> 'llama-cpp'); repos have no shard."""
    if name.startswith("github:"):
        return None
    slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
    return slug if SLUG.match(slug) else None


@dataclass
class FeedSnapshot:
    items: list[FeedItem] = field(default_factory=list)
    generated_at: datetime | None = None
    warnings: list[str] = field(default_factory=list)


def _parse_shard(raw: bytes, label: str, warnings: list[str]) -> list[FeedItem]:
    try:
        data = json.loads(raw)
    except ValueError:
        warnings.append(f"{label}: not valid JSON, skipped")
        return []
    entries = data.get("items") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        warnings.append(f"{label}: no items list, skipped")
        return []
    items: list[FeedItem] = []
    bad = 0
    for entry in entries:
        try:
            items.append(FeedItem.model_validate(entry))
        except ValidationError:
            bad += 1
    if bad:
        warnings.append(f"{label}: skipped {bad} item(s) that did not validate")
    return items


def load_feed(reader: FeedReader, interests: Interests) -> FeedSnapshot:
    """Shards for the user's topics and entities, plus latest.json (keyword matching)."""
    snap = FeedSnapshot()
    paths = [f"tags/{t}.json" for t in interests.topics]
    paths += [f"entities/{s}.json" for e in interests.follow_entities if (s := entity_slug(e))]
    paths.append("latest.json")
    meta = reader.read_bytes("meta.json")
    if meta is not None:
        try:
            raw = json.loads(meta).get("generated_at")
            snap.generated_at = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            snap.warnings.append("meta.json: unreadable generated_at")
    seen: set[str] = set()
    read_any = False
    for rel in paths:
        raw_shard = reader.read_bytes(rel)
        if raw_shard is None:
            continue  # no shard for this topic/entity yet: normal
        read_any = True
        for item in _parse_shard(raw_shard, rel, snap.warnings):
            if item.id not in seen:
                seen.add(item.id)
                snap.items.append(item)
    if not read_any:
        raise FeedError("no feed files found (is FEED_BASE correct and the feed published?)")
    return snap


WEEK_DAYS = 7


def load_week(reader: FeedReader, now: datetime, *, days: int = WEEK_DAYS) -> FeedSnapshot:
    """Every topic shard plus latest.json, keeping items from the last `days` days (trends).

    The shared feed keeps a 168-hour window in each tag shard, so no feed change is needed.
    A missing shard is normal; if none can be read at all the caller gets a FeedError."""
    snap = FeedSnapshot()
    cutoff, horizon = now - timedelta(days=days), now + timedelta(days=1)
    seen: set[str] = set()
    read_any = False
    for rel in [*(f"tags/{t}.json" for t in TOPIC_SLUGS), "latest.json"]:
        raw = reader.read_bytes(rel)
        if raw is None:
            continue
        read_any = True
        for item in _parse_shard(raw, rel, snap.warnings):
            if item.id in seen or not cutoff <= item.published_at <= horizon:
                continue
            seen.add(item.id)
            snap.items.append(item)
    if not read_any:
        raise FeedError("no feed files found for the weekly trends")
    return snap


def read_excerpt(reader: FeedReader, item_id: str) -> str | None:
    """`excerpt` from /feed/items/<id>.json (shards carry only the snippet)."""
    if not ITEM_ID.match(item_id):
        return None
    raw = reader.read_bytes(f"items/{item_id}.json")
    if raw is None:
        return None
    try:
        value = json.loads(raw).get("excerpt")
    except (ValueError, AttributeError):
        return None
    return value if isinstance(value, str) and value.strip() else None


# ---- extra_rss (spec 6.1 sources.extra_rss) -------------------------------------------


def canonical(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k)])
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", query, ""))


def _entry_time(entry: Any, now: datetime) -> datetime:
    for key in ("published_parsed", "updated_parsed"):
        value = entry.get(key)
        if isinstance(value, struct_time):
            return datetime(*value[:6], tzinfo=UTC)
    return now


def _clean(text: str, limit: int) -> str:
    return _WS.sub(" ", _TAGS.sub(" ", text)).strip()[:limit]


def fetch_extra_rss(
    urls: list[str], fetcher: SafeFetcher, now: datetime, *, per_feed: int = 30
) -> tuple[list[FeedItem], list[str]]:
    items: list[FeedItem] = []
    warnings: list[str] = []
    for url in urls:
        try:
            body = fetcher.get(url).body
        except FetchError as exc:
            warnings.append(f"extra_rss {url}: {exc}")
            continue
        parsed = feedparser.parse(body)
        entries = list(parsed.entries)[:per_feed]
        if not entries:
            warnings.append(f"extra_rss {url}: no entries")
        for entry in entries:
            link = str(entry.get("link") or "")
            title = _clean(str(entry.get("title") or ""), 500)
            if not title or urlsplit(link).scheme not in {"http", "https"}:
                continue
            canon = canonical(link)
            published = _entry_time(entry, now)
            age_h = max(0.0, (now - published).total_seconds() / 3600)
            try:
                items.append(
                    FeedItem(
                        id=hashlib.sha256(canon.encode()).hexdigest()[:16],
                        url=link[:2048],
                        canonical_url=canon[:2048],
                        title=title,
                        source_id=EXTRA_RSS_SOURCE,
                        published_at=published,
                        snippet=_clean(str(entry.get("summary") or ""), 280),
                        base_score=round(0.40 * math.exp(-age_h / 36) + 0.25 * 0.5, 4),
                        from_extra_rss=True,
                    )
                )
            except ValidationError:
                continue
    return items, warnings
