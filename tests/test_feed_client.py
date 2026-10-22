from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from digest.feed_client import (
    FeedError,
    FeedItem,
    HttpFeed,
    LocalFeed,
    canonical,
    entity_slug,
    fetch_extra_rss,
    load_feed,
    make_reader,
    read_excerpt,
)
from digest.profile import Interests
from digest.safe_http import SafeFetcher
from tests.conftest import NOW

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
<item><title>Own blog post</title><link>https://blog.example.com/p?utm_source=x#frag</link>
<description>&lt;p&gt;Hello &lt;b&gt;world&lt;/b&gt;&lt;/p&gt;</description>
<pubDate>Tue, 20 Oct 2026 04:00:00 GMT</pubDate></item>
<item><title>Bad link</title><link>javascript:alert(1)</link></item></channel></rss>"""


def test_load_feed_merges_shards_and_dedupes(feed_dir: Path) -> None:
    interests = Interests(
        topics=["agents-tooling", "open-source-llms"], follow_entities=["llama.cpp"]
    )
    snap = load_feed(LocalFeed(feed_dir), interests)
    ids = [i.id for i in snap.items]
    assert len(ids) == len(set(ids)) == 10
    assert snap.generated_at is not None and snap.generated_at.year == 2026
    assert snap.warnings == []


def test_unknown_shards_are_skipped_but_nothing_at_all_is_an_error(
    feed_dir: Path, tmp_path: Path
) -> None:
    snap = load_feed(LocalFeed(feed_dir), Interests(topics=["robotics"]))
    assert snap.items  # latest.json still loads
    with pytest.raises(FeedError, match="no feed files"):
        load_feed(LocalFeed(tmp_path), Interests(topics=["rag"]))


def test_invalid_items_and_files_are_skipped_with_warnings(tmp_path: Path) -> None:
    good = json.loads(
        (Path(__file__).parent / "fixtures/feed-out/latest.json").read_text(encoding="utf-8")
    )
    good["items"] = [good["items"][0], {"id": "nope"}, "junk"]
    (tmp_path / "latest.json").write_text(json.dumps(good))
    (tmp_path / "tags").mkdir()
    (tmp_path / "tags" / "rag.json").write_text("{not json")
    snap = load_feed(LocalFeed(tmp_path), Interests(topics=["rag"]))
    assert len(snap.items) == 1
    assert any("skipped 2 item" in w for w in snap.warnings)
    assert any("not valid JSON" in w for w in snap.warnings)


def test_consumer_view_ignores_unknown_fields(feed_dir: Path) -> None:
    raw = json.loads((feed_dir / "latest.json").read_text(encoding="utf-8"))["items"][0]
    raw["brand_new_field"] = {"x": 1}
    item = FeedItem.model_validate(raw)
    assert item.id == raw["id"] and item.from_extra_rss is False


def test_read_excerpt(feed_dir: Path) -> None:
    reader = LocalFeed(feed_dir)
    assert "Flow-1" in (read_excerpt(reader, "6a23445afa1896da") or "")
    assert read_excerpt(reader, "aa4ccde86d8a3d2e") is None
    assert read_excerpt(reader, "../../etc/passwd") is None  # not an id: never read


def test_local_feed_rejects_path_traversal(feed_dir: Path) -> None:
    with pytest.raises(FeedError):
        LocalFeed(feed_dir).read_bytes("../meta.json")
    with pytest.raises(FeedError):
        LocalFeed(feed_dir).read_bytes("a/b/c.json")


def test_http_feed_status_handling() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if path.endswith("meta.json"):
            return httpx.Response(200, json={"generated_at": "2026-10-20T00:00:00Z"})
        if path.endswith("boom.json"):
            return httpx.Response(500)
        return httpx.Response(404)

    feed = HttpFeed(
        "https://x.github.io/axon-feed/feed/", httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert feed.read_bytes("meta.json") is not None
    assert feed.read_bytes("tags/none.json") is None
    with pytest.raises(FeedError, match="HTTP 500"):
        feed.read_bytes("tags/boom.json")


def test_make_reader_kinds(feed_dir: Path, tmp_path: Path) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    assert isinstance(make_reader(str(feed_dir), client), LocalFeed)
    assert isinstance(make_reader("https://example.com/feed", client), HttpFeed)
    with pytest.raises(FeedError, match="not found"):
        make_reader(str(tmp_path / "missing"), client)


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("llama.cpp", "llama-cpp"),
        ("Hugging Face Transformers", "hugging-face-transformers"),
        ("Qwen", "qwen"),
        ("github:ollama/ollama", None),
        ("***", None),
    ],
)
def test_entity_slug(name: str, slug: str | None) -> None:
    assert entity_slug(name) == slug


def test_canonical_strips_tracking_and_fragment() -> None:
    assert (
        canonical("https://Blog.Example.com/p?utm_source=x&a=1#frag")
        == "https://blog.example.com/p?a=1"
    )


def test_extra_rss_items() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=RSS)))
    fetcher = SafeFetcher(client, resolver=lambda h: ["93.184.216.34"])
    items, warnings = fetch_extra_rss(["https://blog.example.com/feed.xml"], fetcher, NOW)
    assert [i.title for i in items] == ["Own blog post"]  # javascript: link dropped
    first = items[0]
    assert first.from_extra_rss and first.canonical_url == "https://blog.example.com/p"
    assert first.snippet == "Hello world" and 0 < first.base_score <= 1
    assert warnings == []


def test_extra_rss_failure_is_a_warning() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    fetcher = SafeFetcher(client, resolver=lambda h: ["93.184.216.34"])
    items, warnings = fetch_extra_rss(["https://blog.example.com/feed.xml"], fetcher, NOW)
    assert items == [] and "HTTP 500" in warnings[0]
