from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import httpx

from digest.extract import TextEnricher
from digest.feed_client import LocalFeed
from digest.pipeline import DigestRun, run_digest
from digest.profile import Profile
from digest.safe_http import SafeFetcher
from digest.state import StateStore
from digest.summarizers.base import AuthError
from digest.summarizers.router import SummarizerRouter
from tests.conftest import FEED, NOW, FakeChat, empty_json, make_profile, summary_json

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>t</title><item>
<title>My own post</title><link>https://blog.example.com/p</link>
<description>Own feed summary</description><pubDate>Tue, 20 Oct 2026 04:00:00 GMT</pubDate>
</item></channel></rss>"""


def go(
    profile: Profile,
    state: StateStore,
    chat: FakeChat | None = None,
    *,
    fallback: FakeChat | None = None,
    use_seen: bool = True,
    now_shift: int = 0,
    handler: httpx.MockTransport | None = None,
) -> DigestRun:
    reader = LocalFeed(FEED)
    client = httpx.Client(transport=handler or httpx.MockTransport(lambda r: httpx.Response(404)))
    fetcher = SafeFetcher(client, resolver=lambda h: ["93.184.216.34"])
    router = SummarizerRouter(chat, fallback) if chat is not None else None
    return run_digest(
        profile, reader=reader, fetcher=fetcher, enricher=TextEnricher(reader, fetcher),
        router=router, state=state, now=NOW + timedelta(hours=now_shift), use_seen=use_seen,
    )  # fmt: skip


def agents(**delivery: object) -> Profile:
    return make_profile(
        interests={"topics": ["agents-tooling"], "keywords_exclude": []},
        delivery={"timezone": "UTC", **delivery},
    )


def test_extractive_only_run(tmp_path: Path) -> None:
    run = go(agents(max_items=3), StateStore(tmp_path))
    assert [e.item.id for e in run.entries][0] == "aa4ccde86d8a3d2e" and len(run.entries) == 3
    assert all(e.summary is None and e.fallback_snippet for e in run.entries)
    assert run.candidates == 10 and run.usage.calls == 0


def test_summaries_are_cached_and_reused(tmp_path: Path) -> None:
    state = StateStore(tmp_path)
    chat = FakeChat()
    first = go(agents(max_items=3), state, chat)
    assert len(chat.calls) == 3 and all(e.summary for e in first.entries)
    assert first.usage.prompt_tokens == 300 and first.cache_hits == 0
    second = go(agents(max_items=3), state, chat)
    assert len(chat.calls) == 3 and second.cache_hits == 3 and all(e.cached for e in second.entries)
    other_language = go(agents(max_items=3, language="de"), state, chat)
    assert len(chat.calls) == 6 and other_language.cache_hits == 0  # language is part of the key


def test_cache_is_found_under_the_fallback_identity(tmp_path: Path) -> None:
    state = StateStore(tmp_path)
    fb = FakeChat(provider="huggingface", model="o/m")
    go(
        agents(max_items=1), state, FakeChat(["bad", "bad"]), fallback=None
    )  # primary output rejected
    go(agents(max_items=1), state, FakeChat([AuthError("k")]), fallback=fb)
    again = go(agents(max_items=1), state, FakeChat(), fallback=fb)
    assert again.cache_hits == 1 and again.entries[0].provider == "huggingface"


def test_seen_items_are_dropped_unless_disabled(tmp_path: Path) -> None:
    state = StateStore(tmp_path)
    first = go(agents(max_items=2), state)
    state.mark_seen([e.item.id for e in first.entries], NOW)
    second = go(agents(max_items=2), state)
    assert not {e.item.id for e in first.entries} & {e.item.id for e in second.entries}
    again = go(agents(max_items=2), state, use_seen=False)
    assert {e.item.id for e in again.entries} == {e.item.id for e in first.entries}


def test_injection_verdict_never_reaches_the_digest_as_a_summary(tmp_path: Path) -> None:
    def hostile(user: str) -> str:
        return empty_json(user, "suspected_injection", injection_flag=True)

    run = go(agents(max_items=1), StateStore(tmp_path), FakeChat([hostile]))
    entry = run.entries[0]
    assert entry.summary is None and "instructions aimed at AI" in entry.note
    assert any("suspected prompt injection" in w for w in run.warnings)


def test_flagged_but_ok_summary_is_kept_with_a_note(tmp_path: Path) -> None:
    def flagged(user: str) -> str:
        return summary_json(user, injection_flag=True)

    entry = go(agents(max_items=1), StateStore(tmp_path), FakeChat([flagged])).entries[0]
    assert entry.summary is not None and "aimed at AI systems" in entry.note


def test_bad_model_output_degrades_to_the_snippet(tmp_path: Path) -> None:
    run = go(agents(max_items=1), StateStore(tmp_path), FakeChat(["x", "y"]))
    assert run.entries[0].summary is None and "not AI-summarized" in run.entries[0].note
    assert any("summary rejected" in w for w in run.warnings)


def test_insufficient_content_uses_the_snippet(tmp_path: Path) -> None:
    def empty(user: str) -> str:
        return empty_json(user, "insufficient_content")

    entry = go(agents(max_items=1), StateStore(tmp_path), FakeChat([empty])).entries[0]
    assert entry.summary is None and entry.note == "too little content to summarize"


def test_stale_feed_warns(tmp_path: Path) -> None:
    run = go(agents(max_items=1), StateStore(tmp_path), now_shift=72)
    assert any("feed is" in w and "hours old" in w for w in run.warnings)


def test_extra_rss_items_always_qualify(tmp_path: Path) -> None:
    profile = make_profile(
        interests={"topics": ["agents-tooling"], "keywords_exclude": []},
        sources={"extra_rss": ["https://blog.example.com/feed.xml"]},
        delivery={"timezone": "UTC", "max_items": 10},
    )

    def handler(req: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(200, content=RSS)
            if req.url.path.endswith("feed.xml")
            else httpx.Response(404)
        )

    run = go(profile, StateStore(tmp_path), handler=httpx.MockTransport(handler))
    assert "My own post" in [e.item.title for e in run.entries]
