"""P6.3 Atom delivery: valid Atom, hostile text escaped, stable ids, caps, tolerant state,
and the CLI for the rss channel (dry run, real run, Tier 3 owner_channel). Offline."""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from digest.delivery.rss import (
    ATOM_NS,
    ENTRIES_FILE,
    FEED_FILE,
    MAX_CONTENT,
    MAX_ENTRIES,
    MAX_STORED,
    MAX_TITLE,
    AtomEntry,
    build_feed,
    entry_uuid,
    feed_uuid,
    load_entries,
    make_entry,
    merge_entries,
    render_feed,
    stamp,
    store_feed,
    xml_text,
)
from digest.render import build_view, render_email
from digest.state import StateStore
from tests.chat_helpers import refusing
from tests.conftest import NOW, make_profile
from tests.test_chat_delivery import STEM, go, last_run, printed, stored_text, t3
from tests.test_cli import seen
from tests.test_render import entry, run_of

A = "{" + ATOM_NS + "}"
URN = re.compile(r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
ZULU = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
HOSTILE = '</content><script>alert("AXON-RSS-1")</script> & ]]> <![CDATA[x]]> \x00\x1b'
IDS = ("a" * 16, "b" * 16)


def parse(xml: str) -> ET.Element:
    return ET.fromstring(xml.encode("utf-8"))


def feed(
    tmp_path: Path,
    text: str = "body text",
    *,
    subject: str = "Axon digest: Tue Oct 20, 2026 (2 items)",
    ids: tuple[str, ...] = IDS,
    repo: str = "o/r",
    repo_url: str = "https://github.com/o/r",
    self_url: str = "",
) -> tuple[str, list[AtomEntry], list[str]]:
    return build_feed(
        StateStore(tmp_path), subject, text, ids, NOW,
        repo=repo, repo_url=repo_url, self_url=self_url,
    )  # fmt: skip


def make(day: int = 0, ids: tuple[str, ...] = IDS, text: str = "t") -> AtomEntry:
    return make_entry("subject", text, ids, NOW + timedelta(days=day), feed_key="o/r")


# ---- the document ----------------------------------------------------------------------


def test_the_feed_has_every_element_atom_requires(tmp_path: Path) -> None:
    xml, _, _ = feed(tmp_path, self_url="https://raw.example/state/rss/feed.xml")
    root = parse(xml)
    assert root.tag == A + "feed" and xml.startswith('<?xml version="1.0" encoding="utf-8"?>')
    for name in ("title", "id", "updated"):
        assert root.find(A + name) is not None
    assert root.findtext(f"{A}author/{A}name") == "Axon"
    assert URN.match(root.findtext(A + "id") or "")
    assert ZULU.match(root.findtext(A + "updated") or "")
    (node,) = root.findall(A + "entry")
    for name in ("title", "id", "updated", "content"):
        assert node.find(A + name) is not None
    content = node.find(A + "content")
    assert content is not None and content.attrib == {"type": "text"}
    assert node.findtext(A + "id") != root.findtext(A + "id")


def test_the_feed_updated_time_is_the_newest_entry() -> None:
    entries = [make(0), make(3)]
    root = parse(render_feed(entries, feed_key="o/r", now=NOW))
    assert root.findtext(A + "updated") == stamp(NOW + timedelta(days=3))


def test_an_empty_feed_is_still_valid() -> None:
    root = parse(render_feed([], feed_key="local", now=NOW))
    assert root.findtext(A + "updated") == stamp(NOW) and root.findall(A + "entry") == []


def test_links_only_when_known(tmp_path: Path) -> None:
    root = parse(feed(tmp_path, repo_url="", self_url="")[0])
    assert root.findall(A + "link") == []
    assert root.find(f"{A}entry/{A}link") is None
    full = parse(feed(tmp_path, self_url="https://raw.example/f.xml")[0])
    rels = {link.get("rel"): link.get("href") for link in full.findall(A + "link")}
    assert rels == {"alternate": "https://github.com/o/r", "self": "https://raw.example/f.xml"}
    entry_link = full.find(f"{A}entry/{A}link")
    assert entry_link is not None and entry_link.get("href") == "https://github.com/o/r"


def test_a_hostile_link_cannot_break_the_attribute(tmp_path: Path) -> None:
    url = 'https://example.com/?a=1&b="2"<x>'
    root = parse(feed(tmp_path, repo_url=url)[0])
    link = root.find(f"{A}link")
    assert link is not None and link.get("href") == url


def test_the_same_inputs_give_the_same_bytes(tmp_path: Path) -> None:
    assert feed(tmp_path)[0] == feed(tmp_path)[0]


# ---- hostile text ----------------------------------------------------------------------


def test_hostile_digest_text_is_escaped_never_parsed(tmp_path: Path) -> None:
    run = run_of(entry(title=HOSTILE))
    subject, text, _ = render_email(build_view(run, make_profile(), NOW))
    xml, _, _ = feed(tmp_path, text, subject=subject)
    assert "<script>" not in xml and "<![CDATA[" not in xml
    root = parse(xml)
    assert root.findall(".//script") == [] and len(root.findall(A + "entry")) == 1
    content = root.findtext(f"{A}entry/{A}content") or ""
    assert 'alert("AXON-RSS-1")' in content and "</content>" in content  # text, not markup
    assert "\x00" not in content and "\x1b" not in content


def test_a_hostile_subject_stays_in_the_title(tmp_path: Path) -> None:
    xml, _, _ = feed(tmp_path, subject='</title><b x="1">AXON-TITLE</b>\n\t  spaced')
    node = parse(xml).find(A + "entry")
    assert node is not None and node.findall(".//b") == []
    assert node.findtext(A + "title") == '</title><b x="1">AXON-TITLE</b> spaced'


def test_characters_xml_cannot_carry_are_removed(tmp_path: Path) -> None:
    assert xml_text("a\x00b\x1fc\ufffed\uffffe\ud800f\tg\nh") == "abcdef\tg\nh"
    xml, _, _ = feed(tmp_path, "ok\x00\x07\x1b\ufffe more\ud800 end", subject="\x00 \x07")
    node = parse(xml).find(A + "entry")
    assert node is not None and node.findtext(A + "content") == "ok more end"
    assert node.findtext(A + "title") == "Axon digest"  # nothing usable left: the fallback


# ---- ids, merge, caps ------------------------------------------------------------------


def test_entry_ids_are_stable_and_specific() -> None:
    base = entry_uuid("o/r", IDS, "2026-10-20")
    assert URN.match(base) and URN.match(feed_uuid("o/r"))
    assert entry_uuid("o/r", tuple(reversed(IDS)), "2026-10-20") == base  # order does not matter
    assert entry_uuid("o/r", IDS, "2026-10-21") != base
    assert entry_uuid("o/r", IDS[:1], "2026-10-20") != base
    assert entry_uuid("x/y", IDS, "2026-10-20") != base
    assert feed_uuid("o/r") != feed_uuid("x/y") and feed_uuid("o/r") == feed_uuid("o/r")


def test_a_rerun_of_the_same_digest_replaces_its_entry() -> None:
    first, again = make(text="old"), make(text="new")
    assert first.id == again.id
    merged = merge_entries([first], again)
    assert [e.content for e in merged] == ["new"]


def test_entries_are_newest_first_and_capped() -> None:
    entries: list[AtomEntry] = []
    for day in range(MAX_ENTRIES + 10):
        entries = merge_entries(entries, make(day, ids=(f"{day:016x}",)))
    assert len(entries) == MAX_ENTRIES
    assert entries[0].updated > entries[-1].updated
    assert entries[0].updated == NOW + timedelta(days=MAX_ENTRIES + 9)


def test_title_and_content_are_capped() -> None:
    e = make_entry("s" * 500, "x" * (MAX_CONTENT + 5000), IDS, NOW, feed_key="o/r")
    assert len(e.title) <= MAX_TITLE and len(e.content) <= MAX_CONTENT
    assert e.content.endswith("[digest shortened for the feed]")
    exact = make_entry("s", "y" * MAX_CONTENT, IDS, NOW, feed_key="o/r")
    assert exact.content == "y" * MAX_CONTENT


def test_a_blank_subject_falls_back() -> None:
    assert make_entry("  \n ", "t", IDS, NOW, feed_key="k").title == "Axon digest"


# ---- state files -----------------------------------------------------------------------


def test_missing_entries_file_is_normal(tmp_path: Path) -> None:
    assert load_entries(StateStore(tmp_path)) == ([], [])


@pytest.mark.parametrize("raw", ["{not json", '"a string"', '{"a": 1}', "42"])
def test_unreadable_or_wrongly_shaped_state_is_ignored_with_a_warning(
    tmp_path: Path, raw: str
) -> None:
    state = StateStore(tmp_path)
    (tmp_path / "rss").mkdir()
    (tmp_path / ENTRIES_FILE).write_text(raw, encoding="utf-8")
    entries, warnings = load_entries(state)
    assert entries == [] and (warnings or state.warnings)
    xml, _, _ = build_feed(
        state, "s", "t", IDS, NOW, repo="o/r", repo_url="", self_url=""
    )  # the run still produces a feed
    assert len(parse(xml).findall(A + "entry")) == 1


def test_invalid_stored_entries_are_skipped_and_counted(tmp_path: Path) -> None:
    good = make().model_dump(mode="json")
    bad = {**good, "id": "not-a-urn"}
    extra = {**good, "surprise": 1}
    (tmp_path / "rss").mkdir()
    (tmp_path / ENTRIES_FILE).write_text(json.dumps([good, bad, extra, 7]), encoding="utf-8")
    entries, warnings = load_entries(StateStore(tmp_path))
    assert [e.id for e in entries] == [good["id"]]
    assert warnings == [f"state/{ENTRIES_FILE}: skipped 3 invalid entr(y/ies)"]


def test_only_the_first_thousand_stored_entries_are_read(tmp_path: Path) -> None:
    rows = [make(ids=(f"{i:016x}",)).model_dump(mode="json") for i in range(MAX_STORED + 5)]
    (tmp_path / "rss").mkdir()
    (tmp_path / ENTRIES_FILE).write_text(json.dumps(rows), encoding="utf-8")
    assert len(load_entries(StateStore(tmp_path))[0]) == MAX_STORED


def test_store_feed_writes_both_files_with_lf_and_no_leftovers(tmp_path: Path) -> None:
    state = StateStore(tmp_path)
    xml, entries, _ = feed(tmp_path)
    store_feed(state, entries, xml)
    assert (tmp_path / FEED_FILE).read_bytes() == xml.encode("utf-8") and b"\r" not in xml.encode()
    assert [e.id for e in load_entries(StateStore(tmp_path))[0]] == [e.id for e in entries]
    assert not list((tmp_path / "rss").glob("*.tmp"))


def test_the_next_run_builds_on_the_stored_entries(tmp_path: Path) -> None:
    state = StateStore(tmp_path)
    xml, entries, _ = feed(tmp_path, ids=("1" * 16,))
    store_feed(state, entries, xml)
    xml2, entries2, _ = feed(tmp_path, ids=("2" * 16,))
    assert len(entries2) == 2 and len(parse(xml2).findall(A + "entry")) == 2


# ---- the CLI ---------------------------------------------------------------------------


def feed_in(root: Path) -> ET.Element:
    return parse((root / "state" / FEED_FILE).read_text(encoding="utf-8"))


def test_dry_run_writes_the_outbox_and_touches_no_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert go(tmp_path, channel="rss", env={}, http=refusing(), extra=("--dry-run",)) == 0
    files = sorted(p.name for p in (tmp_path / "outbox").iterdir())
    assert files == [f"{STEM}.rss-01.xml"]
    root = parse((tmp_path / "outbox" / files[0]).read_text(encoding="utf-8"))
    assert len(root.findall(A + "entry")) == 1
    assert not (tmp_path / "state" / "rss").exists() and seen(tmp_path) == set()
    assert last_run(tmp_path)["delivered"] == "dry-run" and "dry run: wrote" in printed(capsys)


def test_a_real_run_stores_the_feed_and_consumes_the_items(tmp_path: Path) -> None:
    assert go(tmp_path, channel="rss", env={}, http=refusing()) == 0
    assert len(feed_in(tmp_path).findall(A + "entry")) == 1
    assert len(seen(tmp_path)) == 2 and last_run(tmp_path)["delivered"] == "rss"
    assert len(load_entries(StateStore(tmp_path / "state"))[0]) == 1


def test_a_second_run_with_new_items_adds_an_entry(tmp_path: Path) -> None:
    assert go(tmp_path, channel="rss", env={}, http=refusing()) == 0
    first = [e.id for e in load_entries(StateStore(tmp_path / "state"))[0]]
    assert go(tmp_path, channel="rss", env={}, http=refusing()) == 0
    after = load_entries(StateStore(tmp_path / "state"))[0]
    assert len(after) == 2 and first[0] in {e.id for e in after}
    assert len({e.id for e in after}) == 2 and len(seen(tmp_path)) == 4


def test_the_self_link_needs_github_dot_com_and_a_repository(tmp_path: Path) -> None:
    env = {"GITHUB_REPOSITORY": "o/r", "GITHUB_SERVER_URL": "https://github.com"}
    assert go(tmp_path, channel="rss", env=env, http=refusing()) == 0
    rels = {link.get("rel"): link.get("href") for link in feed_in(tmp_path).findall(A + "link")}
    assert rels["self"] == "https://raw.githubusercontent.com/o/r/state/rss/feed.xml"
    assert rels["alternate"] == "https://github.com/o/r"


@pytest.mark.parametrize(
    "env",
    [
        {"GITHUB_REPOSITORY": "o/r", "GITHUB_SERVER_URL": "https://ghe.example.com"},
        {"GITHUB_SERVER_URL": "https://github.com"},
        {"GITHUB_REPOSITORY": "not a repo", "GITHUB_SERVER_URL": "https://github.com"},
    ],
)
def test_no_self_link_for_other_servers_or_bad_repositories(
    tmp_path: Path, env: dict[str, str]
) -> None:
    assert go(tmp_path, channel="rss", env=env, http=refusing()) == 0
    rels = {link.get("rel") for link in feed_in(tmp_path).findall(A + "link")}
    assert "self" not in rels


def test_no_secret_reaches_the_feed_or_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    secrets = {
        "LLM_API_KEY": "AXON-LLM-KEY-777777", "SMTP_PASS": "AXON-SMTP-PASS-777777",
        "GITHUB_TOKEN": "AXON-GH-TOKEN-777777", "EMBED_API_KEY": "AXON-EMBED-KEY-777777",
    }  # fmt: skip
    assert go(tmp_path, channel="rss", env=secrets, http=refusing()) == 0
    haystack = printed(capsys) + stored_text(tmp_path)
    assert "AXON-" not in haystack and "<entry" in haystack


def test_owner_channel_delivers_through_the_feed(tmp_path: Path) -> None:
    code, result = t3(tmp_path, "rss", {}, refusing())
    assert code == 0 and len(result["items"]) == 1
    assert last_run(tmp_path)["delivered"] == "owner_channel:rss"
    assert len(feed_in(tmp_path).findall(A + "entry")) == 1
    assert not (tmp_path / "state" / "seen.json").exists()  # Tier 3 never consumes items
