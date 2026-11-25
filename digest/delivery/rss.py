"""Atom delivery (spec 4.2, Phase 6, Appendix C 95): the digest becomes one entry of an Atom
feed kept on the state branch as rss/feed.xml. The entries are kept as JSON next to it
(rss/entries.json) and the XML is rebuilt from them, so an old feed file is never parsed.

Everything in an entry comes from the digest text, which holds untrusted feed and model text.
It is written as plain text content (type="text"), escaped by the XML serializer, stripped of
characters XML 1.0 cannot carry, and capped. Ids are stable: the same digest of the same items
on the same day always has the same id, so a rerun replaces its entry instead of adding one."""

from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from xml.etree import ElementTree as ET

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, ValidationError

from ..state import StateStore

ATOM_NS = "http://www.w3.org/2005/Atom"
ENTRIES_FILE = "rss/entries.json"
FEED_FILE = "rss/feed.xml"
MAX_ENTRIES = 50
MAX_TITLE = 200
MAX_CONTENT = 20_000
MAX_STORED = 1000  # entries read back from a state file; anything beyond is ignored
_TRUNCATED = "\n[digest shortened for the feed]"
_XML_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f￾￿\ud800-\udfff]")
_URN = r"^urn:uuid:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"


class AtomEntry(BaseModel):
    """One stored entry (strict: this file is ours)."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=_URN)
    title: str = Field(min_length=1, max_length=MAX_TITLE)
    updated: AwareDatetime
    link: str = Field("", max_length=300)
    content: str = Field(max_length=MAX_CONTENT + 100)


def xml_text(value: str) -> str:
    """Drop what XML 1.0 cannot carry (controls, lone surrogates, U+FFFE, U+FFFF)."""
    return _XML_ILLEGAL.sub("", value)


def _clip(value: str, limit: int, suffix: str = "") -> str:
    value = xml_text(value)
    if len(value) <= limit:
        return value
    return value[: max(0, limit - len(suffix))].rstrip() + suffix


def feed_uuid(feed_key: str) -> str:
    return f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, f'axon:feed:{feed_key}')}"


def entry_uuid(feed_key: str, item_ids: Sequence[str], day: str) -> str:
    raw = f"axon:entry:{feed_key}:{day}:{','.join(sorted(item_ids))}"
    return f"urn:uuid:{uuid.uuid5(uuid.NAMESPACE_URL, raw)}"


def stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_entry(
    subject: str,
    text: str,
    item_ids: Sequence[str],
    now: datetime,
    *,
    feed_key: str,
    link: str = "",
) -> AtomEntry:
    day = now.astimezone(UTC).strftime("%Y-%m-%d")
    title = _clip(" ".join(subject.split()), MAX_TITLE).strip() or "Axon digest"
    return AtomEntry(
        id=entry_uuid(feed_key, item_ids, day),
        title=title,
        updated=now,
        link=link,
        content=_clip(text, MAX_CONTENT, _TRUNCATED),
    )


def merge_entries(existing: Sequence[AtomEntry], new: AtomEntry) -> list[AtomEntry]:
    """`new` first, an older entry with the same id replaced, newest first, at most 50."""
    rest = [e for e in existing if e.id != new.id]
    merged = sorted([new, *rest], key=lambda e: (e.updated, e.id), reverse=True)
    return merged[:MAX_ENTRIES]


def load_entries(state: StateStore) -> tuple[list[AtomEntry], list[str]]:
    """Stored entries; anything unreadable or invalid is skipped with a warning."""
    raw = state.read_json(ENTRIES_FILE)
    warnings: list[str] = []
    if raw is None:
        return [], warnings
    if not isinstance(raw, list):
        return [], [f"state/{ENTRIES_FILE}: not a list, ignored"]
    entries: list[AtomEntry] = []
    bad = 0
    for item in raw[:MAX_STORED]:
        try:
            entries.append(AtomEntry.model_validate(item))
        except ValidationError:
            bad += 1
    if bad:
        warnings.append(f"state/{ENTRIES_FILE}: skipped {bad} invalid entr(y/ies)")
    return entries, warnings


def render_feed(
    entries: Sequence[AtomEntry],
    *,
    feed_key: str,
    now: datetime,
    alternate: str = "",
    self_url: str = "",
) -> str:
    """The Atom document. Required elements (RFC 4287): id, title, updated, and an author."""
    ET.register_namespace("", ATOM_NS)
    root = ET.Element(f"{{{ATOM_NS}}}feed")

    def add(parent: ET.Element, name: str, text: str | None = None, **attrs: str) -> ET.Element:
        node = ET.SubElement(parent, f"{{{ATOM_NS}}}{name}", attrs)
        if text is not None:
            node.text = xml_text(text)
        return node

    add(root, "title", "Axon digest", type="text")
    add(root, "id", feed_uuid(feed_key))
    newest = max((e.updated for e in entries), default=now)
    add(root, "updated", stamp(newest))
    add(add(root, "author"), "name", "Axon")
    if alternate:
        add(root, "link", rel="alternate", href=xml_text(alternate))
    if self_url:
        add(root, "link", rel="self", href=xml_text(self_url))
    for entry in entries:
        node = add(root, "entry")
        add(node, "title", entry.title, type="text")
        add(node, "id", entry.id)
        add(node, "updated", stamp(entry.updated))
        if entry.link:
            add(node, "link", rel="alternate", href=xml_text(entry.link))
        add(node, "content", entry.content, type="text")
    ET.indent(root, space="  ")
    body = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + body + "\n"


def build_feed(
    state: StateStore,
    subject: str,
    text: str,
    item_ids: Sequence[str],
    now: datetime,
    *,
    repo: str,
    repo_url: str,
    self_url: str,
) -> tuple[str, list[AtomEntry], list[str]]:
    """Read-only: the feed with this digest added, the entries behind it, and warnings."""
    feed_key = repo or "local"
    existing, warnings = load_entries(state)
    entry = make_entry(subject, text, item_ids, now, feed_key=feed_key, link=repo_url)
    entries = merge_entries(existing, entry)
    xml = render_feed(entries, feed_key=feed_key, now=now, alternate=repo_url, self_url=self_url)
    return xml, entries, warnings


def store_feed(state: StateStore, entries: Sequence[AtomEntry], xml: str) -> None:
    state.write_json(ENTRIES_FILE, [e.model_dump(mode="json") for e in entries])
    state.write_text(FEED_FILE, xml)
