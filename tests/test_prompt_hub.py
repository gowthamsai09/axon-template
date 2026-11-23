"""Prompts equal the hub's canonical copies (Appendix C 82).

The first test is skipped only when the hub is absent and REQUIRE_HUB_PROMPTS is not 1 (CI sets
it). Every other case works on temporary copies."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from digest.summarizers.prompting import prompt_version
from scripts.check_prompt_sync import FILES, ROOT, check, main

HUB = Path(os.environ.get("HUB_DIR", str(ROOT.parent / "Axon")))
REQUIRED = os.environ.get("REQUIRE_HUB_PROMPTS") == "1"
VERSION = "summarizer_system.version"
needs_hub = pytest.mark.skipif(
    not (HUB / "prompts").is_dir() and not REQUIRED, reason="hub not found (set HUB_DIR)"
)


@needs_hub
def test_the_real_prompts_match_the_hub() -> None:
    assert (HUB / "prompts").is_dir()  # with REQUIRE_HUB_PROMPTS=1 a missing hub fails here
    assert check(HUB) == []
    assert main(["--hub", str(HUB)]) == 0


@pytest.fixture
def hub_copy(tmp_path: Path) -> Path:
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in FILES:
        shutil.copy(ROOT / "digest" / "prompts" / name, prompts / name)
    (prompts / VERSION).write_text(prompt_version() + "\n", encoding="utf-8")
    return tmp_path


def test_identical_copies_pass(hub_copy: Path) -> None:
    assert check(hub_copy) == []
    assert main(["--hub", str(hub_copy)]) == 0


def test_crlf_in_the_hub_copy_is_not_drift(hub_copy: Path) -> None:
    p = hub_copy / "prompts" / "summarizer_system.txt"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert b"\r\n" in p.read_bytes()  # the premise holds
    assert check(hub_copy) == []


@pytest.mark.parametrize("name", FILES)
def test_a_one_character_edit_is_drift(hub_copy: Path, name: str) -> None:
    p = hub_copy / "prompts" / name
    before = p.read_text(encoding="utf-8")
    p.write_text(before.replace("e", "E", 1), encoding="utf-8")
    assert p.read_text(encoding="utf-8") != before  # the edit really happened
    problems = check(hub_copy)
    assert len(problems) >= 1 and name in problems[0]
    assert main(["--hub", str(hub_copy)]) == 1


def test_a_stale_id_file_is_drift(hub_copy: Path) -> None:
    (hub_copy / "prompts" / VERSION).write_text("p-0000000000\n", encoding="utf-8")
    problems = check(hub_copy)
    assert len(problems) == 1 and "p-0000000000" in problems[0]
    assert main(["--hub", str(hub_copy)]) == 1


@pytest.mark.parametrize("name", [*FILES, VERSION])
def test_each_missing_hub_file_is_named_and_exits_2(hub_copy: Path, name: str) -> None:
    """Removing exactly one file leaves the others intact, so nothing else can mask the result."""
    target = hub_copy / "prompts" / name
    target.unlink()
    assert not target.exists()
    assert check(hub_copy) == [f"hub copy missing: {target}"]
    assert main(["--hub", str(hub_copy)]) == 2


def test_an_empty_hub_exits_2_and_never_passes_silently(tmp_path: Path) -> None:
    assert check(tmp_path)[0].startswith("hub copy missing")
    assert main(["--hub", str(tmp_path)]) == 2


def test_drift_message_names_the_file_only(
    hub_copy: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    p = hub_copy / "prompts" / "onboarding_system.txt"
    p.write_text(p.read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
    assert main(["--hub", str(hub_copy)]) == 1
    err = capsys.readouterr().err
    assert "prompt drift: onboarding_system.txt differs" in err and "extra" not in err
