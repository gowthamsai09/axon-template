from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from digest.cli import main as cli_main
from digest.profile import TOPIC_SLUGS, Profile, ProfileError, load_profile

ROOT = Path(__file__).resolve().parent.parent


def good() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / "profile.yaml").read_text())
    return copy.deepcopy(data)


def set_path(d: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    *parents, last = path.split(".")
    node = d
    for p in parents:
        node = node[p]
    node[last] = value
    return d


def test_shipped_profile_is_valid() -> None:
    p = load_profile(ROOT / "profile.yaml")
    assert p.delivery.timezone == "Asia/Kolkata" and p.delivery.max_items == 10
    assert cli_main(["validate-profile", str(ROOT / "profile.yaml")]) == 0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("version", 2),
        ("delivery.max_items", 0),
        ("delivery.max_items", 26),
        ("delivery.channel", "sms"),
        ("delivery.frequency", "hourly"),
        ("delivery.depth", "long"),
        ("delivery.timezone", "Mars/Olympus"),
        ("delivery.timezone", "../../etc/passwd"),
        ("delivery.timezone", "IST"),
        ("delivery.send_time_local", "7:30"),
        ("delivery.send_time_local", "24:00"),
        ("delivery.language", "english"),
        ("summarizer.provider", "openai"),
        ("summarizer.model", ""),
        ("summarizer.base_url", "https://x.example/v1"),  # only for openai_compatible
        ("summarizer.fallback_provider", "huggingface"),  # model missing
        ("summarizer.fallback_provider", "openai_compatible"),
        ("interests.topics", ["made-up-topic"]),
        ("interests.topics", ["rag", "rag"]),
        ("interests.follow_entities", ["github:not-a-repo"]),
        ("interests.keywords_include", ["crypto"]),  # also in keywords_exclude
        ("sources.extra_rss", ["ftp://x.example/feed"]),
        ("sources.extra_rss", ["https://user:pw@x.example/feed"]),
        ("agent_access.max_tier3_runs_per_day", 0),
        ("agent_access.max_tier3_runs_per_day", 21),
        ("agent_access.allowed_modes", ["data", "root"]),
        ("agent_access.allowed_modes", ["data", "data"]),
    ],
)
def test_invalid(path: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        Profile.model_validate(set_path(good(), path, value))


def test_unknown_key_rejected() -> None:
    d = good()
    d["delivery"]["api_key"] = "sk-123"
    with pytest.raises(ValidationError):
        Profile.model_validate(d)


def test_openai_compatible_rules() -> None:
    d = set_path(good(), "summarizer.provider", "openai_compatible")
    with pytest.raises(ValidationError):  # base_url required
        Profile.model_validate(d)
    for bad in ("http://x.example/v1", "https://u:p@x.example/v1"):
        with pytest.raises(ValidationError):
            Profile.model_validate(set_path(copy.deepcopy(d), "summarizer.base_url", bad))
    ok = set_path(d, "summarizer.base_url", "https://x.example/v1")
    Profile.model_validate(ok)


def test_fallback_pair_ok() -> None:
    d = set_path(good(), "summarizer.fallback_provider", "huggingface")
    Profile.model_validate(set_path(d, "summarizer.fallback_model", "some/model"))


def test_weekly_needs_weekday() -> None:
    d = set_path(good(), "delivery.frequency", "weekly")
    with pytest.raises(ValidationError):
        Profile.model_validate(d)
    Profile.model_validate(set_path(d, "delivery.weekday", "mon"))
    with pytest.raises(ValidationError):  # weekday with daily
        Profile.model_validate(set_path(good(), "delivery.weekday", "mon"))


def test_empty_interests_allowed_and_slugs() -> None:
    d = good()
    d["interests"] = {}
    Profile.model_validate(d)
    assert len(TOPIC_SLUGS) == 16


def test_readable_error_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    d = set_path(good(), "delivery.max_items", 99)
    f = tmp_path / "p.yaml"
    f.write_text(yaml.safe_dump(d))
    with pytest.raises(ProfileError, match=r"delivery\.max_items"):
        load_profile(f)
    assert cli_main(["validate-profile", str(f)]) == 1
    assert "delivery.max_items" in capsys.readouterr().err
    f.write_text("a: [unclosed")
    with pytest.raises(ProfileError, match="invalid YAML"):
        load_profile(f)


def test_cli_run_without_a_feed_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FEED_BASE", raising=False)
    monkeypatch.chdir(tmp_path)  # a developer's own .env must not leak into this test
    args = ["run", "--profile", str(ROOT / "profile.yaml"), "--state-dir", str(tmp_path / "state")]
    assert cli_main(args) == 2
    assert not (tmp_path / "state").exists()
