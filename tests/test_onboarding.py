"""Onboarding form parser: valid forms, every field rule, hostile bodies (Appendix C 101)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from digest import onboarding as o
from digest.cli import DEFAULT_MODELS, SUGGESTED_EMBEDDING_MODEL
from digest.profile import load_profile
from tests.conftest import NOW
from tests.onboard_helpers import body, section


def build(text: str, access: dict[str, Any] | None = None) -> o.Onboarded:
    return o.build(
        text,
        now=NOW,
        default_models=DEFAULT_MODELS,
        default_embedding_model=SUGGESTED_EMBEDDING_MODEL,
        agent_access=access,
    )


def problems(text: str) -> list[str]:
    with pytest.raises(o.OnboardError) as caught:
        build(text)
    return caught.value.problems


# ---- happy paths ----------------------------------------------------------------------


def test_valid_form_builds_a_profile_that_loads_back(tmp_path: Path) -> None:
    result = build(body())
    path = tmp_path / "profile.yaml"
    path.write_text(result.yaml_text, encoding="utf-8")
    loaded = load_profile(path)
    assert loaded == result.profile
    assert loaded.interests.topics == ["agents-tooling", "rag"]
    assert loaded.delivery.max_items == 10 and loaded.summarizer.model == "deepseek-flash"


def test_cron_and_secret_names_follow_the_channel() -> None:
    assert build(body()).cron == "0 2 * * *"
    assert build(body()).secrets == ("LLM_API_KEY", "SMTP_USER", "SMTP_PASS", "MAIL_TO")
    tg = build(body(L_CHANNEL="telegram"))
    assert tg.secrets == ("LLM_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")
    assert build(body(L_CHANNEL="slack")).secrets == ("LLM_API_KEY", "SLACK_WEBHOOK_URL")
    assert build(body(L_CHANNEL="rss")).secrets == ("LLM_API_KEY",)


def test_weekly_form_needs_and_uses_the_weekday() -> None:
    result = build(body(L_FREQUENCY="weekly", L_WEEKDAY="fri", L_ZONE="UTC", L_TIME="08:00"))
    assert result.profile.delivery.weekday == "fri"
    assert result.cron == "0 8 * * 5"
    assert any("Weekday" in p for p in problems(body(L_FREQUENCY="weekly")))


def test_weekday_is_ignored_for_a_daily_form() -> None:
    assert build(body(L_WEEKDAY="mon")).profile.delivery.weekday is None


def test_cron_crosses_midnight_in_utc() -> None:
    result = build(body(L_ZONE="Asia/Kolkata", L_TIME="01:00"))
    assert result.cron == "30 19 * * *"


def test_extras_switch_features_on() -> None:
    extras = list(o.EXTRAS)
    result = build(body(L_EXTRAS=extras))
    assert result.profile.embeddings.enabled and result.profile.trends.enabled
    assert result.profile.feedback.enabled
    assert result.profile.embeddings.model == SUGGESTED_EMBEDDING_MODEL
    assert result.secrets[-1] == "EMBED_API_KEY"


def test_extras_default_to_off() -> None:
    result = build(body())
    assert not result.profile.embeddings.enabled
    assert not result.profile.trends.enabled and not result.profile.feedback.enabled


def test_openai_compatible_needs_a_base_url_and_embeddings_can_use_one() -> None:
    assert any(o.L_BASE in p for p in problems(body(L_PROVIDER="openai_compatible", L_MODEL="m")))
    ok = build(
        body(
            L_PROVIDER="openai_compatible",
            L_MODEL="m",
            L_BASE="https://llm.example.org/v1",
            L_EXTRAS=[o.EXTRAS[0]],
            L_EPROVIDER="openai_compatible",
            L_EMODEL="e",
            L_EBASE="https://emb.example.org/v1",
        )
    )
    assert ok.profile.embeddings.base_url == "https://emb.example.org/v1"


def test_existing_agent_access_is_kept_on_a_rerun(tmp_path: Path) -> None:
    first = build(body())
    path = tmp_path / "profile.yaml"
    data = yaml.safe_load(first.yaml_text)
    data["agent_access"]["max_tier3_runs_per_day"] = 3
    data["agent_access"]["allowed_modes"] = ["data"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    kept = o.existing_agent_access(path)
    assert kept is not None and kept["max_tier3_runs_per_day"] == 3
    assert build(body(), kept).profile.agent_access.allowed_modes == ["data"]


def test_unreadable_existing_profile_falls_back_to_defaults(tmp_path: Path) -> None:
    assert o.existing_agent_access(tmp_path / "missing.yaml") is None
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: [valid", encoding="utf-8")
    assert o.existing_agent_access(bad) is None


def test_entries_accept_bullets_and_blank_lines() -> None:
    text = body(L_ENTITIES=["- Qwen", "", "* LangChain"])
    assert build(text).profile.interests.follow_entities == ["Qwen", "LangChain"]


def test_crlf_bodies_parse_like_lf_bodies() -> None:
    assert build(body().replace("\n", "\r\n")).profile == build(body()).profile


def test_rendered_profile_is_deterministic() -> None:
    assert build(body()).yaml_text == build(body()).yaml_text


# ---- required fields and shapes -------------------------------------------------------


@pytest.mark.parametrize(
    "label", [o.L_CHANNEL, o.L_FREQUENCY, o.L_TIME, o.L_ZONE, o.L_DEPTH, o.L_MAX, o.L_LANG]
)
def test_required_fields_missing_or_empty(label: str) -> None:
    key = next(k for k in dir(o) if k.startswith("L_") and getattr(o, k) == label)
    assert any(label in p and "required" in p for p in problems(body(**{key: None})))
    dropped = body().replace(f"### {label}", "### Something else")
    assert any(label in p for p in problems(dropped))


def test_acknowledgement_is_required() -> None:
    assert any(o.L_ACK in p for p in problems(body(L_ACK=[])))


def test_empty_body_lists_every_required_field() -> None:
    found = problems("")
    assert len(found) >= 8 and all("required" in p or o.L_ACK in p for p in found)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("L_CHANNEL", "sms"),
        ("L_FREQUENCY", "hourly"),
        ("L_DEPTH", "epic"),
        ("L_PROVIDER", "openai"),
        ("L_TIME", "7:30"),
        ("L_TIME", "24:00"),
        ("L_ZONE", "Mars/Base"),
        ("L_MAX", "0"),
        ("L_MAX", "26"),
        ("L_MAX", "-3"),
        ("L_MAX", "1e1"),
        ("L_MAX", "٣"),
        ("L_LANG", "English"),
    ],
)
def test_bad_values_are_rejected_without_echo(key: str, value: str) -> None:
    found = problems(body(**{key: value}))
    assert found and all(value not in p for p in found)


def test_unlisted_topic_and_unlisted_extra_are_rejected() -> None:
    forged = body().replace("- [ ] multimodal", "- [x] not-a-topic")
    assert any(o.L_TOPICS in p for p in problems(forged))
    assert any(o.L_EXTRAS in p for p in problems(body().replace("Weekly trends", "Evil")))


def test_non_checkbox_line_inside_a_checkbox_section_is_rejected() -> None:
    assert any(o.L_TOPICS in p for p in problems(body().replace("- [ ] multimodal", "multimodal")))


def test_multiline_text_field_is_rejected() -> None:
    assert any(o.L_TIME in p for p in problems(body(L_TIME="07:30\n08:30")))


def test_repeated_section_is_rejected() -> None:
    assert any("more than once" in p for p in problems(body() + section(o.L_ZONE, "UTC")))


def test_too_many_entries_and_too_long_entries() -> None:
    assert any(o.L_ENTITIES in p for p in problems(body(L_ENTITIES=["a"] * 51)))
    assert any(o.L_ENTITIES in p for p in problems(body(L_ENTITIES=["x" * 3000])))


def test_profile_rules_map_back_to_form_fields() -> None:
    found = problems(body(L_INCLUDE=["same"], L_EXCLUDE=["same"], L_RSS=["ftp://x.example/a"]))
    assert any(o.L_INCLUDE in p for p in found) and any(o.L_RSS in p for p in found)
    assert all(p.endswith("is missing or not valid") for p in found)


def test_credentials_in_urls_are_rejected() -> None:
    found = problems(body(L_RSS=["https://user:pw-9f3a@feeds.example.org/a"]))
    assert found and all("pw-9f3a" not in p for p in found)


# ---- hostile input --------------------------------------------------------------------

EVIL = "ZQ7-INJECT-91c4"
LISTS = {"L_ENTITIES", "L_INCLUDE", "L_EXCLUDE", "L_RSS"}


@pytest.mark.parametrize(
    "key",
    ["L_ENTITIES", "L_INCLUDE", "L_EXCLUDE", "L_RSS", "L_TIME", "L_ZONE", "L_LANG", "L_MODEL"],
)
def test_untrusted_text_never_appears_in_any_message(key: str) -> None:
    hostile = f"{EVIL} `$(id)` ${{{{ secrets.X }}}} <script>"
    value: str | list[str] = [hostile] if key in LISTS else hostile
    try:
        result = build(body(**{key: value}))
    except o.OnboardError as exc:
        text = o.comment_problems(exc.problems)
        assert EVIL not in text and "secrets.X" not in text and "$(id)" not in text
    else:
        comment = o.comment_ok(result)
        assert EVIL not in comment and "$(id)" not in comment


def test_ok_comment_contains_only_fixed_text_and_validated_values() -> None:
    comment = o.comment_ok(build(body(L_ENTITIES=[EVIL])))
    assert EVIL not in comment and '"0 2 * * *"' in comment and "`LLM_API_KEY`" in comment


def test_oversize_body_is_refused_before_parsing() -> None:
    assert problems(body() + "x" * o.MAX_BODY) == ["The form is too large"]


def test_control_characters_are_rejected() -> None:
    assert any(o.L_ENTITIES in p for p in problems(body(L_ENTITIES=["a\x00b"])))
    assert any(o.L_ZONE in p for p in problems(body(L_ZONE="UTC\x1b[31m")))


def test_fake_heading_inside_an_answer_cannot_override_a_field() -> None:
    sneaky = body(L_ENTITIES=["Qwen", f"### {o.L_CHANNEL}", "telegram"])
    found = problems(sneaky)
    assert found, "a repeated known heading must be refused, not merged"


def test_unknown_headings_are_ignored_not_trusted() -> None:
    extra = body() + "### Admin override\n\ntrue\n\n"
    assert build(extra).profile == build(body()).profile


def test_yaml_tricks_stay_plain_strings() -> None:
    for trick in ("{a: b}", "[x, y]", "&anchor value", "!!python/object:os.system x", "? key: v"):
        result = build(body(L_ENTITIES=[trick]))
        reloaded = yaml.safe_load(result.yaml_text)
        assert reloaded["interests"]["follow_entities"] == [trick]


def test_unicode_and_emoji_entries_round_trip() -> None:
    result = build(body(L_ENTITIES=["Äpfel 🍎", "日本語"]))
    assert yaml.safe_load(result.yaml_text)["interests"]["follow_entities"] == [
        "Äpfel 🍎",
        "日本語",
    ]


def test_bidi_and_zero_width_do_not_break_the_label_match() -> None:
    spoof = body().replace(f"### {o.L_ZONE}", f"### {o.L_ZONE}​")
    assert any(o.L_ZONE in p for p in problems(spoof))


def test_placeholder_text_counts_as_empty() -> None:
    assert build(body(L_ENTITIES=[o.NO_RESPONSE])).profile.interests.follow_entities == []


def test_write_atomic_replaces_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "profile.yaml"
    o.write_atomic(target, "one\n")
    o.write_atomic(target, "two\n")
    assert target.read_text(encoding="utf-8") == "two\n"
    assert [p.name for p in tmp_path.iterdir()] == ["profile.yaml"]


def test_now_is_used_for_the_cron() -> None:
    summer = datetime(2026, 7, 1, tzinfo=UTC)
    result = o.build(
        body(L_ZONE="Europe/Berlin", L_TIME="09:00"),
        now=summer,
        default_models=DEFAULT_MODELS,
        default_embedding_model="m",
    )
    assert result.cron == "0 7 * * *"


def test_unlisted_choice_is_refused_by_the_form_check_itself() -> None:
    found = problems(body(L_CHANNEL="sms"))
    assert found == [f"{o.L_CHANNEL}: not a listed option"]
