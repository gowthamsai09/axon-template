from __future__ import annotations

import smtplib
from email import message_from_bytes
from email.message import EmailMessage
from pathlib import Path

import pytest

from digest.delivery.email import (
    DeliveryConfigError,
    DeliveryError,
    SmtpConfig,
    build_message,
    send_email,
    smtp_config_from_env,
    write_outbox,
)

ENV = {"SMTP_USER": "me@gmail.com", "SMTP_PASS": "app-password-123", "MAIL_TO": "you@example.com"}
CFG = SmtpConfig("smtp.example.com", 587, "me@gmail.com", "app-password-123", "you@example.com")


class FakeSmtp:
    log: list[str]

    def __init__(self, fail: Exception | None = None) -> None:
        self.log = []
        self.fail = fail

    def starttls(self) -> object:
        self.log.append("starttls")
        return None

    def login(self, user: str, password: str) -> object:
        self.log.append(f"login:{user}")
        if self.fail:
            raise self.fail
        return None

    def send_message(self, msg: EmailMessage) -> object:
        self.log.append(f"send:{msg['To']}")
        return None

    def quit(self) -> object:
        self.log.append("quit")
        return None


def msg() -> EmailMessage:
    return build_message("Subject", "plain", "<p>html</p>", sender=CFG.user, to=CFG.to)


def test_config_names_missing_secrets_only() -> None:
    with pytest.raises(DeliveryConfigError) as info:
        smtp_config_from_env({"SMTP_USER": "me@gmail.com"})
    assert "SMTP_PASS, MAIL_TO" in str(info.value) and "me@gmail.com" not in str(info.value)


def test_config_defaults_and_validation() -> None:
    cfg = smtp_config_from_env(ENV)
    assert (cfg.host, cfg.port) == ("smtp.gmail.com", 465)
    assert smtp_config_from_env({**ENV, "SMTP_HOST": "h", "SMTP_PORT": "587"}).port == 587
    for bad in ({"MAIL_TO": "a@b.c, d@e.f"}, {"MAIL_TO": "no-at-sign"}, {"SMTP_PORT": "abc"}):
        with pytest.raises(DeliveryConfigError):
            smtp_config_from_env({**ENV, **bad})


def test_message_is_multipart_and_rejects_header_injection() -> None:
    m = msg()
    assert m.get_body(("plain",)) is not None and m.get_body(("html",)) is not None
    with pytest.raises(DeliveryConfigError):
        build_message("s", "t", None, sender="a@b.c", to="x@y.z\nBcc: evil@example.com")
    assert "\n" not in build_message("a\nBcc: x", "t", None, sender="a@b.c", to="x@y.z")["Subject"]


def test_send_uses_starttls_on_587_and_always_quits() -> None:
    smtp = FakeSmtp()
    send_email(msg(), CFG, factory=lambda h, p, t: smtp)
    assert smtp.log == ["starttls", "login:me@gmail.com", "send:you@example.com", "quit"]


def test_no_starttls_on_465() -> None:
    smtp = FakeSmtp()
    cfg = SmtpConfig("h", 465, "me@gmail.com", "pw-pw-pw", "you@example.com")
    send_email(msg(), cfg, factory=lambda h, p, t: smtp)
    assert "starttls" not in smtp.log


def test_auth_failure_is_actionable_and_secret_free() -> None:
    smtp = FakeSmtp(smtplib.SMTPAuthenticationError(535, b"bad creds app-password-123"))
    with pytest.raises(DeliveryError) as info:
        send_email(msg(), CFG, factory=lambda h, p, t: smtp)
    assert "SMTP_USER and SMTP_PASS" in str(info.value) and "app-password-123" not in str(
        info.value
    )
    assert smtp.log[-1] == "quit"


def test_connection_error_is_scrubbed() -> None:
    def boom(h: str, p: int, t: float) -> FakeSmtp:
        raise OSError("cannot reach app-password-123")

    with pytest.raises(DeliveryError) as info:
        send_email(msg(), CFG, factory=boom, env=ENV)
    assert "app-password-123" not in str(info.value) and "OSError" in str(info.value)


def test_dry_run_writes_eml_and_html_without_network(tmp_path: Path) -> None:
    paths = write_outbox(msg(), "<p>html</p>", tmp_path / "out", "digest-x")
    assert [p.suffix for p in paths] == [".eml", ".html"]
    parsed = message_from_bytes(paths[0].read_bytes())
    assert parsed["To"] == "you@example.com" and parsed["Subject"] == "Subject"
