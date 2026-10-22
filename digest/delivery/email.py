"""Email delivery (spec 6.3 step 9): SMTP over SSL (port 465) or STARTTLS (other ports), or a
dry-run that writes the message to a folder and touches no network. Credentials come from the
environment (GitHub Secrets) and are never logged; errors name the secret, not its value."""

from __future__ import annotations

import contextlib
import re
import smtplib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from typing import Protocol

from ..redact import redact, secret_values

_ADDRESS = re.compile(r"^[^@\s<>,;\"']+@[^@\s<>,;\"']+\.[^@\s<>,;\"']+$")
DEFAULT_HOST, DEFAULT_PORT = "smtp.gmail.com", 465
TIMEOUT = 30.0


class DeliveryError(Exception):
    """Delivery failed; the message is safe to print."""


class DeliveryConfigError(DeliveryError):
    """Missing or invalid delivery configuration."""


@dataclass(frozen=True)
class SmtpConfig:
    host: str
    port: int
    user: str
    password: str
    to: str


class SmtpClient(Protocol):
    def starttls(self) -> object: ...
    def login(self, user: str, password: str) -> object: ...
    def send_message(self, msg: EmailMessage) -> object: ...
    def quit(self) -> object: ...


SmtpFactory = Callable[[str, int, float], SmtpClient]


def default_factory(host: str, port: int, timeout: float) -> SmtpClient:
    if port == 465:
        return smtplib.SMTP_SSL(host, port, timeout=timeout)
    return smtplib.SMTP(host, port, timeout=timeout)


def smtp_config_from_env(env: Mapping[str, str]) -> SmtpConfig:
    missing = [n for n in ("SMTP_USER", "SMTP_PASS", "MAIL_TO") if not env.get(n)]
    if missing:
        raise DeliveryConfigError("missing secret(s): " + ", ".join(missing))
    to = env["MAIL_TO"].strip()
    if not _ADDRESS.match(to):
        raise DeliveryConfigError("MAIL_TO must be a single email address")
    port_raw = env.get("SMTP_PORT", "") or str(DEFAULT_PORT)
    if not port_raw.isdigit() or not 0 < int(port_raw) < 65536:
        raise DeliveryConfigError("SMTP_PORT must be a port number")
    return SmtpConfig(
        host=env.get("SMTP_HOST", "") or DEFAULT_HOST,
        port=int(port_raw),
        user=env["SMTP_USER"],
        password=env["SMTP_PASS"],
        to=to,
    )


def build_message(
    subject: str, text: str, html: str | None, *, sender: str, to: str
) -> EmailMessage:
    msg = EmailMessage()
    try:
        msg["Subject"] = subject.replace("\n", " ")
        msg["From"] = sender
        msg["To"] = to
    except ValueError as exc:
        raise DeliveryConfigError("invalid address or header") from exc
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    return msg


def send_email(
    msg: EmailMessage,
    cfg: SmtpConfig,
    *,
    factory: SmtpFactory = default_factory,
    env: Mapping[str, str] | None = None,
) -> None:
    secrets = secret_values(env or {}) + [cfg.password]
    try:
        client = factory(cfg.host, cfg.port, TIMEOUT)
        try:
            if cfg.port != 465:
                client.starttls()
            client.login(cfg.user, cfg.password)
            client.send_message(msg)
        finally:
            with contextlib.suppress(smtplib.SMTPException, OSError):
                client.quit()
    except smtplib.SMTPAuthenticationError:
        raise DeliveryError("SMTP login rejected: check SMTP_USER and SMTP_PASS") from None
    except (smtplib.SMTPException, OSError) as exc:
        raise DeliveryError(redact(f"SMTP error ({type(exc).__name__})", secrets)) from None


def write_outbox(msg: EmailMessage, html: str | None, outdir: Path, stem: str) -> list[Path]:
    """Dry run: nothing is sent. Writes <stem>.eml (and <stem>.html for a browser preview)."""
    outdir.mkdir(parents=True, exist_ok=True)
    eml = outdir / f"{stem}.eml"
    eml.write_bytes(bytes(msg))
    written = [eml]
    if html:
        page = outdir / f"{stem}.html"
        page.write_text(html, encoding="utf-8")
        written.append(page)
    return written
