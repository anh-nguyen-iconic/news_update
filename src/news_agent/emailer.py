"""Send digests via Gmail SMTP (app password)."""
from __future__ import annotations

import logging
import smtplib
import time
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465


class EmailNotConfigured(RuntimeError):
    """Raised when Gmail credentials or recipient are missing."""


def build_message(settings, subject: str, html: str, text: str, to: str | None = None) -> EmailMessage:
    recipient = to or settings.digest_to
    if not settings.gmail_user or not settings.gmail_app_password:
        raise EmailNotConfigured("GMAIL_USER and GMAIL_APP_PASSWORD must be set to send email")
    if not recipient:
        raise EmailNotConfigured("No recipient: set DIGEST_TO (or GMAIL_USER) or pass `to`")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr(("News Agent", settings.gmail_user))
    msg["To"] = recipient
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=settings.gmail_user.split("@")[-1])
    msg.set_content(text or "", subtype="plain", charset="utf-8")
    msg.add_alternative(html, subtype="html", charset="utf-8")
    return msg


def send_email(
    settings,
    subject: str,
    html: str,
    text: str,
    to: str | None = None,
    smtp_factory: Callable[..., smtplib.SMTP] = smtplib.SMTP_SSL,
    retry_delay: float = 5.0,
) -> None:
    """Send a multipart/alternative (text + html) email via Gmail SMTP over SSL.

    Retries once on a transient ``smtplib.SMTPException`` / ``OSError``.
    Authentication errors are not retried.
    """
    msg = build_message(settings, subject, html, text, to)
    attempts = 2
    for attempt in range(1, attempts + 1):
        try:
            with smtp_factory(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
                smtp.login(settings.gmail_user, settings.gmail_app_password)
                smtp.send_message(msg)
            log.info("Sent email %r to %s", subject, msg["To"])
            return
        except smtplib.SMTPAuthenticationError:
            raise
        except (smtplib.SMTPException, OSError) as exc:
            if attempt >= attempts:
                raise
            log.warning("SMTP send failed (%s); retrying in %.0fs", exc, retry_delay)
            time.sleep(retry_delay)


def write_preview(path: Path, html: str) -> Path:
    """Write HTML to ``path`` (creating parent dirs) for --dry-run previews."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
