from __future__ import annotations

import smtplib
from pathlib import Path

import pytest

from news_agent.config import Settings, load_yaml
from news_agent.emailer import EmailNotConfigured, send_email, write_preview

ROOT = Path(__file__).resolve().parents[1]


def make_settings(**kw) -> Settings:
    base = dict(root=ROOT, sources={}, interests=load_yaml(ROOT / "config" / "interests.yaml"),
                gmail_user="me@gmail.com", gmail_app_password="app-pw", digest_to="inbox@example.com")
    base.update(kw)
    return Settings(**base)


class FakeSMTP:
    instances: list["FakeSMTP"] = []
    fail_times = 0

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.logins: list[tuple[str, str]] = []
        self.sent = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, pw):
        self.logins.append((user, pw))

    def send_message(self, msg):
        if FakeSMTP.fail_times > 0:
            FakeSMTP.fail_times -= 1
            raise smtplib.SMTPServerDisconnected("boom")
        self.sent.append(msg)


@pytest.fixture(autouse=True)
def reset():
    FakeSMTP.instances = []
    FakeSMTP.fail_times = 0


def test_send_email_builds_multipart():
    send_email(make_settings(), "Subject ✓", "<p>Hello</p>", "Hello", smtp_factory=FakeSMTP)
    smtp = FakeSMTP.instances[0]
    assert (smtp.host, smtp.port) == ("smtp.gmail.com", 465)
    assert smtp.logins == [("me@gmail.com", "app-pw")]
    msg = smtp.sent[0]
    assert msg["Subject"] == "Subject ✓"
    assert msg["From"] == "News Agent <me@gmail.com>"
    assert msg["To"] == "inbox@example.com"
    assert msg.get_content_type() == "multipart/alternative"
    parts = list(msg.iter_parts())
    assert [p.get_content_type() for p in parts] == ["text/plain", "text/html"]
    assert "Hello" in parts[0].get_content()
    assert "<p>Hello</p>" in parts[1].get_content()


def test_send_email_to_override():
    send_email(make_settings(), "s", "<p>x</p>", "x", to="other@example.com", smtp_factory=FakeSMTP)
    assert FakeSMTP.instances[0].sent[0]["To"] == "other@example.com"


def test_send_email_retries_once():
    FakeSMTP.fail_times = 1
    send_email(make_settings(), "s", "<p>x</p>", "x", smtp_factory=FakeSMTP, retry_delay=0)
    assert len(FakeSMTP.instances) == 2
    assert len(FakeSMTP.instances[1].sent) == 1


def test_send_email_gives_up_after_retry():
    FakeSMTP.fail_times = 5
    with pytest.raises(smtplib.SMTPException):
        send_email(make_settings(), "s", "<p>x</p>", "x", smtp_factory=FakeSMTP, retry_delay=0)
    assert len(FakeSMTP.instances) == 2


@pytest.mark.parametrize("kw", [{"gmail_user": None}, {"gmail_app_password": None},
                                {"digest_to": None}])
def test_missing_config_raises(kw):
    with pytest.raises(EmailNotConfigured):
        send_email(make_settings(**kw), "s", "<p>x</p>", "x", smtp_factory=FakeSMTP)
    assert FakeSMTP.instances == []


def test_write_preview():
    import shutil

    base = ROOT / "out" / "_test_write_preview"
    try:
        p = write_preview(base / "a" / "b" / "preview.html", "<p>hi</p>")
        assert p.read_text(encoding="utf-8") == "<p>hi</p>"
    finally:
        shutil.rmtree(base, ignore_errors=True)
