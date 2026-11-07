import imaplib
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from email.header import Header

import pytest

from news_agent.config import load_settings
from news_agent.feedback import FeedbackStore, ingest_from_gmail, parse_feedback_subject
from news_agent.models import ScoredItem
from news_agent.store import History

TAG = "[news-agent-feedback]"
CFG = {"step": 0.5, "min_multiplier": 0.5, "max_multiplier": 2.0, "subject_tag": TAG}


def si(key="arxiv:1", topics=("recsys",), source="arxiv"):
    return ScoredItem(key=key, source=source, source_name=source, kind="paper", title=f"T {key}",
                      url="https://x", topics=list(topics), score=8, section="papers")


@pytest.mark.parametrize("subject,expected", [
    ("[news-agent-feedback] up 287ce25f6b17", ("up", "287ce25f6b17")),
    ("Re: [News-Agent-Feedback]   DOWN   287CE25F6B17 ", ("down", "287ce25f6b17")),
    ("RE: Fwd: [news-agent-feedback] 👍 287ce25f6b17", ("up", "287ce25f6b17")),
    ("[news-agent-feedback] -1 287ce25f6b17", ("down", "287ce25f6b17")),
    ("[news-agent-feedback] +1 287ce25f6b17", ("up", "287ce25f6b17")),
    ("[news-agent-feedback] maybe 287ce25f6b17", None),
    ("hello up 287ce25f6b17", None),
    ("", None),
])
def test_parse_subject(subject, expected):
    assert parse_feedback_subject(subject, TAG) == expected


def test_apply_vote_clamps_and_dedupes(tmp_path):
    fb = FeedbackStore(tmp_path / "fb.json", CFG)
    a, b, c = si("a"), si("b"), si("c")
    assert fb.apply_vote(a, "up")
    assert fb.topic_multiplier("recsys") == 1.5 and fb.source_multiplier("arxiv") == 1.5
    assert not fb.apply_vote(a, "up")  # duplicate ignored
    assert fb.topic_multiplier("recsys") == 1.5
    fb.apply_vote(b, "up")
    fb.apply_vote(c, "up")
    assert fb.topic_multiplier("recsys") == 2.0  # clamped
    for k in "defgh":
        fb.apply_vote(si(k), "down")
    assert fb.topic_multiplier("recsys") == 0.5
    assert fb.topic_multiplier("unknown") == 1.0
    st = fb.stats()
    assert st["up"] == 3 and st["down"] == 5 and st["top_up_topics"] == ["recsys"]


def test_matched_topics_used_when_no_topics(tmp_path):
    fb = FeedbackStore(tmp_path / "fb.json", CFG)
    it = si(topics=())
    it.matched_topics = ["fashion"]
    fb.apply_vote(it, "down")
    assert fb.topic_multiplier("fashion") == 0.5


def test_persistence_and_prompt_context(tmp_path):
    path = tmp_path / "sub" / "fb.json"
    fb = FeedbackStore(path, CFG)
    assert fb.prompt_context() == ""
    fb.apply_vote(si("a"), "up")
    fb.apply_vote(si("b", topics=["ai_industry"], source="hn"), "down")
    fb.save()
    fb2 = FeedbackStore(path, CFG)
    assert fb2.multipliers == fb.multipliers and len(fb2.votes) == 2
    ctx = fb2.prompt_context()
    assert "recsys" in ctx and "ai_industry" in ctx and "hn" in ctx
    assert fb2.stats(since=datetime.now(timezone.utc) + timedelta(hours=1))["up"] == 0
    assert fb2.stats(since=date.today() - timedelta(days=1))["up"] == 1


class FakeIMAP:
    def __init__(self, subjects, fail_on=None):
        self.subjects = subjects
        self.fail_on = fail_on
        self.stored, self.expunged, self.logged_out, self.search_args = [], False, False, None

    def _maybe_fail(self, op):
        if self.fail_on == op:
            raise imaplib.IMAP4.error(f"{op} failed")

    def login(self, user, pw):
        self._maybe_fail("login")
        return "OK", [b""]

    def select(self, box):
        assert box == "INBOX"
        return "OK", [str(len(self.subjects)).encode()]

    def search(self, charset, *criteria):
        self.search_args = criteria
        return "OK", [" ".join(str(i + 1) for i in range(len(self.subjects))).encode()]

    def fetch(self, num, what):
        subj = self.subjects[int(num) - 1]
        raw = f"Subject: {Header(subj, 'utf-8').encode()}\r\n\r\n".encode()
        return "OK", [(b"1 (BODY[HEADER.FIELDS (SUBJECT)] {40}", raw), b")"]

    def store(self, num, op, flags):
        self.stored.append((num, op, flags))
        return "OK", []

    def expunge(self):
        self.expunged = True
        return "OK", []

    def logout(self):
        self.logged_out = True


def _settings(tmp_path, user="me@gmail.com", pw="secret"):
    s = load_settings()
    return replace(s, root=tmp_path, gmail_user=user, gmail_app_password=pw)


def test_ingest_from_gmail(tmp_path):
    s = _settings(tmp_path)
    history = History(tmp_path / "items")
    a, b = si("a"), si("b", topics=["fashion"])
    history.save(date.today(), [a, b])
    fb = FeedbackStore(tmp_path / "fb.json", s.interests["feedback"])
    imap = FakeIMAP([
        f"{TAG} up {a.id}",
        f"Re: {TAG} down {b.id}",
        f"{TAG} up {a.id}",  # duplicate
        f"{TAG} up 000000000000",  # unknown id
        f"{TAG} 👎 nothex",  # unparseable -> left alone
    ])
    n = ingest_from_gmail(s, fb, history, imap_factory=lambda: imap)
    assert n == 2
    assert fb.topic_multiplier("recsys") > 1 and fb.topic_multiplier("fashion") < 1
    assert [x[0] for x in imap.stored] == [b"1", b"2", b"3", b"4"]
    assert all("\\Seen" in x[2] and "\\Deleted" in x[2] for x in imap.stored)
    assert imap.expunged and imap.logged_out
    assert "UNSEEN" in imap.search_args and any(TAG in str(c) for c in imap.search_args)


def test_ingest_missing_credentials(tmp_path):
    s = _settings(tmp_path, user=None, pw=None)
    fb = FeedbackStore(tmp_path / "fb.json", {})

    def boom():
        raise AssertionError("should not connect")

    assert ingest_from_gmail(s, fb, History(tmp_path / "items"), imap_factory=boom) == 0


def test_ingest_imap_error_is_soft(tmp_path):
    s = _settings(tmp_path)
    fb = FeedbackStore(tmp_path / "fb.json", {})
    imap = FakeIMAP([], fail_on="login")
    assert ingest_from_gmail(s, fb, History(tmp_path / "items"), imap_factory=lambda: imap) == 0
    assert imap.logged_out
