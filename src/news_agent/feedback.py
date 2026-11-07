"""Thumbs up/down feedback: persistent topic/source multipliers and Gmail ingestion."""
from __future__ import annotations

import email
import imaplib
import json
import logging
import re
from collections import Counter
from datetime import date, datetime, timezone
from email.header import decode_header, make_header
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .models import ScoredItem

if TYPE_CHECKING:
    from .config import Settings
    from .store import History

log = logging.getLogger(__name__)

DEFAULT_TAG = "[news-agent-feedback]"
_UP = {"up", "+1", "👍", "👍🏻", "👍🏼", "👍🏽", "👍🏾", "👍🏿"}
_DOWN = {"down", "-1", "👎", "👎🏻", "👎🏼", "👎🏽", "👎🏾", "👎🏿"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _as_aware(value: datetime | date) -> datetime:
    if not isinstance(value, datetime):
        value = datetime(value.year, value.month, value.day)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


class FeedbackStore:
    """Persists votes and the derived multipliers in ``data/feedback.json``."""

    def __init__(self, path: Path, cfg: dict | None = None):
        cfg = cfg or {}
        self.path = Path(path)
        self.step = float(cfg.get("step", 0.05))
        self.min_multiplier = float(cfg.get("min_multiplier", 0.5))
        self.max_multiplier = float(cfg.get("max_multiplier", 2.0))
        self.subject_tag = cfg.get("subject_tag", DEFAULT_TAG)
        self.data: dict[str, Any] = {"multipliers": {"topics": {}, "sources": {}}, "votes": []}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Could not read %s (%s); starting with empty feedback", self.path, exc)
                loaded = {}
            mult = loaded.get("multipliers") or {}
            self.data["multipliers"]["topics"].update(mult.get("topics") or {})
            self.data["multipliers"]["sources"].update(mult.get("sources") or {})
            self.data["votes"] = list(loaded.get("votes") or [])

    @property
    def multipliers(self) -> dict:
        return self.data["multipliers"]

    @property
    def votes(self) -> list[dict]:
        return self.data["votes"]

    def topic_multiplier(self, topic_id: str) -> float:
        return float(self.multipliers["topics"].get(topic_id, 1.0))

    def source_multiplier(self, source: str) -> float:
        return float(self.multipliers["sources"].get(source, 1.0))

    def _clamp(self, value: float) -> float:
        return max(self.min_multiplier, min(self.max_multiplier, value))

    def has_vote(self, item_id: str, direction: str) -> bool:
        return any(v.get("item_id") == item_id and v.get("direction") == direction for v in self.votes)

    def apply_vote(self, item: ScoredItem, direction: str) -> bool:
        """Apply an up/down vote. Returns False if ignored (duplicate or invalid direction)."""
        direction = direction.lower()
        if direction not in ("up", "down"):
            raise ValueError(f"invalid vote direction: {direction!r}")
        if self.has_vote(item.id, direction):
            log.info("Ignoring duplicate %s vote for %s", direction, item.id)
            return False
        factor = 1 + self.step if direction == "up" else 1 - self.step
        topics = list(dict.fromkeys(item.topics or item.matched_topics))
        tmult = self.multipliers["topics"]
        for t in topics:
            tmult[t] = round(self._clamp(tmult.get(t, 1.0) * factor), 4)
        smult = self.multipliers["sources"]
        smult[item.source] = round(self._clamp(smult.get(item.source, 1.0) * factor), 4)
        self.votes.append(
            {
                "ts": _now().isoformat(timespec="seconds"),
                "item_id": item.id,
                "direction": direction,
                "topics": topics,
                "source": item.source,
                "title": item.title,
            }
        )
        return True

    def _votes_since(self, since: datetime | date | None) -> list[dict]:
        if since is None:
            return list(self.votes)
        cutoff = _as_aware(since)
        out = []
        for v in self.votes:
            try:
                ts = _as_aware(datetime.fromisoformat(v["ts"]))
            except (KeyError, ValueError):
                continue
            if ts >= cutoff:
                out.append(v)
        return out

    def stats(self, since: datetime | date | None = None) -> dict:
        votes = self._votes_since(since)
        up = [v for v in votes if v.get("direction") == "up"]
        down = [v for v in votes if v.get("direction") == "down"]
        up_topics = Counter(t for v in up for t in v.get("topics", []))
        down_topics = Counter(t for v in down for t in v.get("topics", []))
        return {
            "up": len(up),
            "down": len(down),
            "multipliers": json.loads(json.dumps(self.multipliers)),
            "top_up_topics": [t for t, _ in sorted(up_topics.items(), key=lambda kv: (-kv[1], kv[0]))[:5]],
            "top_down_topics": [t for t, _ in sorted(down_topics.items(), key=lambda kv: (-kv[1], kv[0]))[:5]],
        }

    def prompt_context(self) -> str:
        if not self.votes:
            return ""

        def fmt(d: dict[str, float], liked: bool) -> str:
            sel = [(k, m) for k, m in d.items() if (m > 1.0 if liked else m < 1.0)]
            sel.sort(key=lambda kv: (-kv[1], kv[0]) if liked else (kv[1], kv[0]))
            return ", ".join(f"{k} (x{m:.2f})" for k, m in sel[:8])

        lines = ["Reader feedback so far (thumbs up/down on past digest items):"]
        for label, value in (
            ("Liked topics", fmt(self.multipliers["topics"], True)),
            ("Disliked topics", fmt(self.multipliers["topics"], False)),
            ("Liked sources", fmt(self.multipliers["sources"], True)),
            ("Disliked sources", fmt(self.multipliers["sources"], False)),
        ):
            if value:
                lines.append(f"- {label}: {value}")
        recent_up = [v["title"] for v in self.votes if v.get("direction") == "up" and v.get("title")][-5:]
        recent_down = [v["title"] for v in self.votes if v.get("direction") == "down" and v.get("title")][-5:]
        if recent_up:
            lines.append("- Recently liked: " + " | ".join(recent_up))
        if recent_down:
            lines.append("- Recently disliked: " + " | ".join(recent_down))
        lines.append("Nudge scores towards what the reader liked and away from what they disliked.")
        return "\n".join(lines)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")


def parse_feedback_subject(subject: str, tag: str = DEFAULT_TAG) -> tuple[str, str] | None:
    """Parse ``"[news-agent-feedback] up 287ce25f6b17"`` → ``("up", "287ce25f6b17")``."""
    if not subject:
        return None
    subject = " ".join(subject.split())
    pattern = re.escape(" ".join(tag.split())) + r"\s*(\S+)\s+([0-9a-fA-F]{6,40})\b"
    m = re.search(pattern, subject, flags=re.IGNORECASE)
    if not m:
        return None
    word = m.group(1).lower().strip(":")
    if word in _UP:
        direction = "up"
    elif word in _DOWN:
        direction = "down"
    else:
        return None
    return direction, m.group(2).lower()


def _decode_subject(raw: bytes | str) -> str:
    if isinstance(raw, bytes):
        msg = email.message_from_bytes(raw)
    else:
        msg = email.message_from_string(raw)
    value = msg.get("Subject", "")
    try:
        return str(make_header(decode_header(value)))
    except Exception:  # noqa: BLE001
        return str(value)


def _fetch_subject(imap: Any, num: bytes | str) -> str:
    typ, data = imap.fetch(num, "(BODY.PEEK[HEADER.FIELDS (SUBJECT)])")
    if typ != "OK" or not data:
        return ""
    for part in data:
        if isinstance(part, tuple) and len(part) >= 2:
            return _decode_subject(part[1])
    return ""


def ingest_from_gmail(
    settings: "Settings",
    store: FeedbackStore,
    history: "History",
    imap_factory: Callable[[], Any] | None = None,
) -> int:
    """Read unread feedback mails from INBOX, apply votes, mark them read and archive them.

    Fail-soft: returns 0 on missing credentials or any IMAP error.
    """
    if not (settings.gmail_user and settings.gmail_app_password):
        log.info("Gmail credentials not set; skipping feedback ingestion")
        return 0
    tag = (settings.interests.get("feedback") or {}).get("subject_tag", store.subject_tag)
    factory = imap_factory or (lambda: imaplib.IMAP4_SSL("imap.gmail.com"))
    applied = 0
    imap = None
    try:
        imap = factory()
        imap.login(settings.gmail_user, settings.gmail_app_password)
        typ, _ = imap.select("INBOX")
        if typ != "OK":
            raise imaplib.IMAP4.error("could not select INBOX")
        quoted = '"' + tag.replace("\\", "\\\\").replace('"', '\\"') + '"'
        typ, data = imap.search(None, "UNSEEN", "SUBJECT", quoted)
        if typ != "OK":
            raise imaplib.IMAP4.error("search failed")
        nums = data[0].split() if data and data[0] else []
        processed = []
        for num in nums:
            subject = _fetch_subject(imap, num)
            parsed = parse_feedback_subject(subject, tag)
            if parsed is None:
                log.info("Skipping feedback mail with unparseable subject: %r", subject)
                continue
            direction, item_id = parsed
            item = history.find(item_id)
            if item is None:
                log.warning("Feedback for unknown item id %s (not in recent history)", item_id)
            elif store.apply_vote(item, direction):
                applied += 1
            processed.append(num)
        for num in processed:
            imap.store(num, "+FLAGS", "(\\Seen \\Deleted)")
        if processed:
            # In Gmail, removing a message from INBOX via \Deleted + EXPUNGE archives it.
            imap.expunge()
        log.info("Feedback ingestion: %d mails processed, %d votes applied", len(processed), applied)
    except Exception as exc:  # noqa: BLE001 - fail-soft by design
        log.warning("Feedback ingestion from Gmail failed: %s", exc)
    finally:
        if imap is not None:
            try:
                imap.logout()
            except Exception:  # noqa: BLE001
                pass
    return applied
