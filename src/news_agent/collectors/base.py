"""Shared helpers for collectors."""
from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Any, Protocol

import httpx

from ..models import Item


class ItemList(list):
    """A list of items that can also report partial failures (e.g. one RSS feed out of many).

    ``failures`` maps a sub-source id (e.g. ``rss:openai``) to an error message.
    """

    def __init__(self, items=(), failures: dict[str, str] | None = None):
        super().__init__(items)
        self.failures: dict[str, str] = failures or {}


class Collector(Protocol):
    def __call__(self, cfg: dict[str, Any], since: datetime, client: httpx.Client) -> list[Item]: ...


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str | None, max_len: int = 1200) -> str:
    """Strip HTML tags/entities, collapse whitespace and truncate."""
    if not text:
        return ""
    text = html.unescape(_TAG_RE.sub(" ", text))
    text = _WS_RE.sub(" ", text).strip()
    return text[: max_len - 1] + "…" if len(text) > max_len else text


def to_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def make_client(http_cfg: dict[str, Any] | None = None) -> httpx.Client:
    http_cfg = http_cfg or {}
    return httpx.Client(
        timeout=http_cfg.get("timeout_s", 20),
        follow_redirects=True,
        headers={"User-Agent": http_cfg.get("user_agent", "news-agent/0.1")},
    )
