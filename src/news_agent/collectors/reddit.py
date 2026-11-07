"""Reddit collector using the public ``top/.rss?t=day`` feed.

The ``.json`` endpoints return 403 without OAuth, while RSS works unauthenticated but is limited to
roughly one request per minute per IP. All subreddits are therefore fetched in a single multireddit
request (``/r/a+b+c/top/.rss``) and entries are grouped back per subreddit (top ``limit`` each).
On HTTP 429 we wait for ``x-ratelimit-reset`` (capped by ``max_wait_s``) and retry once.
RSS carries no score, so ``min_score`` cannot be applied: we keep the top posts of the day instead.
Configured subreddits with no posts in the day's top are simply absent (not failures).
"""
from __future__ import annotations

import html
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx

from ..models import Item
from .base import clean_text, to_utc

log = logging.getLogger(__name__)

FEED_URL = "https://www.reddit.com/r/{sub}/top/.rss"
_SUBMITTED_RE = re.compile(r"submitted by\s+/u/\S+.*$", re.S)
_SUB_RE = re.compile(r"/r/([^/]+)/")
_LINK_RE = re.compile(r'<a href="([^"]+)">\[link\]</a>')


def _entry_dt(entry) -> datetime | None:
    st = entry.get("published_parsed") or entry.get("updated_parsed")
    return to_utc(datetime(*st[:6], tzinfo=timezone.utc)) if st else None


def _entry_sub(entry) -> str | None:
    for t in entry.get("tags", []):
        if t.get("term"):
            return t["term"]
    m = _SUB_RE.search(entry.get("link") or "")
    return m.group(1) if m else None


def parse_feed(content: bytes | str, sub: str | None = None, since: datetime | None = None, limit: int = 10) -> list[Item]:
    """Parse a (multi)subreddit RSS feed, keeping at most ``limit`` posts per subreddit (feed order = top).

    ``sub`` forces the subreddit name; otherwise it is read from each entry's category.
    """
    parsed = feedparser.parse(content)
    if not parsed.entries and parsed.get("bozo"):
        raise ValueError(f"not a feed: {parsed.get('bozo_exception')}")
    items: list[Item] = []
    per_sub: dict[str, int] = {}
    for e in parsed.entries:
        permalink = e.get("link") or ""
        pid = (e.get("id") or "").removeprefix("t3_")
        name = sub or _entry_sub(e)
        if not permalink.startswith(("http://", "https://")) or not pid or not name or per_sub.get(name.lower(), 0) >= limit:
            continue
        published = _entry_dt(e)
        if since is not None and published is not None and published < since:
            continue
        per_sub[name.lower()] = per_sub.get(name.lower(), 0) + 1
        body = e.content[0].get("value", "") if e.get("content") else e.get("summary", "")
        tags, links = [], {}
        m = _LINK_RE.search(body)
        linked = html.unescape(m.group(1)) if m else ""
        if linked.startswith(("http://", "https://")) and not any(d in linked for d in ("reddit.com", "redd.it")):
            tags.append(f"link:{linked}")
            links["Linked article"] = linked
        author = (e.get("author") or "").removeprefix("/u/")
        items.append(
            Item(
                key=f"reddit:{pid}",
                source=f"reddit:{name}",
                source_name=f"r/{name}",
                kind="community",
                title=clean_text(e.get("title"), 300),
                url=permalink,
                summary=_SUBMITTED_RE.sub("", clean_text(body, 4000)).strip()[:1200],
                authors=[author] if author else [],
                published=published,
                tags=tags,
                links=links,
            )
        )
    return items


def _wait_seconds(resp: httpx.Response, cap: float) -> float:
    for header in ("retry-after", "x-ratelimit-reset"):
        try:
            return min(float(resp.headers[header]), cap)
        except (KeyError, ValueError):
            continue
    return 0.0


def collect(cfg: dict[str, Any], since: datetime, client: httpx.Client) -> list[Item]:
    subs = cfg.get("subreddits", [])
    if not subs:
        return []
    limit = int(cfg.get("limit", 10))
    url = FEED_URL.format(sub="+".join(subs))
    # Ask for enough entries to cover every subreddit; reddit caps RSS at 100.
    params = {"t": "day", "limit": min(100, max(25, limit * len(subs) * 3))}
    resp = client.get(url, params=params)
    if resp.status_code == 429:
        wait = _wait_seconds(resp, float(cfg.get("max_wait_s", 60)))
        log.info("reddit: rate limited, retrying in %.0fs", wait)
        time.sleep(wait + 1)
        resp = client.get(url, params=params)
    resp.raise_for_status()
    return parse_feed(resp.content, None, since, limit)
