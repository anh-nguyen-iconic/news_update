"""RSS/Atom collector for blogs, news sites and newsletters.

Each feed in ``cfg["feeds"]`` supports: ``id``, ``name``, ``url``, ``kind`` (default ``blog``),
``enabled`` (default true), ``fallback_urls`` (tried in order when ``url`` fails) and
``max_items`` (overrides ``cfg["max_items_per_feed"]``).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import feedparser
import httpx

from ..models import Item
from .base import ItemList, clean_text, to_utc

log = logging.getLogger(__name__)

_TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "ref", "ref_src", "referrer", "source",
    "cmpid", "ncid", "sr_share", "_hsenc", "_hsmi", "mkt_tok", "igshid", "yclid", "guccounter", "_sm_nck",
}
_VALID_KINDS = {"paper", "blog", "news", "model", "repo", "community"}


def canonical_url(url: str) -> str:
    """Normalise a link for dedupe: lowercase host, drop fragment, tracking params and trailing slash."""
    url = (url or "").strip()
    if not url:
        return url
    parts = urlsplit(url)
    query = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith(("utm_", "at_", "pk_")) and k.lower() not in _TRACKING_PARAMS
    ]
    path = parts.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    scheme = (parts.scheme or "https").lower()
    if scheme == "http":
        scheme = "https"
    return urlunsplit((scheme, parts.netloc.lower(), path, urlencode(query), ""))


def _entry_dt(entry) -> datetime | None:
    st = entry.get("published_parsed") or entry.get("updated_parsed") or entry.get("created_parsed")
    return to_utc(datetime(*st[:6], tzinfo=timezone.utc)) if st else None


def _entry_summary(entry) -> str:
    text = entry.get("summary") or ""
    if not text and entry.get("content"):
        text = entry.content[0].get("value", "")
    return clean_text(text)


def parse_feed(content: bytes | str, feed_cfg: dict[str, Any], since: datetime | None = None, max_items: int = 20) -> list[Item]:
    """Parse one feed body into Items (newest first, at most ``max_items``, published at/after ``since``).

    Undated entries are kept (the dedupe store prevents repeats). Raises ``ValueError`` when the
    body is not a feed with entries.
    """
    parsed = feedparser.parse(content)
    if not parsed.entries:
        reason = f": {parsed.get('bozo_exception')}" if parsed.get("bozo") else ""
        raise ValueError(f"no feed entries parsed{reason}")
    fid = feed_cfg["id"]
    name = feed_cfg.get("name", fid)
    kind = feed_cfg.get("kind", "blog")
    if kind not in _VALID_KINDS:
        kind = "blog"
    entries = [(e, _entry_dt(e)) for e in parsed.entries]
    oldest = datetime.min.replace(tzinfo=timezone.utc)
    entries.sort(key=lambda p: p[1] or oldest, reverse=True)
    items: list[Item] = []
    seen: set[str] = set()
    for e, published in entries:
        if len(items) >= max_items:
            break
        if since is not None and published is not None and published < since:
            continue
        link = e.get("link") or ""
        if not link:
            continue
        url = canonical_url(link)
        public_url = link if link.startswith(("http://", "https://")) else url
        if url in seen or not public_url.startswith(("http://", "https://")) or not urlsplit(public_url).netloc:
            continue
        seen.add(url)
        authors = [a.get("name") for a in e.get("authors", []) if a.get("name")] or ([e.author] if e.get("author") else [])
        items.append(
            Item(
                key=f"url:{url}",
                source=f"rss:{fid}",
                source_name=name,
                kind=kind,
                title=clean_text(e.get("title") or url, 300),
                url=public_url,
                summary=_entry_summary(e),
                authors=authors,
                published=published,
                tags=[t.get("term") for t in e.get("tags", []) if t.get("term")][:8],
            )
        )
    return items


def fetch_feed(feed_cfg: dict[str, Any], since: datetime, client: httpx.Client, max_items: int) -> list[Item]:
    urls = [feed_cfg["url"], *(feed_cfg.get("fallback_urls") or [])]
    errors: list[str] = []
    for url in urls:
        try:
            resp = client.get(url)
            resp.raise_for_status()
            return parse_feed(resp.content, feed_cfg, since, max_items)
        except (httpx.HTTPError, ValueError) as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}"[:200])
    raise RuntimeError(" | ".join(errors))


def collect(cfg: dict[str, Any], since: datetime, client: httpx.Client) -> list[Item]:
    feeds = [f for f in cfg.get("feeds", []) if f.get("enabled", True) and f.get("url") and f.get("id")]
    default_max = int(cfg.get("max_items_per_feed", 20))
    out = ItemList()
    if not feeds:
        return out
    with ThreadPoolExecutor(max_workers=int(cfg.get("max_workers", 6))) as pool:
        futures = {
            f["id"]: pool.submit(fetch_feed, f, since, client, int(f.get("max_items", default_max))) for f in feeds
        }
        for fid, fut in futures.items():
            try:
                got = fut.result()
                log.debug("rss:%s -> %d items", fid, len(got))
                out.extend(got)
            except Exception as exc:  # noqa: BLE001 - per-feed fail-soft
                log.warning("rss:%s failed: %s", fid, exc)
                out.failures[f"rss:{fid}"] = str(exc)[:300]
    if len(out.failures) == len(feeds):
        raise RuntimeError(f"all {len(feeds)} RSS feeds failed")
    return out
