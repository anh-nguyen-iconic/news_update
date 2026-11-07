"""arXiv API collector (Atom feed of the newest submissions in the configured categories).

Since-handling: arXiv announces papers in daily batches (~20:00 US/Eastern) and the API only
exposes a paper once it is announced, so a paper's ``published`` timestamp is typically 6-30h
older than the moment it becomes visible (longer over weekends). We therefore keep entries whose
``published`` date is at/after ``since - grace_hours`` (default 24h). ``updated`` is deliberately
ignored so that revised versions (v2, v3, ...) of old papers do not resurface. Any repeats caused
by the grace window are removed by the dedupe store.
"""
from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import httpx

from ..models import Item
from .base import clean_text, to_utc

log = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
_VERSION_RE = re.compile(r"v\d+$")
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


def _struct_to_dt(st) -> datetime | None:
    return datetime(*st[:6], tzinfo=timezone.utc) if st else None


def arxiv_id(entry_id: str) -> str:
    """``http://arxiv.org/abs/2411.01234v2`` -> ``2411.01234`` (also handles old-style ids)."""
    tail = entry_id.split("/abs/", 1)[-1]
    return _VERSION_RE.sub("", tail)


def parse_affiliations(content: bytes | str) -> dict[str, list[str]]:
    """Map arXiv id -> unique ``<arxiv:affiliation>`` values in author order.

    feedparser keeps only the last affiliation of an entry, so the raw XML is read directly.
    """
    try:
        root = ET.fromstring(content.encode("utf-8") if isinstance(content, str) else content)
    except ET.ParseError:
        return {}
    out: dict[str, list[str]] = {}
    for entry in root.iter(f"{_ATOM}entry"):
        eid = (entry.findtext(f"{_ATOM}id") or "").strip()
        affs: list[str] = []
        for author in entry.iter(f"{_ATOM}author"):
            for aff in author.iter(f"{_ARXIV}affiliation"):
                name = clean_text(aff.text, 120)
                if name and name not in affs:
                    affs.append(name)
        if eid and affs:
            out[arxiv_id(eid)] = affs
    return out


def parse_feed(content: bytes | str, since: datetime | None = None) -> list[Item]:
    """Parse an arXiv API Atom response into Items, keeping entries published at/after ``since``.

    Fills ``affiliations`` from ``<arxiv:affiliation>``, adds a ``comment:<text>`` tag from
    ``<arxiv:comment>`` (often the venue, e.g. "Accepted by KDD 2026") and a PDF link.
    """
    parsed = feedparser.parse(content)
    affiliations = parse_affiliations(content) if parsed.entries else {}
    items: list[Item] = []
    for e in parsed.entries:
        if not e.get("id"):
            continue
        aid = arxiv_id(e.id)
        published = to_utc(_struct_to_dt(e.get("published_parsed") or e.get("updated_parsed")))
        if since is not None and published is not None and published < since:
            continue
        tags = [t.get("term") for t in e.get("tags", []) if t.get("term")]
        primary = (e.get("arxiv_primary_category") or {}).get("term")
        if primary and primary in tags:
            tags.remove(primary)
            tags.insert(0, primary)
        comment = clean_text(e.get("arxiv_comment"), 200)
        if comment:
            tags.append(f"comment:{comment}")
        affs = affiliations.get(aid)
        if affs is None and e.get("arxiv_affiliation"):
            affs = [clean_text(e["arxiv_affiliation"], 120)]
        items.append(
            Item(
                key=f"arxiv:{aid}",
                source="arxiv",
                source_name="arXiv",
                kind="paper",
                title=clean_text(e.get("title"), 300),
                url=f"https://arxiv.org/abs/{aid}",
                summary=clean_text(e.get("summary")),
                authors=[a.get("name") for a in e.get("authors", []) if a.get("name")],
                published=published,
                tags=tags,
                affiliations=affs or [],
                links={"PDF": f"https://arxiv.org/pdf/{aid}"},
            )
        )
    return items


def build_query(categories: list[str]) -> str:
    return " OR ".join(f"cat:{c}" for c in categories)


RSS_URL = "https://rss.arxiv.org/rss/"
RSS_TYPES = {"new", "cross"}  # skip "replace" / "replace-cross" (revisions of older papers)
_ABSTRACT = re.compile(r"^.*?Abstract:\s*", re.S)


def parse_rss(content: bytes | str) -> list[Item]:
    """Parse the arXiv RSS daily announcement (new + cross-listed papers only)."""
    parsed = feedparser.parse(content)
    items: list[Item] = []
    for e in parsed.entries:
        if (e.get("arxiv_announce_type") or "new").strip() not in RSS_TYPES:
            continue
        aid = arxiv_id(e.get("link") or e.get("id") or "")
        if not aid or not re.match(r"^\d{4}\.\d{4,5}$|^[a-z-]+(\.[A-Z]{2})?/\d{7}$", aid):
            continue
        authors = [a.strip() for a in (e.get("author") or "").split(",") if a.strip()]
        items.append(
            Item(
                key=f"arxiv:{aid}",
                source="arxiv",
                source_name="arXiv",
                kind="paper",
                title=clean_text(e.get("title"), 300),
                url=f"https://arxiv.org/abs/{aid}",
                summary=clean_text(_ABSTRACT.sub("", e.get("summary") or "")),
                authors=authors,
                published=to_utc(_struct_to_dt(e.get("published_parsed"))),
                tags=[t.get("term") for t in e.get("tags", []) if t.get("term")],
                links={"PDF": f"https://arxiv.org/pdf/{aid}"},
            )
        )
    return items


def collect_rss(cfg: dict[str, Any], client: httpx.Client) -> list[Item]:
    categories = cfg.get("categories") or ["cs.IR", "cs.LG"]
    resp = client.get(cfg.get("rss_url", RSS_URL) + "+".join(categories), timeout=float(cfg.get("timeout_s", 60)))
    resp.raise_for_status()
    items = parse_rss(resp.content)
    if not items:
        raise RuntimeError("arXiv RSS returned no new papers")
    return items


def collect(cfg: dict[str, Any], since: datetime, client: httpx.Client) -> list[Item]:
    """RSS (latest daily announcement; not rate-limited like the API) first, API as fallback.

    GitHub-hosted runners share IPs that the arXiv API often answers with 429. Each weekday run sees
    a different announcement, and the paper pool keeps relevant unshown papers for 30 days.
    """
    if cfg.get("mode", "rss") == "rss":
        try:
            return collect_rss(cfg, client)
        except Exception as exc:  # noqa: BLE001
            log.warning("arxiv: RSS failed (%s); falling back to the API", exc)
    return collect_api(cfg, since, client)


def collect_api(cfg: dict[str, Any], since: datetime, client: httpx.Client) -> list[Item]:
    categories = cfg.get("categories") or ["cs.IR", "cs.LG"]
    params = {
        "search_query": build_query(categories),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": int(cfg.get("max_results", 300)),
    }
    grace = timedelta(hours=float(cfg.get("grace_hours", 24)))
    url = cfg.get("api_url", API_URL)
    # arXiv occasionally answers 5xx or an empty feed under load: retry once after a pause.
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            resp = client.get(url, params=params, timeout=float(cfg.get("timeout_s", 60)))
            resp.raise_for_status()
            if feedparser.parse(resp.content).entries:
                return parse_feed(resp.content, since - grace)
            last_exc = RuntimeError("arXiv returned an empty feed")
        except httpx.HTTPError as exc:
            last_exc = exc
        if attempt == 0:
            log.info("arxiv: retrying after %s", last_exc)
            time.sleep(float(cfg.get("retry_sleep_s", 3)))
    raise RuntimeError(f"arXiv API failed: {last_exc}")
