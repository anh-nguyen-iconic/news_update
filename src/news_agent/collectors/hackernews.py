"""Hacker News collector via the Algolia search API."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from ..models import Item
from .base import ItemList, clean_text

log = logging.getLogger(__name__)

API = "https://hn.algolia.com/api/v1"


def hn_item_url(object_id: str) -> str:
    return f"https://news.ycombinator.com/item?id={object_id}"


def parse_hits(data: dict[str, Any], since: datetime | None = None) -> list[Item]:
    items: list[Item] = []
    for h in (data or {}).get("hits", []):
        oid = str(h.get("objectID") or h.get("story_id") or "")
        title = h.get("title") or h.get("story_title")
        if not oid or not title:
            continue
        ts = h.get("created_at_i")
        published = datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
        if since is not None and published is not None and published < since:
            continue
        discussion = hn_item_url(oid)
        ext = h.get("url") if str(h.get("url") or "").startswith(("http://", "https://")) else None
        text = clean_text(h.get("story_text"), 1000)
        summary = f"{text} (HN discussion: {discussion})" if text else f"HN discussion: {discussion}"
        items.append(
            Item(
                key=f"hn:{oid}",
                source="hn",
                source_name="Hacker News",
                kind="community",
                title=clean_text(title, 300),
                url=ext or discussion,
                summary=summary,
                authors=[h["author"]] if h.get("author") else [],
                published=published,
                tags=[f"hn_discussion:{discussion}"],
                metrics={"points": float(h.get("points") or 0), "comments": float(h.get("num_comments") or 0)},
                links={"HN discussion": discussion} if ext else {},
            )
        )
    return items


def collect(cfg: dict[str, Any], since: datetime, client: httpx.Client) -> list[Item]:
    min_points = int(cfg.get("min_points", 50))
    ts = int(since.timestamp())
    requests: list[tuple[str, str, dict[str, Any]]] = [
        ("front_page", f"{API}/search", {"tags": "front_page", "hitsPerPage": int(cfg.get("front_page_limit", 50))})
    ]
    for q in cfg.get("queries", []):
        requests.append((q, f"{API}/search_by_date", {
            "query": q,
            "tags": "story",
            "numericFilters": f"created_at_i>{ts},points>={min_points}",
            "hitsPerPage": int(cfg.get("per_query_limit", 20)),
        }))
    out = ItemList()
    seen: set[str] = set()
    for label, url, params in requests:
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            got = parse_hits(resp.json(), since)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("hn query %r failed: %s", label, exc)
            out.failures[f"hn:{label}"] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        for it in got:
            if it.key not in seen:
                seen.add(it.key)
                out.append(it)
    if len(out.failures) == len(requests):
        raise RuntimeError("all Hacker News queries failed")
    return out
