"""Bluesky collector using the unauthenticated AppView ``searchPosts`` endpoint.

Note: ``public.api.bsky.app`` answers 403 for searchPosts, while ``api.bsky.app`` works without
auth, so the latter is the default ``api_base``.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from ..models import Item
from .base import ItemList, clean_text, to_utc

log = logging.getLogger(__name__)

API_BASE = "https://api.bsky.app"


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _external_links(post: dict[str, Any]) -> list[str]:
    record = post.get("record") or {}
    links: list[str] = []
    for embed in (record.get("embed"), post.get("embed")):
        if not embed:
            continue
        for e in (embed, embed.get("media") or {}):
            uri = (e.get("external") or {}).get("uri")
            if uri:
                links.append(uri)
    for facet in record.get("facets") or []:
        for feat in facet.get("features") or []:
            if feat.get("$type", "").endswith("#link") and feat.get("uri"):
                links.append(feat["uri"])
    return list(dict.fromkeys(links))


def post_url(uri: str, handle: str) -> str:
    return f"https://bsky.app/profile/{handle}/post/{uri.rstrip('/').rsplit('/', 1)[-1]}"


def parse_posts(data: dict[str, Any], since: datetime | None = None, min_likes: int = 0) -> list[Item]:
    items: list[Item] = []
    for p in (data or {}).get("posts", []):
        uri = p.get("uri")
        author = p.get("author") or {}
        handle = author.get("handle") or author.get("did")
        if not uri or not handle:
            continue
        likes = int(p.get("likeCount") or 0)
        if likes < min_likes:
            continue
        record = p.get("record") or {}
        published = _parse_dt(record.get("createdAt")) or _parse_dt(p.get("indexedAt"))
        if since is not None and published is not None and published < since:
            continue
        text = record.get("text") or ""
        ext_links = [u for u in _external_links(p) if u.startswith(("http://", "https://"))]
        one_line = " ".join(text.split())
        title = one_line if len(one_line) <= 120 else one_line[:119].rstrip() + "…"
        items.append(
            Item(
                key=f"bsky:{uri}",
                source="bluesky",
                source_name="Bluesky",
                kind="community",
                title=title or f"Post by @{handle}",
                url=post_url(uri, handle),
                summary=clean_text(text),
                authors=[author.get("displayName") or handle],
                published=published,
                tags=[f"link:{u}" for u in ext_links],
                metrics={
                    "likes": float(likes),
                    "reposts": float(p.get("repostCount") or 0),
                    "replies": float(p.get("replyCount") or 0),
                    "quotes": float(p.get("quoteCount") or 0),
                },
                links={"Linked article": ext_links[0]} if ext_links else {},
            )
        )
    return items


def collect(cfg: dict[str, Any], since: datetime, client: httpx.Client) -> list[Item]:
    url = f"{cfg.get('api_base', API_BASE).rstrip('/')}/xrpc/app.bsky.feed.searchPosts"
    min_likes = int(cfg.get("min_likes", 25))
    since_iso = since.strftime("%Y-%m-%dT%H:%M:%SZ")
    queries = cfg.get("queries", [])
    out = ItemList()
    seen: set[str] = set()
    for q in queries:
        params = {"q": q, "sort": cfg.get("sort", "top"), "since": since_iso, "limit": int(cfg.get("limit", 25))}
        try:
            resp = client.get(url, params=params)
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"Bluesky searchPosts at {url} requires authentication now (HTTP {resp.status_code}); "
                    "set bluesky.enabled: false or change bluesky.api_base"
                )
            resp.raise_for_status()
            got = parse_posts(resp.json(), since, min_likes)
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("bluesky query %r failed: %s", q, exc)
            out.failures[f"bluesky:{q}"] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        for it in got:
            if it.key not in seen:
                seen.add(it.key)
                out.append(it)
    if queries and len(out.failures) == len(queries):
        raise RuntimeError("all Bluesky queries failed")
    return out
