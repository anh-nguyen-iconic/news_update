"""goodailist.com collector: fast-rising and newly discovered AI GitHub repos."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from ..models import Item
from .base import clean_text, to_utc

BASE_URL = "https://goodailist.com/api/repos"


def _num(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _split(value: str | None) -> list[str]:
    return [p.strip() for p in (value or "").split(",") if p.strip()]


def _date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _richness(r: dict[str, Any]) -> int:
    return sum(r.get(k) is not None for k in ("star_1d", "star_7d", "star_7d_pct"))


def merge_repos(*payloads: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Merge several ``/api/repos`` responses by lowercase repo name (prefers the most complete record)."""
    merged: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for r in (payload or {}).get("repos", []) or []:
            name = (r.get("repo") or "").strip()
            if "/" not in name:
                continue
            key = name.lower()
            prev = merged.get(key)
            if prev is None or _richness(r) > _richness(prev):
                rec = dict(r)
                if prev is not None:
                    rec["is_new"] = bool(prev.get("is_new")) or bool(r.get("is_new"))
                merged[key] = rec
            else:
                prev["is_new"] = bool(prev.get("is_new")) or bool(r.get("is_new"))
    return merged


def parse_repos(
    *payloads: dict[str, Any], min_star_1d: float = 300, min_star_7d_pct: float = 50
) -> list[Item]:
    items: list[Item] = []
    for key, r in merge_repos(*payloads).items():
        is_new = bool(r.get("is_new"))
        star_1d, star_7d_pct = _num(r.get("star_1d")), _num(r.get("star_7d_pct"))
        if not (is_new or star_1d >= min_star_1d or star_7d_pct >= min_star_7d_pct):
            continue
        name = r["repo"].strip()
        tags = [r.get("category")] + _split(r.get("subcat")) + _split(r.get("keywords")) + [r.get("language")]
        tags = list(dict.fromkeys(t for t in tags if t))
        homepage = str(r.get("homepage") or "").strip()
        items.append(
            Item(
                key=f"repo:{key}",
                source="goodailist",
                source_name="GoodAIList",
                kind="repo",
                title=name,
                url=f"https://github.com/{name}",
                summary=clean_text(r.get("description")),
                authors=[name.split("/", 1)[0]],
                published=_date(r.get("created_at")) or _date(r.get("first_seen")),
                tags=tags,
                metrics={
                    "stars": _num(r.get("stars")),
                    "star_1d": star_1d,
                    "star_7d": _num(r.get("star_7d")),
                    "star_7d_pct": star_7d_pct,
                    "forks": _num(r.get("forks")),
                    "is_new": 1.0 if is_new else 0.0,
                },
                links={"Homepage": homepage} if homepage.startswith(("http://", "https://")) else {},
            )
        )
    return items


def collect(cfg: dict[str, Any], since: datetime, client: httpx.Client) -> list[Item]:
    # No `since` filtering: trending repos are evaluated daily; the dedupe store handles repeats.
    base = cfg.get("base_url", BASE_URL)
    limit = int(cfg.get("limit", 100))
    payloads = []
    for params in ({"sort": "star_1d", "limit": limit}, {"is_new": 1, "limit": limit}):
        resp = client.get(base, params=params)
        resp.raise_for_status()
        payloads.append(resp.json())
    return parse_repos(
        *payloads,
        min_star_1d=float(cfg.get("min_star_1d", 300)),
        min_star_7d_pct=float(cfg.get("min_star_7d_pct", 50)),
    )
