"""Persistent state: dedupe (`data/seen.json`) and scored item history (`data/items/`)."""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import Item, ScoredItem


def _now() -> datetime:
    return datetime.now(timezone.utc)


_TRACKING_PARAMS = {"ref", "ref_src", "source", "fbclid", "gclid", "mc_cid", "mc_eid"}
_TITLE_RE = re.compile(r"\W+")
MIN_TITLE_KEY = 12  # shorter normalised titles are too generic to treat as duplicates


def canonical_url(url: str) -> str:
    """Normalise a URL for duplicate detection: lowercase host, no www/fragment/tracking params/trailing slash."""
    parts = urlsplit(url.strip())
    host = parts.netloc.lower().removeprefix("www.")
    query = urlencode(
        [(k, v) for k, v in parse_qsl(parts.query) if not (k.lower().startswith("utm_") or k.lower() in _TRACKING_PARAMS)]
    )
    return urlunsplit(("https", host, parts.path.rstrip("/"), query, ""))


# When the same item comes from several sources, the first source in this list is kept as the base.
SOURCE_PRIORITY = ("arxiv", "hf_papers", "openalex")


def merge_duplicates(items: list[Item]) -> list[Item]:
    """Merge items sharing a key (e.g. one arXiv paper from arXiv, HF Papers and OpenAlex).

    The base copy comes from the highest-priority source; affiliations, links and tags are
    unioned, metrics take the max, and the longest summary wins.
    """
    def prio(it: Item) -> int:
        return SOURCE_PRIORITY.index(it.source) if it.source in SOURCE_PRIORITY else len(SOURCE_PRIORITY)

    groups: dict[str, list[Item]] = {}
    for it in items:
        groups.setdefault(it.key, []).append(it)
    out: list[Item] = []
    for group in groups.values():
        if len(group) == 1:
            out.append(group[0])
            continue
        group = sorted(group, key=prio)
        base = group[0].model_copy(deep=True)
        for other in group[1:]:
            base.affiliations = list(dict.fromkeys(base.affiliations + other.affiliations))
            base.tags = list(dict.fromkeys(base.tags + other.tags))
            for label, url in other.links.items():
                if url != base.url and url not in base.links.values():
                    base.links.setdefault(label, url)
            for k, v in other.metrics.items():
                base.metrics[k] = max(v, base.metrics.get(k, v))
            if len(other.summary) > len(base.summary):
                base.summary = other.summary
            if base.published is None:
                base.published = other.published
        out.append(base)
    return out


class SeenStore:
    """Tracks reported item keys.

    File format: ``{key: {"first_seen": iso8601, "stars": int | null}}``.
    Repos (key prefix ``repo:``) resurface when their stars reach
    ``resurface_multiplier`` × the star count when first reported.
    """

    def __init__(self, path: Path, resurface_multiplier: float = 2.0):
        self.path = path
        self.resurface_multiplier = resurface_multiplier
        self.data: dict[str, dict] = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8") or "{}")

    def is_new(self, item: Item) -> bool:
        entry = self.data.get(item.key)
        if entry is None:
            return True
        if item.key.startswith("repo:"):
            prev = entry.get("stars")
            stars = item.metrics.get("stars")
            if prev and stars and stars >= prev * self.resurface_multiplier:
                return True
        return False

    def filter_new(self, items: list[Item]) -> list[Item]:
        """Drop already-reported items and in-run duplicates (same key or same canonical URL).

        When two sources share a URL (e.g. an OpenAI post also on Hacker News), the
        primary source wins over community reposts.
        """
        ordered = sorted(items, key=lambda it: it.kind == "community")
        out, seen_ids = [], set()
        for it in ordered:
            ids = {it.key, canonical_url(it.url)}
            norm = _TITLE_RE.sub("", it.title.casefold()).replace("_", "")
            if len(norm) >= MIN_TITLE_KEY:
                ids.add("title:" + norm)
            if ids & seen_ids:
                continue
            seen_ids |= ids
            if self.is_new(it):
                out.append(it)
        return out

    def mark(self, items: list[Item]) -> None:
        now = _now().isoformat(timespec="seconds")
        for it in items:
            stars = it.metrics.get("stars")
            self.data[it.key] = {"first_seen": now, "stars": int(stars) if stars else None}

    def prune(self, days: int = 60) -> int:
        cutoff = _now() - timedelta(days=days)
        stale = [k for k, v in self.data.items() if datetime.fromisoformat(v["first_seen"]) < cutoff]
        for k in stale:
            del self.data[k]
        return len(stale)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True), encoding="utf-8")


class History:
    """Daily scored items, one JSON file per day: ``data/items/YYYY-MM-DD.json``."""

    def __init__(self, items_dir: Path):
        self.dir = items_dir

    def path_for(self, day: date) -> Path:
        return self.dir / f"{day.isoformat()}.json"

    def save(self, day: date, items: list[ScoredItem]) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(day)
        existing: dict[str, dict] = {}
        if path.exists():
            existing = {d["key"]: d for d in json.loads(path.read_text(encoding="utf-8"))}
        for it in items:
            existing[it.key] = json.loads(it.model_dump_json())
        path.write_text(json.dumps(list(existing.values()), indent=1, ensure_ascii=False), encoding="utf-8")
        return path

    def load_range(self, start: date, end: date) -> list[ScoredItem]:
        out: list[ScoredItem] = []
        day = start
        while day <= end:
            path = self.path_for(day)
            if path.exists():
                out.extend(ScoredItem.model_validate(d) for d in json.loads(path.read_text(encoding="utf-8")))
            day += timedelta(days=1)
        return out

    def load_recent(self, days: int, today: date | None = None) -> list[ScoredItem]:
        today = today or _now().date()
        return self.load_range(today - timedelta(days=days - 1), today)

    def find(self, item_id: str, lookback_days: int = 30) -> ScoredItem | None:
        for it in self.load_recent(lookback_days):
            if it.id == item_id:
                return it
        return None

    def prune(self, days: int = 90, today: date | None = None) -> int:
        today = today or _now().date()
        cutoff = today - timedelta(days=days)
        n = 0
        for p in self.dir.glob("*.json"):
            try:
                if date.fromisoformat(p.stem) < cutoff:
                    p.unlink()
                    n += 1
            except ValueError:
                continue
        return n


class PaperPool:
    """Relevant papers not yet shown, kept for a rolling window so arXiv (fetched daily) competes for 30 days.

    File format: ``{key: {"first_seen": "YYYY-MM-DD", "item": <Item JSON>}}``.
    """

    def __init__(self, path: Path, rolling_days: int = 30):
        self.path = path
        self.rolling_days = rolling_days
        self.data: dict[str, dict] = {}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8") or "{}")

    def items(self) -> list[Item]:
        out = []
        for entry in self.data.values():
            try:
                out.append(Item.model_validate(entry["item"]))
            except Exception:  # noqa: BLE001 - skip entries from an older schema
                continue
        return out

    def merge_into(self, collected: list[Item]) -> list[Item]:
        """Today's copies win (fresh metrics); pooled papers fill in the rest."""
        keys = {it.key for it in collected}
        return collected + [it for it in self.items() if it.key not in keys]

    def update(self, ranked: list[ScoredItem], shown: set[str], today: date, min_score: float) -> None:
        fields = set(Item.model_fields)
        for it in ranked:
            if it.kind != "paper":
                continue
            if it.key in shown:
                self.data.pop(it.key, None)
            elif it.score >= min_score:
                prev = self.data.get(it.key, {})
                self.data[it.key] = {
                    "first_seen": prev.get("first_seen", today.isoformat()),
                    "item": json.loads(it.model_dump_json(include=fields)),
                }
        cutoff = today - timedelta(days=self.rolling_days)
        for key in [k for k, v in self.data.items() if date.fromisoformat(v["first_seen"]) < cutoff]:
            del self.data[key]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
