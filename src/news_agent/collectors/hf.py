"""Hugging Face collectors: Daily Papers and trending models."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from ..models import Item
from .base import ItemList, clean_text, to_utc

log = logging.getLogger(__name__)

PAPERS_URL = "https://huggingface.co/api/daily_papers"
MODELS_URL = "https://huggingface.co/api/models"

# Tags that carry no information for a reader.
_NOISE_TAG_PREFIXES = ("region:", "endpoints_compatible", "autotrain_compatible", "text-generation-inference",
                       "deploy:", "eval-results", "custom_code", "safetensors", "arxiv:", "base_model:", "doi:")


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return to_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


# --------------------------------------------------------------------------- papers

def _paper_affiliations(entry: dict[str, Any], paper: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for org in (entry.get("organization"), paper.get("organization")):
        if not isinstance(org, dict):
            continue
        name = clean_text(org.get("fullname") or org.get("name"), 120)
        if name and name not in out:
            out.append(name)
    return out


def _http(url: Any) -> str | None:
    return url if isinstance(url, str) and url.startswith(("http://", "https://")) else None


def parse_papers(data: list[dict[str, Any]], since: datetime | None = None, min_upvotes: int = 0) -> list[Item]:
    """Parse ``/api/daily_papers``. A paper's date is when it was featured on Daily Papers.

    Duplicates (same paper id) are merged keeping the highest upvote/comment/star counts and the
    union of affiliations/links. ``summary`` is the abstract (HF's ai_summary only as a fallback).
    """
    by_id: dict[str, Item] = {}
    for entry in data or []:
        paper = entry.get("paper") or {}
        pid = paper.get("id")
        if not pid:
            continue
        upvotes = int(paper.get("upvotes") or 0)
        if upvotes < min_upvotes:
            continue
        published = _parse_dt(paper.get("submittedOnDailyAt") or entry.get("publishedAt") or paper.get("publishedAt"))
        if since is not None and published is not None and published < since:
            continue
        metrics = {"upvotes": float(upvotes), "comments": float(entry.get("numComments") or 0)}
        if paper.get("githubStars") is not None:
            metrics["github_stars"] = float(paper.get("githubStars") or 0)
        affiliations = _paper_affiliations(entry, paper)
        links = {"HF": f"https://huggingface.co/papers/{pid}", "PDF": f"https://arxiv.org/pdf/{pid}"}
        if code := _http(paper.get("githubRepo")):
            links["Code"] = code
        if project := _http(paper.get("projectPage")):
            links["Project"] = project
        prev = by_id.get(pid)
        if prev is not None:
            for k, v in metrics.items():
                prev.metrics[k] = max(prev.metrics.get(k, 0.0), v)
            prev.affiliations += [a for a in affiliations if a not in prev.affiliations]
            prev.links = {**links, **prev.links}
            continue
        abstract = paper.get("summary") or entry.get("summary") or paper.get("ai_summary") or ""
        by_id[pid] = Item(
            key=f"arxiv:{pid}",
            source="hf_papers",
            source_name="HF Daily Papers",
            kind="paper",
            title=clean_text(paper.get("title") or entry.get("title"), 300),
            url=f"https://arxiv.org/abs/{pid}",
            summary=clean_text(abstract),
            authors=[a.get("name") for a in paper.get("authors", []) if a.get("name")],
            published=published,
            tags=list(paper.get("ai_keywords") or [])[:10],
            metrics=metrics,
            affiliations=affiliations,
            links=links,
        )
    return list(by_id.values())


def iso_weeks(start: datetime, end: datetime) -> list[str]:
    """ISO week labels (``YYYY-Www``) of every week overlapping ``[start, end]``, oldest first."""
    weeks: list[str] = []
    day = start.date()
    while day <= end.date():
        y, w, _ = day.isocalendar()
        label = f"{y}-W{w:02d}"
        if label not in weeks:
            weeks.append(label)
        day += timedelta(days=1)
    return weeks


def collect_papers(cfg: dict[str, Any], since: datetime, client: httpx.Client, now: datetime | None = None) -> list[Item]:
    """Rolling-window HF Daily Papers.

    ``since`` is ignored: papers featured in the last ``rolling_days`` (default 30) are returned, so a
    paper that keeps gaining upvotes stays eligible; the pipeline decides what was already shown.
    Fetches one ``?week=YYYY-Www`` list per ISO week overlapping the window (paged with ``p`` while a
    full page still has papers above ``min_upvotes``, at most ``pages_per_week``) plus the latest-day
    list (today's papers are not in a weekly list yet). Each request fails soft.
    """
    now = now or datetime.now(timezone.utc)
    url = cfg.get("url", PAPERS_URL)
    min_upvotes = int(cfg.get("min_upvotes", 0))
    limit = min(int(cfg.get("limit", 100)), 100)  # the API answers 400 above 100
    cutoff = now - timedelta(days=float(cfg.get("rolling_days", 30)))
    pages = max(1, int(cfg.get("pages_per_week", 2)))

    out = ItemList()
    entries: list[dict[str, Any]] = []
    requests = [("latest", {"limit": limit}, 1)] + [(w, {"week": w, "limit": limit}, pages) for w in iso_weeks(cutoff, now)]
    for label, params, max_pages in requests:
        for page in range(max_pages):
            try:
                resp = client.get(url, params={**params, **({"p": page} if page else {})})
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    raise ValueError(f"unexpected payload: {str(data)[:100]}")
            except (httpx.HTTPError, ValueError) as exc:
                log.warning("hf_papers %s (page %d) failed: %s", label, page, exc)
                out.failures[f"hf_papers:{label}"] = f"{type(exc).__name__}: {exc}"[:300]
                break
            entries.extend(data)
            ups = [int((e.get("paper") or {}).get("upvotes") or 0) for e in data]
            if len(data) < limit or not ups or min(ups) < min_upvotes:
                break
    if len(out.failures) == len(requests):
        raise RuntimeError("all HF Daily Papers requests failed")
    out.extend(parse_papers(entries, cutoff, min_upvotes))
    return out


# --------------------------------------------------------------------------- models

def _model_summary(m: dict[str, Any]) -> str:
    parts = []
    if m.get("pipeline_tag"):
        parts.append(f"{m['pipeline_tag']} model")
    if m.get("library_name"):
        parts.append(f"library: {m['library_name']}")
    tags = [t for t in m.get("tags", []) if not t.startswith(_NOISE_TAG_PREFIXES) and t not in (m.get("pipeline_tag"), m.get("library_name"))]
    if tags:
        parts.append("tags: " + ", ".join(tags[:12]))
    return clean_text(" · ".join(parts))


def parse_models(
    data: list[dict[str, Any]],
    now: datetime | None = None,
    max_age_days: float = 21,
    min_likes: int = 0,
) -> list[Item]:
    """Parse ``/api/models``; keep recently created models with enough likes."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
    items: list[Item] = []
    for m in data or []:
        mid = m.get("id") or m.get("modelId")
        if not mid or m.get("private"):
            continue
        likes = int(m.get("likes") or 0)
        created = _parse_dt(m.get("createdAt"))
        if likes < min_likes or (created is not None and created < cutoff):
            continue
        tags = [t for t in [m.get("pipeline_tag"), m.get("library_name")] if t]
        tags += [t for t in m.get("tags", []) if not t.startswith(_NOISE_TAG_PREFIXES) and t not in tags]
        items.append(
            Item(
                key=f"hf_model:{mid}",
                source="hf_models",
                source_name="Hugging Face trending models",
                kind="model",
                title=mid,
                url=f"https://huggingface.co/{mid}",
                summary=_model_summary(m),
                authors=[mid.split("/", 1)[0]] if "/" in mid else [],
                published=created,
                tags=tags[:15],
                metrics={
                    "likes": float(likes),
                    "downloads": float(m.get("downloads") or 0),
                    "trending_score": float(m.get("trendingScore") or 0),
                },
            )
        )
    return items


def collect_models(cfg: dict[str, Any], since: datetime, client: httpx.Client) -> list[Item]:
    url = cfg.get("url", MODELS_URL)
    limit = int(cfg.get("limit", 50))
    sorts = [cfg.get("sort", "trendingScore")] + [s for s in cfg.get("fallback_sorts", ["likes7d", "likes"])]
    last_exc: Exception | None = None
    for sort in dict.fromkeys(sorts):  # ordered, unique
        try:
            resp = client.get(url, params={"sort": sort, "limit": limit})
            resp.raise_for_status()
            data = resp.json()
        except (httpx.HTTPStatusError, ValueError) as exc:
            last_exc = exc
            continue
        return parse_models(data, max_age_days=float(cfg.get("max_age_days", 21)), min_likes=int(cfg.get("min_likes", 0)))
    raise RuntimeError(f"HF models API failed for sorts {sorts}: {last_exc}")
