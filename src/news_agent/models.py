"""Core data models shared by every pipeline stage."""
from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

Kind = Literal["paper", "blog", "news", "model", "repo", "community"]
Section = Literal["ecommerce", "papers", "models", "industry", "engineering", "repos", "community"]

SECTION_ORDER: list[Section] = ["ecommerce", "papers", "models", "industry", "engineering", "repos", "community"]

SECTION_TITLES: dict[str, str] = {
    "top": "🔥 Top must-reads",
    "ecommerce": "🛍 Ecommerce & RecSys",
    "papers": "📄 Papers",
    "models": "🤖 New models & releases",
    "industry": "📰 AI industry news",
    "engineering": "🏢 Engineering blogs",
    "repos": "⭐ Rising GitHub repos",
    "community": "💬 Community buzz",
}

# Default section when the LLM is unavailable, derived from the item kind.
KIND_DEFAULT_SECTION: dict[str, Section] = {
    "paper": "papers",
    "blog": "engineering",
    "news": "industry",
    "model": "models",
    "repo": "repos",
    "community": "community",
}


def make_id(key: str) -> str:
    """Short stable id used in feedback links and history lookups."""
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


class Item(BaseModel):
    """A raw item produced by a collector.

    `key` is the dedupe key and must be stable across runs, e.g.
    ``arxiv:2411.01234``, ``repo:owner/name``, ``hf_model:org/name``,
    ``hn:12345`` or ``url:<canonical url>``.
    `source` is the machine id from config/sources.yaml (e.g. ``rss:openai``,
    ``arxiv``, ``hn``, ``reddit:MachineLearning``, ``goodailist``).
    """

    key: str
    source: str
    source_name: str
    kind: Kind
    title: str
    url: str
    summary: str = ""
    authors: list[str] = Field(default_factory=list)
    published: datetime | None = None
    tags: list[str] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    # Paper/author organisations (OpenAlex institutions, HF organisation, arXiv affiliation).
    affiliations: list[str] = Field(default_factory=list)
    # Extra labelled links back to the source, beyond the primary `url`
    # (e.g. {"PDF": ..., "HF": ..., "Code": ..., "Discussion": ...}).
    links: dict[str, str] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        return make_id(self.key)


class ScoredItem(Item):
    prefilter_score: float = 0.0
    matched_topics: list[str] = Field(default_factory=list)
    score: float = 0.0
    section: Section | None = None
    topics: list[str] = Field(default_factory=list)
    tldr: str = ""
    why_iconic: str = ""
    llm_scored: bool = False
    llm_score: float | None = None  # raw LLM score before the paper priority boost
    boost: float = 0.0
    orgs: list[str] = Field(default_factory=list)  # big companies detected (canonical names)
    boost_reasons: list[str] = Field(default_factory=list)
    first_seen: date | None = None  # first day the item was ranked (kept across rolling-window days)
    shown_on: date | None = None  # last day the item appeared in a daily digest
    # Deep read (full-text summary) for the top items of a digest.
    deep_summary: str = ""
    key_ideas: list[str] = Field(default_factory=list)
    results: str = ""
    apply_at_iconic: list[str] = Field(default_factory=list)
    deep_source: str = ""  # e.g. "full text (arXiv HTML)", "PDF", "README", "article", "abstract only"
    deep_llm: bool = False
    problem: str = ""  # best-matching THE ICONIC problem (label), shown as 🎯
    image_url: str = ""  # raster figure / preview image for the long-read layout
    image_caption: str = ""


class RunStats(BaseModel):
    window_hours: int = 26
    collected: int = 0
    after_dedupe: int = 0
    candidates: int = 0
    ranked: int = 0
    selected: int = 0
    llm_used: bool = False
    feedback_ingested: int = 0
    per_source: dict[str, int] = Field(default_factory=dict)
    failed_sources: dict[str, str] = Field(default_factory=dict)


class DailyDigest(BaseModel):
    date: date
    top: list[ScoredItem] = Field(default_factory=list)
    sections: dict[str, list[ScoredItem]] = Field(default_factory=dict)
    stats: RunStats = Field(default_factory=RunStats)
    # "Today in brief" (shown before the details): overview + highlights citing item ids.
    overview: str = ""
    highlights: list[dict] = Field(default_factory=list)  # [{"text": str, "item_ids": [str]}]


class WeeklyDigest(BaseModel):
    week: str  # ISO week label, e.g. "2026-W45"
    start: date
    end: date
    trends_summary: str = ""
    # Structured, linkable versions (preferred by templates when present):
    trends: list[dict] = Field(default_factory=list)  # [{"text": str, "item_ids": [str]}]
    themes: list[dict] = Field(default_factory=list)  # [{"title": str, "summary": str, "item_ids": [str]}]
    try_at_iconic: list = Field(default_factory=list)  # [{"idea": str, "item_ids": [str]}] (str accepted)
    refs: dict[str, ScoredItem] = Field(default_factory=dict)  # id -> item for every cited id
    top: list[ScoredItem] = Field(default_factory=list)
    models: list[ScoredItem] = Field(default_factory=list)
    repos: list[ScoredItem] = Field(default_factory=list)
    feedback_stats: dict = Field(default_factory=dict)  # {"up": int, "down": int, "multipliers": {...}}
    total_items: int = 0
    llm_used: bool = False
