"""Paper prioritisation: big-company (industry) detection and popularity signals."""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any

from .models import Item, ScoredItem

# A title/abstract company mention only counts in a sentence that also signals real deployment,
# so "experiments on Amazon reviews" does not count but "deployed in Taobao" does.
_DEPLOY_CUE = re.compile(
    r"\b(deploy(?:ed|ment|ing)?|online a/?b|a/?b test(?:s|ing)?|in production|production (?:system|traffic|environment)|"
    r"serv(?:es|ing) (?:hundreds|millions|billions)|launched|industrial|real-world platform|live traffic)\b",
    re.I,
)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@lru_cache(maxsize=8)
def _patterns(companies_key: tuple) -> list[tuple[str, re.Pattern]]:
    out = []
    for name, aliases in companies_key:
        alts = "|".join(re.escape(a.lower()) for a in aliases if a)
        if alts:
            out.append((name, re.compile(rf"(?<![\w.-]){'(?:' + alts + ')'}(?![\w-])", re.I)))
    return out


def _company_patterns(cfg: dict[str, Any]) -> list[tuple[str, re.Pattern]]:
    companies = cfg.get("companies") or {}
    return _patterns(tuple((k, tuple(v or [])) for k, v in companies.items()))


def detect_companies(item: Item, cfg: dict[str, Any]) -> list[str]:
    """Return canonical company names found in affiliations, or in a deployment-cue sentence of the text."""
    found: list[str] = []
    pats = _company_patterns(cfg)
    aff_text = " ; ".join(item.affiliations)
    for name, pat in pats:
        if aff_text and pat.search(aff_text):
            found.append(name)
    text = f"{item.title}. {item.summary}"
    sentences = [s for s in _SENT_SPLIT.split(text) if _DEPLOY_CUE.search(s)]
    if sentences:
        for name, pat in pats:
            if name not in found and any(pat.search(s) for s in sentences):
                found.append(name)
    return found


def _log_ratio(value: float, full_at: float) -> float:
    """0..1, log-scaled: reaches 1.0 when value >= full_at."""
    if value <= 0 or full_at <= 0:
        return 0.0
    return min(1.0, math.log1p(value) / math.log1p(full_at))


def popularity(item: Item, cfg: dict[str, Any]) -> tuple[float, list[str]]:
    """0..1 popularity from HF upvotes, citations and GitHub stars, plus human-readable reasons."""
    m = item.metrics
    up = float(m.get("upvotes") or 0)
    cit = float(m.get("citations") or 0)
    gh = float(m.get("github_stars") or 0)
    parts = [
        _log_ratio(up, float(cfg.get("popular_upvotes", 100))),
        _log_ratio(cit, float(cfg.get("popular_citations", 20))),
        _log_ratio(gh, float(cfg.get("popular_github_stars", 500))),
    ]
    reasons = []
    if up >= 10:
        reasons.append(f"▲ {int(up)} HF upvotes")
    if cit >= 3:
        reasons.append(f"{int(cit)} citations")
    if gh >= 50:
        reasons.append(f"⭐ {int(gh)} code stars")
    return max(parts), reasons


def annotate(item: ScoredItem, cfg: dict[str, Any]) -> ScoredItem:
    """Fill `orgs` for papers (idempotent)."""
    if item.kind == "paper":
        item.orgs = detect_companies(item, cfg)
    return item


def apply_boost(item: ScoredItem, cfg: dict[str, Any]) -> ScoredItem:
    """score = min(10, llm_score + industry boost + popularity boost) for papers; records reasons."""
    if item.llm_score is None:
        item.llm_score = item.score
    item.boost, item.boost_reasons = 0.0, []
    if item.kind == "paper":
        pop, reasons = popularity(item, cfg)
        if item.orgs:
            item.boost += float(cfg.get("industry_boost", 0.7))
            item.boost_reasons.append("🏢 " + ", ".join(item.orgs[:3]))
        item.boost += round(pop * float(cfg.get("popularity_boost_max", 1.0)), 2)
        item.boost_reasons.extend(reasons)
    item.score = round(min(10.0, item.llm_score + item.boost), 1)
    return item
