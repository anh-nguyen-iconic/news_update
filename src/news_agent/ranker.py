"""Semantic ranking (local embeddings vs. THE ICONIC problem statements) and digest selection.

No LLM is used here: every item is embedded and scored by cosine similarity to each topic's
``query`` in config/interests.yaml. Gemini is only used later to summarise the selected items.
"""
from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Any

from .llm import first_sentences
from .models import KIND_DEFAULT_SECTION, SECTION_ORDER, Item, ScoredItem
from .papers import annotate, apply_boost
from .semantic import cosine_matrix, get_embedder, item_text

if TYPE_CHECKING:
    from .config import Settings
    from .feedback import FeedbackStore
    from .semantic import Embedder

log = logging.getLogger(__name__)

ENGAGEMENT_METRICS = ("points", "likes", "star_1d", "upvotes")


def weight_factor(weight: float) -> float:
    """Topic weight 1..3 -> score multiplier 0.7..1.0 (soft, so lower-weight interests still surface)."""
    return 0.55 + 0.15 * max(1.0, min(3.0, float(weight)))


def relevance(sim: float, floor: float, ceil: float) -> float:
    return max(0.0, min(1.0, (sim - floor) / max(1e-6, ceil - floor)))


def engagement(metrics: dict[str, float]) -> float:
    """0..1 log-scaled engagement (1.0 at ~1000 points/likes/stars today)."""
    best = max((float(metrics.get(m) or 0.0) for m in ENGAGEMENT_METRICS), default=0.0)
    return min(1.0, math.log10(1 + best) / 3) if best > 0 else 0.0


def fallback_tldr(item: Item) -> str:
    # Collector summaries can be just links (e.g. "HN discussion: https://..."); links are shown separately.
    return first_sentences(re.sub(r"(?:HN discussion:\s*)?https?://\S+", " ", item.summary or "")).strip(" :·-")


def rank(
    items: list[Item],
    settings: "Settings",
    feedback: "FeedbackStore | None" = None,
    embedder: "Embedder | None" = None,
) -> list[ScoredItem]:
    """Score every item 0-10 by semantic relevance to the topics; assign topics, 🎯 problem and section."""
    if not items:
        return []
    cfg = settings.interests.get("semantic") or {}
    floor, ceil = float(cfg.get("floor", 0.70)), float(cfg.get("ceil", 0.82))
    match_min = float(cfg.get("match_min", 0.35))
    problem_min = float(cfg.get("problem_min_short", 0.6))
    ecommerce_min = float(cfg.get("ecommerce_min", 0.5))
    engagement_max = float(cfg.get("engagement_max", 0.5))
    topics = settings.topics
    papers_cfg = settings.papers

    embedder = embedder or get_embedder(settings)
    context_query = cfg.get("context_query")
    query_texts = [t.get("query") or t.get("label", t["id"]) for t in topics] + ([context_query] if context_query else [])
    queries = embedder.embed_queries(query_texts)
    docs = embedder.embed_documents([item_text(it, int(cfg.get("max_chars", 600))) for it in items])
    sims = cosine_matrix(docs, queries)
    min_factor = float(cfg.get("context_min_factor", 0.6))
    ctx_floor, ctx_ceil = float(cfg.get("ctx_floor", 0.48)), float(cfg.get("ctx_ceil", 0.60))
    general_factor = float(cfg.get("paper_general_factor", 0.6))

    out: list[ScoredItem] = []
    kind_cal = cfg.get("kind_calibration") or {}
    for item, row in zip(items, sims):
        k_floor, k_ceil = (float(x) for x in kind_cal.get(item.kind, (floor, ceil)))
        rel = {t["id"]: relevance(s, k_floor, k_ceil) for t, s in zip(topics, row)}
        if context_query:
            ctx = relevance(row[len(topics)], ctx_floor, ctx_ceil)
            factor = min_factor + (1 - min_factor) * ctx
            for t in topics:
                if t.get("context"):
                    rel[t["id"]] *= factor
        if item.kind == "paper":
            for t in topics:
                if not t.get("iconic"):
                    rel[t["id"]] *= general_factor
        weighted = []
        for t in topics:
            mult = feedback.topic_multiplier(t["id"]) if feedback else 1.0
            weighted.append((rel[t["id"]] * weight_factor(t.get("weight", 1)) * mult, t))
        weighted.sort(key=lambda x: -x[0])
        base = weighted[0][0] + 0.1 * (weighted[1][0] if len(weighted) > 1 else 0.0)
        if feedback:
            base *= feedback.source_multiplier(item.source)
        score = 10.0 * min(1.0, base)
        if item.kind != "paper":  # papers get the popularity boost in papers.apply_boost instead
            score += engagement_max * engagement(item.metrics)

        si = ScoredItem.model_validate(item.model_dump())
        si.prefilter_score = round(max(rel.values(), default=0.0), 4)  # raw best relevance (tie-break)
        si.matched_topics = [t["id"] for t in sorted(topics, key=lambda t: -rel[t["id"]]) if rel[t["id"]] >= match_min][:3]
        si.topics = list(si.matched_topics)
        iconic = [t for t in topics if t.get("iconic")]
        best_iconic = max(iconic, key=lambda t: rel[t["id"]], default=None)
        need = match_min if item.kind in ("paper", "blog") else problem_min
        if best_iconic is not None and rel[best_iconic["id"]] >= need:
            si.problem = best_iconic.get("label", best_iconic["id"])
        si.section = _section(si, rel, best_iconic, ecommerce_min)
        si.score = round(min(10.0, score), 2)
        si.llm_score = si.score
        annotate(si, papers_cfg)
        apply_boost(si, papers_cfg)  # industry / popularity boost for papers (recorded in boost_reasons)
        si.tldr = fallback_tldr(si)
        si.why_iconic, si.llm_scored = "", False
        out.append(si)
    out.sort(key=_sort_key)
    return out


def _section(item: ScoredItem, rel: dict[str, float], best_iconic: dict | None, ecommerce_min: float) -> str:
    if item.kind in ("paper", "blog") and best_iconic is not None and rel[best_iconic["id"]] >= ecommerce_min:
        return "ecommerce"
    if item.kind == "news" and rel.get("model_releases", 0) >= 0.5 and rel["model_releases"] == max(rel.values()):
        return "models"
    return KIND_DEFAULT_SECTION[item.kind]


# ---------------------------------------------------------------- select
def _sort_key(it: ScoredItem) -> tuple:
    return (-it.score, -it.prefilter_score, it.key)


def select(items: list[ScoredItem], settings: "Settings") -> tuple[list[ScoredItem], dict[str, list[ScoredItem]]]:
    """Top N (max 2 per section) + per-section caps among items scoring >= ranking.score_threshold."""
    ranking = settings.ranking
    top_n = int(ranking.get("top_n", 5))
    caps = ranking.get("section_caps") or {}
    threshold = float(ranking.get("score_threshold", 5))
    seen: set[str] = set()
    pool: list[ScoredItem] = []
    for it in sorted((i for i in items if i.section in SECTION_ORDER and i.score >= threshold), key=_sort_key):
        if it.key not in seen:
            seen.add(it.key)
            pool.append(it)

    max_top_source = int(ranking.get("max_top_per_source", 3))
    max_top_section = int(ranking.get("max_top_per_section", 3))
    top: list[ScoredItem] = []
    per_section: dict[str, int] = {}
    per_source: dict[str, int] = {}
    for it in pool:
        if len(top) >= top_n:
            break
        if per_section.get(it.section, 0) >= max_top_section or per_source.get(it.source, 0) >= max_top_source:
            continue
        per_section[it.section] = per_section.get(it.section, 0) + 1
        per_source[it.source] = per_source.get(it.source, 0) + 1
        top.append(it)
    if len(top) < top_n:  # not enough diversity available: fill with best remaining
        keys = {t.key for t in top}
        top.extend([it for it in pool if it.key not in keys][: top_n - len(top)])
    top.sort(key=_sort_key)
    top_keys = {t.key for t in top}
    share = float(ranking.get("max_source_share", 0.5))
    sections: dict[str, list[ScoredItem]] = {}
    for sec in SECTION_ORDER:
        cands = [it for it in pool if it.section == sec and it.key not in top_keys]
        cap = int(caps.get(sec, 5))
        chosen = _mix_sources(cands, cap, share) if sec in PAPER_SECTIONS else cands[:cap]
        if chosen:
            sections[sec] = chosen
    return top, sections


PAPER_SECTIONS = {"ecommerce", "papers"}


def _mix_sources(cands: list[ScoredItem], cap: int, share: float) -> list[ScoredItem]:
    """Best items first, but one source fills at most ``share`` of the section while others qualify."""
    limit = max(1, math.ceil(cap * share))
    chosen: list[ScoredItem] = []
    per_source: dict[str, int] = {}
    for it in cands:
        if len(chosen) >= cap:
            break
        if per_source.get(it.source, 0) < limit:
            chosen.append(it)
            per_source[it.source] = per_source.get(it.source, 0) + 1
    for it in cands:  # fill remaining slots if alternatives ran out
        if len(chosen) >= cap:
            break
        if it not in chosen:
            chosen.append(it)
    chosen.sort(key=_sort_key)
    return chosen
