"""Weekly roll-up of the last 7 days of scored items."""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .models import SECTION_ORDER, SECTION_TITLES, ScoredItem, WeeklyDigest
from .llm import deep_thinking, gemini_json, get_client

if TYPE_CHECKING:
    from .config import Settings
    from .feedback import FeedbackStore
    from .store import History

log = logging.getLogger(__name__)

WEEKLY_MIN_SCORE = 5
WEEKLY_MAX_ITEMS = 120


class ThemeOut(BaseModel):
    title: str
    summary: str
    item_ids: list[str]


class TrendOut(BaseModel):
    text: str
    item_ids: list[str]


class IdeaOut(BaseModel):
    idea: str
    item_ids: list[str]


class WeeklyOut(BaseModel):
    trends: list[TrendOut]
    themes: list[ThemeOut]
    try_at_iconic: list[IdeaOut]


def _sort_key(it: ScoredItem) -> tuple:
    return (-it.score, -it.prefilter_score, it.key)


def _dedupe(items: list[ScoredItem]) -> list[ScoredItem]:
    best: dict[str, ScoredItem] = {}
    for it in items:
        cur = best.get(it.key)
        if cur is None or _sort_key(it) < _sort_key(cur):
            best[it.key] = it
    return sorted(best.values(), key=_sort_key)


def _item_topics(it: ScoredItem) -> list[str]:
    return list(dict.fromkeys(it.topics or it.matched_topics))


def _weekly_prompt(items: list[ScoredItem], settings: "Settings", week: str) -> str:
    lines = []
    for it in items:
        tldr = " ".join((it.tldr or it.summary or "").split())[:300]
        if it.key_ideas:
            tldr += " Key ideas: " + " / ".join(it.key_ideas)[:400]
        lines.append(f"{it.id} | {it.section} | {it.title} | {tldr}")
    return f"""You write the weekly trends section of an AI / agents / data-science digest for one reader.

READER PROFILE:
{settings.interests.get('profile', '').strip()}

Below are this week's ({week}) highest-scoring items, one per line as: id | section | title | tldr.

Return JSON with:
- trends: 3-5 objects {{"text": one sentence describing a main trend of the week for this reader, "item_ids": 1-3 ids of the items that evidence it}}.
- themes: 3-6 themes, each {{"title", "summary" (1-2 sentences), "item_ids" (ids from the list below)}}.
- try_at_iconic: 2-4 objects {{"idea": a concrete, practical idea the reader could try at THE ICONIC (fashion ecommerce), "item_ids": 1-3 ids of the items it builds on}}.
Every trend, theme and idea MUST cite item ids from the list below (they are rendered as links back to the sources).
Only use facts present in the items; do not invent links, numbers or claims.

ITEMS:
{chr(10).join(lines)}
"""


def _fallback_summary(items: list[ScoredItem], settings: "Settings") -> tuple[list[dict], list[dict]]:
    labels = {t["id"]: t.get("label", t["id"]) for t in settings.topics}
    topic_items: dict[str, list[ScoredItem]] = defaultdict(list)
    for it in items:
        for t in _item_topics(it):
            topic_items[t].append(it)
    ranked_topics = sorted(topic_items.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    trends = [
        {"text": f"{len(lst)} items on {labels.get(t, t)}", "item_ids": [it.id for it in sorted(lst, key=_sort_key)[:3]]}
        for t, lst in ranked_topics[:5]
    ]
    sec_counts = Counter(it.section for it in items if it.section)
    if sec_counts:
        parts = [
            f"{sec_counts[s]} {SECTION_TITLES.get(s, s).split(' ', 1)[-1]}" for s in SECTION_ORDER if sec_counts[s]
        ]
        trends.append({"text": "By section: " + ", ".join(parts), "item_ids": []})
    themes = [
        {
            "title": labels.get(t, t),
            "summary": f"{len(lst)} items this week.",
            "item_ids": [it.id for it in sorted(lst, key=_sort_key)[:5]],
        }
        for t, lst in ranked_topics[:5]
    ]
    return trends, themes


def build_weekly(
    history: "History",
    feedback: "FeedbackStore",
    settings: "Settings",
    today: date,
    client: Any = None,
) -> WeeklyDigest:
    start = today - timedelta(days=6)
    # Rolling-window papers are re-saved daily while unshown; only count them in the week they
    # first appeared or were actually shown, so old papers do not repeat in every weekly digest.
    items = _dedupe([
        it for it in history.load_range(start, today)
        if it.first_seen is None or it.first_seen >= start or (it.shown_on is not None and it.shown_on >= start)
    ])
    iso = today.isocalendar()
    week = f"{iso[0]}-W{iso[1]:02d}"
    threshold = float(settings.ranking.get("score_threshold", WEEKLY_MIN_SCORE))

    repos_all = [it for it in items if it.section == "repos"]
    repos_all.sort(key=lambda it: (-(it.score >= threshold), -float(it.metrics.get("star_7d") or 0), -it.score, it.key))

    digest = WeeklyDigest(
        week=week,
        start=start,
        end=today,
        top=items[:10],
        models=[it for it in items if it.section == "models"][:6],
        repos=repos_all[:8],
        feedback_stats=feedback.stats(since=start) if feedback is not None else {},
        total_items=len(items),
    )

    strong = [it for it in items if it.score >= WEEKLY_MIN_SCORE][:WEEKLY_MAX_ITEMS]
    if client is None and strong:
        client = get_client(settings)
    if client is not None and strong:
        try:
            data = gemini_json(
                client,
                settings.llm.get("weekly_model") or settings.llm.get("model", "gemini-2.5-flash"),
                _weekly_prompt(strong, settings, week),
                WeeklyOut,
                settings,
                thinking=deep_thinking(settings),
            )
            parsed = WeeklyOut.model_validate(data)
            known = {it.id for it in items}

            def ids(lst: list[str]) -> list[str]:
                return [i for i in dict.fromkeys(lst) if i in known]

            # Statements that cite no known item cannot link back to a source, so they are dropped.
            digest.trends = [{"text": t.text.strip(), "item_ids": ids(t.item_ids)} for t in parsed.trends
                             if t.text.strip() and ids(t.item_ids)]
            digest.themes = [{"title": t.title, "summary": t.summary, "item_ids": ids(t.item_ids)}
                             for t in parsed.themes if ids(t.item_ids)]
            digest.try_at_iconic = [{"idea": i.idea.strip(), "item_ids": ids(i.item_ids)} for i in parsed.try_at_iconic
                                    if i.idea.strip() and ids(i.item_ids)]
            if digest.trends or digest.themes:
                digest.trends_summary = "\n".join(f"- {t['text']}" for t in digest.trends)
                digest.llm_used = True
                _attach_refs(digest, items)
                return digest
            log.warning("Weekly LLM summary cited no known items; using fallback")
        except Exception as exc:  # noqa: BLE001
            log.warning("Weekly LLM summary failed: %s; using fallback", exc)

    digest.trends, digest.themes = _fallback_summary(items, settings)
    digest.trends_summary = "\n".join(f"- {t['text']}" for t in digest.trends)
    digest.try_at_iconic = []
    _attach_refs(digest, items)
    return digest


def _attach_refs(digest: WeeklyDigest, items: list[ScoredItem]) -> None:
    """Map every cited id to its item so templates can always link back to the source."""
    by_id = {it.id: it for it in items}
    cited = [i for group in (digest.trends, digest.themes, digest.try_at_iconic) if group
             for entry in group if isinstance(entry, dict) for i in entry.get("item_ids") or []]
    digest.refs = {i: by_id[i] for i in dict.fromkeys(cited) if i in by_id}
