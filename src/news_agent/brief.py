"""'Today in brief': an overview and key highlights shown before the detailed items."""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .models import DailyDigest, ScoredItem
from .llm import deep_thinking, gemini_json, get_client

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)

MAX_HIGHLIGHTS = 6


class HighlightOut(BaseModel):
    text: str
    item_ids: list[str]


class BriefOut(BaseModel):
    overview: str
    highlights: list[HighlightOut]


def digest_items(digest: DailyDigest) -> list[ScoredItem]:
    seen: set[str] = set()
    out = []
    for it in [*digest.top, *(i for lst in digest.sections.values() for i in lst)]:
        if it.id not in seen:
            seen.add(it.id)
            out.append(it)
    return out


def _line(it: ScoredItem) -> str:
    text = it.deep_summary or it.tldr or it.summary
    text = " ".join((text or "").split())[:600]
    ideas = (" Key ideas: " + " / ".join(it.key_ideas))[:500] if it.key_ideas else ""
    results = f" Results: {it.results}" if it.results else ""
    return f"{it.id} | {it.section} | {it.title} | {text}{ideas}{results}"


def build_prompt(items: list[ScoredItem], settings: "Settings") -> str:
    lines = "\n".join(_line(it) for it in items)
    return f"""You write the opening "Today in brief" of a daily AI / agents / data-science digest for one reader.

READER PROFILE:
{settings.interests.get('profile', '').strip()}

Below are today's selected items, one per line as: id | section | title | summary (+ key ideas/results if read in full).

Return JSON with:
- overview: 2-3 sentences on what matters most today for this reader (the big picture, not a list).
- highlights: 4-6 objects {{"text": 1-2 sentences with the single most useful takeaway (concrete: what is new, a number,
  or what to try), "item_ids": 1-3 ids of the items it comes from}}, most important first.
Every highlight MUST cite item ids from the list (they are rendered as links). Use only facts from the items;
do not invent numbers, names or links.

ITEMS:
{lines}
"""


def _first_sentence(text: str, limit: int = 260) -> str:
    text = " ".join((text or "").split())
    m = re.match(r"(.+?[.!?])(?:\s|$)", text)
    s = m.group(1) if m else text
    return s if len(s) <= limit else s[: limit - 1].rsplit(" ", 1)[0] + "…"


def fallback(digest: DailyDigest) -> tuple[str, list[dict]]:
    """Without an LLM: one highlight per deep-read / top item, from its own opening sentence."""
    items = digest_items(digest)
    picked = [it for it in items if it.deep_summary] or digest.top or items
    highlights = []
    for it in picked[:MAX_HIGHLIGHTS - 1]:
        text = _first_sentence(it.results or it.deep_summary or it.tldr or it.summary)
        if text:
            highlights.append({"text": text, "item_ids": [it.id]})
    return "", highlights


def build_brief(digest: DailyDigest, settings: "Settings", client: Any = None) -> DailyDigest:
    """Fill ``digest.overview`` and ``digest.highlights`` in place (LLM, else extractive fallback)."""
    items = digest_items(digest)
    if not items:
        return digest
    if client is None:
        client = get_client(settings)
    if client is not None:
        try:
            data = gemini_json(client, settings.llm.get("model", "gemini-2.5-flash"),
                               build_prompt(items, settings), BriefOut, settings, thinking=deep_thinking(settings))
            parsed = BriefOut.model_validate(data)
            known = {it.id for it in items}
            highlights = []
            for h in parsed.highlights:
                ids = [i for i in dict.fromkeys(h.item_ids) if i in known]
                if h.text.strip() and ids:  # uncited highlights cannot link back to a source
                    highlights.append({"text": " ".join(h.text.split()), "item_ids": ids})
            if highlights:
                digest.overview = " ".join(parsed.overview.split())
                digest.highlights = highlights[:MAX_HIGHLIGHTS]
                return digest
            log.warning("brief: LLM highlights cited no known items; using fallback")
        except Exception as exc:  # noqa: BLE001
            log.warning("brief: LLM failed: %s; using fallback", exc)
    digest.overview, digest.highlights = fallback(digest)
    return digest
