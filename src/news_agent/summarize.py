"""Gemini summaries for the selected (non-deep-read) items: TL;DR + why it matters, in one batched call."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .llm import gemini_json, get_client
from .models import ScoredItem

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)

BATCH = 25
SUMMARY_CHARS = 700


class SummaryOut(BaseModel):
    id: str
    tldr: str
    why_iconic: str


class SummaryBatchOut(BaseModel):
    items: list[SummaryOut]


def _line(it: ScoredItem) -> str:
    rec: dict[str, Any] = {"id": it.id, "kind": it.kind, "source": it.source_name, "title": it.title,
                           "text": " ".join((it.summary or "").split())[:SUMMARY_CHARS]}
    if it.problem:
        rec["iconic_problem"] = it.problem
    if it.orgs:
        rec["industry"] = ", ".join(it.orgs)
    return json.dumps(rec, ensure_ascii=False)


def build_prompt(items: list[ScoredItem], settings: "Settings") -> str:
    lines = "\n".join(_line(it) for it in items)
    return f"""You summarise items for a daily AI / data-science digest for one reader.

READER PROFILE:
{settings.interests.get('profile', '').strip()}

For EVERY item below return {{"id", "tldr", "why_iconic"}}:
- tldr: 2-4 sentences: what it is, the key idea or method, and the main result or claim (use numbers if given).
- why_iconic: 1-2 sentences on concrete relevance to THE ICONIC (online fashion retail); use "iconic_problem" when
  given. For repos say whether it is worth trying and for what.
Use only information in the item; never invent numbers, names or links. Return JSON {{"items": [...]}}.

ITEMS (one JSON object per line):
{lines}
"""


def summarize(items: list[ScoredItem], settings: "Settings", client: Any = None) -> int:
    """Fill tldr/why_iconic in place for items without a deep read. Returns the number summarised by Gemini."""
    todo = [it for it in items if not it.deep_summary and not it.llm_scored]
    if not todo:
        return 0
    if client is None:
        client = get_client(settings)
    if client is None:
        return 0
    model = settings.llm.get("model", "gemini-2.5-flash")
    done = 0
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        try:
            data = gemini_json(client, model, build_prompt(batch, settings), SummaryBatchOut, settings)
            by_id = {str(r.get("id")): r for r in (data.get("items", []) if isinstance(data, dict) else [])}
        except Exception as exc:  # noqa: BLE001 - extractive TL;DRs remain
            log.warning("summaries: Gemini failed for batch %d: %s", start // BATCH, exc)
            continue
        for it in batch:
            r = by_id.get(it.id)
            if r and str(r.get("tldr") or "").strip():
                it.tldr = " ".join(str(r["tldr"]).split())
                it.why_iconic = " ".join(str(r.get("why_iconic") or "").split())
                it.llm_scored = True  # i.e. summarised by Gemini
                done += 1
    log.info("summaries: %d/%d items summarised by Gemini", done, len(todo))
    return done
