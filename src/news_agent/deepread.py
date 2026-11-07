"""Deep read: fetch the full text of the top digest items and write a detailed summary.

With Gemini: one call per item returns a 4-6 sentence summary, key ideas, results and
"how to apply at THE ICONIC". Without an LLM, an extractive fallback picks key sentences from
the full text (no ideas or applications are invented).
"""
from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .content import FullText, fetch_fulltext
from .models import ScoredItem
from .llm import deep_thinking, gemini_json, get_client

if TYPE_CHECKING:
    import httpx

    from .config import Settings

log = logging.getLogger(__name__)


class DeepOut(BaseModel):
    summary: str
    key_ideas: list[str]
    results: str
    apply_at_iconic: list[str]


# ---------------------------------------------------------------- selection
def pick_targets(top: list[ScoredItem], sections: dict[str, list[ScoredItem]], cfg: dict[str, Any]) -> list[ScoredItem]:
    """Top must-reads first, then the best remaining eligible items across sections."""
    kinds = set(cfg.get("kinds") or ["paper", "blog", "news", "repo"])
    limit = int(cfg.get("max_items", 8))
    rest = sorted((it for lst in sections.values() for it in lst), key=lambda it: (-it.score, it.key))
    out: list[ScoredItem] = []
    for it in [*top, *rest]:
        if len(out) >= limit:
            break
        if it.kind in kinds and it not in out:
            out.append(it)
    return out


# ---------------------------------------------------------------- prompt
def build_prompt(item: ScoredItem, ft: FullText, settings: "Settings") -> str:
    profile = settings.interests.get("profile", "").strip()
    body = "(The full text is attached as a PDF.)" if ft.pdf else ft.text
    org = f"\nOrganisations: {', '.join(item.orgs or item.affiliations[:4])}" if (item.orgs or item.affiliations) else ""
    return f"""You are reading a {item.kind} in full for one reader and writing a detailed but skimmable summary.

READER PROFILE:
{profile}

ITEM: {item.title}
Source: {item.source_name} · {item.url}{org}
Text available: {ft.source}

Return JSON with:
- summary: 4-6 sentences in plain English: the problem, the approach, what is new, and the main findings.
- key_ideas: 3-5 bullets; each one concrete idea, technique or design decision (max ~35 words each), specific enough to act on.
- results: 1-2 sentences with concrete numbers from the text (datasets, metrics, offline gains, online A/B lift, scale).
  If there are none, say "No quantitative results reported."
- apply_at_iconic: 1-3 concrete, realistic ways a data scientist at THE ICONIC (online fashion retail: recsys, search,
  size & fit, personalisation, marketing, forecasting, LLM agents) could use this. Use ["Not directly applicable"] if none.
Use only information present in the text below; never invent numbers, names or links.
If only the abstract or a feed summary is available, keep to what it states.

TEXT:
{body}
"""


def _contents(prompt: str, ft: FullText) -> Any:
    if not ft.pdf:
        return prompt
    from google.genai import types

    return [types.Part.from_bytes(data=ft.pdf, mime_type="application/pdf"), prompt]


def _clean(s: Any, limit: int = 600) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "…"


def apply_llm(item: ScoredItem, out: dict, ft: FullText) -> ScoredItem:
    parsed = DeepOut.model_validate(out)
    item.deep_summary = _clean(parsed.summary, 1200)
    item.key_ideas = [_clean(k, 400) for k in parsed.key_ideas if str(k).strip()][:5]
    item.results = _clean(parsed.results, 500)
    apply = [_clean(a, 400) for a in parsed.apply_at_iconic if str(a).strip()][:3]
    item.apply_at_iconic = [] if apply and apply[0].lower().startswith("not directly applicable") else apply
    item.deep_source, item.deep_llm = ft.source, True
    item.image_url, item.image_caption = ft.image, ft.caption
    return item


# ---------------------------------------------------------------- extractive fallback
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
_IDEA = re.compile(
    r"\b(we (?:propose|introduce|present|design|develop|build|formulate)|our (?:method|approach|framework|model|system)|"
    r"key (?:idea|insight)|this (?:paper|post|work|article|repo(?:sitory)?) (?:describes|introduces|presents|proposes|shows)|"
    r"we (?:show|find|demonstrate|observe)|the core idea)\b",
    re.I,
)
_WEAK_IDEA = re.compile(r"\b(is designed to|consists of|is built on|works by)\b", re.I)  # only fill leftover slots
_RESULT = re.compile(r"\b(outperform\w*|improv\w*|increas\w*|reduc\w*|lift|gains?|state-of-the-art|sota|a/b|speed-?up|achiev\w*)\b", re.I)
# A result sentence must contain an actual quantity: a percentage, a multiplier, a precise metric or "from X to Y".
_QUANTITY = re.compile(r"\d(?:\.\d+)?\s*(?:%|×|x\b|pp\b|points?\b)|\b\d+\.\d{2,}\b|\bfrom\s+\d", re.I)
# Sentences that cite other work describe related work, not the item's own ideas.
_CITATION = re.compile(r"et al\.|\[\d+(?:[,–-]\s*\d+)*\]|\(\w+,? \d{4}\)")
_WORD = re.compile(r"[a-z0-9]+")


def _similar(a: str, b: str, threshold: float = 0.5) -> bool:
    wa, wb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= threshold


_INTRODUCED = re.compile(r"\b(?i:we) (?:introduce|propose|present|develop|design|build)\s+(?:a |an |the )?([A-Z][\w-]+)")


def _distinct(candidates: list[str], avoid: list[str], limit: int) -> list[str]:
    """Pick up to ``limit`` sentences, skipping near-duplicates and repeat introductions of the same name."""
    out: list[str] = []
    names = {m.group(1) for s in avoid for m in [_INTRODUCED.search(s)] if m}
    for s in candidates:
        m = _INTRODUCED.search(s)
        if (m and m.group(1) in names) or any(_similar(s, o) for o in [*avoid, *out]):
            continue
        if m:
            names.add(m.group(1))
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _delatex(s: str) -> str:
    """Remove inline LaTeX left in arXiv HTML text: \\( \\), \\%, \\bar{R}, {,} ..."""
    s = re.sub(r"\\[()\[\]]", "", s).replace("{,}", ",")
    s = re.sub(r"\\(?:bar|hat|tilde|mathrm|mathbf|mathit|text|textbf|textit|emph|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\([%_&$#])", r"\1", s)
    s = re.sub(r"([_^])\{([^{}]*)\}", r"\1\2", s)  # s_{t} -> s_t
    return re.sub(r"\\[a-zA-Z]+\s?", "", s)


def _plain_lines(text: str) -> list[str]:
    """Prose lines only: drop markdown/README noise (code, badges, tables, html) and headings."""
    text = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    # Re-join hard-wrapped prose: a line without end punctuation followed by a lowercase/digit start continues.
    joined: list[str] = []
    for raw in text.splitlines():
        raw = raw.strip()
        if joined and joined[-1] and raw and re.match(r"[a-z0-9(]", raw) and not re.search(r"[.!?:)\"”]$", joined[-1]):
            joined[-1] += " " + raw
        else:
            joined.append(raw)
    out = []
    for ln in joined:
        if re.match(r"\s*(!\[|\[!\[|<|\||#|[-*]\s*\[)", ln):
            continue
        ln = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", ln)
        ln = re.sub(r"\s*\((?:Figure|Fig\.|Table|Eq\.)\s*[\w.]*\)", "", ln)  # dangling figure/table refs
        ln = _delatex(ln)
        ln = " ".join(ln.split())
        # Short lines without end punctuation are headings/labels; long ones are prose.
        if ln and (re.search(r"[.!?:)\"”…]$", ln) or len(ln) >= 100):
            out.append(ln)
    return out


_ENUM = re.compile(r"^(?:\(?\d+[).]|\(?[ivx]+\)|[-•*])\s+", re.I)


def _sentences(text: str, min_len: int = 40) -> list[str]:
    out = []
    for ln in _plain_lines(text):
        for s in _SENT.split(ln):
            s = _ENUM.sub("", s.strip())
            if min_len <= len(s) <= 350:
                out.append(s)
    return out


def _join_sentences(sentences: list[str], limit: int) -> str:
    """Whole sentences only, up to ``limit`` characters (never cut mid-sentence unless the first is too long)."""
    out = ""
    for s in sentences:
        if out and len(out) + 1 + len(s) > limit:
            break
        out = f"{out} {s}".strip()
    return _clean(out, limit)


def apply_extractive(item: ScoredItem, ft: FullText) -> ScoredItem:
    is_paper = item.kind == "paper" and bool(item.summary)
    lead_src = item.summary if is_paper else (ft.text or item.summary)
    lead = _sentences(lead_src, min_len=15)[: 5 if is_paper else 3]
    item.deep_summary = _join_sentences(lead, 900) if lead else _clean(lead_src, 600)
    pool = [s for s in _sentences(ft.text or item.summary) if not _CITATION.search(s)]
    strong = [s for s in pool if _IDEA.search(s)]
    weak = [s for s in pool if _WEAK_IDEA.search(s) and s not in strong]
    item.key_ideas = _distinct(strong + weak, lead, 4)
    results = [s for s in pool if _RESULT.search(s) and _QUANTITY.search(s)]
    item.results = _clean(" ".join(_distinct(results, [*lead, *item.key_ideas], 2)), 500)
    item.apply_at_iconic = []
    item.deep_source, item.deep_llm = ft.source, False
    item.image_url, item.image_caption = ft.image, ft.caption
    return item


# ---------------------------------------------------------------- main entry
def deep_read(
    top: list[ScoredItem],
    sections: dict[str, list[ScoredItem]],
    settings: "Settings",
    client: Any = None,
    http: "httpx.Client | None" = None,
) -> int:
    """Fill deep-read fields in place for the selected items. Returns the number of items summarised."""
    cfg = settings.interests.get("deep_read") or {}
    if not cfg.get("enabled", True):
        return 0
    targets = [it for it in pick_targets(top, sections, cfg) if not it.deep_summary]
    if not targets:
        return 0
    if client is None:
        client = get_client(settings)

    own_http = http is None
    if own_http:
        from .collectors.base import make_client

        http = make_client(settings.sources.get("http"))
    t0 = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            texts = list(pool.map(lambda it: fetch_fulltext(it, http, cfg, want_pdf=client is not None), targets))
    finally:
        if own_http:
            http.close()
    log.info("deep read: fetched %d full texts in %.1fs", len(texts), time.monotonic() - t0)
    t0 = time.monotonic()

    model = settings.llm.get("model", "gemini-2.5-flash")
    pause = float(cfg.get("sleep_between_s", 4))
    done = 0
    for n, (item, ft) in enumerate(zip(targets, texts)):
        if client is not None:
            if n and pause > 0:
                time.sleep(pause)
            try:
                out = gemini_json(client, model, _contents(build_prompt(item, ft, settings), ft), DeepOut, settings,
                                  thinking=deep_thinking(settings))
                apply_llm(item, out, ft)
                done += 1
                continue
            except Exception as exc:  # noqa: BLE001
                log.warning("deep read: Gemini failed for %s: %s; using extractive summary", item.url, exc)
                if ft.pdf:  # PDF can't be summarised extractively; fall back to the abstract
                    ft = FullText(item.summary or "", "abstract only", item.url, image=ft.image)
        apply_extractive(item, ft)
        done += 1
    log.info("deep read: summarised %d items in %.1fs (%s)", done, time.monotonic() - t0,
             ", ".join(sorted({t.source for t in texts})))
    return done
