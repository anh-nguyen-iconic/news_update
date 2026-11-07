"""OpenAlex collector (recent ecommerce/RecSys works with affiliations) and arXiv enrichment.

``collect`` searches ``/works`` for each configured query over the last ``rolling_days`` (``since`` is
ignored, like HF papers: the rolling window replaces it). arXiv preprints are keyed ``arxiv:<id>`` so
they merge with the arXiv/HF copies; other works are keyed ``doi:<lowercase doi>``.

``enrich`` fills ``affiliations`` and ``metrics["citations"]`` of arXiv paper items in place via a
batched DOI lookup (arXiv DOIs are ``10.48550/arxiv.<id>``).

OpenAlex indexes arXiv preprints with a lag of a few days, and unauthenticated requests share a
small daily budget per IP; set ``api_key`` (or the ``OPENALEX_API_KEY`` env var) to raise it.
"""
from __future__ import annotations

import html
import logging
import os
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

import httpx

from ..models import Item
from .base import ItemList, clean_text

log = logging.getLogger(__name__)

API_URL = "https://api.openalex.org/works"
ARXIV_SOURCE_ID = "S4306400194"
ARXIV_DOI_PREFIX = "10.48550/arxiv."
SELECT = "id,doi,title,publication_date,cited_by_count,authorships,abstract_inverted_index,primary_location,ids,type"
ENRICH_SELECT = "doi,authorships,cited_by_count"
MAX_AFFILIATIONS = 8
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^?#\s]+?)(?:v\d+)?(?:\.pdf)?$", re.I)
_VERSION_RE = re.compile(r"v\d+$")


def _base_params(cfg: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if cfg.get("mailto"):
        params["mailto"] = cfg["mailto"]
    api_key = cfg.get("api_key") or os.environ.get("OPENALEX_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def bare_doi(doi: str | None) -> str:
    """``https://doi.org/10.48550/arXiv.2611.1`` -> ``10.48550/arxiv.2611.1`` (lowercase)."""
    doi = (doi or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        doi = doi.removeprefix(prefix)
    return doi


def arxiv_id_of(work: dict[str, Any]) -> str | None:
    doi = bare_doi(work.get("doi") or (work.get("ids") or {}).get("doi"))
    if doi.startswith(ARXIV_DOI_PREFIX):
        return _VERSION_RE.sub("", doi[len(ARXIV_DOI_PREFIX):]) or None
    loc = work.get("primary_location") or {}
    for url in (loc.get("landing_page_url"), loc.get("pdf_url"), (work.get("ids") or {}).get("arxiv")):
        m = _ARXIV_URL_RE.search(url or "")
        if m:
            return _VERSION_RE.sub("", m.group(1))
    return None


def abstract_text(inverted: dict[str, list[int]] | None) -> str:
    """Rebuild an abstract from OpenAlex's ``abstract_inverted_index``."""
    if not inverted:
        return ""
    words: dict[int, str] = {}
    for word, positions in inverted.items():
        for pos in positions or []:
            words[pos] = word
    return clean_text(" ".join(words[i] for i in sorted(words)))


def _clean_aff(text: str | None) -> str:
    # Raw affiliation strings are sometimes double-escaped ("Taobao &amp;#38; Tmall").
    return clean_text(html.unescape(text or ""), 120).strip(" ,;")


def affiliations_of(work: dict[str, Any]) -> list[str]:
    """Unique institution names, then raw affiliation strings; at most ``MAX_AFFILIATIONS``."""
    names: list[str] = []
    authorships = work.get("authorships") or []
    for a in authorships:
        names += [i.get("display_name") for i in a.get("institutions") or [] if i]
    for a in authorships:
        names += list(a.get("raw_affiliation_strings") or [])
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        n = _clean_aff(n)
        if n and n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
        if len(out) >= MAX_AFFILIATIONS:
            break
    return out


def _published(value: str | None) -> datetime | None:
    try:
        return datetime.combine(date.fromisoformat(value), time(), tzinfo=timezone.utc) if value else None
    except ValueError:
        return None


def parse_work(work: dict[str, Any]) -> Item | None:
    title = clean_text(work.get("title") or work.get("display_name"), 300)
    if not title:
        return None
    work_url = work.get("id") if str(work.get("id") or "").startswith("http") else None
    aid = arxiv_id_of(work)
    links: dict[str, str] = {}
    if aid:
        key, url = f"arxiv:{aid}", f"https://arxiv.org/abs/{aid}"
        links["PDF"] = f"https://arxiv.org/pdf/{aid}"
    else:
        doi = bare_doi(work.get("doi"))
        landing = (work.get("primary_location") or {}).get("landing_page_url") or ""
        url = f"https://doi.org/{doi}" if doi else (landing if landing.startswith(("http://", "https://")) else "")
        if not url:
            return None
        key = f"doi:{doi}" if doi else f"url:{url}"
    if work_url:
        links["OpenAlex"] = work_url
    authors = [((a.get("author") or {}).get("display_name")) for a in work.get("authorships") or []]
    return Item(
        key=key,
        source="openalex",
        source_name="OpenAlex",
        kind="paper",
        title=title,
        url=url,
        summary=abstract_text(work.get("abstract_inverted_index")),
        authors=[a for a in authors if a],
        published=_published(work.get("publication_date")),
        tags=[t for t in [work.get("type")] if t],
        metrics={"citations": float(work.get("cited_by_count") or 0)},
        affiliations=affiliations_of(work),
        links=links,
    )


def parse_works(data: dict[str, Any]) -> list[Item]:
    return [it for it in (parse_work(w) for w in (data or {}).get("results") or []) if it is not None]


def collect(cfg: dict[str, Any], since: datetime, client: httpx.Client, now: datetime | None = None) -> list[Item]:
    now = now or datetime.now(timezone.utc)
    start = (now - timedelta(days=float(cfg.get("rolling_days", 30)))).date().isoformat()
    flt = f"from_publication_date:{start}"
    if cfg.get("arxiv_only"):
        flt += f",primary_location.source.id:{ARXIV_SOURCE_ID}"
    url = cfg.get("api_url", API_URL)
    queries = cfg.get("queries") or []
    out = ItemList()
    seen: set[str] = set()
    excluded = tuple(str(x).lower() for x in cfg.get("exclude_doi_prefixes") or [])
    for q in queries:
        params = {**_base_params(cfg), "search": q, "filter": flt, "per_page": int(cfg.get("per_page", 50)), "select": SELECT}
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            got = parse_works(resp.json())
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("openalex query %r failed: %s", q, exc)
            out.failures[f"openalex:{q}"] = f"{type(exc).__name__}: {exc}"[:300]
            continue
        for it in got:
            if excluded and it.key.startswith("doi:") and it.key[4:].startswith(excluded):
                continue
            if it.key not in seen:
                seen.add(it.key)
                # Journal works can carry a future issue date (e.g. next month): clamp to now.
                if it.published is not None and it.published > now:
                    it.published = now
                out.append(it)
    if queries and len(out.failures) == len(queries):
        raise RuntimeError("all OpenAlex queries failed")
    return out


def enrich(items: list[Item], client: httpx.Client, cfg: dict[str, Any] | None = None) -> int:
    """Fill ``affiliations`` and ``metrics["citations"]`` of arXiv papers lacking affiliations, in place.

    Looks works up by arXiv DOI in batches of ``enrich_batch`` (max 50). Fails soft: on an error the
    remaining batches are skipped and the number of items enriched so far is returned.
    """
    cfg = cfg or {}
    targets: dict[str, list[Item]] = {}
    for it in items:
        if it.kind == "paper" and it.key.startswith("arxiv:") and not it.affiliations:
            targets.setdefault(f"{ARXIV_DOI_PREFIX}{it.key[6:].lower()}", []).append(it)
    if not targets:
        return 0
    url = cfg.get("api_url", API_URL)
    batch = max(1, min(int(cfg.get("enrich_batch", 50)), 50))
    dois = list(targets)
    enriched = 0
    for i in range(0, len(dois), batch):
        chunk = dois[i:i + batch]
        params = {**_base_params(cfg), "filter": "doi:" + "|".join(chunk), "per_page": len(chunk), "select": ENRICH_SELECT}
        try:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            results = resp.json().get("results") or []
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            log.warning("openalex enrich failed after %d items: %s", enriched, exc)
            return enriched
        for work in results:
            affs = affiliations_of(work)
            for it in targets.get(bare_doi(work.get("doi")), []):
                if work.get("cited_by_count") is not None:
                    it.metrics["citations"] = max(it.metrics.get("citations", 0.0), float(work["cited_by_count"]))
                if affs and not it.affiliations:
                    it.affiliations = affs
                    enriched += 1
    log.info("openalex enrich: %d/%d arXiv papers got affiliations", enriched, sum(map(len, targets.values())))
    return enriched
