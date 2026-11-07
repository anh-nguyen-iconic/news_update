"""Full-text fetching for the deep read: arXiv HTML / PDF, GitHub README, article pages."""
from __future__ import annotations

import html as html_lib
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import httpx

from .models import ScoredItem

log = logging.getLogger(__name__)

MAX_HTML_BYTES = 8 * 1024 * 1024
MIN_USEFUL_CHARS = 600
# Paper back matter is not useful for a summary and wastes the token budget.
_BACK_MATTER = re.compile(r"\n\s*(?:\d+\.?\s*)?(References|Bibliography|Acknowledg(?:e)?ments?)\s*\n", re.I)


@dataclass
class FullText:
    text: str = ""
    source: str = "abstract only"  # what the summary was based on (kept for logs/history)
    url: str = ""
    pdf: bytes | None = None
    image: str = ""
    caption: str = ""


# Gmail does not render SVG, so only raster images are used.
_RASTER = re.compile(r"\.(?:png|jpe?g|gif|webp)(?:$|\?)", re.I)
_FIGURE = re.compile(r'<figure\b[^>]*\bid="([SA][^"]*)"[^>]*class="[^"]*ltx_figure[^"]*"[^>]*>(.*?)</figure>', re.S)
_IMG_SRC = re.compile(r'<img\b[^>]*\bsrc="([^"]+)"', re.I)
_OVERVIEW = re.compile(r"\b(overview|framework|architecture|pipeline|illustration|system|workflow|approach)\b", re.I)
_META_IMAGE = re.compile(
    r'<meta\b[^>]*(?:property|name)=["\'](?:og:image(?::secure_url)?|twitter:image(?::src)?)["\'][^>]*>', re.I
)
_CONTENT_ATTR = re.compile(r'content=["\']([^"\']+)["\']', re.I)


# MathML carries the LaTeX source in <annotation> next to the rendered text; drop it to avoid "73.6 % 73.6\%".
_ANNOTATION = re.compile(r"<annotation(?:-xml)?\b.*?</annotation(?:-xml)?>", re.S)


def _strip_tags(s: str) -> str:
    s = _ANNOTATION.sub(" ", s)
    text = " ".join(html_lib.unescape(re.sub(r"<[^>]+>", " ", s)).split())
    return re.sub(r"\s+([,.;:%)])", r"\1", text).replace("( ", "(")


def _resolve_arxiv_src(src: str, base_url: str) -> str:
    # arXiv HTML image paths look like "2610.14609v1/fig.png" and live under /html/.
    if re.match(r"^\d{4}\.\d{4,5}v\d+/", src):
        return "https://arxiv.org/html/" + src
    return urljoin(base_url, src)


def arxiv_figure(page_html: str, base_url: str) -> tuple[str, str]:
    """Most relevant raster figure of an arXiv HTML paper.

    Order: main-body overview/architecture figure, any main-body figure, appendix overview figure.
    Returns ("", "") when nothing suitable exists (no image is better than an irrelevant one).
    """
    figures, appendix = [], []
    for m in _FIGURE.finditer(page_html):
        body = m.group(2)
        src = next((s for s in _IMG_SRC.findall(body) if _RASTER.search(s)), None)
        if not src:
            continue
        cap = re.search(r"<figcaption.*?</figcaption>", body, re.S)
        caption = _strip_tags(cap.group(0)) if cap else ""
        (figures if m.group(1).startswith("S") else appendix).append((_resolve_arxiv_src(src, base_url), caption))
    best = next((f for f in figures if _OVERVIEW.search(f[1])), figures[0] if figures else None)
    if best is None:
        best = next((f for f in appendix if _OVERVIEW.search(f[1])), None)
    if best is None:
        return "", ""
    caption = best[1] if len(best[1]) <= 240 else best[1][:239].rsplit(" ", 1)[0] + "…"
    return best[0], caption


def meta_image(page_html: str, base_url: str) -> str:
    for tag in _META_IMAGE.findall(page_html[:200_000]):
        m = _CONTENT_ATTR.search(tag)
        if m:
            url = urljoin(base_url, html_lib.unescape(m.group(1).strip()))
            if url.startswith("https://") and not url.lower().split("?")[0].endswith(".svg"):
                return url
    return ""


def _extract_html(html: str, url: str) -> str:
    import trafilatura

    html = _ANNOTATION.sub(" ", html)
    return trafilatura.extract(html, url=url, include_comments=False, include_tables=False, favor_recall=True) or ""


def trim(text: str, max_chars: int) -> str:
    """Drop references/back matter and keep the beginning plus the conclusion within ``max_chars``."""
    m = _BACK_MATTER.search(text, len(text) // 3)
    if m:
        text = text[: m.start()]
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return text
    head = int(max_chars * 0.75)
    tail = max_chars - head
    return text[:head] + "\n\n[…]\n\n" + text[-tail:]


def _get(client: httpx.Client, url: str, timeout: float) -> httpx.Response | None:
    try:
        resp = client.get(url, timeout=timeout)
    except httpx.HTTPError as exc:
        log.info("deep read: fetch failed %s: %s", url, exc)
        return None
    if resp.status_code != 200:
        log.info("deep read: %s -> HTTP %s", url, resp.status_code)
        return None
    return resp


def _arxiv_id(item: ScoredItem) -> str | None:
    return item.key[6:] if item.key.startswith("arxiv:") else None


def fetch_fulltext(item: ScoredItem, client: httpx.Client, cfg: dict[str, Any], want_pdf: bool) -> FullText:
    """Best-effort full text for an item. Never raises; falls back to the item's own summary."""
    timeout = float(cfg.get("fetch_timeout_s", 30))
    max_chars = int(cfg.get("max_chars", 40000))
    try:
        aid = _arxiv_id(item)
        if aid:
            resp = _get(client, f"https://arxiv.org/html/{aid}", timeout)
            if resp is not None and len(resp.content) <= MAX_HTML_BYTES:
                text = _extract_html(resp.text, str(resp.url))
                if len(text) >= MIN_USEFUL_CHARS:
                    image, caption = arxiv_figure(resp.text, str(resp.url))
                    return FullText(trim(text, max_chars), "full text (arXiv HTML)", f"https://arxiv.org/html/{aid}",
                                    image=image, caption=caption)
            if want_pdf:
                resp = _get(client, f"https://arxiv.org/pdf/{aid}", timeout)
                limit = float(cfg.get("max_pdf_mb", 15)) * 1024 * 1024
                if resp is not None and resp.content[:5] == b"%PDF-" and len(resp.content) <= limit:
                    return FullText("", "full text (PDF)", f"https://arxiv.org/pdf/{aid}", pdf=resp.content)
        elif item.kind == "repo" and item.key.startswith("repo:"):
            repo = item.key[5:]
            for name in ("README.md", "readme.md", "README.rst", "README"):
                resp = _get(client, f"https://raw.githubusercontent.com/{repo}/HEAD/{name}", timeout)
                if resp is not None and resp.text.strip():
                    return FullText(trim(resp.text, max_chars), "README", f"https://github.com/{repo}",
                                    image=f"https://opengraph.githubassets.com/1/{repo}")
        elif item.kind in ("blog", "news") or (item.kind == "paper" and not item.url.startswith("https://doi.org/")):
            resp = _get(client, item.url, timeout)
            if resp is not None and "html" in resp.headers.get("content-type", "") and len(resp.content) <= MAX_HTML_BYTES:
                text = _extract_html(resp.text, str(resp.url))
                if len(text) >= MIN_USEFUL_CHARS:
                    return FullText(trim(text, max_chars), "full article", item.url, image=meta_image(resp.text, str(resp.url)))
    except Exception as exc:  # noqa: BLE001 - deep read is best-effort
        log.warning("deep read: extraction failed for %s: %s", item.url, exc)
    label = "abstract only" if item.kind == "paper" else "feed summary only"
    return FullText(item.summary or "", label, item.url)
