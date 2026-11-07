"""Render daily/weekly digests to email HTML (also archived as HTML).

``render_daily`` / ``render_weekly`` return ``(subject, html, text)``; the plain-text
email part is derived from the HTML by :func:`html_to_text`.
"""
from __future__ import annotations

import html as html_lib
import re
from datetime import date, datetime, timezone
from typing import Any
from urllib.parse import urlsplit, quote

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from .models import SECTION_ORDER, SECTION_TITLES, DailyDigest, ScoredItem, WeeklyDigest

FEEDBACK_BODY = "Just hit send."
DEFAULT_SUBJECT_TAG = "[news-agent-feedback]"
SUBJECT_MAX = 118


# --------------------------------------------------------------------------- filters

def human_number(value: Any) -> str:
    """1234 -> '1.2k', 2_500_000 -> '2.5M'."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return str(value)
    sign = "-" if n < 0 else ""
    n = abs(n)
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            txt = f"{n / div:.1f}".rstrip("0").rstrip(".")
            if n / div >= 100:
                txt = f"{n / div:.0f}"
            return f"{sign}{txt}{suffix}"
    return f"{sign}{int(n)}" if n == int(n) else f"{sign}{n:.1f}"


def short_date(value: Any, now: datetime | None = None) -> str:
    """Relative for recent datetimes ('3h ago'), else '6 Nov' / '6 Nov 2025'."""
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        now = now or datetime.now(timezone.utc)
        delta = now - dt
        secs = delta.total_seconds()
        if 0 <= secs < 3600:
            return f"{max(1, int(secs // 60))}m ago"
        if 0 <= secs < 48 * 3600:
            return f"{int(secs // 3600)}h ago"
        d = dt.date()
    elif isinstance(value, date):
        d = value
        now = now or datetime.now(timezone.utc)
    else:
        return str(value)
    if d.year == now.year:
        return f"{d.day} {d.strftime('%b')}"
    return f"{d.day} {d.strftime('%b %Y')}"


def long_date(value: date | datetime) -> str:
    return f"{value.strftime('%a')} {value.day} {value.strftime('%b %Y')}"


def score_color(score: Any) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        s = 0.0
    if s >= 8.5:
        return "#15803d"  # green
    if s >= 7:
        return "#2563eb"  # blue
    if s >= 5:
        return "#d97706"  # amber
    return "#6b7280"  # grey


def score_fmt(score: Any) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "–"
    return f"{s:.0f}" if s == int(s) else f"{s:.1f}"


def make_topic_label(settings) -> Any:
    labels = {t.get("id"): t.get("label", t.get("id")) for t in (settings.topics or []) if isinstance(t, dict)}

    def topic_label(topic_id: str) -> str:
        return labels.get(topic_id) or str(topic_id).replace("_", " ").capitalize()

    return topic_label


def _env(settings) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(settings.templates_dir)),
        autoescape=select_autoescape(enabled_extensions=("html", "html.j2"), default_for_string=False, default=False),
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    env.filters["human"] = human_number
    env.filters["short_date"] = short_date
    env.filters["long_date"] = long_date
    env.filters["score_color"] = score_color
    env.filters["score_fmt"] = score_fmt
    env.filters["topic_label"] = make_topic_label(settings)
    env.filters["md_escape"] = md_escape
    return env


def md_escape(text: Any) -> str:
    """Escape characters that would break a Markdown link label."""
    return re.sub(r"([\[\]])", r"\\\1", str(text or "")).replace("\n", " ")


# --------------------------------------------------------------------------- feedback

def feedback_link(settings, item_id: str, direction: str) -> str:
    """mailto: link that, when sent, records a 👍/👎 vote for ``item_id``."""
    if direction not in ("up", "down"):
        raise ValueError(f"direction must be 'up' or 'down', got {direction!r}")
    if not getattr(settings, "gmail_user", None):
        return ""
    tag = ((settings.interests or {}).get("feedback") or {}).get("subject_tag") or DEFAULT_SUBJECT_TAG
    subject = f"{tag} {direction} {item_id}"
    return f"mailto:{settings.gmail_user}?subject={quote(subject, safe='')}&body={quote(FEEDBACK_BODY, safe='')}"


# --------------------------------------------------------------------------- item view helpers

def _m(item: ScoredItem, *keys: str) -> float | None:
    for k in keys:
        v = item.metrics.get(k)
        if v is not None:
            return v
    return None


def _tag_value(item: ScoredItem, *prefixes: str) -> str | None:
    for tag in item.tags:
        for p in prefixes:
            if tag.startswith(p):
                return tag[len(p):]
    return None


def metrics_line(item: ScoredItem, include_org: bool = True) -> str:
    """Kind-specific one-liner, e.g. '⭐ 12.3k (+2.2k today, +14k this week) · Python'.

    Recognised metric keys (first match wins):
      repo:      stars | stars_total; stars_today | stars_1d | stars_day; stars_week | stars_7d;
                 language from tag 'lang:'/'language:'
      paper:     upvotes; comments
      community: points | score | ups; comments | num_comments
      model:     likes; downloads
    """
    parts: list[str] = []
    if item.kind == "repo":
        stars = _m(item, "stars", "stars_total", "stargazers")
        today = _m(item, "stars_today", "star_1d", "stars_1d", "stars_day")
        week = _m(item, "stars_week", "star_7d", "stars_7d", "stars_this_week")
        if stars is not None or today is not None or week is not None:
            s = f"⭐ {human_number(stars)}" if stars is not None else "⭐"
            growth = []
            if today:
                growth.append(f"+{human_number(today)} today")
            if week:
                growth.append(f"+{human_number(week)} this week")
            if growth:
                s += f" ({', '.join(growth)})" if stars is not None else " " + ", ".join(growth)
            parts.append(s)
        lang = _tag_value(item, "lang:", "language:")
        if lang:
            parts.append(lang)
    elif item.kind == "paper":
        if item.orgs:
            if include_org:
                parts.append("🏢 " + ", ".join(item.orgs[:3]))
        elif item.affiliations:
            parts.append(_truncate(item.affiliations[0], 60))
        up = _m(item, "upvotes", "hf_upvotes")
        if up:
            parts.append(f"▲ {human_number(up)} upvotes")
        cit = _m(item, "citations")
        if cit:
            parts.append(f"{human_number(cit)} citations")
        gh = _m(item, "github_stars")
        if gh:
            parts.append(f"⭐ {human_number(gh)} code")
        c = _m(item, "comments", "num_comments")
        if c:
            parts.append(f"💬 {human_number(c)} comments")
    elif item.kind == "community":
        pts = _m(item, "points", "score", "ups", "likes")
        if pts is not None:
            parts.append(f"▲ {human_number(pts)} points")
        c = _m(item, "comments", "num_comments")
        if c is not None:
            parts.append(f"💬 {human_number(c)} comments")
    elif item.kind == "model":
        likes = _m(item, "likes")
        dls = _m(item, "downloads")
        if likes is not None:
            parts.append(f"❤ {human_number(likes)} likes")
        if dls is not None:
            parts.append(f"⬇ {human_number(dls)} downloads")
    return " · ".join(parts)


def primary_label(item: ScoredItem) -> str:
    host = urlsplit(item.url).netloc.lower().removeprefix("www.")
    if item.kind == "paper":
        return "arXiv" if "arxiv.org" in host else ("DOI" if "doi.org" in host else "Paper")
    if item.kind == "repo":
        return "GitHub" if "github.com" in host else "Repo"
    if item.kind == "model":
        return "Model card"
    if item.kind == "community":
        for needle, label in (("ycombinator", "HN"), ("reddit", "Reddit post"), ("bsky", "Bluesky post")):
            if needle in host:
                return label
        return "Article"
    return "Read article"


def extra_links(item: ScoredItem) -> list[dict[str, str]]:
    """Every link back to the source: the primary URL first, then labelled extras (deduped)."""
    links: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(label: str, url: str | None) -> None:
        if url and url.startswith(("http://", "https://")) and url not in seen:
            seen.add(url)
            links.append({"label": label, "url": url})

    add(primary_label(item), item.url)
    for label, url in (item.links or {}).items():
        add(label, url)
    for tag in item.tags:
        if tag.startswith("hn_discussion:"):
            url = tag[len("hn_discussion:"):]
            add("HN discussion", f"https://news.ycombinator.com/item?id={url}" if url.isdigit() else url)
        elif tag.startswith("link:"):
            add("Linked article", tag[len("link:"):])
    return links


def _view(item: ScoredItem, settings) -> dict[str, Any]:
    return {
        "item": item,
        "id": item.id,
        "metrics_line": metrics_line(item),
        "links": extra_links(item),
        "up": feedback_link(settings, item.id, "up"),
        "down": feedback_link(settings, item.id, "down"),
    }


def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# --------------------------------------------------------------------------- daily

def _kicker(item: ScoredItem) -> str:
    first = f"🎯 {item.problem}" if item.problem else SECTION_TITLES.get(item.section or "", "").split(" ", 1)[-1]
    parts = [first, item.source_name]
    if item.orgs:
        parts.append(", ".join(item.orgs[:3]))
    return " · ".join(p for p in parts if p)


def _deep_view(item: ScoredItem, settings, n: int, total: int) -> dict[str, Any]:
    v = _view(item, settings)
    v.update(n=n, total=total, kicker=_kicker(item), metrics_line=metrics_line(item, include_org=False))
    return v


def render_daily(digest: DailyDigest, settings) -> tuple[str, str, str]:
    """Return (subject, html, text) for the daily digest.

    Layout: Today in brief → deep reads (long-read articles) → More to read (compact cards by section).
    """
    ordered_sections = list(SECTION_ORDER) + [k for k in digest.sections if k not in SECTION_ORDER and k != "top"]
    all_items: list[ScoredItem] = []
    seen_ids: set[str] = set()
    for it in [*digest.top, *(i for k in ordered_sections for i in digest.sections.get(k) or [])]:
        if it.id not in seen_ids:
            seen_ids.add(it.id)
            all_items.append(it)
    by_id = {it.id: it for it in all_items}

    deep_items = [it for it in all_items if it.deep_summary]
    deep = [_deep_view(it, settings, n, len(deep_items)) for n, it in enumerate(deep_items, 1)]
    deep_ids = {it.id for it in deep_items}

    more: list[dict[str, Any]] = []
    top_rest = [i for i in digest.top if i.id not in deep_ids]
    if top_rest:
        more.append({"key": "top", "title": SECTION_TITLES["top"], "items": [_view(i, settings) for i in top_rest]})
    for key in ordered_sections:
        items = [i for i in digest.sections.get(key) or [] if i.id not in deep_ids]
        if items:
            more.append({
                "key": key,
                "title": SECTION_TITLES.get(key, key.replace("_", " ").title()),
                "items": [_view(i, settings) for i in items],
            })

    highlights = []
    for h in digest.highlights or []:
        cited = [by_id[i] for i in dict.fromkeys(h.get("item_ids") or []) if i in by_id]
        if h.get("text") and cited:
            highlights.append({"text": h["text"], "items": cited})

    counts = []
    for key in ordered_sections:
        n = len(digest.sections.get(key) or []) + sum(1 for i in digest.top if i.section == key)
        if n:
            counts.append(f"{n} {SECTION_TITLES.get(key, key).split(' ', 1)[-1].lower()}")

    picks = len(all_items)
    stats = digest.stats
    mode = "ranked for THE ICONIC · summaries by Gemini" if stats.llm_used else "ranked for THE ICONIC · extractive summaries (Gemini unavailable)"
    stats_line = f"{stats.collected:,} items scanned · {picks} picks · {mode}"

    subject = f"AI × Ecommerce daily — {long_date(digest.date)} · {picks} picks"
    lead = deep_items[0] if deep_items else (digest.top[0] if digest.top else None)
    if lead is not None:
        room = SUBJECT_MAX - len(subject) - 3
        if room >= 15:
            subject += f" · {_truncate(lead.title.strip(), room)}"

    ctx = {
        "digest": digest,
        "stats": stats,
        "deep": deep,
        "more": more,
        "highlights": highlights,
        "overview": digest.overview,
        "in_this_issue": counts,
        "picks": picks,
        "stats_line": stats_line,
        "date_label": long_date(digest.date),
        "archive_url": settings.archive_url,
        "feedback_enabled": bool(settings.gmail_user),
        "subject": subject,
    }
    html = _env(settings).get_template("daily.html.j2").render(**ctx)
    return subject, html, html_to_text(html)


# --------------------------------------------------------------------------- weekly

def _bullets(md_text: str) -> list[str]:
    out = []
    for line in (md_text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^([-*•]|\d+[.)])\s+", "", line)
        out.append(line)
    return out


def _inline_md_to_html(text: str) -> str:
    """Escape, then support **bold**, *italic*, `code` and [label](url)."""
    t = html_lib.escape(text, quote=True)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>", t)
    t = re.sub(r"`([^`]+)`", r'<code style="background:#f3f4f6;padding:0 3px;border-radius:3px;">\1</code>', t)
    t = re.sub(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        r'<a href="\2" style="color:#1d4ed8;text-decoration:none;">\1</a>',
        t,
    )
    return t


def render_weekly(weekly: WeeklyDigest, settings, items_by_id: dict[str, ScoredItem] | None = None) -> tuple[str, str, str]:
    """Return (subject, html, markdown) for the weekly trends digest.

    Theme ``item_ids`` are resolved against ``items_by_id`` (if given) plus
    weekly.top/models/repos; unknown ids are skipped silently.
    """
    from markupsafe import Markup

    lookup: dict[str, ScoredItem] = {}
    for it in [*weekly.top, *weekly.models, *weekly.repos, *(weekly.refs or {}).values()]:
        lookup.setdefault(it.id, it)
    if items_by_id:
        lookup.update(items_by_id)

    themes = []
    for th in weekly.themes or []:
        if not isinstance(th, dict):
            continue
        resolved = [lookup[i] for i in th.get("item_ids") or [] if i in lookup]
        themes.append({"title": th.get("title", ""), "summary": th.get("summary", ""), "items": resolved})

    def _cited(ids) -> list[ScoredItem]:
        return [lookup[i] for i in dict.fromkeys(ids or []) if i in lookup]

    if weekly.trends:
        trends = [{"text": str(tr.get("text", "")), "items": _cited(tr.get("item_ids"))}
                  for tr in weekly.trends if isinstance(tr, dict) and tr.get("text")]
    else:
        trends = [{"text": b, "items": []} for b in _bullets(weekly.trends_summary)]
    for tr in trends:
        tr["html"] = Markup(_inline_md_to_html(tr["text"]))
    ideas = []
    for idea in weekly.try_at_iconic or []:
        if isinstance(idea, dict) and idea.get("idea"):
            ideas.append({"text": str(idea["idea"]), "items": _cited(idea.get("item_ids"))})
        elif isinstance(idea, str) and idea.strip():
            ideas.append({"text": idea.strip(), "items": []})
    fb = weekly.feedback_stats or {}
    mults = (fb.get("multipliers") or {})
    topic_label = make_topic_label(settings)
    changed = []
    for kind in ("topics", "sources"):
        for k, v in sorted((mults.get(kind) or {}).items(), key=lambda kv: -abs(float(kv[1]) - 1)):
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if abs(fv - 1.0) > 1e-9:
                changed.append({"kind": kind[:-1], "name": topic_label(k) if kind == "topics" else k, "value": f"{fv:.2f}×"})

    def _topic_names(v):
        out = []
        for t in v or []:
            if isinstance(t, (list, tuple)) and t:
                out.append(f"{topic_label(t[0])} ({t[1]})" if len(t) > 1 else topic_label(t[0]))
            elif isinstance(t, dict):
                out.append(topic_label(t.get("topic") or t.get("id") or ""))
            else:
                out.append(topic_label(t))
        return out

    feedback = {
        "up": fb.get("up", 0),
        "down": fb.get("down", 0),
        "top_up": _topic_names(fb.get("top_up_topics")),
        "top_down": _topic_names(fb.get("top_down_topics")),
        "multipliers": changed,
    }

    subject = f"AI × Ecommerce weekly trends — {weekly.week}"
    date_range = f"{weekly.start.day} {weekly.start.strftime('%b')} – {weekly.end.day} {weekly.end.strftime('%b %Y')}"
    ctx = {
        "weekly": weekly,
        "subject": subject,
        "date_range": date_range,
        "trends": trends,
        "ideas": ideas,
        "themes": themes,
        "top": [_view(i, settings) for i in weekly.top[:10]],
        "models": [_view(i, settings) for i in weekly.models],
        "repos": [_view(i, settings) for i in weekly.repos],
        "feedback": feedback,
        "archive_url": settings.archive_url,
        "mode": "Gemini summarised" if weekly.llm_used else "keyword mode (LLM unavailable)",
        "section_titles": SECTION_TITLES,
    }
    html = _env(settings).get_template("weekly.html.j2").render(**ctx)
    return subject, html, html_to_text(html)


# --------------------------------------------------------------------------- text part

def html_to_text(html: str) -> str:
    """Crude but dependency-free HTML → plain text for the email text/plain part."""
    t = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", "", html)
    t = re.sub(r'(?is)<a\s[^>]*href="(?!mailto:)([^"]+)"[^>]*>(.*?)</a>', r"\2 (\1)", t)
    t = re.sub(r"(?is)<a\s[^>]*>(.*?)</a>", r"\1", t)
    t = re.sub(r"(?i)<br\s*/?>", "\n", t)
    t = re.sub(r"(?i)</(p|div|tr|h[1-6]|li|table)>", "\n", t)
    t = re.sub(r"(?i)<li[^>]*>", "• ", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html_lib.unescape(t).replace("\xa0", " ")
    t = re.sub(r"[ \t\r\f\v]+", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip() + "\n"
