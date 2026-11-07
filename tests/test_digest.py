from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote

import pytest

from news_agent.config import Settings, load_yaml
from news_agent.digest import (
    feedback_link,
    html_to_text,
    human_number,
    metrics_line,
    render_daily,
    render_weekly,
    score_color,
)
from news_agent.models import SECTION_TITLES, DailyDigest, RunStats, ScoredItem, WeeklyDigest

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 6, 6, 0, tzinfo=timezone.utc)


def make_settings(**kw) -> Settings:
    base = dict(
        root=ROOT,
        sources={},
        interests=load_yaml(ROOT / "config" / "interests.yaml"),
        gmail_user="me@example.com",
        gmail_app_password="pw",
        digest_to="me@example.com",
        archive_url="https://github.com/me/news_update/tree/main/digests",
    )
    base.update(kw)
    return Settings(**base)


def item(key, kind, section, title, **kw) -> ScoredItem:
    defaults = dict(
        source=kw.pop("source", kind),
        source_name=kw.pop("source_name", kind.title()),
        url=kw.pop("url", f"https://example.com/{key}"),
        published=NOW - timedelta(hours=5),
        score=7.5,
        tldr=f"TLDR for {title}",
        why_iconic="Could improve PDP recommendations.",
        topics=["recsys"],
        llm_scored=True,
    )
    defaults.update(kw)
    return ScoredItem(key=key, kind=kind, section=section, title=title, **defaults)


@pytest.fixture
def items():
    return {
        "repo": item("repo:acme/agentkit", "repo", "repos", "acme/agentkit",
                     metrics={"stars": 12300, "stars_today": 2200, "stars_week": 14000},
                     tags=["lang:Python"], topics=["agents_production"]),
        "paper": item("arxiv:2611.00001", "paper", "papers", "Two-Tower Fashion Recs",
                      metrics={"upvotes": 42}, topics=["recsys", "fashion"], score=9.1),
        "hn": item("hn:123", "community", "community", "Show HN: Agent checkout",
                   metrics={"points": 312, "comments": 87},
                   tags=["hn_discussion:https://news.ycombinator.com/item?id=123"]),
        "model": item("hf_model:org/m", "model", "models", "org/m-7b",
                      metrics={"likes": 1500, "downloads": 250000}, topics=["model_releases"]),
        "news": item("url:news", "news", "industry", "<script>alert('x')</script> Big AI raise", score=4),
        "blog": item("url:blog", "blog", "engineering", "Zalando size & fit",
                     topics=["fashion"], tags=["link:https://bsky.app/x"]),
    }


@pytest.fixture
def daily(items):
    return DailyDigest(
        date=date(2026, 11, 6),
        top=[items["paper"]],
        sections={
            "repos": [items["repo"]],
            "community": [items["hn"]],
            "models": [items["model"]],
            "industry": [items["news"]],
            "engineering": [items["blog"]],
            "ecommerce": [],
        },
        stats=RunStats(collected=412, after_dedupe=380, candidates=120, ranked=80, selected=6, llm_used=True,
                       feedback_ingested=3, failed_sources={"rss:openai": "HTTP 503 Service Unavailable"}),
    )


class _Balance(HTMLParser):
    VOID = {"meta", "br", "img", "hr", "input", "link"}

    def __init__(self):
        super().__init__()
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (stack={self.stack[-3:]})")
        else:
            self.stack.pop()


def assert_balanced(html: str):
    p = _Balance()
    p.feed(html)
    assert not p.errors, p.errors[:5]
    assert not p.stack, p.stack


def test_filters():
    assert human_number(1234) == "1.2k"
    assert human_number(14000) == "14k"
    assert human_number(2_500_000) == "2.5M"
    assert human_number(87) == "87"
    assert score_color(9) != score_color(3)


def test_feedback_link():
    s = make_settings()
    link = feedback_link(s, "abc123def456", "up")
    assert link.startswith("mailto:me@example.com?subject=")
    assert quote("[news-agent-feedback] up abc123def456", safe="") in link
    assert "&body=" in link
    assert feedback_link(make_settings(gmail_user=None), "x", "down") == ""


def test_render_daily(daily, items):
    s = make_settings()
    subject, html, text = render_daily(daily, s)
    assert subject.startswith("AI × Ecommerce daily — Fri 6 Nov 2026 · 6 picks")
    assert len(subject) < 120
    for key in ["top", "papers", "models", "industry", "engineering", "repos", "community"]:
        if key == "papers":
            continue  # paper is only in top
        assert escape(SECTION_TITLES[key]) in html
    # empty section skipped
    assert escape(SECTION_TITLES["ecommerce"]) not in html
    assert escape(SECTION_TITLES["papers"]) not in html
    # feedback links
    for it in items.values():
        enc = quote(f"[news-agent-feedback] up {it.id}", safe="")
        assert enc in html
        assert quote(f"[news-agent-feedback] down {it.id}", safe="") in html
    # escaping
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    # metrics
    assert "⭐ 12.3k (+2.2k today, +14k this week) · Python" in html
    assert "▲ 312 points" in html and "💬 87 comments" in html
    assert "1.5k likes" in html and "250k downloads" in html
    assert "▲ 42 upvotes" in html
    # extra links
    assert 'href="https://news.ycombinator.com/item?id=123"' in html and ">HN discussion&nbsp;↗<" in html
    assert 'href="https://bsky.app/x"' in html
    # R1: every item links back to its source in an explicit 🔗 links row
    assert html.count("🔗 ") == len(items)
    for it in items.values():
        assert f'href="{it.url}"' in html and f"({it.url})" in text
    # topic labels and stats/footer
    assert "412 items scanned · 6 picks · ranked for THE ICONIC · summaries by Gemini" in html
    assert "rss:openai" in html and "HTTP 503" in html and "rss:openai" in text
    assert s.archive_url in html
    assert "Why it matters:" in html
    assert_balanced(html)
    # text part (derived from the HTML; no markdown output any more)
    assert text == html_to_text(html)
    assert "Two-Tower Fashion Recs" in text and "<td" not in text and "mailto:" not in text


def test_render_daily_keyword_mode_and_no_feedback(daily):
    daily.stats.llm_used = False
    daily.stats.failed_sources = {}
    s = make_settings(gmail_user=None, archive_url=None)
    _, html, text = render_daily(daily, s)
    assert "extractive summaries (Gemini unavailable)" in html and "extractive summaries (Gemini unavailable)" in text
    assert "mailto:" not in html
    assert "More like this" not in html
    assert "Failed sources" not in html
    assert_balanced(html)


def test_render_daily_empty():
    subject, html, _ = render_daily(DailyDigest(date=date(2026, 11, 6)), make_settings())
    assert "0 picks" in subject
    assert "Nothing made the cut" in html
    assert_balanced(html)


def test_render_weekly(items):
    w = WeeklyDigest(
        week="2026-W45",
        start=date(2026, 11, 2),
        end=date(2026, 11, 8),
        trends_summary="- **Agents** are moving to checkout\n- Two-tower models are back\n",
        themes=[{"title": "Agentic checkout", "summary": "Shopping agents mature.",
                 "item_ids": [items["paper"].id, "deadbeef0000", items["repo"].id]}],
        try_at_iconic=["Prototype an outfit-completion agent", "A/B test reranker"],
        top=[items["paper"], items["news"]],
        models=[items["model"]],
        repos=[items["repo"]],
        feedback_stats={"up": 5, "down": 2, "multipliers": {"topics": {"recsys": 1.1, "fashion": 1.0},
                                                            "sources": {"hn": 0.9}},
                        "top_up_topics": ["recsys"], "top_down_topics": ["ai_industry"]},
        total_items=210,
        llm_used=True,
    )
    subject, html, text = render_weekly(w, make_settings())
    assert subject == "AI × Ecommerce weekly trends — 2026-W45"
    assert "Trends this week" in html and "<strong>Agents</strong> are moving" in html
    assert "Agentic checkout" in html and "Shopping agents mature." in html
    # known ids resolved in theme list, unknown skipped
    theme_block = html.split("Agentic checkout", 1)[1].split("Try at THE ICONIC", 1)[0]
    assert items["paper"].url in theme_block and items["repo"].url in theme_block
    assert "deadbeef0000" not in html
    assert "Prototype an outfit-completion agent" in html
    assert "Top 10 of the week" in html and "Repos of the week" in html
    assert escape(SECTION_TITLES["models"]) in html
    assert "👍 5" in html and "👎 2" in html
    assert "1.10×" in html and "0.90×" in html and "1.00×" not in html
    assert "Recommendations &amp; personalisation" in html
    assert "<script>" not in html
    assert_balanced(html)
    assert "Agents are moving to checkout" in text and "Prototype an outfit-completion agent" in text
    assert "Repos of the week" in text and "<td" not in text


def test_render_weekly_minimal():
    w = WeeklyDigest(week="2026-W45", start=date(2026, 11, 2), end=date(2026, 11, 8))
    subject, html, _ = render_weekly(w, make_settings())
    assert "2026-W45" in html
    assert_balanced(html)


def test_metrics_line_repo_without_total():
    it = item("repo:x/y", "repo", "repos", "x/y", metrics={"stars_today": 500})
    assert metrics_line(it) == "⭐ +500 today"


def test_paper_links_orgs_and_signals():
    s = make_settings()
    paper = ScoredItem(
        key="arxiv:2610.39426", source="arxiv", source_name="arXiv", kind="paper",
        title="UGR: Unified Generative Ranking", url="https://arxiv.org/abs/2610.39426",
        links={"PDF": "https://arxiv.org/pdf/2610.39426", "HF": "https://huggingface.co/papers/2610.39426",
               "Code": "https://github.com/x/ugr"},
        affiliations=["Alibaba Group (China)"], orgs=["Alibaba"],
        metrics={"upvotes": 143, "citations": 4, "github_stars": 526}, score=9.1, section="papers",
    )
    digest = DailyDigest(date=date(2026, 11, 6), top=[], sections={"papers": [paper]}, stats=RunStats())
    _, html, _ = render_daily(digest, s)
    for label, url in [("arXiv", paper.url), *paper.links.items()]:
        assert f'href="{url}" style="color:#1d4ed8;text-decoration:none;font-weight:600;">{label}&nbsp;↗</a>' in html
    assert "🏢 Alibaba" in html and "▲ 143 upvotes" in html and "4 citations" in html and "526 code" in html


def test_weekly_trends_and_ideas_link_to_sources():
    s = make_settings()
    it = ScoredItem(key="url:a", source="rss:x", source_name="X", kind="news", title="Big launch",
                    url="https://example.com/a", score=8, section="industry")
    w = WeeklyDigest(week="2026-W45", start=date(2026, 10, 31), end=date(2026, 11, 6),
                     trends=[{"text": "Agents are shipping", "item_ids": [it.id]}],
                     try_at_iconic=[{"idea": "Pilot a shopping agent", "item_ids": [it.id, "unknown"]}],
                     refs={it.id: it})
    _, html, text = render_weekly(w, s)
    assert html.count('href="https://example.com/a"') >= 2
    assert text.count("(https://example.com/a)") >= 2
    assert "Agents are shipping" in html and "Pilot a shopping agent" in text


def _deep_paper(**kw):
    base = dict(
        key="arxiv:2610.30932", source="hf_papers", source_name="HF Daily Papers", kind="paper",
        title="UserArena: Benchmarking Shopping Agents", url="https://arxiv.org/abs/2610.30932",
        links={"PDF": "https://arxiv.org/pdf/2610.30932"}, orgs=["Alibaba"], metrics={"upvotes": 55},
        score=9.5, section="ecommerce", tldr="Short TL;DR that must not repeat.",
        deep_summary="A long summary of the paper.", key_ideas=["Idea one", "Idea two"],
        results="64,729 sessions from 11,010 users.", apply_at_iconic=["Simulate PDP sessions"],
        deep_source="full text (arXiv HTML)", deep_llm=False,
        image_url="https://arxiv.org/html/2610.30932v1/fig1.png", image_caption="Figure 1: Overview of UserArena.",
    )
    base.update(kw)
    return ScoredItem(**base)


def test_long_read_layout_brief_and_deep_article():
    s = make_settings()
    deep = _deep_paper()
    other = ScoredItem(key="url:n", source="rss:x", source_name="TechCrunch", kind="news", title="Agent news",
                       url="https://tc.com/n", score=7, section="industry", tldr="News TL;DR.")
    digest = DailyDigest(date=date(2026, 11, 7), top=[deep], sections={"industry": [other]}, stats=RunStats(),
                         overview="Shopping agents had a big day.",
                         highlights=[{"text": "UserArena gives real sessions.", "item_ids": [deep.id, "unknown"]},
                                     {"text": "Uncited highlight is dropped.", "item_ids": ["nope"]}])
    subject, html, text = render_daily(digest, s)
    # brief comes first, before the in-depth articles
    assert html.index("Today in brief") < html.index("In depth") < html.index("More to read")
    assert "Shopping agents had a big day." in html and "UserArena gives real sessions." in html
    assert "Uncited highlight is dropped." not in html
    brief = html.split("Today in brief", 1)[1].split("In depth", 1)[0]
    assert f'href="{deep.url}"' in brief and "In this issue:" in brief
    # long-read article: kicker, image + caption, numbered ideas, results, apply; no source-type label
    article = html.split("In depth", 1)[1].split("More to read", 1)[0]
    assert "Ecommerce &amp; RecSys · HF Daily Papers · Alibaba" in article
    assert f'<img src="{deep.image_url}"' in article and "Figure 1: Overview of UserArena." in article
    assert "Idea one" in article and "📊 Results" in article and "How to apply at THE ICONIC" in article
    assert "full text (arXiv HTML)" not in html and "key sentences extracted" not in html
    assert "Short TL;DR that must not repeat." not in html  # deep summary replaces the TL;DR
    # deep-read item is not repeated in More to read; others are
    more = html.split("More to read", 1)[1]
    assert deep.url not in more.split("Run stats", 1)[0] and "Agent news" in more
    assert_balanced(html)
    assert "Today in brief" in text


def test_image_only_when_present_and_escaped():
    s = make_settings()
    p = _deep_paper(image_url="", image_caption="", title="<b>bold</b> title")
    _, html, _ = render_daily(DailyDigest(date=date(2026, 11, 7), top=[p], stats=RunStats()), s)
    assert "<img" not in html and "&lt;b&gt;bold&lt;/b&gt; title" in html
