from datetime import date

from news_agent.main import DAILY_WEEKDAYS, WEEKLY_WEEKDAY, build_parser, default_window_hours


def test_schedule_constants():
    assert date(2026, 11, 7).weekday() == WEEKLY_WEEKDAY  # Saturday
    assert date(2026, 11, 8).weekday() not in DAILY_WEEKDAYS  # Sunday
    assert date(2026, 11, 9).weekday() in DAILY_WEEKDAYS  # Monday


def test_monday_window_covers_weekend():
    assert default_window_hours(date(2026, 11, 9)) == 74
    assert default_window_hours(date(2026, 11, 10)) == 26


def test_parser_flags():
    args = build_parser().parse_args(["daily", "--dry-run", "--no-llm", "--since", "48", "--only", "arxiv", "rss"])
    assert args.dry_run and args.no_llm and args.since == 48 and args.only == ["arxiv", "rss"]
    args = build_parser().parse_args(["weekly", "--force"])
    assert args.command == "weekly" and args.force


def _setup(tmp_path, monkeypatch, items, section_caps=None):
    import shutil
    from dataclasses import replace

    import news_agent.main as m
    import news_agent.ranker as r
    from news_agent.config import load_settings
    from news_agent.semantic import HashEmbedder

    shutil.copytree(load_settings().root / "config", tmp_path / "config")
    shutil.copytree(load_settings().root / "templates", tmp_path / "templates")
    s = replace(load_settings(tmp_path), google_api_key=None, use_vertex=False,
                gmail_user="me@x.com", gmail_app_password="pw", digest_to="me@x.com")
    s.sources["openalex"] = {"enabled": False}
    s.interests["deep_read"] = {"enabled": False}
    s.interests["semantic"].update(floor=0.25, ceil=0.6, match_min=0.3, ecommerce_min=0.4)  # HashEmbedder scale
    if section_caps:
        s.interests["ranking"]["section_caps"] = section_caps
    sent = []
    monkeypatch.setattr(r, "get_embedder", lambda settings: HashEmbedder())  # no model download in tests
    monkeypatch.setattr(m, "load_settings", lambda: s)
    monkeypatch.setattr(m, "run_all", lambda *a, **k: (list(items), {}))
    monkeypatch.setattr(m, "ingest_from_gmail", lambda *a, **k: 0)
    monkeypatch.setattr(m, "send_email", lambda st, subject, html, text: sent.append(html))
    return m, s, sent


SIZE_FIT = "Size and fit for clothing and shoes: predicting the right size, fit feedback and reducing returns"


def test_daily_marks_only_shown_papers_and_links_everything(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone

    from news_agent.models import Item

    now = datetime.now(timezone.utc)
    items = [
        Item(key="arxiv:1", source="arxiv", source_name="arXiv", kind="paper", url="https://arxiv.org/abs/1",
             title=SIZE_FIT, summary="The system is deployed in Taobao with online A/B gains.", published=now),
        Item(key="arxiv:2", source="arxiv", source_name="arXiv", kind="paper", url="https://arxiv.org/abs/2",
             title="Protein folding with diffusion", summary="Molecules.", published=now),
        Item(key="url:n", source="rss:x", source_name="X", kind="news", url="https://x.com/n",
             title="Release of a new AI model: new open-weight frontier LLM launch with model card", summary="y",
             published=now),
        Item(key="url:bad", source="rss:x", source_name="X", kind="news", url="javascript:alert(1)",
             title="Release of a new AI model broken link", summary="y", published=now),
    ]
    m, s, sent = _setup(tmp_path, monkeypatch, items)
    assert m.main(["daily", "--force", "--no-llm"]) == 0
    seen = json.loads((tmp_path / "data" / "seen.json").read_text())
    assert "arxiv:1" in seen and "url:n" in seen
    assert "arxiv:2" not in seen  # unshown paper is not marked seen
    html = sent[0]
    assert "javascript:" not in html
    assert 'href="https://arxiv.org/abs/1"' in html and "🏢 Alibaba" in html and "🎯 Size &amp; fit, returns" in html


def test_paper_pool_keeps_unshown_relevant_arxiv_papers_for_later_days(tmp_path, monkeypatch):
    import json
    from datetime import datetime, timezone

    from news_agent.models import Item

    now = datetime.now(timezone.utc)
    papers = [Item(key=f"arxiv:{i}", source="arxiv", source_name="arXiv", kind="paper",
                   url=f"https://arxiv.org/abs/{i}", title=f"{SIZE_FIT} variant {i}", summary="", published=now)
              for i in range(4)]
    caps = {"ecommerce": 1, "papers": 1, "models": 1, "industry": 1, "engineering": 1, "repos": 1, "community": 1}
    m, s, sent = _setup(tmp_path, monkeypatch, papers, section_caps=caps)
    s.interests["ranking"]["top_n"] = 1
    assert m.main(["daily", "--force", "--no-llm"]) == 0
    pool = json.loads((tmp_path / "data" / "paper_pool.json").read_text())
    shown_day1 = {k for k in json.loads((tmp_path / "data" / "seen.json").read_text())}
    assert len(shown_day1) == 2 and set(pool) == {p.key for p in papers} - shown_day1

    # Day 2: arXiv returns nothing new, the pooled papers are still considered and shown.
    monkeypatch.setattr(m, "run_all", lambda *a, **k: ([], {}))
    assert m.main(["daily", "--force", "--no-llm"]) == 0
    assert set(json.loads((tmp_path / "data" / "seen.json").read_text())) == {p.key for p in papers}
    assert json.loads((tmp_path / "data" / "paper_pool.json").read_text()) == {}


def test_render_fitted_keeps_email_under_gmail_clip(monkeypatch):
    from datetime import date

    import news_agent.main as m
    from news_agent.config import load_settings
    from news_agent.models import DailyDigest, ScoredItem

    s = load_settings()
    def it(i, sec):
        return ScoredItem(key=f"url:{sec}{i}", source="x", source_name="X", kind="paper", title=f"T{i}",
                          url=f"https://x/{sec}{i}", score=9 - i * 0.1, section=sec, deep_summary="word " * 900,
                          key_ideas=["idea " * 70] * 5)
    sections = {"papers": [it(i, "papers") for i in range(12)], "models": [it(i, "models") for i in range(3)]}
    digest = DailyDigest(date=date(2026, 11, 6), top=[], sections=sections)
    digest.stats.selected = 15
    _, html, _ = m.render_fitted(digest, s)
    assert len(html.encode()) <= m.GMAIL_CLIP_BYTES
    assert digest.sections["papers"][0].key == "url:papers0"  # best items kept
    assert digest.stats.selected == sum(len(v) for v in digest.sections.values())
