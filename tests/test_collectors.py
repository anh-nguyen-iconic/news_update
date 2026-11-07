"""Offline tests for the source collectors (fixtures + respx, no network)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx

from news_agent.collectors import arxiv, bluesky, goodailist, hackernews, hf, reddit, rss

FIX = Path(__file__).parent / "fixtures"
UTC = timezone.utc


def raw(name: str) -> bytes:
    return (FIX / name).read_bytes()


def js(name: str):
    return json.loads(raw(name))


@pytest.fixture
def client():
    with httpx.Client() as c:
        yield c


# --------------------------------------------------------------------------- arXiv

def test_arxiv_parse():
    items = arxiv.parse_feed(raw("arxiv.xml"))
    assert len(items) == 2
    it = items[0]
    assert it.key == "arxiv:2611.08487"
    assert it.url == "https://arxiv.org/abs/2611.08487"
    assert it.kind == "paper" and it.source == "arxiv"
    assert it.title.startswith("User Embeddings are Superpositions")
    assert it.authors[0] == "Jonathan Kozaczuk" and len(it.authors) == 10
    assert it.tags[0] == "cs.LG"
    assert it.published == datetime(2026, 11, 5, 18, 59, 42, tzinfo=UTC)
    assert "\n" not in it.summary
    assert it.links == {"PDF": "https://arxiv.org/pdf/2611.08487"}
    assert "comment:Accepted to NeurIPS 2026" in it.tags and it.affiliations == []


def test_arxiv_affiliations_and_comment():
    (it,) = arxiv.parse_feed(raw("arxiv_affil.xml"))
    assert it.key == "arxiv:2611.01111"
    assert it.affiliations == ["Zalando SE", "TU Berlin"]  # all authors, deduped (feedparser keeps only the last)
    assert it.tags[0] == "cs.IR"
    assert "comment:Accepted by KDD 2026 (Applied Data Science track). 10 pages, 4 figures" in it.tags
    assert it.links["PDF"] == "https://arxiv.org/pdf/2611.01111"


def test_arxiv_since_filter():
    assert arxiv.parse_feed(raw("arxiv.xml"), datetime(2026, 11, 5, 18, 59, tzinfo=UTC))[0].key == "arxiv:2611.08487"
    assert len(arxiv.parse_feed(raw("arxiv.xml"), datetime(2026, 11, 5, 18, 59, tzinfo=UTC))) == 1


def test_arxiv_id_strips_version():
    assert arxiv.arxiv_id("http://arxiv.org/abs/2411.01234v12") == "2411.01234"
    assert arxiv.arxiv_id("http://arxiv.org/abs/hep-th/9901001v1") == "hep-th/9901001"


@respx.mock
def test_arxiv_collect_applies_grace(client):
    route = respx.get(arxiv.API_URL).mock(return_value=httpx.Response(200, content=raw("arxiv.xml")))
    # since is 20h after publication; the default 24h grace keeps both papers.
    items = arxiv.collect({"categories": ["cs.IR", "cs.LG"], "max_results": 5}, datetime(2026, 11, 6, 15, tzinfo=UTC), client)
    assert len(items) == 2
    q = route.calls.last.request.url.params
    assert q["search_query"] == "cat:cs.IR OR cat:cs.LG"
    assert q["sortBy"] == "submittedDate" and q["max_results"] == "5"


# --------------------------------------------------------------------------- Hugging Face

def test_hf_papers_parse_filters_upvotes_and_date():
    items = hf.parse_papers(js("hf_daily_papers.json"), since=None, min_upvotes=5)
    assert [i.key for i in items] == ["arxiv:2611.06728", "arxiv:2611.02075"]
    assert items[0].source == "hf_papers" and items[0].kind == "paper"
    assert items[0].metrics == {"upvotes": 126.0, "comments": 1.0, "github_stars": 63.0}
    assert items[0].url == "https://arxiv.org/abs/2611.06728"
    assert items[0].links == {
        "HF": "https://huggingface.co/papers/2611.06728",
        "PDF": "https://arxiv.org/pdf/2611.06728",
        "Code": "https://github.com/XD111ds/turn-wise-rl",
    }
    assert items[0].affiliations == [] and items[1].affiliations == ["Stable AI"]
    assert items[0].summary.startswith("Streaming video understanding requires")  # abstract, not ai_summary
    later = hf.parse_papers(js("hf_daily_papers.json"), since=datetime(2026, 11, 5, 12, tzinfo=UTC), min_upvotes=5)
    assert [i.key for i in later] == ["arxiv:2611.02075"]


def test_hf_iso_weeks():
    assert hf.iso_weeks(datetime(2026, 10, 8, tzinfo=UTC), datetime(2026, 11, 7, tzinfo=UTC)) == [
        "2026-W41", "2026-W42", "2026-W43", "2026-W44", "2026-W45"]
    assert hf.iso_weeks(datetime(2025, 12, 30, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC)) == ["2026-W01"]


def _hf_entry(pid: str, upvotes: int, day: str, **paper):
    return {"paper": {"id": pid, "title": f"Paper {pid}", "upvotes": upvotes, "summary": "abs",
                      "submittedOnDailyAt": f"{day}T00:00:00.000Z", **paper},
            "publishedAt": f"{day}T00:00:00.000Z", "numComments": 1}


@respx.mock
def test_hf_collect_papers_rolling_window(client):
    now = datetime(2026, 11, 7, 3, tzinfo=UTC)
    week_data = {
        "2026-W41": [_hf_entry("2610.00001", 50, "2026-10-06")],  # featured before the 30-day window
        "2026-W42": [_hf_entry("2610.00002", 40, "2026-10-14", organization={"fullname": "Alibaba Group"},
                               projectPage="https://proj.example.com")],
        "2026-W44": [_hf_entry("2610.00003", 3, "2026-10-28")],  # below min_upvotes
    }

    def handler(request):
        week = request.url.params.get("week")
        if week == "2026-W43":
            return httpx.Response(500)
        if week is None:  # latest day: same paper as W42 with more upvotes
            return httpx.Response(200, json=[_hf_entry("2610.00002", 60, "2026-10-14"), *js("hf_daily_papers.json")])
        return httpx.Response(200, json=week_data.get(week, []))

    route = respx.get(hf.PAPERS_URL).mock(side_effect=handler)
    items = hf.collect_papers({"min_upvotes": 5, "rolling_days": 30}, datetime(2026, 11, 6, tzinfo=UTC), client, now=now)
    weeks = [c.request.url.params.get("week") for c in route.calls]
    assert weeks == [None, "2026-W41", "2026-W42", "2026-W43", "2026-W44", "2026-W45"]
    assert all(c.request.url.params["limit"] == "100" for c in route.calls)
    assert list(items.failures) == ["hf_papers:2026-W43"]
    by_key = {i.key: i for i in items}
    assert set(by_key) == {"arxiv:2610.00002", "arxiv:2611.06728", "arxiv:2611.02075"}  # `since` is ignored
    merged = by_key["arxiv:2610.00002"]
    assert merged.metrics["upvotes"] == 60 and merged.affiliations == ["Alibaba Group"]
    assert merged.links["Project"] == "https://proj.example.com"


@respx.mock
def test_hf_collect_papers_pages_full_weeks(client):
    now = datetime(2026, 11, 7, tzinfo=UTC)
    full = [_hf_entry(f"2611.{i:05d}", 20, "2026-11-05") for i in range(3)]

    def handler(request):
        p = request.url.params
        if p.get("week") == "2026-W45":
            return httpx.Response(200, json=full if "p" not in p else [_hf_entry("2611.99999", 9, "2026-11-04")])
        return httpx.Response(200, json=[])

    route = respx.get(hf.PAPERS_URL).mock(side_effect=handler)
    items = hf.collect_papers({"min_upvotes": 5, "rolling_days": 3, "limit": 3}, now, client, now=now)
    assert len(items) == 4
    assert [c.request.url.params.get("p") for c in route.calls if c.request.url.params.get("week") == "2026-W45"] == [None, "1"]


@respx.mock
def test_hf_collect_papers_raises_when_all_fail(client):
    respx.get(hf.PAPERS_URL).mock(return_value=httpx.Response(503))
    with pytest.raises(RuntimeError):
        hf.collect_papers({"rolling_days": 1}, datetime.now(UTC), client)


def test_hf_models_parse():
    now = datetime(2026, 11, 6, tzinfo=UTC)
    items = hf.parse_models(js("hf_models.json"), now=now, max_age_days=21, min_likes=400)
    ids = [i.key for i in items]
    assert ids == ["hf_model:moonshotai/Kimi-K3.1-Pro", "hf_model:speridlabs/idol", "hf_model:ccore/Qwen3.5-4B-3D-Coder"]
    kimi = items[0]
    assert kimi.url == "https://huggingface.co/moonshotai/Kimi-K3.1-Pro"
    assert kimi.kind == "model" and kimi.source == "hf_models"
    assert kimi.metrics["likes"] == 1271 and kimi.metrics["trending_score"] == 925
    assert "image-text-to-text" in kimi.summary
    assert not any(t.startswith("region:") for t in kimi.tags)


@respx.mock
def test_hf_models_falls_back_on_bad_sort(client):
    def handler(request):
        if request.url.params["sort"] == "trendingScore":
            return httpx.Response(400, json={"error": "bad sort"})
        return httpx.Response(200, json=[])
    respx.get(hf.MODELS_URL).mock(side_effect=handler)
    assert hf.collect_models({"limit": 5}, datetime.now(UTC), client) == []


# --------------------------------------------------------------------------- RSS

FEED_CFG = {"id": "example", "name": "Example Eng", "url": "https://eng.example.com/feed", "kind": "blog"}


def test_canonical_url():
    assert rss.canonical_url("http://Ex.com/a/?utm_source=x&id=3&fbclid=y#frag") == "https://ex.com/a?id=3"
    assert rss.canonical_url("https://medium.com/p/abc?source=rss----1") == "https://medium.com/p/abc"


def test_rss_parse_feed():
    items = rss.parse_feed(raw("sample_feed.xml"), FEED_CFG, since=datetime(2026, 11, 5, tzinfo=UTC))
    assert [i.title for i in items] == ["Feature store migration", "Two-tower retrieval at scale"]
    two = items[1]
    assert two.key == "url:https://eng.example.com/posts/two-tower?id=7"
    assert two.source == "rss:example" and two.source_name == "Example Eng" and two.kind == "blog"
    assert two.summary == "How we scaled retrieval & ranking."
    assert two.authors == ["Jane Doe"] and two.tags == ["RecSys"]
    assert len(rss.parse_feed(raw("sample_feed.xml"), FEED_CFG, max_items=1)) == 1


def test_rss_parse_rejects_non_feed():
    with pytest.raises(ValueError):
        rss.parse_feed(b"<html><body>blocked</body></html>", FEED_CFG)


@respx.mock
def test_rss_collect_partial_failure_and_fallback(client):
    respx.get("https://eng.example.com/feed").mock(return_value=httpx.Response(200, content=raw("sample_feed.xml")))
    respx.get("https://broken.example.com/feed").mock(return_value=httpx.Response(403))
    respx.get("https://primary.example.com/feed").mock(return_value=httpx.Response(429))
    respx.get("https://mirror.example.com/feed.xml").mock(return_value=httpx.Response(200, content=raw("sample_feed.xml")))
    cfg = {
        "max_items_per_feed": 10,
        "feeds": [
            FEED_CFG,
            {"id": "broken", "name": "Broken", "url": "https://broken.example.com/feed", "kind": "news"},
            {"id": "mirrored", "name": "Mirrored", "url": "https://primary.example.com/feed",
             "fallback_urls": ["https://mirror.example.com/feed.xml"], "kind": "news"},
            {"id": "off", "name": "Off", "url": "https://off.example.com/feed", "enabled": False},
        ],
    }
    items = rss.collect(cfg, datetime(2026, 11, 1, tzinfo=UTC), client)
    assert {i.source for i in items} == {"rss:example", "rss:mirrored"}
    assert len(items) == 6
    assert list(items.failures) == ["rss:broken"]


@respx.mock
def test_rss_collect_raises_when_all_fail(client):
    respx.get("https://broken.example.com/feed").mock(return_value=httpx.Response(500))
    cfg = {"feeds": [{"id": "broken", "name": "B", "url": "https://broken.example.com/feed"}]}
    with pytest.raises(RuntimeError):
        rss.collect(cfg, datetime(2026, 11, 1, tzinfo=UTC), client)


# --------------------------------------------------------------------------- goodailist

def test_goodailist_parse_and_merge():
    trending = js("goodailist_star1d.json")
    new = js("goodailist_new.json")
    weak = {"repos": [{"repo": "Some/Weak", "stars": 10, "star_1d": 5, "star_7d_pct": 3.0, "is_new": False},
                      {"repo": "MORLUTO/rea", "is_new": True, "star_1d": None}]}
    items = goodailist.parse_repos(trending, new, weak, min_star_1d=300, min_star_7d_pct=50)
    keys = [i.key for i in items]
    assert "repo:some/weak" not in keys
    assert keys.count("repo:morluto/rea") == 1
    rea = next(i for i in items if i.key == "repo:morluto/rea")
    assert rea.metrics["is_new"] == 1.0 and rea.metrics["star_1d"] == 4436  # richer record kept, is_new OR-ed
    assert rea.url == "https://github.com/morluto/rea" and rea.kind == "repo"
    assert rea.tags[:3] == ["AI Engineering", "Give agent tools", "Protocols"] and "TypeScript" in rea.tags
    new_repo = next(i for i in items if i.key == "repo:blazux/qwen3.8-flash-dgx")
    assert new_repo.metrics["star_7d_pct"] == 0.0 and new_repo.metrics["is_new"] == 1.0


@respx.mock
def test_goodailist_collect(client):
    base = "https://goodailist.com/api/repos"
    route = respx.get(base).mock(side_effect=lambda r: httpx.Response(
        200, json=js("goodailist_new.json") if r.url.params.get("is_new") else js("goodailist_star1d.json")))
    items = goodailist.collect({"base_url": base, "limit": 3}, datetime.now(UTC), client)
    assert len(items) == 5 and route.call_count == 2


# --------------------------------------------------------------------------- Hacker News

def test_hn_parse():
    items = hackernews.parse_hits(js("hn_front.json"))
    assert items[0].key == "hn:50378228"
    assert items[0].url == "https://sshhole.com"
    assert items[0].metrics == {"points": 2700.0, "comments": 648.0}
    assert "hn_discussion:https://news.ycombinator.com/item?id=50378228" in items[0].tags
    assert "news.ycombinator.com/item?id=50378228" in items[0].summary
    assert items[0].links == {"HN discussion": "https://news.ycombinator.com/item?id=50378228"}
    assert hackernews.parse_hits({"hits": [{"objectID": "1", "title": "Ask HN: x", "created_at_i": 1}]})[0].url \
        == "https://news.ycombinator.com/item?id=1"
    assert hackernews.parse_hits({"hits": [{"objectID": "1", "title": "Ask HN: x", "created_at_i": 1}]})[0].links == {}
    assert len(hackernews.parse_hits(js("hn_front.json"), since=datetime(2026, 11, 5, 21, tzinfo=UTC))) == 1


@respx.mock
def test_hn_collect_dedupes_and_fails_soft(client):
    respx.get(f"{hackernews.API}/search").mock(return_value=httpx.Response(200, json=js("hn_front.json")))
    search = respx.get(f"{hackernews.API}/search_by_date").mock(side_effect=[
        httpx.Response(200, json=js("hn_search.json")),
        httpx.Response(200, json=js("hn_front.json")),
        httpx.Response(503),
    ])
    since = datetime(2026, 11, 5, tzinfo=UTC)
    items = hackernews.collect({"min_points": 50, "queries": ["LLM", "dup", "boom"]}, since, client)
    assert len(items) == 4 and len({i.key for i in items}) == 4
    assert list(items.failures) == ["hn:boom"]
    nf = search.calls[0].request.url.params["numericFilters"]
    assert nf == f"created_at_i>{int(since.timestamp())},points>=50"


# --------------------------------------------------------------------------- Reddit

def test_reddit_parse_groups_by_subreddit():
    items = reddit.parse_feed(raw("reddit_multi.rss"), limit=10)
    assert len(items) == 7
    assert {i.source for i in items} == {"reddit:LocalLLaMA", "reddit:MachineLearning", "reddit:datascience"}
    it = next(i for i in items if i.key == "reddit:1xorgq0")
    assert it.source == "reddit:MachineLearning" and it.source_name == "r/MachineLearning"
    assert it.url.startswith("https://www.reddit.com/r/MachineLearning/comments/1xorgq0/")
    assert it.authors == ["Norlax_42"] and it.kind == "community"
    assert it.summary and "submitted by" not in it.summary
    assert all(i.url.startswith("https://www.reddit.com/") for i in items)
    for i in items:
        assert i.links == ({"Linked article": i.tags[0][5:]} if i.tags else {})
    assert len(reddit.parse_feed(raw("reddit_multi.rss"), limit=2)) == 5  # 2 + 2 + 1
    assert len(reddit.parse_feed(raw("reddit_multi.rss"), since=datetime(2026, 11, 5, 15, tzinfo=UTC))) == 5


@respx.mock
def test_reddit_collect_single_request_and_429_retry(client, monkeypatch):
    monkeypatch.setattr(reddit.time, "sleep", lambda s: None)
    route = respx.get("https://www.reddit.com/r/MachineLearning+LocalLLaMA/top/.rss").mock(side_effect=[
        httpx.Response(429, headers={"x-ratelimit-reset": "30"}),
        httpx.Response(200, content=raw("reddit_multi.rss")),
    ])
    items = reddit.collect({"subreddits": ["MachineLearning", "LocalLLaMA"], "limit": 2},
                           datetime(2026, 11, 1, tzinfo=UTC), client)
    assert len(items) == 5 and route.call_count == 2
    assert route.calls.last.request.url.params["t"] == "day"


@respx.mock
def test_reddit_collect_raises_when_blocked(client):
    respx.get("https://www.reddit.com/r/blocked/top/.rss").mock(return_value=httpx.Response(403))
    with pytest.raises(httpx.HTTPStatusError):
        reddit.collect({"subreddits": ["blocked"]}, datetime(2026, 11, 1, tzinfo=UTC), client)


# --------------------------------------------------------------------------- Bluesky

def test_bluesky_parse():
    items = bluesky.parse_posts(js("bsky_search.json"), min_likes=0)
    assert len(items) == 3
    p = items[1]
    assert p.key == "bsky:at://did:plc:gd6m4mw3km2betcnbbs6362q/app.bsky.feed.post/3mzlr7xnx6j2x"
    assert p.url == "https://bsky.app/profile/minomobi.com/post/3mzlr7xnx6j2x"
    assert "link:https://arxiv.org/abs/2611.06696" in p.tags
    assert p.links == {"Linked article": "https://arxiv.org/abs/2611.06696"}
    assert len(p.title) <= 120 and p.kind == "community"
    assert [i.metrics["likes"] for i in bluesky.parse_posts(js("bsky_search.json"), min_likes=2)] == [3.0, 2.0]
    assert len(bluesky.parse_posts(js("bsky_search.json"), since=datetime(2026, 11, 5, tzinfo=UTC))) == 1


@respx.mock
def test_bluesky_collect_auth_error_is_clear(client):
    respx.get(f"{bluesky.API_BASE}/xrpc/app.bsky.feed.searchPosts").mock(return_value=httpx.Response(403))
    with pytest.raises(RuntimeError, match="requires authentication"):
        bluesky.collect({"queries": ["llm"]}, datetime(2026, 11, 1, tzinfo=UTC), client)


@respx.mock
def test_bluesky_collect_dedupes(client):
    route = respx.get(f"{bluesky.API_BASE}/xrpc/app.bsky.feed.searchPosts").mock(
        return_value=httpx.Response(200, json=js("bsky_search.json")))
    items = bluesky.collect({"queries": ["a", "b"], "min_likes": 0}, datetime(2026, 11, 1, tzinfo=UTC), client)
    assert len(items) == 3 and route.call_count == 2
    assert route.calls[0].request.url.params["since"] == "2026-11-01T00:00:00Z"
