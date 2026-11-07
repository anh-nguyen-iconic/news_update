import copy
import json
from dataclasses import replace
from datetime import date, timedelta
from types import SimpleNamespace

from news_agent.config import load_settings
from news_agent.feedback import FeedbackStore
from news_agent.models import ScoredItem
from news_agent.store import History
from news_agent.weekly import build_weekly

TODAY = date(2026, 11, 6)


def make_settings(tmp_path):
    s = load_settings()
    interests = copy.deepcopy(s.interests)
    interests["llm"].update({"retry_min_wait_s": 0, "max_retries": 2, "sleep_between_batches_s": 0})
    return replace(s, root=tmp_path, interests=interests, google_api_key=None, use_vertex=False)


def si(key, section, score, topics=("recsys",), **metrics):
    kind = {"models": "model", "repos": "repo"}.get(section, "paper")
    return ScoredItem(key=key, source="src", source_name="Src", kind=kind, title=f"T {key}", url="https://x",
                      section=section, score=score, topics=list(topics), tldr=f"tldr {key}", metrics=metrics)


def seed(tmp_path):
    h = History(tmp_path / "items")
    h.save(TODAY - timedelta(days=7), [si("old", "papers", 10)])  # outside window
    h.save(TODAY - timedelta(days=6), [si("p1", "papers", 9), si("dup", "papers", 5)])
    h.save(TODAY - timedelta(days=2), [si("dup", "papers", 8.5)] + [si(f"m{i}", "models", 7 + i * 0.1,
                                                                      topics=["model_releases"]) for i in range(8)])
    h.save(TODAY, [si(f"r{i}", "repos", 6 + i * 0.1, star_7d=100 * (5 - i)) for i in range(10)]
           + [si("rlow", "repos", 3, star_7d=10000), si("e1", "ecommerce", 9.5, topics=["fashion"])])
    return h


def test_weekly_fallback(tmp_path):
    s = make_settings(tmp_path)
    h = seed(tmp_path)
    fb = FeedbackStore(tmp_path / "fb.json", s.interests["feedback"])
    w = build_weekly(h, fb, s, TODAY)
    assert w.week == "2026-W45" and w.start == TODAY - timedelta(days=6) and w.end == TODAY
    assert w.llm_used is False and w.try_at_iconic == []
    keys = [t.key for t in w.top]
    assert len(keys) == 10 and len(set(keys)) == 10 and "old" not in keys
    assert keys[:3] == ["e1", "p1", "dup"]
    assert len(w.models) == 6 and all(m.section == "models" for m in w.models)
    assert [r.key for r in w.repos][:2] == ["r0", "r1"] and len(w.repos) == 8 and "rlow" not in [r.key for r in w.repos]
    assert w.total_items == 2 + 8 + 12  # deduped by key
    assert "- " in w.trends_summary and "Recommendations" in w.trends_summary
    assert w.themes and w.themes[0]["item_ids"]
    assert w.trends[0]["item_ids"] and set(w.trends[0]["item_ids"]) <= set(w.refs)
    assert w.feedback_stats["up"] == 0


class FakeClient:
    def __init__(self, fail_times=0):
        self.calls = []
        self.fail_times = fail_times
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents))
        if len(self.calls) <= self.fail_times:
            raise RuntimeError("503")
        ids = [line.split(" | ")[0] for line in contents.split("ITEMS:")[1].strip().splitlines()]
        return SimpleNamespace(text=json.dumps({
            "trends": [{"text": "Agents everywhere", "item_ids": ids[:1]},
                       {"text": "Uncited claim", "item_ids": ["nope"]}],
            "themes": [{"title": "RecSys", "summary": "s", "item_ids": ids[:2] + ["nope"]}],
            "try_at_iconic": [{"idea": "Try X", "item_ids": ids[1:2]}, {"idea": " ", "item_ids": ids[:1]}],
        }))


def test_weekly_llm(tmp_path):
    s = make_settings(tmp_path)
    h = seed(tmp_path)
    fb = FeedbackStore(tmp_path / "fb.json", s.interests["feedback"])
    client = FakeClient(fail_times=1)
    w = build_weekly(h, fb, s, TODAY, client=client)
    assert w.llm_used is True and len(client.calls) == 2
    assert client.calls[0][0] == s.llm["weekly_model"]
    prompt = client.calls[0][1]
    assert "rlow" not in prompt and "T e1" in prompt  # only score >= 6
    assert w.trends_summary.startswith("- Agents")
    assert [t["text"] for t in w.trends] == ["Agents everywhere"]  # uncited statement dropped
    assert w.themes[0]["item_ids"] and "nope" not in w.themes[0]["item_ids"]
    assert [i["idea"] for i in w.try_at_iconic] == ["Try X"]
    cited = {i for g in (w.trends, w.themes, w.try_at_iconic) for e in g for i in e["item_ids"]}
    assert cited and cited <= set(w.refs)  # every cited id resolves to a linkable item


def test_weekly_llm_failure_falls_back(tmp_path):
    s = make_settings(tmp_path)
    h = seed(tmp_path)
    fb = FeedbackStore(tmp_path / "fb.json", s.interests["feedback"])
    w = build_weekly(h, fb, s, TODAY, client=FakeClient(fail_times=99))
    assert w.llm_used is False and "items on" in w.trends_summary


def test_weekly_empty(tmp_path):
    s = make_settings(tmp_path)
    w = build_weekly(History(tmp_path / "items"), FeedbackStore(tmp_path / "fb.json", {}), s, TODAY)
    assert w.total_items == 0 and w.top == [] and w.llm_used is False


def test_weekly_skips_rolling_papers_from_earlier_weeks(tmp_path):
    from datetime import timedelta
    from news_agent.models import ScoredItem
    from news_agent.store import History

    s = make_settings(tmp_path)
    h = History(tmp_path / "items")
    def p(key, **kw):
        return ScoredItem(key=key, source="hf_papers", source_name="HF", kind="paper", title=key, url=f"https://x/{key}",
                          score=9, section="papers", **kw)
    old_unshown = p("old", first_seen=TODAY - timedelta(days=20))
    old_shown = p("shown", first_seen=TODAY - timedelta(days=20), shown_on=TODAY - timedelta(days=1))
    fresh = p("fresh", first_seen=TODAY)
    h.save(TODAY, [old_unshown, old_shown, fresh])
    w = build_weekly(h, None, s, TODAY)
    assert {i.key for i in w.top} == {"shown", "fresh"}
