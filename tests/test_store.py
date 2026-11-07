from datetime import date, datetime, timedelta, timezone

from news_agent.models import Item, ScoredItem
from news_agent.store import History, SeenStore


def _item(key: str, **kw) -> Item:
    return Item(key=key, source="test", source_name="Test", kind=kw.pop("kind", "blog"), title=key, url="https://x/" + key, **kw)


def test_seen_dedupes_within_run_and_across_runs(tmp_path):
    store = SeenStore(tmp_path / "seen.json")
    items = [_item("url:a"), _item("url:a"), _item("url:b")]
    fresh = store.filter_new(items)
    assert [i.key for i in fresh] == ["url:a", "url:b"]
    store.mark(fresh)
    store.save()

    reloaded = SeenStore(tmp_path / "seen.json")
    assert reloaded.filter_new([_item("url:a"), _item("url:c")])[0].key == "url:c"


def test_cross_source_url_duplicate_prefers_primary_source(tmp_path):
    store = SeenStore(tmp_path / "seen.json")
    hn = _item("hn:1", kind="community").model_copy(update={"url": "https://openai.com/index/gpt/?utm_source=hn"})
    rss = _item("url:https://openai.com/index/gpt", kind="news").model_copy(update={"url": "https://openai.com/index/gpt"})
    fresh = store.filter_new([hn, rss])
    assert [i.key for i in fresh] == ["url:https://openai.com/index/gpt"]

    same_title = [_item("url:a").model_copy(update={"title": "Gemini 2 Flash!"}),
                  _item("url:b").model_copy(update={"title": "gemini 2 flash"})]
    assert len(store.filter_new(same_title)) == 1


def test_repo_resurfaces_after_star_multiplier(tmp_path):
    store = SeenStore(tmp_path / "seen.json", resurface_multiplier=2.0)
    store.mark([_item("repo:o/r", kind="repo", metrics={"stars": 1000})])
    assert not store.is_new(_item("repo:o/r", kind="repo", metrics={"stars": 1500}))
    assert store.is_new(_item("repo:o/r", kind="repo", metrics={"stars": 2000}))


def test_seen_prune(tmp_path):
    store = SeenStore(tmp_path / "seen.json")
    old = (datetime.now(timezone.utc) - timedelta(days=61)).isoformat()
    store.data = {"url:old": {"first_seen": old, "stars": None}}
    store.mark([_item("url:new")])
    assert store.prune(60) == 1
    assert list(store.data) == ["url:new"]


def test_history_roundtrip_find_and_prune(tmp_path):
    hist = History(tmp_path / "items")
    today = date(2026, 11, 6)
    a = ScoredItem(**_item("url:a").model_dump(), score=8, section="papers")
    hist.save(today, [a])
    hist.save(today, [ScoredItem(**_item("url:b").model_dump(), score=5)])  # merges, not overwrites
    loaded = hist.load_range(today, today)
    assert {i.key for i in loaded} == {"url:a", "url:b"}
    assert hist.load_recent(7, today)[0].key in {"url:a", "url:b"}

    hist.save(today - timedelta(days=100), [a])
    assert hist.prune(90, today) == 1


def test_non_latin_titles_are_not_collapsed(tmp_path):
    store = SeenStore(tmp_path / "seen.json")
    items = [_item(f"bsky:{i}", kind="community").model_copy(update={"title": t, "url": f"https://bsky.app/{i}"})
             for i, t in enumerate(["大規模言語モデルの新しい論文について", "エージェントを試した結果について", "🔥🔥🔥"])]
    assert len(store.filter_new(items)) == 3
