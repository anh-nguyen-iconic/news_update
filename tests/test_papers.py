from datetime import datetime, timezone

from news_agent.config import load_settings
from news_agent.models import Item, ScoredItem
from news_agent.papers import apply_boost, detect_companies, popularity
from news_agent.store import merge_duplicates

CFG = load_settings().papers


def paper(key="arxiv:1", source="arxiv", **kw) -> Item:
    base = dict(key=key, source=source, source_name=source, kind="paper", title="A recommender paper",
                url=f"https://arxiv.org/abs/{key.split(':')[1]}", summary="We study sequential recommendation.")
    base.update(kw)
    return Item(**base)


def test_company_from_affiliation():
    assert detect_companies(paper(affiliations=["Alibaba Group (China)"]), CFG) == ["Alibaba"]
    assert detect_companies(paper(affiliations=["Walmart Global Tech, Sunnyvale"]), CFG) == ["Walmart"]
    assert detect_companies(paper(affiliations=["Meta, Menlo Park"]), CFG) == ["Meta"]


def test_company_from_text_requires_deployment_cue():
    dataset_only = paper(summary="Experiments on Amazon reviews and Taobao datasets show gains.")
    assert detect_companies(dataset_only, CFG) == []
    deployed = paper(summary="We propose X. X has been deployed in Taobao search, with online A/B gains of 2%.")
    assert detect_companies(deployed, CFG) == ["Alibaba"]


def test_short_alias_whole_word_only():
    assert detect_companies(paper(affiliations=["University of Metadata Studies"]), CFG) == []
    assert detect_companies(paper(affiliations=["Ubermensch Institute"]), CFG) == []


def test_popularity_and_boost():
    pop, reasons = popularity(paper(metrics={"upvotes": 143, "github_stars": 526}), CFG)
    assert pop == 1.0 and "▲ 143 HF upvotes" in reasons
    it = ScoredItem(**paper(metrics={"upvotes": 143}).model_dump(), score=7.5, orgs=["Alibaba"])
    apply_boost(it, CFG)
    assert it.llm_score == 7.5 and it.boost > 1 and it.score == min(10, round(7.5 + it.boost, 1))
    assert any("Alibaba" in r for r in it.boost_reasons)
    # idempotent: re-applying does not stack
    apply_boost(it, CFG)
    assert it.score == min(10, round(7.5 + it.boost, 1))
    # non-papers are never boosted
    repo = ScoredItem(key="repo:a/b", source="goodailist", source_name="g", kind="repo", title="t",
                      url="https://github.com/a/b", score=7, metrics={"upvotes": 999})
    assert apply_boost(repo, CFG).score == 7


def test_merge_duplicates_unions_signals():
    ax = paper(source="arxiv", summary="Short.", links={"PDF": "https://arxiv.org/pdf/1"})
    hf = paper(source="hf_papers", summary="A much longer abstract text.", affiliations=["Google"],
               metrics={"upvotes": 50}, links={"HF": "https://huggingface.co/papers/1", "PDF": "https://arxiv.org/pdf/1"})
    oa = paper(source="openalex", summary="x", affiliations=["Google DeepMind"], metrics={"citations": 3, "upvotes": 1})
    other = paper("arxiv:2")
    merged = {m.key: m for m in merge_duplicates([hf, oa, ax, other])}
    m = merged["arxiv:1"]
    assert m.source == "arxiv" and m.summary == "A much longer abstract text."
    assert m.affiliations == ["Google", "Google DeepMind"]
    assert m.metrics == {"upvotes": 50, "citations": 3}
    assert m.links == {"PDF": "https://arxiv.org/pdf/1", "HF": "https://huggingface.co/papers/1"}
    assert len(merged) == 2


def test_meta_alias_ignores_meta_learning():
    s = paper(summary="Our meta-learning recommender is deployed in production at a regional retailer.")
    assert detect_companies(s, CFG) == []
    assert detect_companies(paper(affiliations=["Meta-Analysis Lab"]), CFG) == []


def test_alias_with_punctuation():
    assert detect_companies(paper(affiliations=["JD.com, Beijing"]), CFG) == ["JD.com"]
    assert detect_companies(paper(affiliations=["Huawei Noah's Ark Lab"]), CFG) == ["Huawei"]
