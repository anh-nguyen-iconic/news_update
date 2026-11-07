import copy
from dataclasses import replace

from news_agent.config import load_settings
from news_agent.models import Item, ScoredItem
from news_agent.ranker import fallback_tldr, rank, relevance, select, weight_factor
from news_agent.semantic import HashEmbedder, cosine_matrix, item_text


def make_settings(**ranking):
    s = load_settings()
    interests = copy.deepcopy(s.interests)
    # HashEmbedder similarities are lower than bge's; calibrate for the fake.
    interests["semantic"].update(floor=0.25, ceil=0.6, match_min=0.3, ecommerce_min=0.4)
    interests["ranking"].update(ranking)
    return replace(s, interests=interests, google_api_key=None, use_vertex=False)


def item(key, kind="paper", title="t", summary="", source="arxiv", **kw):
    return Item(key=key, source=source, source_name=source, kind=kind, title=title,
                url=f"https://x/{key}", summary=summary, **kw)


def si(key, score, section="papers", **kw):
    return ScoredItem(key=key, source="arxiv", source_name="arXiv", kind="paper", title=key,
                      url=f"https://x/{key}", score=score, section=section, **kw)


EMB = HashEmbedder()


def test_relevance_and_weight_factor():
    assert relevance(0.5, 0.7, 0.82) == 0 and relevance(0.9, 0.7, 0.82) == 1
    assert abs(relevance(0.76, 0.7, 0.82) - 0.5) < 1e-9
    assert weight_factor(3) == 1.0 and abs(weight_factor(1) - 0.7) < 1e-9 and weight_factor(9) == 1.0


def test_rank_scores_iconic_problems_higher_and_sets_problem_and_section():
    s = make_settings()
    rec = item("arxiv:1", title="Size and fit for clothing and shoes: predicting the right size",
               summary="Fit feedback, reducing returns, return prediction and return reasons.")
    off = item("arxiv:2", title="Protein folding with diffusion", summary="We study proteins and molecules.")
    out = {o.key: o for o in rank([rec, off], s, embedder=EMB)}
    assert out["arxiv:1"].score > out["arxiv:2"].score
    assert out["arxiv:1"].problem == "Size & fit, returns"
    assert out["arxiv:1"].section == "ecommerce" and "size_fit" in out["arxiv:1"].topics
    assert out["arxiv:2"].problem == "" and out["arxiv:2"].section == "papers"
    assert all(not o.llm_scored and o.why_iconic == "" for o in out.values())  # Gemini only summarises later


def test_rank_feedback_multiplier_changes_order():
    class FB:
        def topic_multiplier(self, t):
            return 2.0 if t == "forecasting_pricing" else 0.5

        def source_multiplier(self, s):
            return 1.0

    s = make_settings()
    a = item("arxiv:a", title="Demand forecasting for retail sales and inventory optimisation pricing")
    b = item("arxiv:b", title="Personalised product recommendations two-tower retrieval ranking models")
    plain = {o.key: o.score for o in rank([a, b], s, embedder=EMB)}
    tuned = {o.key: o.score for o in rank([a, b], s, feedback=FB(), embedder=EMB)}
    assert tuned["arxiv:a"] - tuned["arxiv:b"] > plain["arxiv:a"] - plain["arxiv:b"]


def test_rank_news_about_model_release_goes_to_models():
    s = make_settings()
    n = item("url:n", kind="news", source="rss:openai",
             title="Release of a new AI model: new open-weight frontier LLM launch with model card and benchmark results")
    (o,) = rank([n], s, embedder=EMB)
    assert o.section == "models"


def test_rank_applies_paper_industry_boost():
    s = make_settings()
    base = dict(title="Sequential models for shopping sessions", summary="We study cold-start items.")
    plain = item("arxiv:p", **base)
    industry = item("arxiv:q", affiliations=["Alibaba Group"], **base)
    out = {o.key: o for o in rank([plain, industry], s, embedder=EMB)}
    assert out["arxiv:q"].orgs == ["Alibaba"] and out["arxiv:q"].score > out["arxiv:p"].score


def test_fallback_tldr_drops_bare_links():
    assert fallback_tldr(item("h", summary="HN discussion: https://news.ycombinator.com/item?id=1")) == ""
    assert fallback_tldr(item("r", summary="https://hf.co/x from Nous: Arete is a finetune.")) == "from Nous: Arete is a finetune."


def test_item_text_and_cosine():
    it = item("repo:a/b", kind="repo", title="tool", summary="desc", tags=["Agents", "lang:Python"])
    assert item_text(it) == "tool. desc. Agents"
    m = cosine_matrix([[1, 0], [0, 2]], [[2, 0]])
    assert m == [[1.0], [0.0]]


def test_select_threshold_caps_top_diversity_and_no_duplicates():
    s = make_settings(top_n=3, score_threshold=5, max_top_per_source=9, max_top_per_section=2, section_caps={"papers": 2, "models": 5, "ecommerce": 5})
    items = [si(f"p{i}", 9.5 - i * 0.1) for i in range(5)]
    items += [si("m0", 7, "models"), si("low", 4, "models"), si("e0", 6, "ecommerce"), si("p0", 1)]
    top, sections = select(items, s)
    assert [t.key for t in top] == ["p0", "p1", "m0"]  # max 2 per section in top
    assert [i.key for i in sections["papers"]] == ["p2", "p3"]  # cap 2, top items not repeated
    assert "low" not in {i.key for lst in sections.values() for i in lst}  # below threshold
    assert [i.key for i in sections["ecommerce"]] == ["e0"]


def test_select_fills_top_when_not_diverse():
    s = make_settings(top_n=3, score_threshold=5)
    top, _ = select([si(f"p{i}", 9 - i * 0.1) for i in range(4)], s)
    assert [t.key for t in top] == ["p0", "p1", "p2"]


def _src(key, score, source, section="papers"):
    return ScoredItem(key=key, source=source, source_name=source, kind="paper", title=key,
                      url=f"https://x/{key}", score=score, section=section)


def test_select_mixes_paper_sources_and_limits_top_per_source():
    s = make_settings(top_n=3, score_threshold=5, max_top_per_source=1, max_source_share=0.5,
                      section_caps={"papers": 4})
    items = [_src(f"h{i}", 9.9 - i * 0.1, "hf_papers") for i in range(6)] + [_src(f"a{i}", 7 - i * 0.1, "arxiv") for i in range(3)]
    items.append(_src("m", 6, "rss:x", "models"))
    top, sections = select(items, s)
    assert [t.source for t in top].count("hf_papers") == 1
    papers = sections["papers"]
    assert len(papers) == 4 and sum(p.source == "arxiv" for p in papers) >= 2


def test_context_factor_downweights_iconic_match_without_retail_context():
    s = make_settings()
    s.interests["semantic"].update(context_min_factor=0.2, ctx_floor=0.05, ctx_ceil=0.4)
    retail = item("arxiv:r", title="Agentic checkout and payment protocols for shoppers buying products online")
    generic = item("arxiv:g", title="Agentic checkout and payment protocols")
    out = {o.key: o.score for o in rank([retail, generic], s, embedder=EMB)}
    assert out["arxiv:r"] > out["arxiv:g"]
