import httpx
import pytest

from news_agent.config import load_settings
from news_agent.content import FullText, fetch_fulltext, trim
from news_agent.deepread import apply_extractive, build_prompt, deep_read, pick_targets
from news_agent.models import ScoredItem

ARTICLE_HTML = """<html><head><title>Reflex</title></head><body><nav>menu</nav><article>
<h1>Reflex: an architecture for building mobile apps with agents</h1>
<p>We introduce Reflex, an architecture that lets coding agents inspect and drive our mobile app without a simulator.
The key idea is that every ViewModel exposes an introspection node that agents can read and act on.</p>
<p>Reflex consists of three layers: an introspection kit, a command bus and a CLI that agents call from the terminal.
Agents use it to verify their changes against the real app state instead of screenshots.</p>
<p>Since rollout, agent-written pull requests that pass review on the first attempt increased from 41% to 68%,
and median verification time dropped by 3x across 1,200 screens.</p>
<p>We plan to open source the introspection kit next year and extend it to web surfaces as well.</p>
</article><footer>footer</footer></body></html>"""

PAPER_HTML = """<html><body><article><h1>UGR: Unified Generative Ranking</h1>
<h2>Abstract</h2><p>We propose UGR, a unified generative ranking model for heterogeneous e-commerce recommendation.
It replaces a cascade of retrieval and ranking models with one decoder that generates item semantic ids.</p>
<h2>1 Introduction</h2><p>Industrial recommenders at Alibaba use multi-stage cascades that are hard to optimise jointly.
Our approach trains a single generative model with a multi-task objective over clicks, add-to-carts and purchases.</p>
<h2>5 Experiments</h2><p>UGR outperforms the production cascade by 4.2% in GMV and 3.1% in CTR in an online A/B test over 14 days.</p>
<h2>References</h2><p>[1] Some citation that should be removed from the text sent to the model.</p>
""" + "<p>Filler paragraph about related work and details of the model architecture.</p>" * 20 + "</article></body></html>"


def item(key, kind="paper", url=None, summary="An abstract. It has two sentences about the method.", **kw):
    return ScoredItem(key=key, source="x", source_name="X", kind=kind, title=f"Title {key}",
                      url=url or f"https://arxiv.org/abs/{key.split(':')[1]}", summary=summary, score=8, **kw)


def test_trim_drops_references_and_keeps_head_and_tail():
    text = "Intro " * 50 + "\nReferences\n" + "cite " * 100
    assert "cite" not in trim(text, 10_000)
    long = "a" * 1000 + "z" * 1000
    out = trim(long, 400)
    assert out.startswith("a") and out.endswith("z") and "[…]" in out and len(out) < 420


def test_fetch_arxiv_html(respx_mock):
    respx_mock.get("https://arxiv.org/html/2610.1").mock(return_value=httpx.Response(200, html=PAPER_HTML))
    with httpx.Client() as c:
        ft = fetch_fulltext(item("arxiv:2610.1"), c, {"max_chars": 40000}, want_pdf=True)
    assert ft.source == "full text (arXiv HTML)" and "generative ranking" in ft.text
    assert "should be removed" not in ft.text


def test_fetch_arxiv_pdf_fallback_only_with_llm(respx_mock):
    respx_mock.get("https://arxiv.org/html/2610.2").mock(return_value=httpx.Response(404))
    respx_mock.get("https://arxiv.org/pdf/2610.2").mock(return_value=httpx.Response(200, content=b"%PDF-1.7 ..."))
    with httpx.Client() as c:
        with_llm = fetch_fulltext(item("arxiv:2610.2"), c, {}, want_pdf=True)
        without = fetch_fulltext(item("arxiv:2610.2"), c, {}, want_pdf=False)
    assert with_llm.pdf and with_llm.source == "full text (PDF)"
    assert without.pdf is None and without.source == "abstract only" and without.text.startswith("An abstract")


def test_fetch_readme_and_article_and_403(respx_mock):
    respx_mock.get("https://raw.githubusercontent.com/o/r/HEAD/README.md").mock(
        return_value=httpx.Response(200, text="# R\nA useful tool for agents. " * 10))
    respx_mock.get("https://blog.example.com/reflex").mock(return_value=httpx.Response(200, html=ARTICLE_HTML))
    respx_mock.get("https://openai.com/index/x").mock(return_value=httpx.Response(403))
    with httpx.Client() as c:
        repo = fetch_fulltext(item("repo:o/r", kind="repo", url="https://github.com/o/r"), c, {}, want_pdf=False)
        blog = fetch_fulltext(item("url:b", kind="blog", url="https://blog.example.com/reflex"), c, {}, want_pdf=False)
        blocked = fetch_fulltext(item("url:o", kind="news", url="https://openai.com/index/x", summary="Feed text."),
                                 c, {}, want_pdf=False)
    assert repo.source == "README" and "useful tool" in repo.text
    assert blog.source == "full article" and "introspection" in blog.text and "menu" not in blog.text
    assert blocked.source == "feed summary only" and blocked.text == "Feed text."


def test_doi_paper_is_not_fetched(respx_mock):
    with httpx.Client() as c:
        ft = fetch_fulltext(item("doi:10.1145/1", url="https://doi.org/10.1145/1"), c, {}, want_pdf=True)
    assert ft.source == "abstract only" and not respx_mock.calls


def test_pick_targets_top_first_then_best_eligible():
    top = [item("arxiv:1"), item("url:c", kind="community", url="https://x/c")]
    sections = {"papers": [item("arxiv:2")], "repos": [item("repo:a/b", kind="repo", url="https://github.com/a/b")]}
    sections["papers"][0].score, sections["repos"][0].score = 7, 9
    got = pick_targets(top, sections, {"max_items": 3, "kinds": ["paper", "repo"]})
    assert [g.key for g in got] == ["arxiv:1", "repo:a/b", "arxiv:2"]


def test_extractive_fallback_picks_ideas_and_results():
    from news_agent.content import _extract_html

    blog = item("url:b", kind="blog", url="https://blog.example.com/reflex")
    ft = FullText(_extract_html(ARTICLE_HTML, blog.url), "full article", blog.url)
    apply_extractive(blog, ft)
    assert blog.deep_summary and not blog.deep_llm and blog.apply_at_iconic == []
    assert "We introduce Reflex" in blog.deep_summary and "Reflex:" not in blog.deep_summary  # heading dropped
    assert "consists of three layers" in blog.deep_summary + " ".join(blog.key_ideas)
    assert not set(blog.key_ideas) & set(blog.deep_summary.split(". "))
    assert "68%" in blog.results


class FakeModels:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def generate_content(self, model, contents, config):
        self.calls.append(contents)
        if self.fail:
            raise RuntimeError("boom")
        import json

        class R:
            text = json.dumps({"summary": "Detailed summary.", "key_ideas": ["Idea 1", "Idea 2", " "],
                               "results": "+4.2% GMV online.", "apply_at_iconic": ["Try it for PDP recs"]})
        return R()


class FakeClient:
    def __init__(self, fail=False):
        self.models = FakeModels(fail)


def _settings():
    s = load_settings()
    s.interests["deep_read"] = {**s.interests["deep_read"], "sleep_between_s": 0, "max_items": 2}
    s.interests["llm"] = {**s.interests["llm"], "retry_min_wait_s": 0, "max_retries": 1}
    return s


def test_deep_read_with_llm(respx_mock):
    respx_mock.get("https://arxiv.org/html/2610.1").mock(return_value=httpx.Response(200, html=PAPER_HTML))
    p = item("arxiv:2610.1", orgs=["Alibaba"])
    client = FakeClient()
    with httpx.Client() as http:
        assert deep_read([p], {}, _settings(), client=client, http=http) == 1
    assert p.deep_llm and p.deep_source == "full text (arXiv HTML)"
    assert p.key_ideas == ["Idea 1", "Idea 2"] and p.apply_at_iconic == ["Try it for PDP recs"]
    prompt = client.models.calls[0]
    assert "generative ranking" in prompt and "Alibaba" in prompt and "never invent numbers" in prompt


def test_deep_read_llm_failure_falls_back(respx_mock):
    respx_mock.get("https://arxiv.org/html/2610.1").mock(return_value=httpx.Response(200, html=PAPER_HTML))
    p = item("arxiv:2610.1")
    with httpx.Client() as http:
        deep_read([p], {}, _settings(), client=FakeClient(fail=True), http=http)
    assert not p.deep_llm and p.deep_summary and p.apply_at_iconic == []


def test_deep_read_skips_already_summarised_and_disabled():
    s = _settings()
    p = item("arxiv:1", deep_summary="cached")
    assert deep_read([p], {}, s, client=FakeClient()) == 0
    s.interests["deep_read"]["enabled"] = False
    assert deep_read([item("arxiv:2")], {}, s, client=FakeClient()) == 0


def test_prompt_mentions_pdf_when_attached():
    p = item("arxiv:1")
    prompt = build_prompt(p, FullText("", "full text (PDF)", p.url, pdf=b"%PDF"), _settings())
    assert "attached as a PDF" in prompt


def test_extractive_dedupes_repeat_introductions_and_latex():
    text = ("We introduce UserArena, a real-world benchmark with 64,729 sessions from shoppers.\n"
            "To address these limitations, we introduce UserArena, an interactive benchmark on a large platform.\n"
            "We propose OPTD, which distills teacher guidance into student rollouts for long horizons.\n"
            "Each package consists of \\(R=10\\) rounds and \\(19{,}419\\) judgments in total.\n")
    p = item("arxiv:9", summary="Short abstract about shopping agents and their evaluation.")
    apply_extractive(p, FullText(text, "full text (arXiv HTML)", p.url))
    assert sum("UserArena" in k for k in p.key_ideas) == 1
    assert any("OPTD" in k for k in p.key_ideas)
    assert all("\\(" not in k for k in p.key_ideas) and any("19,419" in k for k in p.key_ideas)


def test_delatex():
    from news_agent.deepread import _delatex

    assert _delatex(r"achieves \bar{R}=0.754 and 41.0\% with \(R=10\) and \(19{,}419\)") == "achieves R=0.754 and 41.0% with R=10 and 19,419"


def test_extractive_handles_hard_wrapped_abstract_and_enumerators():
    abstract = ("Agentic recommendation pursues shopper goals through sequences of context-dependent\n"
                "decisions in a store. We introduce UserArena, a real-world e-commerce interactive benchmark\n"
                "containing 64,729 session trajectories from 11,010 users.")
    full = ("We introduce UserArena, a real-world e-commerce interactive benchmark containing 64,729 sessions.\n"
            "(2) We propose OPTD, which uses trajectory-level intent to guide student-generated rollouts.\n")
    p = item("arxiv:7", summary=abstract)
    apply_extractive(p, FullText(full, "full text (arXiv HTML)", p.url))
    assert "UserArena" in p.deep_summary and p.deep_summary.endswith("users.")
    assert p.key_ideas == ["We propose OPTD, which uses trajectory-level intent to guide student-generated rollouts."]


def test_truncated_abstract_is_not_dropped():
    from news_agent.deepread import _sentences

    abstract = "First sentence of the abstract is here. Second sentence explains the method in detail and goes on…"
    assert _sentences(abstract, min_len=15)[0] == "First sentence of the abstract is here."
