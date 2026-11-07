import json
from datetime import date

from news_agent.brief import build_brief, build_prompt, digest_items
from news_agent.config import load_settings
from news_agent.content import arxiv_figure, meta_image
from news_agent.models import DailyDigest, ScoredItem


def it(key, **kw):
    base = dict(key=key, source="x", source_name="X", kind="paper", title=f"T {key}", url=f"https://x/{key}",
                score=8, section="papers", tldr="TLDR sentence. Second.")
    base.update(kw)
    return ScoredItem(**base)


class Fake:
    def __init__(self, payload=None, fail=False):
        self.payload, self.fail, self.calls = payload, fail, []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(contents)
        if self.fail:
            raise RuntimeError("down")
        payload = self.payload

        class R:
            text = json.dumps(payload)
        return R()


def settings():
    s = load_settings()
    s.interests["llm"] = {**s.interests["llm"], "retry_min_wait_s": 0, "max_retries": 1}
    return s


def test_brief_with_llm_keeps_only_cited_highlights():
    a, b = it("a", deep_summary="Deep A.", key_ideas=["k1"]), it("b")
    d = DailyDigest(date=date(2026, 11, 7), top=[a], sections={"papers": [b, a]})
    client = Fake({"overview": "Big  day.", "highlights": [
        {"text": "A matters.", "item_ids": [a.id, a.id, "zzz"]},
        {"text": "Uncited.", "item_ids": ["zzz"]}]})
    build_brief(d, settings(), client=client)
    assert d.overview == "Big day."
    assert d.highlights == [{"text": "A matters.", "item_ids": [a.id]}]
    assert len(digest_items(d)) == 2
    prompt = client.calls[0]
    assert "Deep A." in prompt and "k1" in prompt and "MUST cite" in prompt


def test_brief_fallback_uses_deep_read_items():
    a = it("a", deep_summary="First sentence of A. More.", results="Gains of 4.2% GMV. Also more.")
    b = it("b", deep_summary="B opens here. Then.")
    c = it("c")
    d = DailyDigest(date=date(2026, 11, 7), top=[a, c], sections={"papers": [b]})
    build_brief(d, settings(), client=Fake(fail=True))
    assert d.overview == ""
    assert d.highlights == [{"text": "Gains of 4.2% GMV.", "item_ids": [a.id]},
                            {"text": "B opens here.", "item_ids": [b.id]}]


def test_build_prompt_lists_every_item():
    items = [it("a"), it("b")]
    p = build_prompt(items, settings())
    assert all(i.id in p for i in items)


def test_arxiv_figure_prefers_raster_overview_in_main_body():
    page = """
    <figure id="S1.F1" class="ltx_figure"><object type="image/svg+xml" data="v1/a.svg"></object>
      <figcaption>Figure 1: Overview of X (svg only).</figcaption></figure>
    <figure id="S2.F2" class="ltx_figure"><img src="v1/results.png"><figcaption>Figure 2: Results.</figcaption></figure>
    <figure id="S3.F3" class="ltx_figure"><img src="2610.1v1/arch.png"><figcaption>Figure 3: The <b>architecture</b> of X.</figcaption></figure>
    <figure id="A1.F9" class="ltx_figure"><img src="v1/appendix.png"><figcaption>Appendix overview.</figcaption></figure>
    """
    appendix_only = """
    <figure id="S1.F1" class="ltx_figure"><object data="a.svg"></object><figcaption>Overview.</figcaption></figure>
    <figure id="A1.F8" class="ltx_figure"><img src="2610.1v1/case.png"><figcaption>A case study.</figcaption></figure>
    <figure id="A1.F9" class="ltx_figure"><img src="2610.1v1/pipe.png"><figcaption>Annotation pipeline.</figcaption></figure>
    """
    assert arxiv_figure(appendix_only, "https://arxiv.org/html/2610.1")[0] == "https://arxiv.org/html/2610.1v1/pipe.png"
    assert arxiv_figure(appendix_only.replace("pipeline", "screen"), "https://arxiv.org/html/2610.1") == ("", "")
    url, cap = arxiv_figure(page, "https://arxiv.org/html/2610.1")
    assert url == "https://arxiv.org/html/2610.1v1/arch.png" and cap == "Figure 3: The architecture of X."
    assert arxiv_figure("<p>no figures</p>", "https://arxiv.org/html/1") == ("", "")


def test_meta_image_https_raster_only():
    assert meta_image('<meta property="og:image" content="/img/a.png">', "https://b.com/post") == "https://b.com/img/a.png"
    assert meta_image('<meta name="twitter:image" content="https://b.com/a.svg">', "https://b.com/") == ""
    assert meta_image('<meta property="og:image" content="http://insecure.com/a.png">', "https://b.com/") == ""


def test_arxiv_caption_drops_mathml_annotation():
    page = ('<figure id="S1.F1" class="ltx_figure"><img src="2610.1v1/a.png"><figcaption>Figure 1: rises from '
            '<math alttext="73.6\\%"><semantics><mn>73.6</mn><mo>%</mo><annotation encoding="application/x-tex">73.6\\%</annotation>'
            '</semantics></math> overall ( see text ).</figcaption></figure>')
    _, cap = arxiv_figure(page, "https://arxiv.org/html/2610.1")
    assert cap == "Figure 1: rises from 73.6% overall (see text)."
