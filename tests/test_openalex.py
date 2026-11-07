"""Offline tests for the OpenAlex collector and arXiv enrichment (fixtures + respx)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import respx

from news_agent.collectors import COLLECTORS, openalex
from news_agent.models import Item

FIX = Path(__file__).parent / "fixtures"
UTC = timezone.utc
NOW = datetime(2026, 11, 7, 3, tzinfo=UTC)


def works():
    return json.loads((FIX / "openalex_works.json").read_text())


@pytest.fixture
def client():
    with httpx.Client() as c:
        yield c


def test_registered():
    assert COLLECTORS["openalex"] is openalex.collect


def test_arxiv_id_and_doi_helpers():
    assert openalex.bare_doi("https://doi.org/10.48550/arXiv.2611.02453") == "10.48550/arxiv.2611.02453"
    assert openalex.arxiv_id_of({"doi": "https://doi.org/10.48550/ARXIV.2611.02453"}) == "2611.02453"
    assert openalex.arxiv_id_of({"doi": None, "primary_location": {"landing_page_url": "http://arxiv.org/abs/2611.00001v3"}}) \
        == "2611.00001"
    assert openalex.arxiv_id_of({"doi": "https://doi.org/10.1145/1"}) is None


def test_abstract_reconstruction():
    assert openalex.abstract_text({"world": [1], "Hello": [0], "again": [3], "hello": [2]}) == "Hello world hello again"
    assert openalex.abstract_text(None) == ""


def test_parse_works():
    items = openalex.parse_works(works())
    assert len(items) == 5  # untitled and URL-less works skipped
    rec = items[0]
    assert rec.key == "arxiv:2611.02453" and rec.url == "https://arxiv.org/abs/2611.02453"
    assert rec.links == {"PDF": "https://arxiv.org/pdf/2611.02453", "OpenAlex": "https://openalex.org/W7225161487"}
    assert rec.kind == "paper" and rec.source == "openalex" and rec.source_name == "OpenAlex"
    assert rec.published == datetime(2026, 11, 2, tzinfo=UTC)
    assert rec.summary and "  " not in rec.summary
    ugr = items[1]
    assert ugr.affiliations[0] == "Alibaba Group (China)"
    assert len(ugr.affiliations) == len(set(ugr.affiliations)) <= openalex.MAX_AFFILIATIONS
    assert all(len(a) <= 120 for a in ugr.affiliations)
    cikm = next(i for i in items if i.key == "doi:10.1145/3799682.3840628")
    assert cikm.url == "https://doi.org/10.1145/3799682.3840628"
    assert cikm.links == {"OpenAlex": "https://openalex.org/W7225122106"}
    assert "Taobao & Tmall Group of Alibaba, Beijing, China" in cikm.affiliations  # double-escaped raw string
    assert cikm.metrics == {"citations": 0.0}
    assert all(i.url.startswith("https://") for i in items)


@respx.mock
def test_collect_params_dedupe_and_soft_failure(client):
    route = respx.get(openalex.API_URL).mock(side_effect=[
        httpx.Response(200, json=works()),
        httpx.Response(200, json=works()),
        httpx.Response(503),
    ])
    cfg = {"queries": ["recsys", "dup", "boom"], "rolling_days": 30, "per_page": 50, "mailto": "a@b.c", "arxiv_only": True}
    items = openalex.collect(cfg, NOW, client, now=NOW)
    assert len(items) == 5 and len({i.key for i in items}) == 5
    assert list(items.failures) == ["openalex:boom"]
    q = route.calls[0].request.url.params
    assert q["search"] == "recsys" and q["per_page"] == "50" and q["mailto"] == "a@b.c"
    assert q["filter"] == "from_publication_date:2026-10-08,primary_location.source.id:S4306400194"
    assert "abstract_inverted_index" in q["select"]


@respx.mock
def test_collect_raises_when_all_fail(client):
    respx.get(openalex.API_URL).mock(return_value=httpx.Response(429))
    with pytest.raises(RuntimeError):
        openalex.collect({"queries": ["a"]}, NOW, client, now=NOW)


def _paper(aid: str, **kw) -> Item:
    return Item(key=f"arxiv:{aid}", source="arxiv", source_name="arXiv", kind="paper", title=aid,
                url=f"https://arxiv.org/abs/{aid}", **kw)


@respx.mock
def test_enrich_fills_in_place(client):
    a, b, c = _paper("2610.39426"), _paper("2611.02453"), _paper("2611.99999", affiliations=["Zalando"])
    route = respx.get(openalex.API_URL).mock(return_value=httpx.Response(200, json={"results": [
        {"doi": "https://doi.org/10.48550/arxiv.2610.39426", "cited_by_count": 4,
         "authorships": [{"institutions": [{"display_name": "Alibaba Group (China)"}], "raw_affiliation_strings": []}]},
        {"doi": "https://doi.org/10.48550/arxiv.2611.02453", "cited_by_count": 1, "authorships": []},
    ]}))
    assert openalex.enrich([a, b, c], client, {"mailto": "a@b.c"}) == 1
    assert a.affiliations == ["Alibaba Group (China)"] and a.metrics["citations"] == 4
    assert b.affiliations == [] and b.metrics["citations"] == 1
    assert c.affiliations == ["Zalando"] and "citations" not in c.metrics  # already had affiliations: not looked up
    flt = route.calls[0].request.url.params["filter"]
    assert flt == "doi:10.48550/arxiv.2610.39426|10.48550/arxiv.2611.02453"


@respx.mock
def test_enrich_batches_and_fails_soft(client):
    papers = [_paper(f"2611.{i:05d}") for i in range(5)]
    route = respx.get(openalex.API_URL).mock(side_effect=[
        httpx.Response(200, json={"results": [
            {"doi": "https://doi.org/10.48550/arxiv.2611.00000", "cited_by_count": 0,
             "authorships": [{"institutions": [], "raw_affiliation_strings": ["Walmart Global Tech"]}]}]}),
        httpx.Response(500),
    ])
    assert openalex.enrich(papers, client, {"enrich_batch": 2}) == 1
    assert route.call_count == 2
    assert papers[0].affiliations == ["Walmart Global Tech"]
    assert openalex.enrich([], client, {}) == 0


def test_collect_excludes_doi_prefixes(respx_mock):
    import httpx
    from datetime import datetime, timezone
    from news_agent.collectors import openalex

    data = {"results": [
        {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.5281/zenodo.1", "title": "Zenodo upload"},
        {"id": "https://openalex.org/W2", "doi": "https://doi.org/10.1145/1.2", "title": "KDD paper"},
    ]}
    respx_mock.get(openalex.API_URL).mock(return_value=httpx.Response(200, json=data))
    with httpx.Client() as client:
        items = openalex.collect({"queries": ["q"], "exclude_doi_prefixes": ["10.5281/"]},
                                 datetime.now(timezone.utc), client)
    assert [i.key for i in items] == ["doi:10.1145/1.2"]
