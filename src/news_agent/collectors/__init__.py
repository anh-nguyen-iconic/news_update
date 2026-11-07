"""Collector registry and fail-soft parallel runner."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from ..models import Item
from . import arxiv, bluesky, goodailist, hackernews, hf, openalex, reddit, rss
from .base import Collector, make_client

log = logging.getLogger(__name__)

# Name in config/sources.yaml -> collector function.
COLLECTORS: dict[str, Collector] = {
    "arxiv": arxiv.collect,
    "hf_papers": hf.collect_papers,
    "openalex": openalex.collect,
    "hf_models": hf.collect_models,
    "rss": rss.collect,
    "goodailist": goodailist.collect,
    "hackernews": hackernews.collect,
    "reddit": reddit.collect,
    "bluesky": bluesky.collect,
}


def run_all(sources_cfg: dict[str, Any], since: datetime, only: list[str] | None = None) -> tuple[list[Item], dict[str, str]]:
    """Run every enabled collector in parallel.

    Returns ``(items, failures)`` where ``failures`` maps collector name to an error message.
    A collector that raises never stops the run.
    """
    items: list[Item] = []
    failures: dict[str, str] = {}
    names = [n for n in COLLECTORS if sources_cfg.get(n, {}).get("enabled", False) and (not only or n in only)]
    with make_client(sources_cfg.get("http")) as client, ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(COLLECTORS[n], sources_cfg[n], since, client): n for n in names}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                got = fut.result()
                log.info("collector %-11s -> %d items", name, len(got))
                items.extend(got)
                failures.update(getattr(got, "failures", {}) or {})
            except Exception as exc:  # noqa: BLE001 - fail-soft by design
                log.warning("collector %s failed: %s", name, exc)
                failures[name] = f"{type(exc).__name__}: {exc}"[:300]
    return items, failures
