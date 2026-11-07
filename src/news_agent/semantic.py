"""Local text embeddings for semantic ranking (fastembed / ONNX; no API calls)."""
from __future__ import annotations

import hashlib
import logging
import math
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .config import Settings

log = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class FastEmbedder:
    """bge-style model via fastembed. Queries and documents use their own prompts (query_embed/passage_embed)."""

    def __init__(self, model: str, cache_dir: str | None = None):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name=model, cache_dir=cache_dir)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self.model.query_embed(texts)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self.model.passage_embed(texts, batch_size=64)]


class HashEmbedder:
    """Deterministic bag-of-words embedder (tests / offline fallback). Not semantically smart."""

    def __init__(self, dim: int = 8192):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            if len(w) > 2:
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


_CACHE: dict[tuple, Embedder] = {}


def get_embedder(settings: "Settings") -> Embedder:
    cfg = settings.interests.get("semantic") or {}
    model = cfg.get("model", "BAAI/bge-small-en-v1.5")
    cache_dir = cfg.get("cache_dir")
    if cache_dir and not Path(cache_dir).is_absolute():
        cache_dir = str(settings.root / cache_dir)
    key = (model, cache_dir)
    if key not in _CACHE:
        _CACHE[key] = FastEmbedder(model, cache_dir)
    return _CACHE[key]


def cosine_matrix(docs: list[list[float]], queries: list[list[float]]) -> list[list[float]]:
    """Cosine similarity [doc][query]; vectors are normalised first."""
    def norm(v: list[float]) -> list[float]:
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    try:
        import numpy as np

        d = np.asarray(docs, dtype=float)
        q = np.asarray(queries, dtype=float)
        d /= np.linalg.norm(d, axis=1, keepdims=True).clip(min=1e-12)
        q /= np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
        return (d @ q.T).tolist()
    except ImportError:  # pragma: no cover
        qn = [norm(q) for q in queries]
        return [[sum(a * b for a, b in zip(norm(dv), qv)) for qv in qn] for dv in docs]


def item_text(item: Any, max_chars: int = 900) -> str:
    """Text used to embed an item: title + summary (+ tags for repos/models)."""
    parts = [item.title or "", (item.summary or "")[:max_chars]]
    if item.kind in ("repo", "model") and item.tags:
        parts.append(" ".join(t for t in item.tags if ":" not in t)[:200])
    return ". ".join(p for p in parts if p)
