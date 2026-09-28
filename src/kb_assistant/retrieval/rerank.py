"""Second-stage reranking.

Hybrid search scores the query and each chunk independently (a bi-encoder plus BM25), which is fast
but blind to how the two relate. A cross-encoder reads query and chunk together and scores
relevance directly: more accurate, too slow for the whole corpus, cheap for ~20 candidates.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from kb_assistant.observability import get_logger
from kb_assistant.retrieval.store import Hit

log = get_logger(__name__)


class Reranker(Protocol):
    name: str

    async def rerank(self, query: str, hits: list[Hit]) -> list[Hit]: ...


class NoopReranker:
    name = "none"

    async def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        return hits


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)
        self.name = model_name

    def _score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, texts)]

    async def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        scores = await asyncio.to_thread(self._score, query, [h.text for h in hits])
        for hit, score in zip(hits, scores, strict=True):
            hit.rerank_score = score
        return sorted(hits, key=lambda h: h.rerank_score or 0.0, reverse=True)


def build_reranker(enabled: bool, model_name: str) -> Reranker:
    if not enabled:
        return NoopReranker()
    try:
        return CrossEncoderReranker(model_name)
    except Exception as exc:  # model download blocked, ONNX missing, ...
        log.warning("reranker_unavailable", model=model_name, error=str(exc))
        return NoopReranker()
