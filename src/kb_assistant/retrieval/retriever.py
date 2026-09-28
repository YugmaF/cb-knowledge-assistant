"""Hybrid retrieval with access control, reranking, sanitisation and graceful degradation.

    query ──► dense (bge) ─┐
          └─► sparse (BM25)┴─► per-namespace queries, concurrently ──► merge ──► rerank ──► sanitise
                                  ▲ filter = caller's filters AND access_level ∈ role's levels

If the vector store fails, retrieval falls back to an in-process BM25 index over the same chunks
and marks the result `degraded`, so the answer can say its evidence may be incomplete.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langsmith import traceable

from kb_assistant.errors import VectorStoreError
from kb_assistant.observability import get_logger
from kb_assistant.retrieval.documents import DOCUMENT_TYPES, date_to_ts
from kb_assistant.retrieval.embeddings import Embedder
from kb_assistant.retrieval.rerank import Reranker
from kb_assistant.retrieval.sparse import BM25Encoder
from kb_assistant.retrieval.store import Hit, LocalVectorStore, VectorStore, matches_filter
from kb_assistant.security.guards import sanitize_retrieved
from kb_assistant.security.rbac import Principal

log = get_logger(__name__)


@dataclass
class SearchFilters:
    """Filters the model (or the user) may choose. Access level is deliberately absent: it is
    derived from the caller's role and cannot be supplied."""

    document_types: list[str] | None = None
    department: str | None = None
    date_from: str | None = None
    date_to: str | None = None

    def to_pinecone(self, principal: Principal) -> dict[str, Any]:
        clauses: list[dict[str, Any]] = [{"access_level": {"$in": sorted(principal.access_levels)}}]
        if self.department:
            clauses.append({"department": {"$eq": self.department}})
        if self.date_from:
            clauses.append({"created_ts": {"$gte": date_to_ts(self.date_from)}})
        if self.date_to:
            clauses.append({"created_ts": {"$lte": date_to_ts(self.date_to)}})
        return {"$and": clauses}

    def namespaces(self) -> list[str]:
        wanted = [t for t in (self.document_types or []) if t in DOCUMENT_TYPES]
        return wanted or list(DOCUMENT_TYPES)


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    mode: str  # "hybrid" | "keyword_fallback"
    degraded: bool
    namespaces: list[str]
    filter: dict[str, Any]
    elapsed_ms: int
    errors: list[str] = field(default_factory=list)
    flagged_chunks: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """Compact form for the activity panel and traces."""
        return {
            "query": self.query, "mode": self.mode, "degraded": self.degraded,
            "namespaces": self.namespaces, "elapsed_ms": self.elapsed_ms,
            "hits": [
                {"chunk_id": h.chunk_id, "score": round(h.score, 4),
                 "rerank": None if h.rerank_score is None else round(h.rerank_score, 3),
                 "access_level": h.metadata.get("access_level"), "flags": h.flags}
                for h in self.hits
            ],
            "flagged_chunks": self.flagged_chunks, "errors": self.errors,
        }


class KeywordIndex:
    """In-process BM25 over the chunk cache written at ingest time: the degraded-mode index."""

    def __init__(self, store: LocalVectorStore) -> None:
        self._records = store.records()

    def search(self, sparse_query, flt: dict[str, Any], top_k: int, namespaces: list[str]) -> list[Hit]:
        hits = [
            Hit(rec.id, sparse_query.dot(rec.sparse), rec.metadata, ns, sparse_score=sparse_query.dot(rec.sparse))
            for ns, rec in self._records
            if ns in namespaces and matches_filter(rec.metadata, flt)
        ]
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def fetch(self, ids: set[str]) -> list[Hit]:
        return [Hit(rec.id, 1.0, rec.metadata, ns) for ns, rec in self._records if rec.id in ids]


class HybridRetriever:
    def __init__(
        self, store: VectorStore, embedder: Embedder, bm25: BM25Encoder, reranker: Reranker,
        keyword_index: KeywordIndex, alpha: float, candidates: int,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker
        self.keyword_index = keyword_index
        self.alpha = alpha
        self.candidates = candidates

    @traceable(run_type="retriever", name="hybrid_search")
    async def search(
        self, query: str, principal: Principal, filters: SearchFilters | None = None, top_k: int = 6,
    ) -> RetrievalResult:
        started = time.perf_counter()
        filters = filters or SearchFilters()
        flt = filters.to_pinecone(principal)
        namespaces = filters.namespaces()
        sparse = self.bm25.encode_query(query)
        errors: list[str] = []
        mode = "hybrid"

        try:
            dense = await self.embedder.embed_query(query)
            dense_q = [v * self.alpha for v in dense]
            sparse_q = sparse.scaled(1 - self.alpha)
            results = await asyncio.gather(
                *(self.store.query(ns, dense_q, sparse_q, self.candidates, flt) for ns in namespaces),
                return_exceptions=True,
            )
            hits: list[Hit] = []
            for ns, res in zip(namespaces, results, strict=True):
                if isinstance(res, BaseException):
                    errors.append(f"{ns}: {res}")
                else:
                    hits.extend(res)
            if errors and len(errors) == len(namespaces):
                raise VectorStoreError("; ".join(errors))
        except VectorStoreError as exc:
            log.warning("vector_store_degraded", error=str(exc))
            mode = "keyword_fallback"
            errors = errors or [str(exc)]
            hits = self.keyword_index.search(sparse, flt, self.candidates, namespaces)

        # Defence in depth: even if a store ignored the filter, never return what the role can't read.
        hits = [h for h in hits if principal.can_read(h.metadata.get("access_level", "restricted"))]
        hits.sort(key=lambda h: h.score, reverse=True)
        hits = await self._safe_rerank(query, hits[: self.candidates])
        hits = hits[:top_k]
        flagged = self._sanitize(hits)

        return RetrievalResult(
            query=query, hits=hits, mode=mode, degraded=mode != "hybrid" or bool(errors),
            namespaces=namespaces, filter=flt, elapsed_ms=int((time.perf_counter() - started) * 1000),
            errors=errors, flagged_chunks=flagged,
        )

    async def _safe_rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        try:
            return await self.reranker.rerank(query, hits)
        except Exception as exc:  # a reranker failure costs precision, not the answer
            log.warning("rerank_failed", error=str(exc))
            return hits

    @staticmethod
    def _sanitize(hits: list[Hit]) -> list[str]:
        flagged: list[str] = []
        for hit in hits:
            result = sanitize_retrieved(hit.text)
            if result.flagged:
                hit.metadata = {**hit.metadata, "text": result.text}
                hit.flags.append("prompt_injection_removed")
                flagged.append(hit.chunk_id)
                log.warning("retrieved_injection_neutralised", chunk_id=hit.chunk_id, findings=result.findings)
        return flagged

    @traceable(run_type="retriever", name="fetch_sections")
    async def fetch_chunks(self, principal: Principal, chunk_ids: list[str], namespace: str) -> list[Hit]:
        """Targeted section fetch for the research agent (RLM): read only the sections it chose."""
        try:
            hits = await self.store.fetch(namespace, chunk_ids)
        except VectorStoreError:
            hits = self.keyword_index.fetch(set(chunk_ids))
        hits = [h for h in hits if principal.can_read(h.metadata.get("access_level", "restricted"))]
        self._sanitize(hits)
        return hits


@dataclass
class CatalogEntry:
    doc_id: str
    title: str
    department: str
    document_type: str
    access_level: str
    created_date: str
    tags: list[str]
    sections: dict[str, str]  # section heading -> first chunk id

    @property
    def created_ts(self) -> int:
        return date_to_ts(self.created_date)


class DocumentCatalog:
    """Metadata-only view of the corpus. The research agent explores this (counts, titles,
    dates, section names) before reading any document text: it decides what to read first."""

    def __init__(self, entries: list[CatalogEntry]) -> None:
        self._entries = entries

    @classmethod
    def load(cls, path: Path) -> DocumentCatalog:
        return cls([CatalogEntry(**row) for row in json.loads(path.read_text())])

    def visible(self, principal: Principal) -> list[CatalogEntry]:
        return [e for e in self._entries if principal.can_read(e.access_level)]

    def list_documents(
        self, principal: Principal, document_type: str | None = None, department: str | None = None,
        date_from: str | None = None, date_to: str | None = None, tag: str | None = None,
    ) -> list[CatalogEntry]:
        out = []
        for e in self.visible(principal):
            if document_type and e.document_type != document_type:
                continue
            if department and e.department != department:
                continue
            if date_from and e.created_ts < date_to_ts(date_from):
                continue
            if date_to and e.created_ts > date_to_ts(date_to):
                continue
            if tag and tag not in e.tags:
                continue
            out.append(e)
        return out

    def get(self, principal: Principal, doc_id: str) -> CatalogEntry | None:
        return next((e for e in self.visible(principal) if e.doc_id == doc_id), None)

    def overview(self, principal: Principal) -> dict[str, Any]:
        visible = self.visible(principal)
        by_type: dict[str, int] = {}
        by_dept: dict[str, int] = {}
        for e in visible:
            by_type[e.document_type] = by_type.get(e.document_type, 0) + 1
            by_dept[e.department] = by_dept.get(e.department, 0) + 1
        dates = sorted(e.created_date for e in visible)
        return {
            "documents": len(visible), "by_type": by_type, "by_department": by_dept,
            "date_range": [dates[0], dates[-1]] if dates else None,
        }
