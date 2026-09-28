"""Vector stores.

`PineconeVectorStore` is the production store the brief asks for. `LocalVectorStore` implements the
same interface in-process (numpy + the same filter language) so the system runs with no cloud
account, tests run offline, and there is a keyword-only fallback when Pinecone is unreachable.

Pinecone layout
    index      one serverless index, metric=dotproduct (required for sparse-dense vectors)
    namespace  one per document_type (incident, policy, runbook, ...). Types have different
               lifecycles: incidents are appended daily, policies re-indexed on revision. A
               namespace can be rebuilt alone, and a query that only needs incidents never scans
               policies. Cross-type queries fan out to namespaces concurrently (asyncio.gather).
    metadata   department, document_type, access_level, created_date, created_ts (int for range
               filters), doc_id, title, section, text (for attribution without a second lookup)
    hybrid     query = (alpha * dense, (1 - alpha) * sparse); Pinecone's dot product then equals
               alpha * dense_score + (1 - alpha) * sparse_score, a convex combination.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from kb_assistant import faults
from kb_assistant.errors import VectorStoreError
from kb_assistant.retrieval.sparse import SparseVector


@dataclass
class VectorRecord:
    id: str
    dense: list[float]
    sparse: SparseVector
    metadata: dict[str, Any]


@dataclass
class Hit:
    chunk_id: str
    score: float
    metadata: dict[str, Any]
    namespace: str
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.metadata.get("text", "")

    @property
    def doc_id(self) -> str:
        return self.metadata.get("doc_id", self.chunk_id.split("#")[0])


class VectorStore(Protocol):
    name: str

    async def upsert(self, namespace: str, records: list[VectorRecord]) -> None: ...

    async def query(
        self, namespace: str, dense: list[float], sparse: SparseVector | None, top_k: int,
        filter: dict[str, Any] | None,
    ) -> list[Hit]: ...

    async def fetch(self, namespace: str, ids: list[str]) -> list[Hit]: ...

    async def namespaces(self) -> dict[str, int]: ...


# --------------------------------------------------------------------------------------------
# Pinecone filter language, evaluated locally
# --------------------------------------------------------------------------------------------

def matches_filter(metadata: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    if not flt:
        return True
    for key, cond in flt.items():
        if key == "$and":
            if not all(matches_filter(metadata, sub) for sub in cond):
                return False
            continue
        if key == "$or":
            if not any(matches_filter(metadata, sub) for sub in cond):
                return False
            continue
        value = metadata.get(key)
        ops = cond if isinstance(cond, dict) else {"$eq": cond}
        for op, target in ops.items():
            if not _apply_op(op, value, target):
                return False
    return True


def _apply_op(op: str, value: Any, target: Any) -> bool:
    if isinstance(value, list):  # list metadata (tags): Pinecone matches if any element matches
        return any(_apply_op(op, v, target) for v in value) if op not in ("$ne", "$nin") else all(
            _apply_op(op, v, target) for v in value)
    match op:
        case "$eq":
            return value == target
        case "$ne":
            return value != target
        case "$in":
            return value in target
        case "$nin":
            return value not in target
        case "$gt":
            return value is not None and value > target
        case "$gte":
            return value is not None and value >= target
        case "$lt":
            return value is not None and value < target
        case "$lte":
            return value is not None and value <= target
        case "$exists":
            return (value is not None) == bool(target)
    raise ValueError(f"unsupported filter operator {op}")


# --------------------------------------------------------------------------------------------
# Local store
# --------------------------------------------------------------------------------------------

class LocalVectorStore:
    name = "local"

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._ns: dict[str, list[VectorRecord]] = {}
        self._matrix: dict[str, np.ndarray] = {}
        if path and path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rec = VectorRecord(row["id"], row["dense"], SparseVector(**row["sparse"]), row["metadata"])
            self._ns.setdefault(row["namespace"], []).append(rec)
        self._rebuild()

    def _rebuild(self) -> None:
        self._matrix = {
            ns: np.asarray([r.dense for r in recs], dtype=np.float32) for ns, recs in self._ns.items()
        }

    def save(self) -> None:
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w") as fh:
            for ns, recs in self._ns.items():
                for r in recs:
                    fh.write(json.dumps({
                        "namespace": ns, "id": r.id, "dense": r.dense,
                        "sparse": {"indices": r.sparse.indices, "values": r.sparse.values},
                        "metadata": r.metadata,
                    }) + "\n")

    def records(self) -> list[tuple[str, VectorRecord]]:
        return [(ns, r) for ns, recs in self._ns.items() for r in recs]

    async def upsert(self, namespace: str, records: list[VectorRecord]) -> None:
        existing = {r.id: r for r in self._ns.get(namespace, [])}
        existing.update({r.id: r for r in records})
        self._ns[namespace] = list(existing.values())
        self._rebuild()

    async def query(
        self, namespace: str, dense: list[float], sparse: SparseVector | None, top_k: int,
        filter: dict[str, Any] | None,
    ) -> list[Hit]:
        if faults.is_active("vectordb"):
            raise VectorStoreError("fault injection: vector store unavailable")
        recs = self._ns.get(namespace, [])
        if not recs:
            return []
        dense_scores = self._matrix[namespace] @ np.asarray(dense, dtype=np.float32)
        hits: list[Hit] = []
        for rec, d_score in zip(recs, dense_scores, strict=True):
            if not matches_filter(rec.metadata, filter):
                continue
            s_score = sparse.dot(rec.sparse) if sparse else 0.0
            # Same arithmetic as Pinecone's dot product over the pre-weighted query vectors.
            hits.append(Hit(rec.id, float(d_score) + s_score, rec.metadata, namespace,
                            dense_score=float(d_score), sparse_score=s_score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    async def fetch(self, namespace: str, ids: list[str]) -> list[Hit]:
        if faults.is_active("vectordb"):
            raise VectorStoreError("fault injection: vector store unavailable")
        wanted = set(ids)
        return [Hit(r.id, 1.0, r.metadata, namespace) for r in self._ns.get(namespace, []) if r.id in wanted]

    async def namespaces(self) -> dict[str, int]:
        return {ns: len(recs) for ns, recs in self._ns.items()}


# --------------------------------------------------------------------------------------------
# Pinecone
# --------------------------------------------------------------------------------------------

class PineconeVectorStore:
    name = "pinecone"

    def __init__(self, api_key: str, index_name: str, dim: int, cloud: str, region: str,
                 timeout_s: float) -> None:
        from pinecone import AsyncPinecone

        self._pc = AsyncPinecone(api_key=api_key, timeout=timeout_s)
        self._index_name = index_name
        self._dim = dim
        self._cloud = cloud
        self._region = region
        self._timeout = timeout_s
        self._index = None
        self._lock = asyncio.Lock()

    async def _get_index(self):
        async with self._lock:
            if self._index is None:
                self._index = await self._call(lambda: self._pc.index(self._index_name))
            return self._index

    async def ensure_index(self) -> None:
        from pinecone import ServerlessSpec

        if not await self._pc.has_index(self._index_name):
            await self._pc.create_index(
                name=self._index_name, dimension=self._dim, metric="dotproduct",
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )

    async def _call(self, coro_factory):
        if faults.is_active("vectordb"):
            raise VectorStoreError("fault injection: vector store unavailable")
        try:
            return await asyncio.wait_for(coro_factory(), timeout=self._timeout)
        except TimeoutError as exc:
            raise VectorStoreError("pinecone timeout") from exc
        except Exception as exc:  # the SDK raises many types; callers only need one
            raise VectorStoreError(f"pinecone error: {exc}") from exc

    async def upsert(self, namespace: str, records: list[VectorRecord]) -> None:
        index = await self._get_index()
        vectors = [
            {"id": r.id, "values": r.dense, "sparse_values": r.sparse.to_pinecone(), "metadata": r.metadata}
            for r in records if r.sparse.indices
        ]
        await self._call(lambda: index.upsert(vectors=vectors, namespace=namespace, batch_size=100,
                                              show_progress=False))

    async def query(
        self, namespace: str, dense: list[float], sparse: SparseVector | None, top_k: int,
        filter: dict[str, Any] | None,
    ) -> list[Hit]:
        index = await self._get_index()
        kwargs: dict[str, Any] = {"top_k": top_k, "vector": dense, "namespace": namespace,
                                  "filter": filter or None, "include_metadata": True}
        if sparse and sparse.indices:
            kwargs["sparse_vector"] = sparse.to_pinecone()
        response = await self._call(lambda: index.query(**kwargs))
        return [Hit(m.id, float(m.score), dict(m.metadata or {}), namespace) for m in response.matches]

    async def fetch(self, namespace: str, ids: list[str]) -> list[Hit]:
        index = await self._get_index()
        response = await self._call(lambda: index.fetch(ids=ids, namespace=namespace))
        return [Hit(vid, 1.0, dict(v.metadata or {}), namespace) for vid, v in response.vectors.items()]

    async def namespaces(self) -> dict[str, int]:
        index = await self._get_index()
        stats = await self._call(lambda: index.describe_index_stats())
        return {ns: int(getattr(info, "vector_count", 0)) for ns, info in (stats.namespaces or {}).items()}
