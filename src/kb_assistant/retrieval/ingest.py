"""Ingestion: corpus -> chunks -> (dense, sparse) vectors -> vector store + catalog.

    uv run python -m kb_assistant.retrieval.ingest            # local store (+ Pinecone if keyed)

Writes into data/index/:
    local_store.jsonl   every chunk with its vectors (local store and degraded-mode BM25 index)
    bm25.json           fitted BM25 statistics (the query encoder must match the documents)
    catalog.json        metadata-only document catalog explored by the research agent
    manifest.json       embedding model, counts, timestamp: what the index was built with
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime

from kb_assistant.config import Settings, get_settings
from kb_assistant.observability import configure_logging, get_logger
from kb_assistant.retrieval.documents import chunk_document, load_corpus
from kb_assistant.retrieval.embeddings import build_embedder
from kb_assistant.retrieval.sparse import BM25Encoder
from kb_assistant.retrieval.store import LocalVectorStore, PineconeVectorStore, VectorRecord

log = get_logger(__name__)


async def ingest(settings: Settings) -> dict:
    docs = load_corpus(settings.corpus_dir)
    chunks = [c for d in docs for c in chunk_document(d, settings.chunk_max_chars, settings.chunk_overlap_chars)]
    log.info("ingest_loaded", documents=len(docs), chunks=len(chunks))

    bm25 = BM25Encoder().fit([c.text for c in chunks])
    embedder = build_embedder(settings.embedding_model, settings.embedding_dim)
    dense = await embedder.embed_documents([c.text for c in chunks])

    by_namespace: dict[str, list[VectorRecord]] = defaultdict(list)
    for chunk, vec in zip(chunks, dense, strict=True):
        by_namespace[chunk.document_type].append(
            VectorRecord(chunk.chunk_id, vec, bm25.encode_document(chunk.text), chunk.metadata())
        )

    settings.index_dir.mkdir(parents=True, exist_ok=True)
    local = LocalVectorStore(settings.index_dir / "local_store.jsonl")
    for ns, records in by_namespace.items():
        await local.upsert(ns, records)
    local.save()
    bm25.save(settings.index_dir / "bm25.json")

    catalog = []
    for doc in docs:
        sections: dict[str, str] = {}
        for c in chunks:
            if c.doc_id == doc.doc_id:
                sections.setdefault(c.section, c.chunk_id)
        catalog.append({
            "doc_id": doc.doc_id, "title": doc.title, "department": doc.department,
            "document_type": doc.document_type, "access_level": doc.access_level,
            "created_date": doc.created_date, "tags": doc.tags, "sections": sections,
        })
    (settings.index_dir / "catalog.json").write_text(json.dumps(catalog, indent=1))

    pinecone_status = "skipped (no PINECONE_API_KEY)"
    if settings.use_pinecone:
        store = PineconeVectorStore(
            settings.pinecone_api_key, settings.pinecone_index, embedder.dim,
            settings.pinecone_cloud, settings.pinecone_region, timeout_s=60,
        )
        await store.ensure_index()
        for ns, records in by_namespace.items():
            await store.upsert(ns, records)
        pinecone_status = f"upserted to index '{settings.pinecone_index}'"

    manifest = {
        "built_at": datetime.now(UTC).isoformat(), "embedding_model": embedder.name,
        "documents": len(docs), "chunks": len(chunks),
        "namespaces": {ns: len(r) for ns, r in by_namespace.items()}, "pinecone": pinecone_status,
    }
    (settings.index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("ingest_done", **manifest)
    return manifest


def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    if "--if-missing" in sys.argv and (settings.index_dir / "manifest.json").exists():
        print(f"index already present in {settings.index_dir}, skipping ingestion")
        return
    print(json.dumps(asyncio.run(ingest(settings)), indent=2))


if __name__ == "__main__":
    main()
