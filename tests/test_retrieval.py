"""Chunking, BM25, hybrid retrieval, access control, filters and degradation."""

from __future__ import annotations

from kb_assistant import faults
from kb_assistant.config import PROJECT_ROOT
from kb_assistant.retrieval.documents import chunk_document, load_corpus, split_sections
from kb_assistant.retrieval.retriever import SearchFilters
from kb_assistant.retrieval.sparse import BM25Encoder, tokenize
from kb_assistant.retrieval.store import matches_filter

CORPUS = PROJECT_ROOT / "data" / "corpus"


def test_corpus_loads_with_valid_metadata():
    docs = load_corpus(CORPUS)
    assert len(docs) >= 50
    assert {d.access_level for d in docs} <= {"public", "internal", "confidential", "restricted"}


def test_chunks_follow_sections_and_carry_context_header():
    doc = next(d for d in load_corpus(CORPUS) if d.doc_id == "RB-001")
    sections = [h for h, _ in split_sections(doc.body)]
    chunks = chunk_document(doc)
    assert len(chunks) >= len(sections)
    assert all(c.text.startswith("[RB-001]") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_tokenizer_keeps_ids_and_their_parts():
    tokens = tokenize("INC-2025-014 hit payments-ledger")
    assert "inc-2025-014" in tokens and "payments-ledger" in tokens and "ledger" in tokens


def test_bm25_dot_product_prefers_exact_term():
    enc = BM25Encoder().fit(["pgbouncer pool exhausted", "certificate expired on gateway", "lunch menu"])
    q = enc.encode_query("pgbouncer")
    scores = [q.dot(enc.encode_document(t)) for t in ["pgbouncer pool exhausted", "certificate expired"]]
    assert scores[0] > 0 and scores[1] == 0


def test_filter_language_matches_pinecone_semantics():
    md = {"access_level": "internal", "created_ts": 20260101, "tags": ["tls", "gateway"]}
    assert matches_filter(md, {"$and": [{"access_level": {"$in": ["public", "internal"]}},
                                        {"created_ts": {"$gte": 20250928}}]})
    assert not matches_filter(md, {"created_ts": {"$lt": 20250101}})
    assert matches_filter(md, {"tags": "tls"})


async def test_viewer_never_retrieves_restricted_or_confidential(services, viewer, admin):
    query = "Project Falcon acquisition evaluation maximum offer"
    viewer_hits = await services.retriever.search(query, viewer, top_k=10)
    assert all(h.metadata["access_level"] in ("public", "internal") for h in viewer_hits.hits)
    assert "MTG-2026-008" not in {h.doc_id for h in viewer_hits.hits}
    admin_hits = await services.retriever.search(query, admin, top_k=10)
    assert "MTG-2026-008" in {h.doc_id for h in admin_hits.hits}


async def test_namespace_and_date_filters(services, analyst):
    result = await services.retriever.search(
        "payment outage root cause", analyst,
        SearchFilters(document_types=["incident"], date_from="2025-09-28"), top_k=10)
    assert result.namespaces == ["incident"]
    assert all(h.metadata["document_type"] == "incident" for h in result.hits)
    assert all(h.metadata["created_date"] >= "2025-09-28" for h in result.hits)


async def test_poisoned_document_is_sanitised_and_flagged(services, viewer):
    result = await services.retriever.search("PayCore vendor sync announcements biller hub", viewer,
                                             SearchFilters(document_types=["meeting_notes"]), top_k=10)
    flagged = [h for h in result.hits if "prompt_injection_removed" in h.flags]
    assert flagged, "the poisoned MTG-2026-007 chunk should be retrieved and flagged"
    assert all("exfil.example" not in h.text for h in result.hits)


async def test_vector_store_failure_degrades_to_keyword_search(services, viewer):
    faults.set_faults({"vectordb"})
    result = await services.retriever.search("CardNet Gateway certificate rotation", viewer, top_k=5)
    assert result.mode == "keyword_fallback" and result.degraded
    assert any(h.doc_id == "RB-001" for h in result.hits)


def test_catalog_hides_documents_by_role(services, viewer, admin):
    assert services.catalog.get(viewer, "MTG-2026-008") is None
    assert services.catalog.get(admin, "MTG-2026-008") is not None
    assert services.catalog.overview(viewer)["documents"] < services.catalog.overview(admin)["documents"]


class _FakePineconeIndex:
    """Captures the requests the adapter sends, in the pinecone v10 AsyncIndex call shape."""

    def __init__(self):
        self.queries: list[dict] = []

    async def query(self, **kwargs):
        from types import SimpleNamespace
        self.queries.append(kwargs)
        match = SimpleNamespace(id="RB-001#x", score=0.9, metadata={"doc_id": "RB-001", "access_level": "internal",
                                                                      "text": "t"})
        return SimpleNamespace(matches=[match])


async def test_pinecone_adapter_sends_hybrid_query_with_namespace_and_filter(viewer):
    from kb_assistant.retrieval.sparse import SparseVector
    from kb_assistant.retrieval.store import PineconeVectorStore

    store = PineconeVectorStore("key", "idx", 4, "aws", "us-east-1", timeout_s=2)
    fake = _FakePineconeIndex()
    store._index = fake
    flt = SearchFilters(document_types=["runbook"], date_from="2025-01-01").to_pinecone(viewer)
    hits = await store.query("runbook", [0.1, 0.2, 0.3, 0.4], SparseVector([7, 9], [0.5, 0.5]), 5, flt)
    sent = fake.queries[0]
    assert sent["namespace"] == "runbook" and sent["top_k"] == 5 and sent["include_metadata"]
    assert sent["sparse_vector"] == {"indices": [7, 9], "values": [0.5, 0.5]}
    assert {"access_level": {"$in": ["internal", "public"]}} in sent["filter"]["$and"]
    assert {"created_ts": {"$gte": 20250101}} in sent["filter"]["$and"]
    assert hits[0].chunk_id == "RB-001#x" and hits[0].namespace == "runbook"


async def test_pinecone_errors_become_vector_store_errors(viewer):
    import pytest

    from kb_assistant.errors import VectorStoreError
    from kb_assistant.retrieval.store import PineconeVectorStore

    class Broken:
        async def query(self, **kwargs):
            raise ConnectionError("pinecone down")

    store = PineconeVectorStore("key", "idx", 4, "aws", "us-east-1", timeout_s=2)
    store._index = Broken()
    with pytest.raises(VectorStoreError):
        await store.query("runbook", [0.1] * 4, None, 5, None)
