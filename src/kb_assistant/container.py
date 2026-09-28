"""Service wiring: builds every dependency once at start-up and hands them to the graph as context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kb_assistant.agents.llm import LLM, LLMGateway
from kb_assistant.agents.memory import MemoryStore
from kb_assistant.config import Settings
from kb_assistant.observability import get_logger
from kb_assistant.retrieval.embeddings import Embedder, build_embedder
from kb_assistant.retrieval.rerank import build_reranker
from kb_assistant.retrieval.retriever import DocumentCatalog, HybridRetriever, KeywordIndex
from kb_assistant.retrieval.sparse import BM25Encoder
from kb_assistant.retrieval.store import LocalVectorStore, PineconeVectorStore, VectorStore
from kb_assistant.security.rate_limit import RateLimiter
from kb_assistant.tools.builtin import build_registry
from kb_assistant.tools.mcp_gateway import MCPGateway
from kb_assistant.tools.registry import ToolExecutor, ToolRegistry

log = get_logger(__name__)


@dataclass
class Services:
    settings: Settings
    llm: LLM
    embedder: Embedder
    store: VectorStore
    retriever: HybridRetriever
    catalog: DocumentCatalog
    memory: MemoryStore
    registry: ToolRegistry
    executor: ToolExecutor
    mcp: MCPGateway
    rate_limiter: RateLimiter


def build_services(settings: Settings, *, llm: LLM | None = None, mcp_target: Any = None,
                   embedder: Embedder | None = None, rerank: bool | None = None) -> Services:
    index_dir = settings.index_dir
    if not (index_dir / "manifest.json").exists():
        raise RuntimeError(f"No index in {index_dir}. Run: uv run python -m kb_assistant.retrieval.ingest")

    embedder = embedder or build_embedder(settings.embedding_model, settings.embedding_dim)
    local = LocalVectorStore(index_dir / "local_store.jsonl")
    store: VectorStore = local
    if settings.use_pinecone:
        store = PineconeVectorStore(settings.pinecone_api_key, settings.pinecone_index, embedder.dim,
                                    settings.pinecone_cloud, settings.pinecone_region, settings.vector_timeout_s)
    reranker = build_reranker(settings.rerank_enabled if rerank is None else rerank, settings.rerank_model)
    retriever = HybridRetriever(
        store=store, embedder=embedder, bm25=BM25Encoder.load(index_dir / "bm25.json"), reranker=reranker,
        keyword_index=KeywordIndex(local), alpha=settings.hybrid_alpha, candidates=settings.retrieval_candidates,
    )
    registry = build_registry()
    services = Services(
        settings=settings,
        llm=llm or LLMGateway(settings),
        embedder=embedder,
        store=store,
        retriever=retriever,
        catalog=DocumentCatalog.load(index_dir / "catalog.json"),
        memory=MemoryStore(settings.memory_db_path, embedder),
        registry=registry,
        executor=ToolExecutor(registry, settings.tool_timeout_s),
        mcp=MCPGateway(mcp_target or settings.mcp_url, settings.mcp_timeout_s),
        rate_limiter=RateLimiter(settings.rate_limits),
    )
    log.info("services_ready", vector_store=store.name, reranker=reranker.name, embedder=embedder.name,
             llm_configured=settings.llm_configured)
    return services
