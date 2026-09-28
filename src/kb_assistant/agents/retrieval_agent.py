"""Retrieval agent: hybrid search with a self-check.

No LLM call: routing and query rewriting already happened in the supervisor, and a deterministic
retriever is cheaper, faster and reproducible. The agent part is the corrective loop: it inspects
its own results and, if they are weak, broadens the search once (drops the narrowing filters)
before handing evidence on.
"""

from __future__ import annotations

import asyncio
from typing import Any

from langgraph.runtime import Runtime

from kb_assistant.agents.common import hit_to_evidence, merge_evidence, node_event
from kb_assistant.agents.events import emit
from kb_assistant.agents.state import AgentState, RunContext
from kb_assistant.retrieval.retriever import RetrievalResult, SearchFilters

# Cross-encoder logits below this mean "probably not relevant" for ms-marco-MiniLM.
WEAK_RERANK_SCORE = -2.0


def _is_weak(result: RetrievalResult) -> bool:
    if not result.hits:
        return True
    top = result.hits[0]
    return top.rerank_score is not None and top.rerank_score < WEAK_RERANK_SCORE


async def retrieval_agent(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    node_event("retrieval_agent", "started")
    ctx = runtime.context
    services = ctx.services
    decision = state.get("decision", {})
    step = state.get("current_step") or {}
    filters = SearchFilters(**{k: v for k, v in (decision.get("filters") or {}).items() if v})
    query = decision.get("standalone_query") or state["question"]
    queries = [query]
    task = step.get("task", "")
    if task and task.lower() != query.lower():
        queries.append(task)  # the supervisor's task phrasing often adds useful terms

    top_k = services.settings.retrieval_top_k
    results = await asyncio.gather(*(services.retriever.search(q, ctx.principal, filters, top_k) for q in queries))
    for r in results:
        emit({"type": "retrieval", **r.summary()})

    best = results[0]
    if _is_weak(best) and (filters.department or filters.date_from or filters.date_to or filters.document_types):
        emit({"type": "retrieval", "status": "weak results, retrying without filters", "query": query})
        retry = await services.retriever.search(query, ctx.principal, SearchFilters(), top_k)
        emit({"type": "retrieval", **retry.summary()})
        results.append(retry)

    evidence = list(state.get("evidence", []))
    for r in results:
        evidence = merge_evidence(evidence, [hit_to_evidence(h) for h in r.hits])
    evidence.sort(key=lambda e: e["score"], reverse=True)

    degraded = list(state.get("degraded", []))
    if any(r.degraded for r in results):
        degraded.append("retrieval: vector search unavailable, used keyword-only search")
    flags = sorted({c for r in results for c in r.flagged_chunks})
    security = list(state.get("security_flags", [])) + ([f"injection_in:{c}" for c in flags])
    return {"evidence": evidence[: top_k * 2], "degraded": sorted(set(degraded)), "security_flags": security}
