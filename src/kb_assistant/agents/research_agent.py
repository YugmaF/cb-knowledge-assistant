"""Research agent: a Recursive Language Model (RLM) over the document collection.

Instead of loading whole documents into one context, the agent treats the corpus as an environment
it explores with code and sub-calls:

  1. explore     read the catalog: counts, dates, titles, section names. No document text.
  2. plan        the LLM writes a Python search plan (search() + list_documents()) that selects
                 documents and the sections worth reading. The plan runs in the sandbox.
  3. fetch       read only the chosen sections of the chosen documents ("targeted sections").
  4. decompose   split the documents into batches that fit a sub-agent's context budget.
  5. recurse     one sub-agent (LLM call) per batch, run concurrently. A batch over budget is split
                 in half and each half analysed by a deeper sub-agent. Sub-agents may request
                 follow-up queries; those are searched and analysed one level deeper.
  6. aggregate   code, not the model, counts categories and builds the timeline.
  7. reduce      one LLM call writes the report from the findings and the code-computed numbers.

Every sub-agent output is checked against its input before it is used: a finding for a document that
was not in the batch is discarded, an evidence id that was not in the batch is replaced.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, defaultdict
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langsmith import traceable

from kb_assistant.agents.common import hit_to_evidence, merge_evidence, node_event, today
from kb_assistant.agents.events import emit
from kb_assistant.agents.prompts import (
    RESEARCH_MAP_SYSTEM,
    RESEARCH_MAP_TASK,
    RESEARCH_PLAN_SYSTEM,
    RESEARCH_PLAN_TASK,
    RESEARCH_REDUCE_SYSTEM,
    RESEARCH_REDUCE_TASK,
)
from kb_assistant.agents.state import (
    AgentState,
    BatchFindings,
    DocFinding,
    ResearchReport,
    RunContext,
    SearchPlan,
)
from kb_assistant.errors import AssistantError, SandboxError
from kb_assistant.observability import get_logger
from kb_assistant.retrieval.retriever import SearchFilters
from kb_assistant.security.rbac import Principal
from kb_assistant.tools.sandbox import run_sandboxed

log = get_logger(__name__)

BATCH_CHAR_BUDGET = 9000  # a sub-agent's document budget (~2.3k tokens) before it recurses


class ResearchEnv:
    """The functions a search plan may call. They run in the sandbox's worker thread and call back
    into the event loop for async retrieval. Every function applies the caller's access level."""

    def __init__(self, ctx: RunContext, loop: asyncio.AbstractEventLoop, timeout_s: float) -> None:
        self._ctx = ctx
        self._loop = loop
        self._timeout = timeout_s
        self.calls: list[str] = []

    def _principal(self) -> Principal:
        return self._ctx.principal

    def overview(self) -> dict[str, Any]:
        self.calls.append("overview()")
        return self._ctx.services.catalog.overview(self._principal())

    def list_documents(self, document_type=None, department=None, date_from=None, date_to=None, tag=None):
        self.calls.append(f"list_documents(type={document_type}, dept={department}, from={date_from}, to={date_to})")
        entries = self._ctx.services.catalog.list_documents(
            self._principal(), document_type, department, date_from, date_to, tag)
        return [{"doc_id": e.doc_id, "title": e.title, "document_type": e.document_type,
                 "department": e.department, "created_date": e.created_date, "tags": list(e.tags),
                 "sections": list(e.sections)} for e in entries]

    def search(self, query, document_type=None, department=None, date_from=None, date_to=None, top_k=20):
        self.calls.append(f"search({query!r}, type={document_type}, from={date_from})")
        filters = SearchFilters([document_type] if document_type else None, department, date_from, date_to)
        coro = self._ctx.services.retriever.search(str(query)[:300], self._principal(), filters, min(int(top_k), 30))
        result = asyncio.run_coroutine_threadsafe(coro, self._loop).result(self._timeout)
        return [{"doc_id": h.doc_id, "chunk_id": h.chunk_id, "title": h.metadata.get("title"),
                 "section": h.metadata.get("section"), "created_date": h.metadata.get("created_date"),
                 "department": h.metadata.get("department"), "score": round(h.score, 4)} for h in result.hits]

    def as_env(self) -> dict[str, Any]:
        return {"overview": self.overview, "list_documents": self.list_documents, "search": self.search}


def default_plan(question: str, filters: dict[str, Any], max_docs: int) -> str:
    """Deterministic fallback plan when the LLM's plan fails twice (or the LLM is down)."""
    doc_type = (filters.get("document_types") or ["incident"])[0]
    return (
        f"hits = search({question!r}, document_type={doc_type!r}, date_from={filters.get('date_from')!r}, top_k=30)\n"
        f"listed = list_documents(document_type={doc_type!r}, date_from={filters.get('date_from')!r}, "
        f"date_to={filters.get('date_to')!r})\n"
        "ids = []\n"
        "for h in hits:\n"
        "    if h['doc_id'] not in ids:\n"
        "        ids.append(h['doc_id'])\n"
        "for d in listed:\n"
        "    if d['doc_id'] not in ids:\n"
        "        ids.append(d['doc_id'])\n"
        f"result = {{'doc_ids': ids[:{max_docs}], 'sections': ['Summary', 'Root Cause', 'Action Items']}}\n"
    )


async def _make_plan(ctx: RunContext, question: str, filters: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ask the LLM for a search plan, run it; on failure feed the error back once, then fall back."""
    services = ctx.services
    settings = services.settings
    overview = services.catalog.overview(ctx.principal)
    emit({"type": "rlm", "phase": "explore", "overview": overview})
    env = ResearchEnv(ctx, asyncio.get_running_loop(), settings.vector_timeout_s + 5)
    messages = [
        SystemMessage(content=RESEARCH_PLAN_SYSTEM.format(max_docs=settings.research_max_docs)),
        HumanMessage(content=RESEARCH_PLAN_TASK.format(
            today=today(), overview=json.dumps(overview), question=question, filters=json.dumps(filters))),
    ]
    source = "llm"
    code = ""
    for attempt in (1, 2):
        try:
            plan = await services.llm.structured("research_plan", messages, SearchPlan)
            code = plan.code
            emit({"type": "rlm", "phase": "plan", "source": "llm", "attempt": attempt,
                  "rationale": plan.rationale, "code": code})
            outcome = await run_sandboxed(code, env.as_env(), timeout_s=settings.sandbox_timeout_s * 3)
            selection = _validate_selection(outcome.result, settings.research_max_docs)
            emit({"type": "rlm", "phase": "execute_plan", "ok": True, "calls": env.calls,
                  "selected": len(selection["doc_ids"]), "sections": selection["sections"]})
            return selection, {"source": source, "code": code, "calls": env.calls}
        except SandboxError as exc:
            emit({"type": "rlm", "phase": "execute_plan", "ok": False, "attempt": attempt, "error": str(exc)})
            messages += [HumanMessage(content=f"Your plan failed: {exc}. Return a corrected JSON plan.")]
        except AssistantError as exc:
            emit({"type": "rlm", "phase": "plan", "ok": False, "error": exc.public_message})
            break

    source = "fallback"
    code = default_plan(question, filters, settings.research_max_docs)
    emit({"type": "rlm", "phase": "plan", "source": "fallback", "code": code})
    outcome = await run_sandboxed(code, env.as_env(), timeout_s=settings.sandbox_timeout_s * 3)
    selection = _validate_selection(outcome.result, settings.research_max_docs)
    emit({"type": "rlm", "phase": "execute_plan", "ok": True, "calls": env.calls, "selected": len(selection["doc_ids"])})
    return selection, {"source": source, "code": code, "calls": env.calls}


def _validate_selection(result: Any, max_docs: int) -> dict[str, Any]:
    if not isinstance(result, dict) or not isinstance(result.get("doc_ids"), list):
        raise SandboxError("plan must set result = {'doc_ids': [...], 'sections': [...]}")
    doc_ids = [str(d) for d in result["doc_ids"] if isinstance(d, str)]
    sections = [str(s) for s in result.get("sections") or [] if isinstance(s, str)]
    if not doc_ids:
        raise SandboxError("plan selected no documents. Keep the date range, but drop keyword or title "
                           "filters: list_documents() by type and date, and let sub-agents judge relevance")
    return {"doc_ids": list(dict.fromkeys(doc_ids))[:max_docs], "sections": sections}


async def _fetch_sections(ctx: RunContext, doc_ids: list[str], sections: list[str]) -> list[dict[str, Any]]:
    """Read only the requested sections. Documents the caller may not read are silently absent
    from the catalog view, so a plan cannot reach them by guessing ids."""
    catalog = ctx.services.catalog
    wanted = {s.lower() for s in sections}
    by_namespace: dict[str, list[str]] = defaultdict(list)
    for doc_id in doc_ids:
        entry = catalog.get(ctx.principal, doc_id)
        if entry is None:
            continue
        chunk_ids = [cid for name, cid in entry.sections.items() if not wanted or name.lower() in wanted]
        by_namespace[entry.document_type].extend(chunk_ids or list(entry.sections.values())[:2])
    fetched = await asyncio.gather(*(
        ctx.services.retriever.fetch_chunks(ctx.principal, ids, ns) for ns, ids in by_namespace.items()))
    docs: dict[str, dict[str, Any]] = {}
    for hits in fetched:
        for hit in hits:
            ev = hit_to_evidence(hit)
            doc = docs.setdefault(hit.doc_id, {"doc_id": hit.doc_id, "title": ev["title"],
                                               "created_date": ev["created_date"], "chunks": []})
            doc["chunks"].append(ev)
    order = {d: i for i, d in enumerate(doc_ids)}
    return sorted(docs.values(), key=lambda d: order.get(d["doc_id"], 1_000))


def _enforce_window(ctx: RunContext, doc_ids: list[str], filters: dict[str, Any]) -> list[str]:
    """The date range the user asked for is a rule, so code enforces it; a generated plan that
    widened it (to "find more") would silently change the question being answered."""
    date_from, date_to = filters.get("date_from"), filters.get("date_to")
    if not (date_from or date_to):
        return doc_ids
    kept, dropped = [], []
    for doc_id in doc_ids:
        entry = ctx.services.catalog.get(ctx.principal, doc_id)
        if entry and (not date_from or entry.created_date >= date_from) and (not date_to or entry.created_date <= date_to):
            kept.append(doc_id)
        else:
            dropped.append(doc_id)
    if dropped:
        emit({"type": "validation", "target": "research plan", "ok": False,
              "detail": f"removed {len(dropped)} documents outside {date_from}..{date_to}: {dropped}"})
    return kept


def _doc_chars(doc: dict[str, Any]) -> int:
    return sum(len(c["text"]) for c in doc["chunks"])


def _render_batch(batch: list[dict[str, Any]]) -> str:
    parts = []
    for doc in batch:
        chunks = "\n".join(f'<chunk id="{c["chunk_id"]}">\n{c["text"]}\n</chunk>' for c in doc["chunks"])
        parts.append(f'<document doc_id="{doc["doc_id"]}" date="{doc["created_date"]}" title="{doc["title"]}">\n'
                     f"{chunks}\n</document>")
    return "\n".join(parts)


class RecursiveAnalyzer:
    def __init__(self, ctx: RunContext, question: str) -> None:
        self.ctx = ctx
        self.question = question
        self.settings = ctx.services.settings
        self.semaphore = asyncio.Semaphore(self.settings.research_concurrency)
        self.sub_agent_calls = 0
        self.discarded: list[str] = []

    @traceable(name="rlm_sub_agent", run_type="chain")
    async def analyze(self, batch: list[dict[str, Any]], depth: int, label: str) -> BatchFindings:
        size = sum(_doc_chars(d) for d in batch)
        if size > BATCH_CHAR_BUDGET and len(batch) > 1 and depth < self.settings.research_max_depth + 1:
            mid = len(batch) // 2
            emit({"type": "rlm", "phase": "recurse", "batch": label, "depth": depth, "chars": size,
                  "reason": "batch over context budget, splitting in two"})
            left, right = await asyncio.gather(
                self.analyze(batch[:mid], depth + 1, f"{label}.1"),
                self.analyze(batch[mid:], depth + 1, f"{label}.2"))
            return BatchFindings(findings=left.findings + right.findings,
                                 themes=list(dict.fromkeys(left.themes + right.themes))[:5],
                                 follow_up_queries=list(dict.fromkeys(left.follow_up_queries + right.follow_up_queries))[:2])

        emit({"type": "rlm", "phase": "batch_start", "batch": label, "depth": depth,
              "docs": [d["doc_id"] for d in batch]})
        async with self.semaphore:
            self.sub_agent_calls += 1
            messages = [SystemMessage(content=RESEARCH_MAP_SYSTEM), HumanMessage(content=RESEARCH_MAP_TASK.format(
                question=self.question, depth=depth, n=len(batch), documents=_render_batch(batch)))]
            try:
                findings = await self.ctx.services.llm.structured("research_map", messages, BatchFindings)
            except AssistantError as exc:
                emit({"type": "rlm", "phase": "batch_done", "batch": label, "ok": False, "error": exc.public_message})
                return BatchFindings(findings=[])
        findings = self._ground(findings, batch)
        emit({"type": "rlm", "phase": "batch_done", "batch": label, "depth": depth, "ok": True,
              "relevant": sum(f.relevant for f in findings.findings), "themes": findings.themes})
        return findings

    def _ground(self, findings: BatchFindings, batch: list[dict[str, Any]]) -> BatchFindings:
        """Layer-4 validation: a sub-agent may only report on documents it was given."""
        docs = {d["doc_id"]: d for d in batch}
        kept: list[DocFinding] = []
        for f in findings.findings:
            doc = docs.get(f.doc_id)
            if doc is None:
                self.discarded.append(f.doc_id)
                emit({"type": "validation", "target": "rlm finding", "ok": False,
                      "detail": f"sub-agent reported on {f.doc_id}, which was not in its batch: discarded"})
                continue
            chunk_ids = [c["chunk_id"] for c in doc["chunks"]]
            if f.evidence_chunk_id not in chunk_ids:
                f.evidence_chunk_id = chunk_ids[0]
            f.date = doc["created_date"]  # dates come from metadata, not from the model
            kept.append(f)
        findings.findings = kept
        return findings


def aggregate(findings: list[DocFinding]) -> dict[str, Any]:
    """Deterministic counting. The model interprets; code counts."""
    relevant = [f for f in findings if f.relevant and f.category != "not_applicable"]
    by_category: dict[str, list[str]] = defaultdict(list)
    for f in sorted(relevant, key=lambda f: f.date or ""):
        by_category[f.category].append(f.doc_id)
    by_month = Counter((f.date or "unknown")[:7] for f in relevant)
    return {
        "documents_analysed": len(findings), "relevant": len(relevant),
        "by_category": {k: {"count": len(v), "doc_ids": v}
                        for k, v in sorted(by_category.items(), key=lambda kv: -len(kv[1]))},
        "recurring": [k for k, v in by_category.items() if len(v) >= 2],
        "by_month": dict(sorted(by_month.items())),
    }


async def research_agent(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    node_event("research_agent", "started")
    ctx = runtime.context
    settings = ctx.services.settings
    decision = state.get("decision", {})
    question = decision.get("standalone_query") or state["question"]
    filters = decision.get("filters") or {}
    degraded = list(state.get("degraded", []))

    selection, plan_info = await _make_plan(ctx, question, filters)
    selection["doc_ids"] = _enforce_window(ctx, selection["doc_ids"], filters)
    docs = await _fetch_sections(ctx, selection["doc_ids"], selection["sections"])
    emit({"type": "rlm", "phase": "fetch", "documents": len(docs),
          "chunks": sum(len(d["chunks"]) for d in docs), "sections": selection["sections"] or "all"})

    batches = [docs[i:i + settings.research_batch_size] for i in range(0, len(docs), settings.research_batch_size)]
    emit({"type": "rlm", "phase": "decompose", "batches": len(batches), "batch_size": settings.research_batch_size})
    analyzer = RecursiveAnalyzer(ctx, question)
    results = await asyncio.gather(*(analyzer.analyze(b, 1, f"b{i + 1}") for i, b in enumerate(batches)))
    findings = [f for r in results for f in r.findings]
    themes = list(dict.fromkeys(t for r in results for t in r.themes))[:8]

    # Recursive exploration: follow-up questions raised by sub-agents are searched and analysed one
    # level deeper, bounded by research_max_depth and by documents not already analysed.
    follow_ups = list(dict.fromkeys(q for r in results for q in r.follow_up_queries))[:2]
    context_findings: list[DocFinding] = []
    if follow_ups and settings.research_max_depth > 1:
        seen = {d["doc_id"] for d in docs}
        extra_ids: list[str] = []
        for q in follow_ups:
            res = await ctx.services.retriever.search(q, ctx.principal, SearchFilters(), 4)
            extra_ids += [h.doc_id for h in res.hits if h.doc_id not in seen and h.doc_id not in extra_ids]
        extra_ids = list(dict.fromkeys(extra_ids))
        emit({"type": "rlm", "phase": "follow_up", "queries": follow_ups, "new_documents": extra_ids[:4]})
        if extra_ids:
            extra_docs = await _fetch_sections(ctx, extra_ids[:4], [])
            extra = await analyzer.analyze(extra_docs, 2, "follow-up")
            context_findings = extra.findings
            docs += extra_docs

    # Only the documents the plan selected are counted; follow-up documents (runbooks, reviews that
    # discuss the same incidents) are supporting context and would double-count.
    aggregates = aggregate(findings)
    findings = findings + context_findings
    emit({"type": "rlm", "phase": "aggregate", "by_category": {k: v["count"] for k, v in aggregates["by_category"].items()},
          "relevant": aggregates["relevant"], "sub_agent_calls": analyzer.sub_agent_calls})

    evidence = list(state.get("evidence", []))
    # Every chunk a finding cites must survive into the evidence, or the validator would reject a
    # correct citation as "not retrieved". Cited chunks go first; the rest fill up to the cap.
    relevant_findings = [f.model_dump() for f in findings if f.relevant]
    all_chunks = {c["chunk_id"]: c for d in docs for c in d["chunks"]}
    cited = [all_chunks[f["evidence_chunk_id"]] for f in relevant_findings if f["evidence_chunk_id"] in all_chunks]
    evidence = merge_evidence(evidence, cited, limit=200)
    evidence = merge_evidence(evidence, list(all_chunks.values()), limit=max(120, len(evidence)))

    report: ResearchReport | None = None
    try:
        messages = [SystemMessage(content=RESEARCH_REDUCE_SYSTEM), HumanMessage(content=RESEARCH_REDUCE_TASK.format(
            question=question, aggregates=json.dumps(aggregates, indent=1),
            findings=json.dumps(relevant_findings, indent=1)[:24000]))]
        report = await ctx.services.llm.structured("research_reduce", messages, ResearchReport)
        # Numbers come from code: overwrite any count the model changed.
        for cause in report.recurring_root_causes:
            if cause.category in aggregates["by_category"]:
                cause.count = aggregates["by_category"][cause.category]["count"]
                cause.incidents = aggregates["by_category"][cause.category]["doc_ids"]
        emit({"type": "rlm", "phase": "reduce", "ok": True, "recurring": [c.category for c in report.recurring_root_causes]})
    except AssistantError as exc:
        degraded.append("research: final synthesis unavailable, reporting code-computed aggregates only")
        emit({"type": "rlm", "phase": "reduce", "ok": False, "error": exc.public_message})

    research = {
        "question": question, "plan": plan_info, "documents_analysed": len(docs),
        "sub_agent_calls": analyzer.sub_agent_calls, "aggregates": aggregates, "themes": themes,
        "findings": relevant_findings, "report": report.model_dump() if report else None,
        "discarded_findings": analyzer.discarded,
    }
    return {"research": research, "evidence": evidence, "degraded": sorted(set(degraded))}
