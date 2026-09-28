"""Response agent: writes the final answer from the evidence other agents gathered, streaming tokens.

The streamed text is a *draft*: the validator checks it afterwards and may send it back for one
rewrite. The UI shows the draft live and replaces it with the validated answer. Validating before
streaming would be safer but would remove streaming; this is the stated trade-off.

If the LLM is unavailable the agent still answers: it returns the most relevant passages verbatim,
with citations, and says the answer is extractive.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from kb_assistant.agents.common import format_evidence, node_event, today
from kb_assistant.agents.events import emit
from kb_assistant.agents.prompts import RESPONSE_CONTEXT, RESPONSE_SYSTEM, canary
from kb_assistant.agents.state import AgentState, RunContext
from kb_assistant.errors import AssistantError
from kb_assistant.observability import get_logger

log = get_logger(__name__)


def _research_block(research: dict[str, Any]) -> str:
    if not research:
        return ""
    report = research.get("report")
    agg = research.get("aggregates", {})
    counts = {k: v["count"] for k, v in agg.get("by_category", {}).items()}
    # The chunk id to cite for each analysed document, so the answer cites evidence, not "the report".
    cite_as = {f["doc_id"]: f["evidence_chunk_id"] for f in research.get("findings", [])}
    body = {
        "documents_analysed": research.get("documents_analysed"), "counts_by_root_cause (computed by code)": counts,
        "report": report, "cite_as (doc_id -> chunk id to put in brackets)": cite_as,
    }
    return "<research_report>\n" + json.dumps(body, indent=1)[:8000] + "\n</research_report>"


def _tools_block(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    parts = [f'<tool_result tool="{r["tool"]}" ok="{r["ok"]}">\n{r["content"][:2500]}\n</tool_result>' for r in results]
    return "Tool results (cite as [tool:<name>]):\n" + "\n".join(parts)


def build_messages(state: AgentState, ctx: RunContext) -> list:
    settings = ctx.services.settings
    principal = ctx.principal
    memory = state.get("memory_context", {})
    decision = state.get("decision", {})
    recalled = memory.get("recalled") or []
    feedback = state.get("validation_feedback", "")
    return [
        SystemMessage(content=RESPONSE_SYSTEM.format(
            assistant=settings.assistant_name, brand=settings.brand_name, canary=canary())),
        HumanMessage(content=RESPONSE_CONTEXT.format(
            today=today(), user_name=principal.name, role=principal.role.value, department=principal.department,
            facts="; ".join(memory.get("facts", [])) or "none", summary=memory.get("summary") or "none",
            recalled="; ".join(r["question"] for r in recalled) or "none",
            intent=decision.get("intent", "unknown"),
            degraded="; ".join(state.get("degraded", [])) or "none",
            # A research turn carries its findings in the report block, so fewer raw chunks are needed.
            evidence=format_evidence(state.get("evidence", [])[:8 if state.get("research") else 14]),
            research=_research_block(state.get("research", {})),
            tools=_tools_block(state.get("tool_results", [])),
            feedback=(f"IMPORTANT - your previous draft was rejected by the validator:\n{feedback}\n"
                      "Rewrite the answer fixing every point.\n") if feedback else "",
            question=decision.get("standalone_query") or state["question"])),
    ]


def extractive_answer(state: AgentState) -> str:
    evidence = state.get("evidence", [])[:3]
    research = state.get("research") or {}
    lines = ["The language model is unavailable, so here are the most relevant passages from the documents "
             "you can access, quoted directly:"]
    by_cat = (research.get("aggregates") or {}).get("by_category") or {}
    if by_cat:
        lines.append("\n**Root-cause counts (computed from the documents):**")
        lines += [f"- {k}: {v['count']} ({', '.join(v['doc_ids'])})" for k, v in by_cat.items()]
    for e in evidence:
        snippet = e["text"].split("\n", 1)[-1][:400].strip()
        lines.append(f"\n> {snippet} [{e['chunk_id']}]")
    for r in state.get("tool_results", [])[:2]:
        if r["ok"]:
            lines.append(f"\n{r['tool']}: {r['content'][:300]} [tool:{r['tool']}]")
    if len(lines) == 1:
        lines = ["The language model is unavailable and no documents matched, so I can't answer right now. "
                 "Please try again in a few minutes."]
    return "\n".join(lines)


async def response_agent(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    attempt = state.get("validation_attempts", 0)
    node_event("response_agent", "started", attempt=attempt + 1)
    ctx = runtime.context
    emit({"type": "response", "status": "generating", "attempt": attempt + 1})
    if attempt:
        emit({"type": "token_reset"})  # tell the UI to clear the rejected draft
    parts: list[str] = []
    try:
        async for token in ctx.services.llm.stream("response", build_messages(state, ctx)):
            parts.append(token)
            emit({"type": "token", "text": token})
        draft = "".join(parts)
        degraded = state.get("degraded", [])
    except AssistantError as exc:
        log.warning("response_degraded", error=str(exc))
        draft = extractive_answer(state)
        degraded = list(state.get("degraded", [])) + ["response: LLM unavailable, extractive answer"]
        emit({"type": "token_reset"})
        emit({"type": "token", "text": draft})
    return {"draft": draft, "degraded": degraded}
