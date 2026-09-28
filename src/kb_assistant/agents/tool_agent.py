"""Tools agent: an LLM tool-calling loop, split into three nodes so a human can approve in between.

    tool_agent ──(tool calls)──► approval_gate ──► execute_tools ──► tool_agent ...
        └──(no calls / step limit)──► dispatch

Why three nodes: when a LangGraph node calls `interrupt()`, it re-runs from the top on resume. If the
LLM call and the interrupt shared a node, resuming would call the LLM again and might request a
different tool than the one the human approved. Here the approval gate re-runs cheaply and
execute_tools runs exactly the calls that were approved.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from kb_assistant.agents.common import merge_evidence, node_event, today
from kb_assistant.agents.events import emit
from kb_assistant.agents.prompts import TOOL_AGENT_SYSTEM, TOOL_AGENT_TASK
from kb_assistant.agents.state import AgentState, RunContext
from kb_assistant.errors import AssistantError
from kb_assistant.tools.registry import ToolContext


async def tool_agent(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    ctx = runtime.context
    services = ctx.services
    steps = state.get("tool_steps", 0)
    node_event("tool_agent", "started", step=steps + 1)
    history = list(state.get("tool_messages", []))
    if not history:
        step = state.get("current_step") or {}
        decision = state.get("decision", {})
        history = [
            SystemMessage(content=TOOL_AGENT_SYSTEM.format(brand=services.settings.brand_name)),
            HumanMessage(content=TOOL_AGENT_TASK.format(
                today=today(), task=step.get("task", ""),
                question=decision.get("standalone_query") or state["question"])),
        ]
    if steps >= services.settings.tool_agent_max_steps:
        emit({"type": "tool_agent", "status": "step limit reached", "steps": steps})
        return {"pending_tool_calls": [], "tool_messages": history}

    tools = [t.openai_schema() for t in services.registry.for_principal(ctx.principal)]
    try:
        reply = await services.llm.complete("tool_agent", history, tools=tools)
    except AssistantError as exc:
        degraded = list(state.get("degraded", [])) + [f"tools: {exc.public_message}"]
        return {"pending_tool_calls": [], "degraded": degraded, "tool_messages": history}

    calls = [{"id": c["id"], "name": c["name"], "args": c["args"]} for c in reply.tool_calls or []]
    emit({"type": "tool_agent", "status": "requested tools" if calls else "done",
          "calls": [{"name": c["name"], "args": c["args"]} for c in calls],
          "note": "" if calls else str(reply.content)[:300]})
    return {"tool_messages": history + [reply], "pending_tool_calls": calls, "tool_steps": steps + 1}


def route_after_tool_agent(state: AgentState) -> str:
    return "approval_gate" if state.get("pending_tool_calls") else "dispatch"


async def approval_gate(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Pause the graph for a human when a requested call needs approval (LangGraph interrupt)."""
    ctx = runtime.context
    registry = ctx.services.registry
    needs = []
    for call in state.get("pending_tool_calls", []):
        spec = registry.get(call["name"])
        if spec and spec.requires_approval and ctx.principal.can(spec.permission):
            needs.append(call)
    if not needs:
        return {"approval": None}

    node_event("approval_gate", "waiting_for_human", calls=[c["name"] for c in needs])
    decision = interrupt({
        "type": "approval_required",
        "calls": [{"id": c["id"], "tool": c["name"], "args": c["args"],
                   "description": registry.get(c["name"]).description} for c in needs],
        "message": "An administrative action needs your approval before it runs.",
    })
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    emit({"type": "approval", "approved": approved, "by": ctx.principal.user_id,
          "calls": [c["name"] for c in needs], "comment": (decision or {}).get("comment", "") if isinstance(decision, dict) else ""})
    return {"approval": {"approved": approved, "call_ids": [c["id"] for c in needs]}}


async def execute_tools(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    ctx = runtime.context
    services = ctx.services
    node_event("execute_tools", "started")
    approval = state.get("approval") or {}
    approved_ids = set(approval.get("call_ids", [])) if approval.get("approved") else set()
    rejected_ids = set(approval.get("call_ids", [])) - approved_ids

    tool_ctx = ToolContext(principal=ctx.principal, services=services, emit=emit,
                           datasets=dict(state.get("datasets", {})))
    messages: list[ToolMessage] = []
    results = list(state.get("tool_results", []))
    evidence = list(state.get("evidence", []))
    for call in state.get("pending_tool_calls", []):
        if call["id"] in rejected_ids:
            content = f"REJECTED: a human reviewer declined '{call['name']}'. Do not retry it."
            emit({"type": "tool_call", "tool": call["name"], "status": "rejected_by_human"})
            results.append({"tool": call["name"], "args": call["args"], "ok": False, "content": content})
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
            continue
        result = await services.executor.execute(call["name"], call["args"], tool_ctx,
                                                 approved=call["id"] in approved_ids)
        results.append({"tool": call["name"], "args": call["args"], "ok": result.ok,
                        "content": result.content[:3000], "error": result.error, "flags": result.flags})
        # Every tool call gets its own ToolMessage, or the next API call is rejected.
        messages.append(ToolMessage(content=result.content, tool_call_id=call["id"]))
        if call["name"] == "knowledge_search" and result.ok and isinstance(result.data, dict):
            evidence = merge_evidence(evidence, _search_evidence(result.data))

    datasets = {k: v for k, v in tool_ctx.datasets.items() if _small(v)}
    return {"tool_messages": list(state.get("tool_messages", [])) + messages, "pending_tool_calls": [],
            "tool_results": results, "datasets": datasets, "approval": None, "evidence": evidence}


def _search_evidence(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"chunk_id": r["chunk_id"], "doc_id": r["chunk_id"].split("#")[0], "title": r.get("title"),
             "section": r.get("section"), "document_type": None, "department": None, "access_level": None,
             "created_date": r.get("created_date"), "text": r.get("text", ""), "score": 0.0, "flags": []}
            for r in data.get("results", [])]


def _small(value: Any) -> bool:
    """Only keep datasets that are cheap to checkpoint."""
    try:
        return len(json.dumps(value, default=str)) < 200_000
    except (TypeError, ValueError):
        return False

