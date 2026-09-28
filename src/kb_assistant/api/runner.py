"""Runs one graph turn and turns it into a stream of activity events for the client.

Stream modes used:
  custom   events agents emit (retrieval status, tool calls, RLM phases, tokens, validation ...)
  updates  one event per finished node (the "active node" in the activity panel) and interrupts

The last event is `final` (answer, citations, validation, explanation, usage) or
`approval_required` when the graph paused for a human.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langgraph.types import Command

from kb_assistant.agents.state import RunContext, new_turn
from kb_assistant.container import Services
from kb_assistant.observability import get_logger
from kb_assistant.security.rbac import Principal

log = get_logger(__name__)


def _config(thread_id: str, run_id: str, principal: Principal) -> dict[str, Any]:
    return {
        "configurable": {"thread_id": thread_id},
        "run_id": uuid.UUID(run_id),
        "run_name": "chat_turn",
        "tags": [f"role:{principal.role.value}"],
        "metadata": {"user_id": principal.user_id, "role": principal.role.value, "thread_id": thread_id},
        "recursion_limit": 40,
    }


def explain(values: dict[str, Any]) -> dict[str, Any]:
    """How the answer was produced, reconstructed from state rather than asked of the model: the
    model's own account of its reasoning can be wrong; the recorded steps cannot."""
    decision = values.get("decision") or {}
    research = values.get("research") or {}
    return {
        "intent": decision.get("intent"),
        "routing_reasoning": decision.get("reasoning"),
        "standalone_query": decision.get("standalone_query"),
        "plan": [f"{s['agent']}: {s['task']}" for s in values.get("plan", [])],
        "documents_consulted": sorted({e["doc_id"] for e in values.get("evidence", [])}),
        "tools_used": [f"{r['tool']} ({'ok' if r['ok'] else r.get('error') or 'failed'})"
                       for r in values.get("tool_results", [])],
        "research": {"documents_analysed": research.get("documents_analysed"),
                     "sub_agent_calls": research.get("sub_agent_calls"),
                     "plan_source": (research.get("plan") or {}).get("source")} if research else None,
        "validation": values.get("validation"),
        "degraded": values.get("degraded", []),
        "security_flags": values.get("security_flags", []),
    }


async def stream_turn(
    graph, services: Services, principal: Principal, thread_id: str, *, message: str | None = None,
    resume: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    run_id = str(uuid.uuid4())
    config = _config(thread_id, run_id, principal)
    ctx = RunContext(principal=principal, services=services, thread_id=thread_id)
    graph_input: Any = Command(resume=resume) if resume is not None else new_turn(message or "")
    usage = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    started = time.perf_counter()

    yield {"type": "run_started", "run_id": run_id, "thread_id": thread_id, "resumed": resume is not None}
    try:
        async for mode, chunk in graph.astream(graph_input, config, context=ctx, stream_mode=["updates", "custom"]):
            if mode == "custom":
                if chunk.get("type") == "llm_call":
                    usage["llm_calls"] += 1
                    usage["input_tokens"] += chunk.get("input_tokens", 0)
                    usage["output_tokens"] += chunk.get("output_tokens", 0)
                    usage["cost_usd"] = round(usage["cost_usd"] + chunk.get("cost_usd", 0.0), 6)
                yield chunk
                continue
            for node, update in chunk.items():
                if node == "__interrupt__":
                    continue  # reported below from the checkpoint, with its full payload
                yield {"type": "node_done", "node": node, "wrote": sorted((update or {}).keys())}
    except Exception as exc:  # last line of defence: the stream ends with an error event, not a 500
        log.exception("turn_failed", thread_id=thread_id)
        yield {"type": "error", "component": "graph", "detail": "The assistant failed to finish this turn.",
               "error": type(exc).__name__}
        return

    snapshot = await graph.aget_state(config)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if snapshot.next:
        payload = next((i.value for t in snapshot.tasks for i in t.interrupts), {})
        yield {"type": "approval_required", "thread_id": thread_id, "run_id": run_id, **payload}
        return

    values = snapshot.values
    log.info("turn_done", thread_id=thread_id, intent=(values.get("decision") or {}).get("intent"),
             elapsed_ms=elapsed_ms, **usage)
    yield {
        "type": "final", "run_id": run_id, "thread_id": thread_id, "answer": values.get("answer", ""),
        "citations": values.get("citations", []), "validation": values.get("validation", {}),
        "explanation": explain(values), "usage": usage, "elapsed_ms": elapsed_ms,
    }
