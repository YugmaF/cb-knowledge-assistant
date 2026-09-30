"""Nodes that frame each turn: input guard, memory load, memory update (with condensation)."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.runtime import Runtime

from kb_assistant.agents.common import node_event
from kb_assistant.agents.events import emit
from kb_assistant.agents.memory import estimate_tokens
from kb_assistant.agents.state import AgentState, RunContext
from kb_assistant.errors import AssistantError
from kb_assistant.observability import get_logger
from kb_assistant.security.guards import check_user_input, refusal_for
from kb_assistant.tools.registry import record_security_event

log = get_logger(__name__)


async def input_guard(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    node_event("input_guard", "started")
    ctx = runtime.context
    settings = ctx.services.settings
    result = check_user_input(state["question"], settings.max_message_chars)
    emit({"type": "guard", "stage": "input", "allowed": result.allowed, "categories": result.categories,
          "flags": result.flags, "reasons": result.reasons})
    if result.allowed:
        if result.flags:  # suspicious but not blocked: leave a trace for the audit log and the activity panel
            record_security_event("input_flagged", ctx.principal, categories=result.flags, reasons=result.reasons)
        return {"question": result.text, "guard": {"allowed": True, "flags": result.flags},
                "security_flags": result.flags}

    record_security_event("input_blocked", ctx.principal, categories=result.categories, reasons=result.reasons)
    refusal = refusal_for(result, settings.brand_name)
    return {
        "blocked": True, "guard": {"allowed": False, "categories": result.categories, "reasons": result.reasons},
        "answer": refusal, "security_flags": result.categories,
        "messages": [AIMessage(content=refusal)],
        "validation": {"ok": True, "issues": [], "note": "request blocked by input guard"},
    }


def _previous_questions(messages: list, limit: int = 8) -> list[str]:
    questions = [m.content for m in messages if isinstance(m, HumanMessage)]
    return questions[:-1][-limit:]  # exclude the current question


async def load_memory(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    node_event("load_memory", "started")
    ctx = runtime.context
    memory = ctx.services.memory
    settings = ctx.services.settings
    facts: list[str] = []
    recalled: list[dict[str, Any]] = []
    try:
        facts = await memory.facts(ctx.principal.user_id)
        hits = await memory.recall(ctx.principal.user_id, state["question"], settings.memory_recall_k,
                                   exclude_thread=ctx.thread_id)
        recalled = [{"question": h.question, "answer": h.answer_summary, "doc_ids": h.doc_ids,
                     "similarity": round(h.similarity, 3)} for h in hits]
    except Exception as exc:  # memory is an enhancement: never fail the turn because of it
        log.warning("memory_load_failed", error=str(exc))
    context = {
        "facts": facts, "recalled": recalled, "summary": state.get("summary", ""),
        "previous_questions": _previous_questions(state.get("messages", [])),
    }
    emit({"type": "memory", "action": "loaded", "facts": facts, "recalled": len(recalled),
          "previous_questions": len(context["previous_questions"]), "has_summary": bool(context["summary"])})
    return {"memory_context": context}


def _render_turns(messages: list) -> str:
    lines = []
    for m in messages:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content[:1500]}")
    return "\n".join(lines)


async def update_memory(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    """Persist what this turn taught us and keep the thread's history within budget."""
    node_event("update_memory", "started")
    ctx = runtime.context
    services = ctx.services
    settings = services.settings
    updates: dict[str, Any] = {}
    answer = state.get("answer", "")

    if not state.get("blocked"):
        try:
            profile = f"{ctx.principal.role.value} {ctx.principal.department}".lower()
            # Role and department come from the token; a "fact" restating them adds nothing.
            facts = [f for f in (state.get("decision") or {}).get("user_facts") or []
                     if not (ctx.principal.department.lower() in f.lower() and ctx.principal.role.value in f.lower())
                     and f.lower() != profile]
            added = await services.memory.add_facts(ctx.principal.user_id, facts)
            doc_ids = sorted({c["doc_id"] for c in state.get("citations", []) if not c["id"].startswith("tool:")})
            await services.memory.remember_interaction(ctx.principal.user_id, ctx.thread_id,
                                                       state["question"], answer, doc_ids)
            emit({"type": "memory", "action": "saved", "facts_added": added, "episodic_docs": doc_ids})
        except Exception as exc:
            log.warning("memory_save_failed", error=str(exc))

    # Condense when the history exceeds the token budget. Keep the last N turns verbatim; fold older
    # ASSISTANT answers into the running summary; delete nothing the user wrote.
    messages = state.get("messages", [])
    total = sum(estimate_tokens(str(m.content)) for m in messages)
    keep = settings.memory_recent_turns * 2
    if total > settings.memory_condense_token_budget and len(messages) > keep:
        old_answers = [m for m in messages[:-keep] if isinstance(m, AIMessage)]
        if old_answers:
            summary = await _summarise(services, state.get("summary", ""), messages[:-keep])
            updates["summary"] = summary
            updates["messages"] = [RemoveMessage(id=m.id) for m in old_answers if m.id]
            emit({"type": "memory", "action": "condensed", "tokens_before": total,
                  "assistant_messages_folded": len(old_answers), "user_messages_pinned": True})
    return updates


async def _summarise(services, previous_summary: str, messages: list) -> str:
    prompt = [
        SystemMessage(content=(
            "Update the running summary of a conversation between an employee and an internal assistant. "
            "Keep: facts established, documents or incidents referred to (with ids), decisions, open "
            "questions. Maximum 150 words. Plain text.")),
        HumanMessage(content=f"Current summary:\n{previous_summary or '(none)'}\n\nNew turns:\n{_render_turns(messages)}"),
    ]
    try:
        reply = await services.llm.complete("memory", prompt)
        return str(reply.content)[:1500]
    except AssistantError as exc:
        log.warning("condense_failed_truncating", error=str(exc))
        # Deterministic fallback: keep the tail of the previous summary plus the latest questions.
        questions = "; ".join(m.content[:120] for m in messages if isinstance(m, HumanMessage))
        return (previous_summary[-600:] + f" Earlier questions: {questions}")[-1500:]
