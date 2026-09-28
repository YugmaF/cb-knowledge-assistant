"""Supervisor agent: intent, query rewriting, task decomposition, routing.

The LLM plans once per turn; a deterministic dispatcher then walks the plan. Control flow stays in
code, so a turn's path is predictable and every hop is visible in the trace.

Plans are checked after the model returns them: steps the caller's role cannot use are removed
(a viewer is never routed to the tools agent), and duplicates are dropped. If the LLM is down, a
keyword router produces a usable plan so the turn still completes.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from kb_assistant.agents.common import node_event, today
from kb_assistant.agents.events import emit
from kb_assistant.agents.prompts import SUPERVISOR_CONTEXT, SUPERVISOR_SYSTEM
from kb_assistant.agents.state import AgentState, PlanStep, RunContext, SupervisorDecision
from kb_assistant.errors import AssistantError
from kb_assistant.observability import get_logger
from kb_assistant.security.rbac import Permission, Principal

log = get_logger(__name__)

_RESEARCH_WORDS = re.compile(r"\b(summari[sz]e|recurring|trends?|all (the )?(incidents|outages|reports)|"
                             r"across|over the (last|past)|root causes|patterns?)\b", re.I)
_LOOKUP_WORDS = re.compile(r"\b(who|on[- ]call|owner|owns|contact|status of|service catalog|"
                           r"how many|count|per month|set .* status|security (log|events))\b", re.I)
_GREETING = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b[\s!.]*$", re.I)


def heuristic_decision(question: str, principal: Principal) -> SupervisorDecision:
    """Used when the supervisor LLM is unavailable."""
    if _GREETING.match(question):
        return SupervisorDecision(intent="greeting", standalone_query=question, reasoning="keyword router: greeting")
    if _RESEARCH_WORDS.search(question):
        plan = [PlanStep(agent="research", task=question)]
        intent = "research_summary"
    elif _LOOKUP_WORDS.search(question) and principal.can(Permission.MCP_READ):
        plan = [PlanStep(agent="tools", task=question)]
        intent = "enterprise_lookup"
    else:
        plan = [PlanStep(agent="retrieval", task=question)]
        intent = "knowledge_question"
    return SupervisorDecision(intent=intent, standalone_query=question, plan=plan,
                              reasoning="keyword router (LLM supervisor unavailable)")


def enforce_plan_policy(decision: SupervisorDecision, principal: Principal) -> tuple[SupervisorDecision, list[str]]:
    notes: list[str] = []
    allowed_steps: list[PlanStep] = []
    seen: set[str] = set()
    has_tools = principal.can(Permission.MCP_READ) or principal.can(Permission.ANALYTICS) or principal.can(Permission.ADMIN)
    for step in decision.plan:
        if step.agent == "tools" and not has_tools:
            notes.append("removed 'tools' step: role has no enterprise tools")
            continue
        if step.agent in seen:
            continue
        seen.add(step.agent)
        allowed_steps.append(step)
    if not allowed_steps and decision.intent not in ("greeting", "out_of_scope"):
        # A viewer asking for an enterprise lookup still gets a document search.
        allowed_steps = [PlanStep(agent="retrieval", task=decision.standalone_query)]
        notes.append("no permitted steps left: falling back to document retrieval")
    decision.plan = allowed_steps
    return decision, notes


_DEPARTMENT_WORDS = {
    "payments": ("payments team", "payments department", "payments engineering", "payments squad"),
    "platform": ("platform",), "security": ("security",), "hr": ("hr", "human resources"),
    "risk": ("risk",), "retail-banking": ("retail",), "data": ("data team", "data platform", "data department"),
}


def enforce_filter_policy(decision: SupervisorDecision, question: str) -> list[str]:
    """A department filter hides everything owned elsewhere, so it must come from the user, not be
    inferred from a topic ("payment failures" are also owned by platform). Checked in code."""
    dept = decision.filters.department
    if dept and not any(re.search(rf"\b{re.escape(w)}\b", question, re.I) for w in _DEPARTMENT_WORDS.get(dept, (dept,))):
        decision.filters.department = None
        return [f"dropped department filter '{dept}': the user did not name that department"]
    return []


async def supervisor(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    node_event("supervisor", "started")
    ctx = runtime.context
    services = ctx.services
    principal = ctx.principal
    memory = state.get("memory_context", {})
    tools = [t.name for t in services.registry.for_principal(principal)]
    degraded = list(state.get("degraded", []))

    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM.format(brand=services.settings.brand_name)),
        HumanMessage(content=SUPERVISOR_CONTEXT.format(
            today=today(), user_name=principal.name, role=principal.role.value,
            department=principal.department, tools=", ".join(tools),
            facts="; ".join(memory.get("facts", [])) or "none",
            summary=memory.get("summary") or "none",
            previous_questions=" | ".join(memory.get("previous_questions", [])) or "none",
            question=state["question"])),
    ]
    try:
        decision = await services.llm.structured("supervisor", messages, SupervisorDecision)
    except AssistantError as exc:
        log.warning("supervisor_fallback", error=str(exc))
        decision = heuristic_decision(state["question"], principal)
        degraded.append("supervisor: LLM unavailable, used keyword routing")

    decision, notes = enforce_plan_policy(decision, principal)
    notes += enforce_filter_policy(decision, state["question"])
    emit({"type": "supervisor", "intent": decision.intent, "standalone_query": decision.standalone_query,
          "plan": [s.model_dump() for s in decision.plan], "filters": decision.filters.model_dump(exclude_none=True),
          "reasoning": decision.reasoning, "policy_notes": notes})
    return {"decision": decision.model_dump(), "plan": [s.model_dump() for s in decision.plan],
            "plan_index": 0, "degraded": degraded}


async def dispatch(state: AgentState) -> dict[str, Any]:
    """Deterministic: take the next planned step, or none (-> response)."""
    plan, index = state.get("plan", []), state.get("plan_index", 0)
    step = plan[index] if index < len(plan) else None
    node_event("dispatch", "routing", next=step["agent"] if step else "response", step=index + 1, of=len(plan))
    return {"current_step": step, "plan_index": index + 1 if step else index}


def route_from_dispatch(state: AgentState) -> str:
    step = state.get("current_step")
    return {"retrieval": "retrieval_agent", "research": "research_agent", "tools": "tool_agent"}.get(
        step["agent"] if step else "", "response_agent")
