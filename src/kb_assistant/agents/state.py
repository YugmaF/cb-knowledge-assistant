"""Graph state, runtime context and the agents' structured-output contracts.

State vs context: *state* is what agents produce and hand to each other (and what the checkpointer
persists). *Context* is who is asking and which services to use; it is supplied by the API on each
run from the verified token, so nothing an agent writes into state can change the caller's identity
or permissions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from kb_assistant.security.rbac import Principal

AgentName = Literal["retrieval", "research", "tools"]
Intent = Literal[
    "knowledge_question", "research_summary", "enterprise_lookup", "analytics", "admin_action",
    "greeting", "out_of_scope",
]


@dataclass(frozen=True)
class RunContext:
    principal: Principal
    services: Any  # kb_assistant.container.Services
    thread_id: str


# --- supervisor contract ------------------------------------------------------------------------

class PlanStep(BaseModel):
    agent: AgentName
    task: str = Field(max_length=300, description="What this agent should do, in one sentence.")


class DecisionFilters(BaseModel):
    document_types: list[Literal["policy", "architecture", "runbook", "incident", "product_spec",
                                 "meeting_notes"]] | None = None
    department: Literal["payments", "platform", "security", "hr", "risk", "retail-banking", "data"] | None = None
    date_from: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    date_to: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")


class SupervisorDecision(BaseModel):
    intent: Intent
    standalone_query: str = Field(min_length=1, max_length=400,
                                  description="The question rewritten to stand alone, resolving 'it'/'that' from history.")
    plan: list[PlanStep] = Field(default_factory=list, max_length=3)
    filters: DecisionFilters = Field(default_factory=DecisionFilters)
    user_facts: list[str] = Field(default_factory=list, max_length=3,
                                  description="Durable facts the user stated about themselves, if any.")
    reasoning: str = Field(default="", max_length=400)

    # The prompt tells the model "use JSON null for anything not stated". A Pydantic default only
    # applies when a key is MISSING, not when it is null, so null must be mapped explicitly or the
    # prompt and the schema contradict each other and every such reply fails validation.
    @field_validator("filters", mode="before")
    @classmethod
    def _null_filters(cls, v):
        return v if v is not None else {}

    @field_validator("plan", "user_facts", mode="before")
    @classmethod
    def _null_lists(cls, v):
        return v if v is not None else []


# --- research (RLM) contracts -------------------------------------------------------------------

class SearchPlan(BaseModel):
    rationale: str = Field(max_length=500)
    code: str = Field(min_length=10, max_length=4000)


class DocFinding(BaseModel):
    doc_id: str
    relevant: bool
    date: str | None = None
    category: str = Field(description="snake_case root-cause category, or 'not_applicable'")
    summary: str = Field(max_length=400)
    evidence_chunk_id: str


    @field_validator("summary", mode="before")
    @classmethod
    def _clip_summary(cls, v):
        # Too long is verbose, not wrong: clip it instead of paying for a retry.
        return v[:400] if isinstance(v, str) else v


class BatchFindings(BaseModel):
    findings: list[DocFinding]
    themes: list[str] = Field(default_factory=list, max_length=5)
    follow_up_queries: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("themes", "follow_up_queries", mode="before")
    @classmethod
    def _null_lists(cls, v):
        return (v or [])[:5]


class RecurringCause(BaseModel):
    category: str
    count: int
    incidents: list[str]
    explanation: str = Field(max_length=600)


class ResearchReport(BaseModel):
    summary: str = Field(max_length=3000)
    recurring_root_causes: list[RecurringCause] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list, max_length=6)


# --- graph state ----------------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # Persistent across turns (checkpointed per thread)
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str  # condensed older assistant answers; user questions are never condensed

    # Per turn (reset by `new_turn`)
    question: str
    guard: dict[str, Any]
    blocked: bool
    memory_context: dict[str, Any]
    decision: dict[str, Any]
    plan: list[dict[str, Any]]
    plan_index: int
    current_step: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    datasets: dict[str, Any]
    research: dict[str, Any]
    tool_messages: list[AnyMessage]
    pending_tool_calls: list[dict[str, Any]]
    approval: dict[str, Any] | None
    tool_steps: int
    draft: str
    answer: str
    citations: list[dict[str, Any]]
    validation: dict[str, Any]
    validation_attempts: int
    validation_feedback: str
    degraded: list[str]
    security_flags: list[str]


def new_turn(question: str) -> dict[str, Any]:
    """Input for one user turn: resets every per-turn field, appends the question to history."""
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=question)],
        "question": question, "guard": {}, "blocked": False, "memory_context": {}, "decision": {},
        "plan": [], "plan_index": 0, "current_step": None, "evidence": [], "tool_results": [],
        "datasets": {}, "research": {}, "tool_messages": [], "pending_tool_calls": [], "approval": None,
        "tool_steps": 0, "draft": "", "answer": "", "citations": [], "validation": {},
        "validation_attempts": 0, "validation_feedback": "", "degraded": [], "security_flags": [],
    }
