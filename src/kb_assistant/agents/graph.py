"""The LangGraph graph.

    START
      │
      ▼
    input_guard ──(blocked)──────────────────────────────────────────────┐
      │                                                                  │
      ▼                                                                  │
    load_memory                                                          │
      │                                                                  │
      ▼                                                                  │
    supervisor  (LLM: intent, standalone query, plan, filters)           │
      │                                                                  │
      ▼                                                                  │
    dispatch ◄───────────────┬────────────────┬──────────────┐           │
      │ next plan step       │                │              │           │
      ├──► retrieval_agent ──┘                │              │           │
      ├──► research_agent (RLM) ──────────────┘              │           │
      ├──► tool_agent ──► approval_gate ──► execute_tools ──►┤ (loop)     │
      │         └──(no more calls)───────────────────────────┘           │
      ▼ (plan done)                                                      │
    response_agent (LLM, streamed) ◄──(rejected, retry once)──┐          │
      │                                                        │          │
      ▼                                                        │          │
    validator ─────────────────────────────────────────────────┘          │
      │ (accepted, or final after retry)                                  │
      ▼                                                                   │
    update_memory ◄───────────────────────────────────────────────────────┘
      │
      ▼
     END

Failure containment ("butterfly effect"): agents never raise into the graph. Each one records what
failed in `degraded` and returns whatever partial result it has; later agents read `degraded` and
say so. One failing dependency therefore narrows the answer instead of cascading into a crash.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from kb_assistant.agents.guard_memory import input_guard, load_memory, update_memory
from kb_assistant.agents.research_agent import research_agent
from kb_assistant.agents.response_agent import response_agent
from kb_assistant.agents.retrieval_agent import retrieval_agent
from kb_assistant.agents.state import AgentState, RunContext
from kb_assistant.agents.supervisor import dispatch, route_from_dispatch, supervisor
from kb_assistant.agents.tool_agent import approval_gate, execute_tools, route_after_tool_agent, tool_agent
from kb_assistant.agents.validator import route_after_validation, validator


def build_graph(checkpointer=None):
    g = StateGraph(AgentState, context_schema=RunContext)
    g.add_node("input_guard", input_guard)
    g.add_node("load_memory", load_memory)
    g.add_node("supervisor", supervisor)
    g.add_node("dispatch", dispatch)
    g.add_node("retrieval_agent", retrieval_agent)
    g.add_node("research_agent", research_agent)
    g.add_node("tool_agent", tool_agent)
    g.add_node("approval_gate", approval_gate)
    g.add_node("execute_tools", execute_tools)
    g.add_node("response_agent", response_agent)
    g.add_node("validator", validator)
    g.add_node("update_memory", update_memory)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", lambda s: "update_memory" if s.get("blocked") else "load_memory",
                            ["update_memory", "load_memory"])
    g.add_edge("load_memory", "supervisor")
    g.add_edge("supervisor", "dispatch")
    g.add_conditional_edges("dispatch", route_from_dispatch,
                            ["retrieval_agent", "research_agent", "tool_agent", "response_agent"])
    g.add_edge("retrieval_agent", "dispatch")
    g.add_edge("research_agent", "dispatch")
    g.add_conditional_edges("tool_agent", route_after_tool_agent, ["approval_gate", "dispatch"])
    g.add_edge("approval_gate", "execute_tools")
    g.add_edge("execute_tools", "tool_agent")
    g.add_edge("response_agent", "validator")
    g.add_conditional_edges("validator", route_after_validation, ["update_memory", "response_agent"])
    g.add_edge("update_memory", END)
    return g.compile(checkpointer=checkpointer, name="cb_knowledge_assistant")
