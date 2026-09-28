"""Fault injection.

A guardrail or fallback that has never been triggered is not known to work. An administrator can
switch these faults on through `POST /admin/faults` during a demo to force each failure path:

    llm        every LLM call raises (primary and fallback) -> extractive degraded answer
    llm_primary  only the primary model fails -> fallback model answers
    vectordb   the vector store raises -> keyword-only (BM25) retrieval
    mcp        MCP calls raise -> tool error returned to the agent
    tool_slow  tools sleep past their timeout -> ToolTimeoutError path
"""

from __future__ import annotations

from typing import Literal

Fault = Literal["llm", "llm_primary", "vectordb", "mcp", "tool_slow"]
ALL_FAULTS: tuple[str, ...] = ("llm", "llm_primary", "vectordb", "mcp", "tool_slow")

_active: set[str] = set()


def set_faults(faults: set[str]) -> set[str]:
    unknown = faults - set(ALL_FAULTS)
    if unknown:
        raise ValueError(f"unknown faults: {sorted(unknown)}")
    _active.clear()
    _active.update(faults)
    return set(_active)


def active_faults() -> set[str]:
    return set(_active)


def is_active(fault: str) -> bool:
    return fault in _active
