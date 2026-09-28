"""Activity events: what the agent is doing, streamed to the UI's activity panel.

Nodes call `emit(...)`; inside a LangGraph run this writes to the "custom" stream, which the API
forwards to the browser as server-sent events. Outside a run (unit tests, scripts) it is a no-op.

Event types: node, supervisor, retrieval, tool_call, approval, rlm, memory, validation, llm_call,
token, guard, error.
"""

from __future__ import annotations

from typing import Any

from langgraph.config import get_stream_writer


def emit(event: dict[str, Any]) -> None:
    try:
        writer = get_stream_writer()
    except RuntimeError:  # not inside a graph run
        return
    writer(event)
