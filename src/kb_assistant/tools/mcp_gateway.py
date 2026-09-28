"""MCP client with a timeout and a circuit breaker.

When the MCP server is down, every call would otherwise wait for the full timeout. After
`failure_threshold` consecutive failures the breaker opens and calls fail immediately for
`cooldown_s`; the next call after that is a trial. The agent receives the failure as a tool result
and can answer from documents instead.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from langsmith import traceable
from mcp import Client

from kb_assistant import faults
from kb_assistant.errors import MCPUnavailableError, ToolTimeoutError
from kb_assistant.observability import get_logger

log = get_logger(__name__)


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_s: float = 30.0) -> None:
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_at: float | None = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "open" if time.monotonic() - self.opened_at < self.cooldown_s else "half_open"

    def before_call(self) -> None:
        if self.state == "open":
            raise MCPUnavailableError("circuit open: MCP server failed repeatedly, not retrying yet")

    def record(self, ok: bool) -> None:
        if ok:
            self.failures, self.opened_at = 0, None
            return
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.monotonic()


class MCPGateway:
    """`target` is a URL (streamable HTTP) in production, or an in-process server in tests."""

    def __init__(self, target: Any, timeout_s: float) -> None:
        self._target = target
        self._timeout = timeout_s
        self.breaker = CircuitBreaker()

    @traceable(run_type="tool", name="mcp_call")
    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        if faults.is_active("mcp"):
            self.breaker.record(False)
            raise MCPUnavailableError("fault injection: MCP server unavailable")
        self.breaker.before_call()
        try:
            async with asyncio.timeout(self._timeout):
                async with Client(self._target) as client:
                    result = await client.call_tool(tool, arguments)
        except TimeoutError as exc:
            self.breaker.record(False)
            raise ToolTimeoutError(f"MCP tool {tool} timed out after {self._timeout}s") from exc
        except MCPUnavailableError:
            raise
        except Exception as exc:  # connection refused, protocol errors, ...
            self.breaker.record(False)
            log.warning("mcp_call_failed", tool=tool, error=str(exc))
            raise MCPUnavailableError(f"MCP call {tool} failed: {exc}") from exc
        self.breaker.record(True)
        if getattr(result, "is_error", False):
            text = " ".join(getattr(c, "text", "") for c in result.content or [])
            raise MCPUnavailableError(f"MCP tool {tool} returned an error: {text[:200]}")
        structured = getattr(result, "structured_content", None)
        if structured is not None:
            return structured
        text = "".join(getattr(c, "text", "") for c in result.content or [])
        try:
            return json.loads(text)  # dict-returning tools arrive as one JSON text block
        except ValueError:
            return text
