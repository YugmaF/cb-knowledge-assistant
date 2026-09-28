"""Tool registry and executor: the only path from a model's tool request to running code.

The model only ever *asks* for a tool. `ToolExecutor.execute` decides, in this order:

    1. does the tool exist?                  unknown name        -> denied, logged as security event
    2. may this role run it?                 RBAC permission     -> denied, logged as security event
    3. are the arguments valid?              Pydantic model      -> error returned to the model
    4. does it need a human?                 requires_approval   -> ApprovalRequired (graph interrupts)
    5. run it with a timeout                 asyncio.timeout     -> timeout returned to the model
    6. treat the output as untrusted         truncate + sanitise -> result returned to the model

Errors come back as tool results, not exceptions, so the model can try another way; the graph never
crashes because a tool did.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langsmith import traceable
from pydantic import BaseModel, ValidationError

from kb_assistant import faults
from kb_assistant.errors import AssistantError, ToolTimeoutError
from kb_assistant.observability import get_logger
from kb_assistant.security.guards import sanitize_retrieved
from kb_assistant.security.rbac import Permission, Principal

log = get_logger(__name__)

Emit = Callable[[dict[str, Any]], None]
MAX_TOOL_OUTPUT_CHARS = 6000

# Recent denials and guard hits, readable by administrators through the security_audit_log tool.
SECURITY_EVENTS: deque[dict[str, Any]] = deque(maxlen=200)


def record_security_event(kind: str, principal: Principal | None, **details: Any) -> None:
    event = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"), "kind": kind,
        "user": principal.user_id if principal else None,
        "role": principal.role.value if principal else None, **details,
    }
    SECURITY_EVENTS.append(event)
    log.warning("security_event", **event)


@dataclass
class ToolContext:
    principal: Principal
    services: Any  # kb_assistant.container.Services (avoids an import cycle)
    emit: Emit = lambda event: None
    datasets: dict[str, Any] = field(default_factory=dict)  # earlier tool outputs, for analysis


Handler = Callable[[BaseModel, ToolContext], Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    permission: Permission
    handler: Handler
    timeout_s: float | None = None
    requires_approval: bool = False
    source: str = "local"

    def openai_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": schema}}


@dataclass
class ToolResult:
    tool: str
    ok: bool
    content: str
    data: Any = None
    error: str | None = None
    elapsed_ms: int = 0
    flags: list[str] = field(default_factory=list)


class ApprovalRequired(Exception):
    def __init__(self, spec: ToolSpec, args: dict[str, Any]) -> None:
        super().__init__(f"{spec.name} requires approval")
        self.spec = spec
        self.args = args


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def for_principal(self, principal: Principal) -> list[ToolSpec]:
        """Least privilege: the model is only shown tools the caller may run."""
        return [t for t in self._tools.values() if principal.can(t.permission)]

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, default_timeout_s: float) -> None:
        self.registry = registry
        self.default_timeout_s = default_timeout_s

    def validate(self, name: str, raw_args: dict[str, Any] | str, principal: Principal) -> tuple[ToolSpec, BaseModel]:
        spec = self.registry.get(name)
        if spec is None:
            record_security_event("unknown_tool", principal, tool=name)
            raise PermissionError(f"tool '{name}' does not exist")
        if not principal.can(spec.permission):
            record_security_event("tool_denied", principal, tool=name, needs=spec.permission.value)
            raise PermissionError(
                f"role '{principal.role.value}' is not permitted to use '{name}' (needs {spec.permission.value})")
        if isinstance(raw_args, str):
            raw_args = json.loads(raw_args or "{}")
        return spec, spec.args_model.model_validate(raw_args)

    @traceable(run_type="tool", name="execute_tool")
    async def execute(
        self, name: str, raw_args: dict[str, Any] | str, ctx: ToolContext, *, approved: bool = False,
    ) -> ToolResult:
        started = time.perf_counter()
        ctx.emit({"type": "tool_call", "tool": name, "args": raw_args, "status": "started"})

        def done(result: ToolResult) -> ToolResult:
            result.elapsed_ms = int((time.perf_counter() - started) * 1000)
            ctx.emit({"type": "tool_call", "tool": name, "status": "ok" if result.ok else "error",
                      "error": result.error, "elapsed_ms": result.elapsed_ms, "flags": result.flags})
            log.info("tool_executed", tool=name, ok=result.ok, error=result.error, elapsed_ms=result.elapsed_ms)
            return result

        try:
            spec, args = self.validate(name, raw_args, ctx.principal)
        except PermissionError as exc:
            return done(ToolResult(name, False, f"DENIED: {exc}", error="permission_denied"))
        except (ValidationError, json.JSONDecodeError) as exc:
            return done(ToolResult(name, False, f"INVALID ARGUMENTS: {exc}", error="invalid_arguments"))

        if spec.requires_approval and not approved:
            raise ApprovalRequired(spec, args.model_dump())

        timeout = spec.timeout_s or self.default_timeout_s
        try:
            async with asyncio.timeout(timeout):
                if faults.is_active("tool_slow"):
                    await asyncio.sleep(timeout + 1)
                data = await spec.handler(args, ctx)
        except TimeoutError:
            return done(ToolResult(name, False, f"TIMEOUT: {name} did not finish in {timeout}s", error="timeout"))
        except ToolTimeoutError as exc:
            return done(ToolResult(name, False, f"TIMEOUT: {exc}", error="timeout"))
        except AssistantError as exc:
            return done(ToolResult(name, False, f"ERROR: {exc.public_message} ({exc})", error=type(exc).__name__))
        except Exception as exc:  # a tool bug must not take down the conversation
            log.exception("tool_crashed", tool=name)
            return done(ToolResult(name, False, f"ERROR: tool failed: {type(exc).__name__}", error="tool_crashed"))

        text = data if isinstance(data, str) else json.dumps(data, default=str)
        flags: list[str] = []
        if len(text) > MAX_TOOL_OUTPUT_CHARS:
            text = text[:MAX_TOOL_OUTPUT_CHARS] + " …[truncated]"
            flags.append("truncated")
        # Tool output is untrusted input too: an MCP record could carry an injection.
        sanitized = sanitize_retrieved(text)
        if sanitized.flagged:
            text = sanitized.text
            flags.append("prompt_injection_removed")
        ctx.datasets[name] = data
        return done(ToolResult(name, True, text, data=data, flags=flags))
