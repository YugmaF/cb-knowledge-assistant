"""The tools the agents can request, with their argument schemas and required permissions.

    tool                     permission   source  approval
    knowledge_search         search       local   -
    employee_directory       mcp_read     MCP     -
    service_catalog          mcp_read     MCP     -
    incident_records         mcp_read     MCP     -
    python_analysis          analytics    local   -
    update_service_status    mcp_write    MCP     human approval (admin only)
    security_audit_log       admin        local   -

Argument models are the tool contract: the same schema is sent to the model and used to validate
what it sends back, so the two can never drift apart.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from kb_assistant.retrieval.documents import DOCUMENT_TYPES
from kb_assistant.retrieval.retriever import SearchFilters
from kb_assistant.security.rbac import Permission
from kb_assistant.tools.registry import SECURITY_EVENTS, ToolContext, ToolRegistry, ToolSpec
from kb_assistant.tools.sandbox import run_sandboxed

DocType = Literal["policy", "architecture", "runbook", "incident", "product_spec", "meeting_notes"]
assert set(DocType.__args__) == set(DOCUMENT_TYPES)  # noqa: S101 - contract check at import time

IsoDate = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$", description="ISO date YYYY-MM-DD")


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="What to search for, in plain words.")
    document_types: list[DocType] | None = Field(default=None, max_length=6)
    department: Literal["payments", "platform", "security", "hr", "risk", "retail-banking", "data"] | None = None
    date_from: str | None = IsoDate
    date_to: str | None = IsoDate
    top_k: int = Field(default=6, ge=1, le=10)


async def knowledge_search(args: KnowledgeSearchArgs, ctx: ToolContext) -> dict:
    filters = SearchFilters(args.document_types, args.department, args.date_from, args.date_to)
    result = await ctx.services.retriever.search(args.query, ctx.principal, filters, args.top_k)
    ctx.emit({"type": "retrieval", **result.summary()})
    return {
        "mode": result.mode, "degraded": result.degraded,
        "results": [
            {"chunk_id": h.chunk_id, "title": h.metadata.get("title"), "section": h.metadata.get("section"),
             "created_date": h.metadata.get("created_date"), "text": h.text[:900]}
            for h in result.hits
        ],
    }


class EmployeeDirectoryArgs(BaseModel):
    query: str = Field(default="", max_length=80, description="Name or job title fragment")
    department: str | None = Field(default=None, max_length=40)
    on_call_only: bool = False


async def employee_directory(args: EmployeeDirectoryArgs, ctx: ToolContext) -> dict:
    return await ctx.services.mcp.call("search_employees", args.model_dump(exclude_none=True))


class ServiceCatalogArgs(BaseModel):
    service_id: str | None = Field(default=None, max_length=60, pattern=r"^[a-z0-9-]+$",
                                   description="e.g. payments-ledger; omit to list all services")


async def service_catalog(args: ServiceCatalogArgs, ctx: ToolContext) -> dict:
    if args.service_id:
        return await ctx.services.mcp.call("get_service", {"service_id": args.service_id})
    return await ctx.services.mcp.call("list_services", {})


class IncidentRecordsArgs(BaseModel):
    service_id: str | None = Field(default=None, max_length=60, pattern=r"^[a-z0-9-]+$")
    root_cause_category: str | None = Field(default=None, max_length=40, pattern=r"^[a-z_]+$")
    payment_related: bool | None = None
    since: str | None = IsoDate
    until: str | None = IsoDate


async def incident_records(args: IncidentRecordsArgs, ctx: ToolContext) -> dict:
    return await ctx.services.mcp.call("query_incidents", args.model_dump(exclude_none=True))


class PythonAnalysisArgs(BaseModel):
    code: str = Field(
        min_length=5, max_length=4000,
        description=(
            "Python that analyses `datasets` (a dict of earlier tool outputs keyed by tool name, e.g. "
            "datasets['incident_records']['incidents']) and assigns the answer to `result`. "
            "No imports. Builtins: len, sorted, min, max, sum, Counter, defaultdict, set, list, dict, round."
        ),
    )


async def python_analysis(args: PythonAnalysisArgs, ctx: ToolContext) -> dict:
    outcome = await run_sandboxed(
        args.code, {"datasets": ctx.datasets}, timeout_s=ctx.services.settings.sandbox_timeout_s
    )
    return {"result": outcome.result, "printed": outcome.output, "steps": outcome.lines_executed}


class UpdateServiceStatusArgs(BaseModel):
    service_id: str = Field(max_length=60, pattern=r"^[a-z0-9-]+$")
    status: Literal["operational", "degraded"]
    note: str = Field(min_length=3, max_length=200)


async def update_service_status(args: UpdateServiceStatusArgs, ctx: ToolContext) -> dict:
    return await ctx.services.mcp.call("update_service_status", args.model_dump())


class SecurityAuditLogArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)


async def security_audit_log(args: SecurityAuditLogArgs, ctx: ToolContext) -> dict:
    events = list(SECURITY_EVENTS)[-args.limit:]
    return {"count": len(events), "events": events}


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    specs = [
        ToolSpec("knowledge_search",
                 "Hybrid search over the bank's internal documents (policies, architecture, runbooks, "
                 "incident reports, product specs, meeting notes). Returns cited chunks.",
                 KnowledgeSearchArgs, Permission.SEARCH, knowledge_search),
        ToolSpec("employee_directory", "Look up employees by name, title, department or on-call status.",
                 EmployeeDirectoryArgs, Permission.MCP_READ, employee_directory, source="mcp"),
        ToolSpec("service_catalog",
                 "Service catalog: owner team, tier, on-call engineer, dependencies, runbook, status.",
                 ServiceCatalogArgs, Permission.MCP_READ, service_catalog, source="mcp"),
        ToolSpec("incident_records",
                 "Structured incident records (id, service, severity, dates, root_cause_category, "
                 "payment_related, customer_impact_count). Use for counting and trends.",
                 IncidentRecordsArgs, Permission.MCP_READ, incident_records, source="mcp"),
        ToolSpec("python_analysis",
                 "Run restricted Python over data returned by earlier tool calls in this turn "
                 "(counts, group-by, trends). Call a data tool first.",
                 PythonAnalysisArgs, Permission.ANALYTICS, python_analysis),
        ToolSpec("update_service_status",
                 "Change a service's status in the catalog. Administrative write; a human must approve.",
                 UpdateServiceStatusArgs, Permission.MCP_WRITE, update_service_status,
                 requires_approval=True, source="mcp"),
        ToolSpec("security_audit_log", "Recent security events: denied tools, blocked prompts, injections.",
                 SecurityAuditLogArgs, Permission.ADMIN, security_audit_log),
    ]
    for spec in specs:
        registry.register(spec)
    return registry
