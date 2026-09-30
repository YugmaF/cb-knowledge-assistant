"""MCP server exposing dummy enterprise data: employee directory, service catalog, incident records.

    uv run python -m kb_assistant.mcp_server.server        # streamable HTTP on :8765/mcp

Every HTTP request must carry `Authorization: Bearer <MCP_SERVICE_TOKEN>`; the server refuses to start
without a token. The token identifies the API, not the end user: the API enforces RBAC first and
tells this server which document access levels the caller has (see `query_incidents`).

Read tools are safe to call freely. `update_service_status` is a write: the assistant only calls it
for administrators, after a human approves it in the UI (LangGraph interrupt).
"""

from __future__ import annotations

import hmac
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import uvicorn
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import JSONResponse

from kb_assistant.config import ConfigError, check_secret

DATA_DIR = Path(os.getenv("ENTERPRISE_DATA_DIR", Path(__file__).resolve().parents[3] / "data" / "enterprise"))


def _load(name: str) -> list[dict]:
    return json.loads((DATA_DIR / name).read_text())


EMPLOYEES = _load("employees.json")
SERVICES = {s["service_id"]: s for s in _load("services.json")}
INCIDENTS = _load("incidents.json")

mcp = MCPServer("commercial-bank-enterprise-data")


def _public_employee(e: dict) -> dict:
    return {k: e[k] for k in ("employee_id", "name", "email", "department", "title", "location", "on_call") if k in e}


@mcp.tool()
def search_employees(
    query: str = "", department: str | None = None, on_call_only: bool = False, limit: int = 10
) -> dict:
    """Search the employee directory by name or title, optionally by department or on-call status."""
    q = query.lower().strip()
    rows = [
        e for e in EMPLOYEES
        if (not q or q in e["name"].lower() or q in e["title"].lower())
        and (not department or e["department"] == department)
        and (not on_call_only or e.get("on_call"))
    ]
    return {"count": len(rows), "employees": [_public_employee(e) for e in rows[: max(1, min(limit, 25))]]}


@mcp.tool()
def get_service(service_id: str) -> dict:
    """Get one service from the service catalog: owner, tier, on-call engineer, dependencies, runbook."""
    service = SERVICES.get(service_id)
    if service is None:
        return {"error": f"unknown service_id '{service_id}'", "known": sorted(SERVICES)}
    on_call = next((e for e in EMPLOYEES if e["employee_id"] == service.get("on_call_employee_id")), None)
    return {**service, "on_call": _public_employee(on_call) if on_call else None}


@mcp.tool()
def list_services(owner_team: str | None = None) -> dict:
    """List services in the catalog, optionally filtered by owner team."""
    rows = [s for s in SERVICES.values() if not owner_team or s["owner_team"] == owner_team]
    return {"count": len(rows), "services": rows}


@mcp.tool()
def query_incidents(
    service_id: str | None = None,
    root_cause_category: str | None = None,
    payment_related: bool | None = None,
    since: str | None = None,
    until: str | None = None,
    limit: int = 100,
    access_levels: list[str] | None = None,
) -> dict:
    """Query structured incident records. Dates are ISO (YYYY-MM-DD) and compare against opened_at.

    Each record has the access level of its document. `access_levels` are the levels the caller may
    read (the API sends them after RBAC); without them only public records are returned, and a record
    with no level is treated as restricted."""
    allowed = set(access_levels or ["public"])
    rows = [
        i for i in INCIDENTS
        if i.get("access_level", "restricted") in allowed
        and (not service_id or i["service_id"] == service_id)
        and (not root_cause_category or i["root_cause_category"] == root_cause_category)
        and (payment_related is None or i["payment_related"] == payment_related)
        and (not since or i["opened_at"][:10] >= since)
        and (not until or i["opened_at"][:10] <= until)
    ]
    rows.sort(key=lambda i: i["opened_at"])
    return {"count": len(rows), "incidents": rows[: max(1, min(limit, 200))]}


@mcp.tool()
def update_service_status(
    service_id: str, status: Literal["operational", "degraded"], note: str
) -> dict:
    """WRITE: change a service's status in the catalog. Requires administrator approval upstream."""
    service = SERVICES.get(service_id)
    if service is None:
        return {"error": f"unknown service_id '{service_id}'"}
    previous = service["status"]
    service["status"] = status
    service["status_note"] = note[:200]
    service["status_updated_at"] = datetime.now(UTC).isoformat()
    return {"service_id": service_id, "previous_status": previous, "status": status}


class ServiceTokenMiddleware:
    """Refuse every HTTP request that does not carry the shared service token (constant-time compare)."""

    def __init__(self, app, token: str) -> None:
        self.app = app
        self._expected = f"Bearer {token}".encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            supplied = dict(scope["headers"]).get(b"authorization", b"")
            if not hmac.compare_digest(supplied, self._expected):
                response = JSONResponse({"error": "unauthorized"}, status_code=401,
                                        headers={"WWW-Authenticate": "Bearer"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def build_app(token: str, *, host: str, allowed_hosts: list[str]) -> Starlette:
    """The MCP endpoint (streamable HTTP, /mcp) behind the service-token check."""
    app = mcp.streamable_http_app(
        stateless_http=True, host=host,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=True,
                                                     allowed_hosts=allowed_hosts),
    )
    app.add_middleware(ServiceTokenMiddleware, token=token)
    return app


def main() -> None:
    try:
        token = check_secret("MCP_SERVICE_TOKEN", os.getenv("MCP_SERVICE_TOKEN", ""))
    except ConfigError as exc:
        raise SystemExit(f"MCP server not started: {exc}") from exc
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8765"))
    allowed = os.getenv("MCP_ALLOWED_HOSTS", f"127.0.0.1:{port},localhost:{port},mcp:{port}").split(",")
    uvicorn.run(build_app(token, host=host, allowed_hosts=allowed), host=host, port=port)


if __name__ == "__main__":
    main()
