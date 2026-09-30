"""MCP incident records obey the same access levels as the documents they describe. Before this,
INC-2026-016 (a restricted, admin-only document) was visible to every analyst through the MCP tool."""

from __future__ import annotations

import json
import re
from pathlib import Path

from mcp import Client

from kb_assistant.mcp_server.server import mcp
from kb_assistant.tools.registry import ToolContext

ROOT = Path(__file__).resolve().parents[1]
RESTRICTED_ID = "INC-2026-016"      # restricted document: admins only
CONFIDENTIAL_ID = "INC-2026-007"    # confidential document: analysts and admins


def _records() -> list[dict]:
    return json.loads((ROOT / "data" / "enterprise" / "incidents.json").read_text())


def test_every_incident_record_carries_the_access_level_of_its_document():
    for record in _records():
        doc = (ROOT / "data" / "corpus" / "incident" / f"{record['incident_id']}.md").read_text()
        level = re.search(r"^access_level:\s*(\w+)", doc, re.M).group(1)
        assert record.get("access_level") == level, record["incident_id"]


async def _visible_ids(services, principal) -> set[str]:
    ctx = ToolContext(principal=principal, services=services)
    result = await services.executor.execute("incident_records", {}, ctx)
    assert result.ok, result.content
    return {i["incident_id"] for i in result.data["incidents"]}


async def test_an_analyst_cannot_see_the_restricted_incident_through_mcp(services, analyst):
    ids = await _visible_ids(services, analyst)
    assert RESTRICTED_ID not in ids
    assert CONFIDENTIAL_ID in ids, "analysts may read confidential incidents"


async def test_an_admin_can_see_the_restricted_incident_through_mcp(services, admin):
    assert RESTRICTED_ID in await _visible_ids(services, admin)


async def test_the_mcp_server_fails_closed_when_no_access_levels_are_sent():
    async with Client(mcp) as client:
        result = await client.call_tool("query_incidents", {})
    payload = result.structured_content or json.loads("".join(c.text for c in result.content))
    levels = {i["access_level"] for i in payload["incidents"]}
    assert levels <= {"public"}, f"a call with no caller identity must not return {levels}"


async def test_the_tool_filters_again_even_if_the_server_returns_too_much(services, analyst, monkeypatch):
    everything = {"count": 2, "incidents": [
        {"incident_id": CONFIDENTIAL_ID, "access_level": "confidential"},
        {"incident_id": RESTRICTED_ID, "access_level": "restricted"},
        {"incident_id": "INC-NO-LEVEL"},   # a record without a level is treated as restricted
    ]}

    async def leaky_server(tool, args):
        return everything
    monkeypatch.setattr(services.mcp, "call", leaky_server)
    ids = await _visible_ids(services, analyst)
    assert ids == {CONFIDENTIAL_ID}
