"""The MCP server sits behind a shared service token: without it, every HTTP call is refused, so
nothing can reach the enterprise data or the write tool except through the API (and its RBAC)."""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from conftest import TEST_MCP_TOKEN

from kb_assistant.config import ConfigError, Settings
from kb_assistant.errors import MCPUnavailableError
from kb_assistant.tools.mcp_gateway import MCPGateway

INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-03-26", "capabilities": {},
                         "clientInfo": {"name": "probe", "version": "0"}}}
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture(scope="module")
def mcp_url():
    """The real MCP server over HTTP on a free local port, protected by TEST_MCP_TOKEN."""
    from kb_assistant.mcp_server.server import build_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = build_app(TEST_MCP_TOKEN, host="127.0.0.1", allowed_hosts=[f"127.0.0.1:{port}"])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    assert server.started, "MCP test server did not start"
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=5)


def test_requests_without_the_token_are_refused(mcp_url):
    assert httpx.post(mcp_url, json=INITIALIZE, headers=MCP_HEADERS).status_code == 401


def test_requests_with_a_wrong_token_are_refused(mcp_url):
    headers = {**MCP_HEADERS, "Authorization": "Bearer not-the-token"}
    assert httpx.post(mcp_url, json=INITIALIZE, headers=headers).status_code == 401


def test_requests_with_the_token_reach_the_server(mcp_url):
    headers = {**MCP_HEADERS, "Authorization": f"Bearer {TEST_MCP_TOKEN}"}
    assert httpx.post(mcp_url, json=INITIALIZE, headers=headers).status_code == 200


async def test_the_gateway_sends_the_token_and_calls_tools(mcp_url):
    gateway = MCPGateway(mcp_url, timeout_s=5, service_token=TEST_MCP_TOKEN)
    result = await gateway.call("get_service", {"service_id": "payments-ledger"})
    assert result["service_id"] == "payments-ledger"


@pytest.mark.parametrize("token", ["", "wrong-token"])
async def test_the_gateway_cannot_call_without_the_right_token(mcp_url, token):
    gateway = MCPGateway(mcp_url, timeout_s=5, service_token=token)
    with pytest.raises(MCPUnavailableError):
        await gateway.call("get_service", {"service_id": "payments-ledger"})


def test_the_server_refuses_to_start_without_a_service_token(monkeypatch):
    from kb_assistant.mcp_server import server

    monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    with pytest.raises(SystemExit, match="MCP_SERVICE_TOKEN"):
        server.main()


def test_the_api_refuses_to_start_without_an_mcp_service_token(monkeypatch):
    monkeypatch.delenv("MCP_SERVICE_TOKEN", raising=False)
    settings = Settings(_env_file=None, jwt_secret="j" * 40, mcp_service_token="")
    with pytest.raises(ConfigError, match="MCP_SERVICE_TOKEN"):
        settings.validate_secrets()
