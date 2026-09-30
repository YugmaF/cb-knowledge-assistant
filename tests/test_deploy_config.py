"""Deployment files make claims the docs rely on (required secrets, what is published). Check them."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]


def test_compose_refuses_to_start_without_a_jwt_secret():
    assert COMPOSE["api"]["environment"]["JWT_SECRET"].startswith("${JWT_SECRET:?")


def test_compose_does_not_publish_the_mcp_server_to_the_host():
    mcp = COMPOSE["mcp"]
    assert "ports" not in mcp, "MCP must only be reachable from other containers"
    assert "8765" in [str(p) for p in mcp["expose"]]


def test_compose_requires_the_mcp_service_token_on_both_sides():
    for service in ("mcp", "api"):
        assert COMPOSE[service]["environment"]["MCP_SERVICE_TOKEN"].startswith("${MCP_SERVICE_TOKEN:?")


def test_run_sh_binds_mcp_to_loopback_and_generates_its_token():
    script = (ROOT / "run.sh").read_text()
    assert 'MCP_HOST=127.0.0.1 "$PY" -m kb_assistant.mcp_server.server' in script
    assert "ensure_secret MCP_SERVICE_TOKEN" in script
