"""Deployment files make claims the docs rely on (required secrets, what is published). Check them."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]


def test_compose_refuses_to_start_without_a_jwt_secret():
    assert COMPOSE["api"]["environment"]["JWT_SECRET"].startswith("${JWT_SECRET:?")
