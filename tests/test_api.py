"""HTTP layer: authentication, validation envelope, rate limiting, thread ownership, admin gates, SSE."""

from __future__ import annotations

import json

import pytest
from conftest import cite_first_document, supervisor_says
from fastapi.testclient import TestClient

from kb_assistant.config import RateLimitRule
from kb_assistant.security.rate_limit import RateLimiter


@pytest.fixture
def client(services, fake_llm, monkeypatch):
    from kb_assistant.api import main

    fake_llm.script = {"supervisor": supervisor_says("knowledge_question", ["retrieval"]),
                       "response": cite_first_document}
    monkeypatch.setattr(main, "get_settings", lambda: services.settings)
    main.app.state.services_override = services
    with TestClient(main.app) as c:
        yield c
    main.app.state.services_override = None


@pytest.mark.parametrize("secret", ["", "dev-only-change-me-dev-only-change-me", "too-short"])
def test_api_refuses_to_start_with_a_weak_jwt_secret(services, monkeypatch, secret):
    from kb_assistant.api import main
    from kb_assistant.config import ConfigError

    weak = services.settings.model_copy(update={"jwt_secret": secret})
    monkeypatch.setattr(main, "get_settings", lambda: weak)
    main.app.state.services_override = services
    try:
        with pytest.raises(ConfigError, match="JWT_SECRET"):
            with TestClient(main.app):
                pass
    finally:
        main.app.state.services_override = None


def login(client, user, password):
    resp = client.post("/auth/login", json={"username": user, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def sse_events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def test_login_and_me(client):
    headers = login(client, "anil", "analyst-pass")
    me = client.get("/auth/me", headers=headers).json()
    assert me["role"] == "analyst" and "python_analysis" in me["tools"] and "update_service_status" not in me["tools"]
    bad = client.post("/auth/login", json={"username": "anil", "password": "nope"})
    assert bad.status_code == 401 and bad.json()["error"] == "unauthorized"


def test_requests_without_token_or_with_bad_body_are_rejected(client):
    assert client.post("/chat/stream", json={"message": "hi"}).status_code == 401
    headers = login(client, "vera", "viewer-pass")
    resp = client.post("/chat/stream", json={"message": ""}, headers=headers)
    assert resp.status_code == 422 and resp.json()["error"] == "invalid_request"


def test_chat_stream_returns_events_and_final_answer(client):
    headers = login(client, "vera", "viewer-pass")
    resp = client.post("/chat/stream", json={"message": "How many days can I work remotely?"}, headers=headers)
    events = sse_events(resp)
    assert events[0]["type"] == "run_started" and events[-1]["type"] == "final"
    assert events[-1]["citations"]


def test_threads_are_private_to_their_owner(client):
    vera = login(client, "vera", "viewer-pass")
    events = sse_events(client.post("/chat/stream", json={"message": "remote work days?"}, headers=vera))
    thread_id = events[0]["thread_id"]
    assert client.get(f"/threads/{thread_id}/messages", headers=vera).status_code == 200
    anil = login(client, "anil", "analyst-pass")
    assert client.get(f"/threads/{thread_id}/messages", headers=anil).status_code == 404
    hijack = client.post("/chat/stream", json={"message": "hi there", "thread_id": thread_id}, headers=anil)
    assert hijack.status_code == 404


def test_admin_endpoints_require_admin(client):
    viewer = login(client, "vera", "viewer-pass")
    assert client.post("/admin/faults", json={"faults": ["llm"]}, headers=viewer).status_code == 403
    admin = login(client, "amal", "admin-pass")
    assert client.post("/admin/faults", json={"faults": ["vectordb"]}, headers=admin).json() == {"active": ["vectordb"]}
    assert client.get("/admin/faults", headers=admin).json()["active"] == ["vectordb"]
    assert client.post("/admin/faults", json={"faults": ["bogus"]}, headers=admin).status_code == 422


def test_rate_limit_returns_429_with_retry_after(client, services):
    services.rate_limiter = RateLimiter({**services.settings.rate_limits,
                                         "viewer": RateLimitRule(capacity=1, refill_per_minute=1)})
    headers = login(client, "vera", "viewer-pass")
    assert client.post("/chat/stream", json={"message": "remote work?"}, headers=headers).status_code == 200
    limited = client.post("/chat/stream", json={"message": "remote work?"}, headers=headers)
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1 and limited.json()["error"] == "rate_limited"


def test_feedback_is_stored(client, services):
    headers = login(client, "vera", "viewer-pass")
    events = sse_events(client.post("/chat/stream", json={"message": "remote work?"}, headers=headers))
    final = events[-1]
    resp = client.post("/feedback", headers=headers, json={"thread_id": final["thread_id"], "run_id": final["run_id"],
                                                           "score": -1, "comment": "too vague"})
    assert resp.json()["stored"]


def test_health(client):
    body = client.get("/health").json()
    assert body["vector_store"]["status"] == "ok" and body["llm_configured"]
