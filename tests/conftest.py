"""Test fixtures: a real index built with a deterministic embedder, an in-process MCP server and a
scripted fake LLM. No network, no API keys; the whole suite runs in seconds."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from kb_assistant import faults
from kb_assistant.config import PROJECT_ROOT, Settings
from kb_assistant.errors import LLMUnavailableError
from kb_assistant.retrieval.ingest import ingest
from kb_assistant.security.rbac import Principal


class FakeLLM:
    """Implements the LLM protocol. `script[stage](messages)` returns a dict (structured), a str
    (complete/stream) or an AIMessage (tool calls). Records every call for assertions."""

    def __init__(self, script: dict[str, Callable[[list[BaseMessage]], Any]] | None = None) -> None:
        self.script = script or {}
        self.calls: list[str] = []

    def _answer(self, stage: str, messages: list[BaseMessage]) -> Any:
        if faults.is_active("llm"):
            raise LLMUnavailableError("fault injection")
        self.calls.append(stage)
        if stage not in self.script:
            raise LLMUnavailableError(f"FakeLLM has no script for stage {stage!r}")
        return self.script[stage](messages)

    async def complete(self, stage, messages, *, tools=None, json_mode=False) -> AIMessage:
        out = self._answer(stage, messages)
        return out if isinstance(out, AIMessage) else AIMessage(content=out if isinstance(out, str) else json.dumps(out))

    async def structured(self, stage, messages, schema: type[BaseModel], max_attempts: int = 3):
        out = self._answer(stage, messages)
        return schema.model_validate(out)

    async def stream(self, stage, messages):
        out = self._answer(stage, messages)
        for token in re.findall(r"\S+\s*", str(out)):
            yield token


def evidence_ids(messages: list[BaseMessage]) -> list[str]:
    return re.findall(r'<document id="([^"]+)"', messages[-1].content)


def cite_first_document(messages: list[BaseMessage]) -> str:
    ids = evidence_ids(messages)
    if not ids:
        return "I couldn't find this in the documents available to you."
    return f"According to the documents, this is covered in the policy [{ids[0]}].\n\n**Why this answer:** [{ids[0]}]"


def supervisor_says(intent: str, agents: list[str], query: str | None = None, **filters) -> Callable:
    def _handler(messages):
        question = re.search(r"<message>\s*(.*?)\s*</message>", messages[-1].content, re.S).group(1)
        return {"intent": intent, "standalone_query": query or question,
                "plan": [{"agent": a, "task": query or question} for a in agents],
                "filters": filters or None, "user_facts": None, "reasoning": "scripted"}
    return _handler


@pytest.fixture(scope="session")
def index_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("index")
    settings = Settings(index_dir=path, embedding_model="hashing", embedding_dim=256, pinecone_api_key="")
    asyncio.run(ingest(settings))
    return path


@pytest.fixture
def settings(index_dir, tmp_path) -> Settings:
    return Settings(
        index_dir=index_dir, embedding_model="hashing", embedding_dim=256, rerank_enabled=False,
        pinecone_api_key="", llm_api_key="test-key", memory_db_path=tmp_path / "memory.sqlite",
        checkpoint_db_path=tmp_path / "checkpoints.sqlite", corpus_dir=PROJECT_ROOT / "data" / "corpus",
        log_json=False, sandbox_timeout_s=3.0,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def services(settings, fake_llm):
    from kb_assistant.container import build_services
    from kb_assistant.mcp_server.server import mcp

    return build_services(settings, llm=fake_llm, mcp_target=mcp, rerank=False)


@pytest.fixture(autouse=True)
def _clear_faults():
    faults.set_faults(set())
    yield
    faults.set_faults(set())


@pytest.fixture
def viewer() -> Principal:
    return Principal.for_role("vera", "Vera Perera", "viewer", "retail-banking")


@pytest.fixture
def analyst() -> Principal:
    return Principal.for_role("anil", "Anil Jayasuriya", "analyst", "payments")


@pytest.fixture
def admin() -> Principal:
    return Principal.for_role("amal", "Amal Silva", "admin", "platform")


def load_events(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("type") == kind]
