"""LLM access for every agent: per-stage model routing, fallback, validated structured output,
streaming, and usage accounting.

Structured output never trusts the model:
    parse JSON -> validate against the Pydantic schema -> (callers add business + grounding checks)
    on failure: send the raw output back as `assistant` and the error as `user`, retry (bounded).
Retries are a safety net, not a design; each one is logged and counted.

Transport failures (timeouts, 5xx, an empty body) are retried by the client, then the stage falls
back to a model from a different provider. Only when both fail does the caller degrade.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from kb_assistant import faults
from kb_assistant.agents.events import emit
from kb_assistant.config import Settings
from kb_assistant.errors import LLMOutputError, LLMUnavailableError
from kb_assistant.observability import get_logger

log = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (input, output), for the per-task cost estimate shown in the UI.
PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4.1-mini": (0.40, 1.60),
    "google/gemini-2.5-flash": (0.30, 2.50),
}


class LLM(Protocol):
    async def complete(self, stage: str, messages: list[BaseMessage], *, tools: list[dict] | None = None,
                       json_mode: bool = False) -> AIMessage: ...

    async def structured(self, stage: str, messages: list[BaseMessage], schema: type[T],
                         max_attempts: int = 3) -> T: ...

    def stream(self, stage: str, messages: list[BaseMessage]) -> AsyncIterator[str]: ...


def extract_json(text: str) -> Any:
    """Models sometimes wrap JSON in ```json fences or add a sentence; take the outermost object."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in the response")
    return json.loads(text[start:end + 1])


class LLMGateway:
    def __init__(self, settings: Settings, chat_factory: Callable[[str], BaseChatModel] | None = None) -> None:
        self.settings = settings
        self._factory = chat_factory or self._default_factory
        self._cache: dict[str, BaseChatModel] = {}

    def _default_factory(self, model: str) -> BaseChatModel:
        return ChatOpenAI(
            model=model, base_url=self.settings.llm_base_url, api_key=self.settings.llm_api_key or "missing",
            temperature=0, timeout=self.settings.llm_timeout_s, max_retries=self.settings.llm_max_retries,
            stream_usage=True,
            default_headers={"X-Title": self.settings.assistant_name},
        )

    def _chat(self, model: str) -> BaseChatModel:
        if model not in self._cache:
            self._cache[model] = self._factory(model)
        return self._cache[model]

    def _models_for(self, stage: str) -> list[str]:
        primary = getattr(self.settings.models, stage)
        fallback = self.settings.models.fallback
        return [primary] if primary == fallback else [primary, fallback]

    def _check_faults(self, model: str, is_primary: bool) -> None:
        if not self.settings.llm_configured:
            raise LLMUnavailableError("no LLM API key configured")
        if faults.is_active("llm") or (is_primary and faults.is_active("llm_primary")):
            raise LLMUnavailableError(f"fault injection: {model} unavailable")

    def _record_usage(self, stage: str, model: str, message: AIMessage, started: float, attempt: str) -> None:
        usage = message.usage_metadata or {}
        tokens_in, tokens_out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        price_in, price_out = PRICES.get(model, (0.0, 0.0))
        event = {
            "type": "llm_call", "stage": stage, "model": model, "attempt": attempt,
            "input_tokens": tokens_in, "output_tokens": tokens_out,
            "cached_tokens": (usage.get("input_token_details") or {}).get("cache_read", 0),
            "cost_usd": round((tokens_in * price_in + tokens_out * price_out) / 1e6, 6),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
        # A 200 with no content and no tool call is a transport failure, not an answer.
        if not message.content and not message.tool_calls:
            raise LLMUnavailableError(f"{model} returned an empty response")
        emit(event)
        log.info("llm_call", **{k: v for k, v in event.items() if k != "type"})

    async def complete(self, stage: str, messages: list[BaseMessage], *, tools: list[dict] | None = None,
                       json_mode: bool = False) -> AIMessage:
        errors: list[str] = []
        for i, model in enumerate(self._models_for(stage)):
            started = time.perf_counter()
            try:
                self._check_faults(model, is_primary=i == 0)
                chat: Any = self._chat(model)
                if tools:
                    chat = chat.bind_tools(tools)
                if json_mode:
                    chat = chat.bind(response_format={"type": "json_object"})
                message = await chat.ainvoke(messages)
                self._record_usage(stage, model, message, started, "primary" if i == 0 else "fallback")
                return message
            except Exception as exc:
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:200]}")
                log.warning("llm_call_failed", stage=stage, model=model, error=errors[-1])
                emit({"type": "error", "component": "llm", "stage": stage, "model": model,
                      "detail": errors[-1], "action": "trying fallback model" if i == 0 else "giving up"})
        raise LLMUnavailableError("; ".join(errors))

    async def structured(self, stage: str, messages: list[BaseMessage], schema: type[T],
                         max_attempts: int = 3) -> T:
        convo = list(messages)
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            reply = await self.complete(stage, convo, json_mode=True)
            raw = reply.content if isinstance(reply.content, str) else json.dumps(reply.content)
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:800]
                log.warning("llm_output_invalid", stage=stage, attempt=attempt, error=last_error[:300])
                emit({"type": "validation", "target": f"{stage} output", "ok": False,
                      "attempt": attempt, "detail": last_error[:300]})
                convo += [AIMessage(content=raw), HumanMessage(content=(
                    f"Your previous reply failed validation:\n{last_error}\n"
                    "Reply again with ONLY a JSON object that fixes these errors."))]
        raise LLMOutputError(f"{stage}: invalid output after {max_attempts} attempts: {last_error[:300]}")

    async def stream(self, stage: str, messages: list[BaseMessage]) -> AsyncIterator[str]:
        """Stream tokens. Falls back to the next model only if the failure happens before the first
        token; after that a switch would splice two different answers together."""
        errors: list[str] = []
        for i, model in enumerate(self._models_for(stage)):
            started = time.perf_counter()
            produced = False
            try:
                self._check_faults(model, is_primary=i == 0)
                final: AIMessage | None = None
                async for chunk in self._chat(model).astream(messages):
                    final = chunk if final is None else final + chunk
                    if chunk.content:
                        produced = True
                        yield chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                if final is None:
                    raise LLMUnavailableError(f"{model} returned an empty stream")
                self._record_usage(stage, model, AIMessage(content=final.content or "",
                                   usage_metadata=final.usage_metadata), started,
                                   "primary" if i == 0 else "fallback")
                return
            except Exception as exc:
                if produced:
                    raise LLMUnavailableError(f"{model} failed mid-stream: {exc}") from exc
                errors.append(f"{model}: {type(exc).__name__}: {str(exc)[:200]}")
                log.warning("llm_stream_failed", stage=stage, model=model, error=errors[-1])
                emit({"type": "error", "component": "llm", "stage": stage, "model": model,
                      "detail": errors[-1], "action": "trying fallback model" if i == 0 else "giving up"})
        raise LLMUnavailableError("; ".join(errors))
