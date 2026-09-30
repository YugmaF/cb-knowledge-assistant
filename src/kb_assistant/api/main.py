"""FastAPI application.

    uv run uvicorn kb_assistant.api.main:app --port 8000

Endpoints
  POST /auth/login              username + password -> JWT                     (rate-limited per IP)
  GET  /auth/me                 who am I, which tools may I use
  POST /chat/stream             one turn, as server-sent events                 (rate-limited per user)
  POST /chat/resume             approve / reject a paused admin action, as SSE  (rate-limited per user)
  GET  /threads                 my sessions
  GET  /threads/{id}/messages   history of one of my sessions
  POST /feedback                thumbs up / down on an answer (stored + sent to LangSmith)
  GET  /admin/faults            (admin) active fault injections
  POST /admin/faults            (admin) set fault injections for demoing failure paths
  GET  /admin/security-events   (admin) recent denials and guard hits
  GET  /health                  dependency status
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from kb_assistant import faults
from kb_assistant.agents.graph import build_graph
from kb_assistant.api.runner import stream_turn
from kb_assistant.api.schemas import (
    ChatRequest,
    FaultsRequest,
    FeedbackRequest,
    LoginRequest,
    LoginResponse,
    ResumeRequest,
)
from kb_assistant.config import get_settings
from kb_assistant.container import Services, build_services
from kb_assistant.errors import AssistantError, AuthError, RateLimitedError
from kb_assistant.observability import configure_logging, configure_tracing, get_logger
from kb_assistant.security.auth import authenticate, decode_token, issue_token
from kb_assistant.security.rbac import Permission, Principal
from kb_assistant.tools.registry import SECURITY_EVENTS

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.validate_secrets()  # refuse to start with a missing, short or published secret
    configure_logging(settings)
    tracing = configure_tracing(settings)
    services = getattr(app.state, "services_override", None) or build_services(settings)
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as checkpointer:
        app.state.services = services
        app.state.graph = build_graph(checkpointer)
        app.state.tracing = tracing
        log.info("api_started", tracing=tracing, vector_store=services.store.name)
        yield


app = FastAPI(title="Commercial Bank Knowledge Assistant", version="0.1.0", lifespan=lifespan)
bearer = HTTPBearer(auto_error=False)


# --- middleware and error handling ---------------------------------------------------------------

@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    log.info("http_request", method=request.method, status=response.status_code,
             elapsed_ms=int((time.perf_counter() - started) * 1000))
    return response


def _error(status: int, error: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    request_id = structlog.contextvars.get_contextvars().get("request_id")
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": error, "message": message, "request_id": request_id})


@app.exception_handler(RateLimitedError)
async def _rate_limited(_: Request, exc: RateLimitedError) -> JSONResponse:
    retry = max(1, int(exc.retry_after_s + 0.999))
    return _error(429, "rate_limited", f"{exc.public_message} Retry in {retry}s.", {"Retry-After": str(retry)})


@app.exception_handler(AuthError)
async def _auth(_: Request, exc: AuthError) -> JSONResponse:
    return _error(401, "unauthorized", exc.public_message, {"WWW-Authenticate": "Bearer"})


@app.exception_handler(AssistantError)
async def _assistant(_: Request, exc: AssistantError) -> JSONResponse:
    log.warning("assistant_error", error=str(exc), kind=type(exc).__name__)
    return _error(503, type(exc).__name__, exc.public_message)


@app.exception_handler(RequestValidationError)
async def _invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
    fields = "; ".join(f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors())
    return _error(422, "invalid_request", fields or "invalid request")


@app.exception_handler(Exception)
async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_error")
    return _error(500, "internal_error", "Something went wrong. The error has been logged.")


# --- dependencies ------------------------------------------------------------------------------------

def get_services(request: Request) -> Services:
    return request.app.state.services


async def current_principal(
    request: Request, creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> Principal:
    if creds is None:
        raise AuthError("missing token", public_message="Please log in.")
    principal = decode_token(creds.credentials, get_services(request).settings)
    structlog.contextvars.bind_contextvars(user=principal.user_id, role=principal.role.value)
    return principal


def require(permission: Permission):
    async def _check(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
        if not principal.can(permission):
            raise HTTPException(403, f"Your role ({principal.role.value}) cannot use this endpoint.")
        return principal
    return _check


PrincipalDep = Annotated[Principal, Depends(current_principal)]
ServicesDep = Annotated[Services, Depends(get_services)]


def _sse(events: AsyncIterator[dict[str, Any]]) -> StreamingResponse:
    async def body():
        async for event in events:
            yield f"data: {json.dumps(event, default=str)}\n\n"
    return StreamingResponse(body(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _own_thread(services: Services, thread_id: str, principal: Principal) -> None:
    owner = await services.memory.thread_owner(thread_id)
    if owner != principal.user_id:
        # 404, not 403: do not confirm that someone else's thread id exists.
        raise HTTPException(404, "Conversation not found.")


# --- auth ----------------------------------------------------------------------------------------

@app.post("/auth/login", response_model=LoginResponse)
async def login(body: LoginRequest, request: Request, services: ServicesDep) -> LoginResponse:
    client_ip = request.client.host if request.client else "unknown"
    await services.rate_limiter.consume(client_ip, "login")
    principal = authenticate(body.username, body.password)
    token = issue_token(principal, services.settings)
    log.info("login", user=principal.user_id, role=principal.role.value)
    return LoginResponse(access_token=token, user=_user_payload(principal, services))


def _user_payload(principal: Principal, services: Services) -> dict[str, Any]:
    return {"user_id": principal.user_id, "name": principal.name, "role": principal.role.value,
            "department": principal.department, "access_levels": sorted(principal.access_levels),
            "tools": [t.name for t in services.registry.for_principal(principal)]}


@app.get("/auth/me")
async def me(principal: PrincipalDep, services: ServicesDep) -> dict[str, Any]:
    return _user_payload(principal, services)


# --- chat ----------------------------------------------------------------------------------------

@app.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request, principal: PrincipalDep,
                      services: ServicesDep) -> StreamingResponse:
    if not principal.can(Permission.CHAT):
        raise HTTPException(403, "Your role cannot chat.")
    await services.rate_limiter.consume(principal.user_id, principal.role.value)
    thread_id = body.thread_id or uuid.uuid4().hex
    if not await services.memory.claim_thread(thread_id, principal.user_id, body.message):
        raise HTTPException(404, "Conversation not found.")
    structlog.contextvars.bind_contextvars(thread_id=thread_id)
    return _sse(stream_turn(request.app.state.graph, services, principal, thread_id, message=body.message))


@app.post("/chat/resume")
async def chat_resume(body: ResumeRequest, request: Request, principal: PrincipalDep,
                      services: ServicesDep) -> StreamingResponse:
    await services.rate_limiter.consume(principal.user_id, principal.role.value)
    await _own_thread(services, body.thread_id, principal)
    graph = request.app.state.graph
    snapshot = await graph.aget_state({"configurable": {"thread_id": body.thread_id}})
    if not snapshot.next:
        raise HTTPException(409, "Nothing is waiting for approval in this conversation.")
    decision = {"approved": body.approved, "comment": body.comment}
    return _sse(stream_turn(graph, services, principal, body.thread_id, resume=decision))


@app.get("/threads")
async def threads(principal: PrincipalDep, services: ServicesDep) -> list[dict[str, Any]]:
    return await services.memory.list_threads(principal.user_id)


@app.get("/threads/{thread_id}/messages")
async def thread_messages(thread_id: str, request: Request, principal: PrincipalDep,
                          services: ServicesDep) -> dict[str, Any]:
    await _own_thread(services, thread_id, principal)
    snapshot = await request.app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = [{"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
                for m in snapshot.values.get("messages", []) if isinstance(m, HumanMessage | AIMessage)]
    return {"thread_id": thread_id, "summary": snapshot.values.get("summary", ""), "messages": messages}


# --- feedback --------------------------------------------------------------------------------------

@app.post("/feedback")
async def feedback(body: FeedbackRequest, request: Request, principal: PrincipalDep,
                   services: ServicesDep) -> dict[str, Any]:
    await _own_thread(services, body.thread_id, principal)
    await services.memory.add_feedback(principal.user_id, body.thread_id, body.run_id, body.score,
                                       body.comment, body.question, body.answer)
    sent = False
    if request.app.state.tracing:
        try:
            from langsmith import Client

            await asyncio.to_thread(Client().create_feedback, body.run_id, key="user_rating",
                                    score=body.score, comment=body.comment)
            sent = True
        except Exception as exc:
            log.warning("langsmith_feedback_failed", error=str(exc))
    return {"stored": True, "sent_to_langsmith": sent}


# --- admin ---------------------------------------------------------------------------------------------

@app.get("/admin/faults")
async def get_faults(_: Annotated[Principal, Depends(require(Permission.ADMIN))]) -> dict[str, Any]:
    return {"active": sorted(faults.active_faults()), "available": list(faults.ALL_FAULTS)}


@app.post("/admin/faults")
async def set_faults(body: FaultsRequest, principal: Annotated[Principal, Depends(require(Permission.ADMIN))]
                     ) -> dict[str, Any]:
    active = faults.set_faults(set(body.faults))
    log.warning("faults_changed", active=sorted(active), by=principal.user_id)
    return {"active": sorted(active)}


@app.get("/admin/security-events")
async def security_events(_: Annotated[Principal, Depends(require(Permission.ADMIN))]) -> list[dict[str, Any]]:
    return list(SECURITY_EVENTS)[-100:]


# --- health ----------------------------------------------------------------------------------------------

@app.get("/health")
async def health(services: ServicesDep, request: Request) -> dict[str, Any]:
    try:
        namespaces = await asyncio.wait_for(services.store.namespaces(), timeout=3)
        vector = {"status": "ok", "store": services.store.name, "namespaces": namespaces}
    except Exception as exc:
        vector = {"status": "degraded", "store": services.store.name, "error": str(exc)[:120]}
    return {
        "status": "ok", "vector_store": vector, "mcp_circuit": services.mcp.breaker.state,
        "llm_configured": services.settings.llm_configured, "tracing": request.app.state.tracing,
        "faults": sorted(faults.active_faults()),
    }
