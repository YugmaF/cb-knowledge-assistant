"""Request / response models. Validation here is the first layer of input validation: malformed
requests are rejected with 422 before any agent, tool or LLM is involved."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_-]{8,64}$")


class ResumeRequest(BaseModel):
    thread_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    approved: bool
    comment: str = Field(default="", max_length=300)


class FeedbackRequest(BaseModel):
    thread_id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    run_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    score: Literal[-1, 1]
    comment: str | None = Field(default=None, max_length=500)
    question: str | None = Field(default=None, max_length=4000)
    answer: str | None = Field(default=None, max_length=8000)


class FaultsRequest(BaseModel):
    faults: list[Literal["llm", "llm_primary", "vectordb", "mcp", "tool_slow"]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: str
    message: str
    request_id: str | None = None
