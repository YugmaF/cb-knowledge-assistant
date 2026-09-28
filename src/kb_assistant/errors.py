"""Domain exceptions.

Each dependency has its own error type so that callers can degrade precisely: a vector-store
failure falls back to keyword search, an MCP failure becomes a tool result the model can read,
and only an LLM failure with no fallback left reaches the user as an extractive answer.
"""

from __future__ import annotations


class AssistantError(Exception):
    """Base class. `public_message` is safe to show to an end user; `str(exc)` may not be."""

    public_message = "The assistant hit an internal error."

    def __init__(self, detail: str = "", *, public_message: str | None = None) -> None:
        super().__init__(detail or self.public_message)
        if public_message:
            self.public_message = public_message


class LLMUnavailableError(AssistantError):
    public_message = "The language model is unavailable right now."


class LLMOutputError(AssistantError):
    public_message = "The language model returned an answer that failed validation."


class VectorStoreError(AssistantError):
    public_message = "Semantic search is unavailable; results may be less complete."


class MCPUnavailableError(AssistantError):
    public_message = "The enterprise data service is unavailable."


class ToolTimeoutError(AssistantError):
    public_message = "A tool took too long to respond."


class ToolDeniedError(AssistantError):
    public_message = "You do not have permission to use that tool."


class ToolInputError(AssistantError):
    public_message = "A tool was called with invalid arguments."


class SandboxError(AssistantError):
    public_message = "Generated analysis code was rejected or failed."


class AuthError(AssistantError):
    public_message = "Authentication failed."


class RateLimitedError(AssistantError):
    public_message = "Too many requests. Please wait a moment and try again."

    def __init__(self, retry_after_s: float) -> None:
        super().__init__(f"rate limited, retry after {retry_after_s:.1f}s")
        self.retry_after_s = retry_after_s
