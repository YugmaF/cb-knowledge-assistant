"""Structured logging and LangSmith wiring.

Logs are JSON lines with a request-scoped context (request_id, user, thread_id) bound through
contextvars, so every line written while handling a request can be joined back to its trace.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog

from kb_assistant.config import Settings


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=level, format="%(message)s")
    # Third-party libraries log at INFO on every HTTP call; keep them quiet.
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    renderer = (
        structlog.processors.JSONRenderer() if settings.log_json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def configure_tracing(settings: Settings) -> bool:
    """Turn on LangSmith tracing when a key is present. LangGraph and LangChain pick these
    variables up automatically; our own retrieval / tool / sandbox steps use @traceable."""
    if not (settings.langsmith_tracing and os.getenv("LANGSMITH_API_KEY")):
        return False
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    return True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
