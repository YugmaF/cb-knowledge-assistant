"""Application settings.

Every tunable lives here so that a value's origin is always visible: either the default below or
an environment variable / `.env` entry. Nothing reads `os.environ` directly anywhere else.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """The app is misconfigured in a way that must stop it from starting."""


# The repository is public, so any secret that appears in it is a secret an attacker knows.
MIN_SECRET_BYTES = 32
_KNOWN_SECRETS = frozenset({
    "dev-only-change-me-dev-only-change-me",   # the old built-in default
    "change-me-to-a-long-random-string",       # the old .env.example value
})
_PLACEHOLDER = re.compile(r"change[-_ ]?me|replace[-_ ]?me|dev-only", re.I)


def check_secret(name: str, value: str) -> str:
    """Return `value` if it is usable as a secret, otherwise raise ConfigError saying how to fix it."""
    fix = f"Set {name} to a random value, e.g. `openssl rand -hex 32` (./run.sh does this for you)."
    if not value:
        raise ConfigError(f"{name} is not set. {fix}")
    if value in _KNOWN_SECRETS or _PLACEHOLDER.search(value):
        raise ConfigError(f"{name} is a published default or a placeholder. {fix}")
    if len(value.encode()) < MIN_SECRET_BYTES:
        raise ConfigError(f"{name} is shorter than {MIN_SECRET_BYTES} bytes. {fix}")
    return value


class StageModels(BaseModel):
    """Model per pipeline stage. Cheap models for routing and map work, a stronger one where
    the user reads the output. See docs/DESIGN.md#model-selection for the rationale."""

    supervisor: str = "openai/gpt-4o-mini"
    tool_agent: str = "openai/gpt-4o-mini"
    research_plan: str = "openai/gpt-4o-mini"
    research_map: str = "openai/gpt-4.1-mini"  # classification quality drives the counts
    research_reduce: str = "openai/gpt-4.1-mini"
    response: str = "openai/gpt-4.1-mini"
    memory: str = "openai/gpt-4o-mini"
    # Different provider on purpose: an outage at one vendor should not take out the fallback too.
    fallback: str = "google/gemini-2.5-flash"


class RateLimitRule(BaseModel):
    capacity: int = Field(gt=0, description="Burst size: the most requests allowed back to back.")
    refill_per_minute: float = Field(gt=0, description="Sustained requests per minute.")


DEFAULT_RATE_LIMITS: dict[str, RateLimitRule] = {
    "viewer": RateLimitRule(capacity=5, refill_per_minute=10),
    "analyst": RateLimitRule(capacity=10, refill_per_minute=20),
    "admin": RateLimitRule(capacity=20, refill_per_minute=60),
    "login": RateLimitRule(capacity=5, refill_per_minute=5),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_nested_delimiter="__", extra="ignore"
    )

    # --- brand -----------------------------------------------------------------------------
    brand_name: str = "Commercial Bank"
    assistant_name: str = "CB Knowledge Assistant"

    # --- LLM ----------------------------------------------------------------------------------
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = Field(
        default="", validation_alias=AliasChoices("LLM_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
    )
    llm_timeout_s: float = 45.0
    llm_max_retries: int = 2
    models: StageModels = StageModels()

    # --- retrieval ------------------------------------------------------------------------
    corpus_dir: Path = PROJECT_ROOT / "data" / "corpus"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    rerank_enabled: bool = True
    hybrid_alpha: float = Field(default=0.6, ge=0.0, le=1.0, description="1.0 = dense only, 0.0 = sparse only")
    retrieval_top_k: int = 6
    retrieval_candidates: int = 20
    chunk_max_chars: int = 1400
    chunk_overlap_chars: int = 200

    pinecone_api_key: str = ""
    pinecone_index: str = "cb-knowledge"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    vector_timeout_s: float = 8.0

    # --- MCP --------------------------------------------------------------------------------
    enterprise_data_dir: Path = PROJECT_ROOT / "data" / "enterprise"
    mcp_url: str = "http://127.0.0.1:8765/mcp"
    mcp_timeout_s: float = 6.0
    # Shared secret between the API and the MCP server. The server refuses every call without it.
    mcp_service_token: str = ""

    # --- agents -----------------------------------------------------------------------------
    tool_timeout_s: float = 10.0
    tool_agent_max_steps: int = 5
    research_batch_size: int = 5
    research_max_depth: int = 2
    research_max_docs: int = 40
    research_concurrency: int = 4
    sandbox_timeout_s: float = 5.0
    response_max_validation_retries: int = 1

    # --- memory -------------------------------------------------------------------------------
    memory_db_path: Path = PROJECT_ROOT / "data" / "state" / "memory.sqlite"
    checkpoint_db_path: Path = PROJECT_ROOT / "data" / "state" / "checkpoints.sqlite"
    memory_recent_turns: int = 6
    memory_condense_token_budget: int = 3000
    memory_recall_k: int = 3

    # --- security -------------------------------------------------------------------------
    jwt_secret: str = ""  # no default on purpose: validate_secrets() refuses to start without a real one
    jwt_ttl_minutes: int = 120
    max_message_chars: int = 2000
    rate_limits: dict[str, RateLimitRule] = Field(default_factory=lambda: dict(DEFAULT_RATE_LIMITS))

    # --- observability ------------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    langsmith_tracing: bool = Field(
        default=False, validation_alias=AliasChoices("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    )
    langsmith_project: str = Field(default="cb-knowledge-assistant", validation_alias="LANGSMITH_PROJECT")

    api_url: str = "http://127.0.0.1:8000"

    @field_validator("rate_limits", mode="after")
    @classmethod
    def _merge_rate_limits(cls, value: dict[str, RateLimitRule]) -> dict[str, RateLimitRule]:
        # Overriding one role (RATE_LIMITS__VIEWER__CAPACITY=...) must not delete the other roles.
        return {**DEFAULT_RATE_LIMITS, **value}

    def validate_secrets(self) -> None:
        """Called once at API start-up: a weak secret stops the app instead of running insecurely."""
        check_secret("JWT_SECRET", self.jwt_secret)
        check_secret("MCP_SERVICE_TOKEN", self.mcp_service_token)

    @property
    def use_pinecone(self) -> bool:
        return bool(self.pinecone_api_key)

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
