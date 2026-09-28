FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    FASTEMBED_CACHE_PATH=/cache/fastembed
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

WORKDIR /app
# Dependencies first: this layer is cached until pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY scripts ./scripts
COPY data/corpus ./data/corpus
COPY data/enterprise ./data/enterprise
COPY data/eval ./data/eval
RUN uv sync --frozen --no-dev

RUN useradd --create-home app && mkdir -p /cache /app/data/index /app/data/state && chown -R app /cache /app/data
USER app
ENV PATH="/app/.venv/bin:$PATH"
