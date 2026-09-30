#!/usr/bin/env bash
# Run the Commercial Bank Knowledge Assistant locally.
#
#   ./run.sh                         start MCP + API + UI (builds the index first only if it is missing)
#   ./run.sh --build                 force a rebuild of the index, then start
#   ./run.sh build                   only (re)build the index, then exit
#   ./run.sh check                   only report whether the index is ready
#   ./run.sh ask [--user anil] "…"   one question in the terminal (starts MCP for the duration)
#   ./run.sh test                    offline test suite (no keys needed)
#   ./run.sh eval                    retrieval eval (recall@5, access-control leaks)
#
# Options (any command):
#   --env-file PATH   file with OPENROUTER_API_KEY, PINECONE_API_KEY, LANGSMITH_API_KEY (default: ./.env)
#
# Environment overrides:
#   API_PORT (8010)  UI_PORT (8511)  MCP_PORT (8765)
#   KB_PINECONE_INDEX (cb-knowledge)  KB_LANGSMITH_PROJECT (cb-knowledge-assistant)
#
# The Pinecone index and LangSmith project are always set from KB_* AFTER the env file is loaded,
# so an env file borrowed from another project can never point this app at that project's index.
set -euo pipefail

cd "$(dirname "$0")"
ROOT=$(pwd)
LOG_DIR="$ROOT/logs"

API_PORT=${API_PORT:-8010}
UI_PORT=${UI_PORT:-8511}
MCP_PORT=${MCP_PORT:-8765}
ENV_FILE=${ENV_FILE:-.env}

say()  { printf '\033[1;34m▶ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m! %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- arguments

COMMAND=start
FORCE_BUILD=0
PASSTHROUGH=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    start|build|check|ask|test|eval) COMMAND=$1 ;;
    --build) FORCE_BUILD=1 ;;
    --env-file) [[ $# -ge 2 ]] || die "--env-file needs a path"; ENV_FILE=$2; shift ;;
    --env-file=*) ENV_FILE=${1#*=} ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) PASSTHROUGH+=("$1") ;;
  esac
  shift
done

# ---------------------------------------------------------------- environment

random_hex() {
  openssl rand -hex 32 2>/dev/null || "$PY" -c 'import secrets; print(secrets.token_hex(32))'
}

# A secret the app refuses to start without. Keep a real value from the env file; otherwise make a
# random one for this run. The example placeholder (REPLACE_ME…) counts as "not set".
ensure_secret() {  # NAME
  local name=$1 value=${!1:-}
  if [[ -z $value || $value == *REPLACE_ME* ]]; then
    export "$name=$(random_hex)"
    say "$name was not set: generated a random one for this run (sessions end when run.sh stops)"
  fi
}

load_env() {
  if [[ -f "$ENV_FILE" ]]; then
    say "loading keys from $ENV_FILE"
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
  else
    warn "no env file at '$ENV_FILE': running without LLM/Pinecone/LangSmith keys (local store, no tracing)"
  fi
  export PINECONE_INDEX=${KB_PINECONE_INDEX:-cb-knowledge}
  export LANGSMITH_PROJECT=${KB_LANGSMITH_PROJECT:-cb-knowledge-assistant}
  unset LLM_MODEL || true   # another project's single-model setting; this app routes models per stage
  export MCP_URL="http://127.0.0.1:${MCP_PORT}/mcp" MCP_PORT API_URL="http://127.0.0.1:${API_PORT}"
  ensure_secret JWT_SECRET
  ensure_secret MCP_SERVICE_TOKEN

  local key
  for key in OPENROUTER_API_KEY PINECONE_API_KEY LANGSMITH_API_KEY; do
    if [[ -n "${!key:-}" ]]; then echo "   $key: set"; else echo "   $key: not set"; fi
  done
  echo "   Pinecone index: $PINECONE_INDEX · LangSmith project: $LANGSMITH_PROJECT"
}

PY="$ROOT/.venv/bin/python"
ensure_deps() {
  command -v uv >/dev/null || die "uv is not installed: https://docs.astral.sh/uv/"
  # Fast no-op when the environment already matches uv.lock.
  uv sync --quiet || die "uv sync failed"
}

# ---------------------------------------------------------------- index gate

build_index() {
  say "building the index (chunk → embed → BM25 → local store + Pinecone)"
  "$PY" -m kb_assistant.retrieval.ingest
}

gate_index() {
  if [[ $FORCE_BUILD == 1 ]]; then
    build_index
  elif "$PY" -m kb_assistant.retrieval.ingest --check; then
    :   # ready; --check already printed why
  else
    build_index
  fi
}

# ---------------------------------------------------------------- processes

PIDS=()
cleanup() {
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    say "stopping services"
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }

wait_for() {  # name url seconds
  local name=$1 url=$2 tries=$(( $3 * 2 ))
  while (( tries-- > 0 )); do
    if curl -fs -o /dev/null "$url" 2>/dev/null; then return 0; fi
    sleep 0.5
  done
  die "$name did not start; see $LOG_DIR/$name.log"
}

start_mcp() {
  port_busy "$MCP_PORT" && die "port $MCP_PORT (MCP) is in use; set MCP_PORT=… or stop the other process"
  mkdir -p "$LOG_DIR"
  # Loopback only, and every call needs MCP_SERVICE_TOKEN (generated by ensure_secret when unset).
  MCP_HOST=127.0.0.1 "$PY" -m kb_assistant.mcp_server.server >"$LOG_DIR/mcp.log" 2>&1 &
  PIDS+=($!)
  # The MCP endpoint answers GET with 4xx; any HTTP answer means it is listening.
  local tries=40
  until port_busy "$MCP_PORT"; do (( tries-- > 0 )) || die "MCP did not start; see $LOG_DIR/mcp.log"; sleep 0.5; done
  echo "   MCP  http://127.0.0.1:$MCP_PORT/mcp"
}

start_all() {
  local p
  for p in "$API_PORT" "$UI_PORT"; do
    port_busy "$p" && die "port $p is in use (another app?). Try: API_PORT=8020 UI_PORT=8521 ./run.sh"
  done
  start_mcp

  "$PY" -m uvicorn kb_assistant.api.main:app --host 127.0.0.1 --port "$API_PORT" >"$LOG_DIR/api.log" 2>&1 &
  PIDS+=($!)
  wait_for api "http://127.0.0.1:$API_PORT/health" 90
  echo "   API  http://127.0.0.1:$API_PORT/docs"
  curl -fs "http://127.0.0.1:$API_PORT/health" | "$PY" -c "import json,sys; h=json.load(sys.stdin); v=h['vector_store']; \
print('        vector store:', v['store'], '(' + v['status'] + ') · MCP circuit:', h['mcp_circuit'], \
      '· LLM:', h['llm_configured'], '· tracing:', h['tracing'])"

  "$PY" -m streamlit run src/kb_assistant/ui/streamlit_app.py --server.address 127.0.0.1 \
    --server.port "$UI_PORT" --server.headless true >"$LOG_DIR/ui.log" 2>&1 &
  PIDS+=($!)
  wait_for ui "http://127.0.0.1:$UI_PORT/_stcore/health" 60

  say "ready → open http://127.0.0.1:$UI_PORT"
  echo "   users: vera / viewer-pass · anil / analyst-pass · amal / admin-pass"
  echo "   logs:  $LOG_DIR/{mcp,api,ui}.log · Ctrl+C stops everything"
  # Exit (and clean up) as soon as any service dies.
  wait -n "${PIDS[@]}"
  warn "a service exited; check $LOG_DIR"
}

# ---------------------------------------------------------------- commands

ensure_deps
case "$COMMAND" in
  test)
    "$PY" -m pytest -q "${PASSTHROUGH[@]}"
    ;;
  check)
    load_env
    "$PY" -m kb_assistant.retrieval.ingest --check
    ;;
  build)
    load_env
    build_index
    ;;
  eval)
    load_env
    gate_index
    "$PY" scripts/eval_retrieval.py "${PASSTHROUGH[@]}"
    ;;
  ask)
    load_env
    gate_index
    start_mcp
    "$PY" scripts/ask.py "${PASSTHROUGH[@]}"
    ;;
  start)
    load_env
    gate_index
    start_all
    ;;
esac
