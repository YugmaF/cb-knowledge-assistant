# Design

This document explains *why* the system is built the way it is. The README covers what it does and
how to run it.

## Principles

1. **Understanding goes to the model; rules go in code.** The model classifies, rewrites, plans,
   summarises and writes. Everything that can be checked deterministically is checked in code:
   permissions, document access, date windows, counts, citation ids, numbers in claims, URLs, brand phrases.
2. **Never trust model output.** Every structured output goes through four checks: parse, then
   schema, then business rules, then grounding against the input. A failure is sent back to the
   model with the error, a bounded number of times.
3. **Stable first, dynamic last.** Every prompt starts with the fixed role, rules and schema, and
   ends with the date, user, memory and question. Provider prompt caching reuses identical prefixes;
   the 2,688 cached tokens on the second response call in the RLM run come from this layout.
4. **Degrade, don't crash.** Every dependency has a named failure mode and a fallback. Agents record
   failures in `degraded` instead of raising, and the answer says what was missing.
5. **Make the invisible visible.** Each decision an agent makes is emitted as an event and traced.
   The UI's "How this answer was produced" panel is rebuilt from recorded state, not from the model's
   account of itself.

## Agent architecture

| Agent | LLM? | Responsibility | Output (state keys) |
|---|---|---|---|
| input_guard | no | Normalise, detect injection / exfiltration / tool abuse, block | `guard`, `blocked` |
| load_memory | no | Summary, pinned questions, user facts, recalled episodes | `memory_context` |
| supervisor | yes (cheap) | Intent, standalone query, plan (≤3 steps), filters, user facts | `decision`, `plan` |
| dispatch | no | Walk the plan | `current_step` |
| retrieval | no | Hybrid search (question + task phrasing concurrently), corrective retry | `evidence` |
| research (RLM) | yes (plan, map, reduce) | Multi-document investigation | `research`, `evidence` |
| tool agent | yes (cheap) | Tool-calling loop over role-filtered tools | `tool_messages`, `pending_tool_calls` |
| approval_gate | no | `interrupt()` for writes | `approval` |
| execute_tools | no | RBAC, validation, timeout, sanitise | `tool_results`, `datasets` |
| response | yes (stronger) | Grounded, cited, streamed answer | `draft` |
| validator | no | Citations, grounding, output guard, retry routing | `answer`, `citations`, `validation` |
| update_memory | sometimes | Facts, episode, condensation | `summary`, `messages` |

**State vs context.** `AgentState` holds what agents produce (and what the checkpointer persists).
`RunContext` holds who is asking and which services to use; the API builds it from the verified
token on every run. The model can write into state, but nothing it writes can change the caller's
identity or permissions.

**Why three nodes for tools.** On resume, LangGraph re-runs an interrupted node from the start. If the
LLM call and the interrupt were in one node, approving would call the LLM again, and it might request
a *different* tool than the one approved. The split (`tool_agent → approval_gate → execute_tools`)
makes the gate cheap to re-run and guarantees the approved call is the one executed. A test checks
that resuming does not re-run the supervisor.

### Multi-agent collaboration and failure containment

Agents never call each other directly; they communicate only through state keys (`evidence`,
`research`, `tool_results`, `degraded`). That keeps the blast radius of a failure local:

- An agent that fails writes a line to `degraded` and returns what it has. The next agent still runs.
- The response agent is told what degraded ("Degraded: retrieval: vector search unavailable…")
  and says so in the answer.
- The validator is the last line of defence: whatever went wrong upstream, a claim must cite
  something that was actually retrieved or computed.
- Recursion is bounded everywhere: plan ≤3 steps, tool loop ≤5 steps, research depth ≤2, validator
  retry ≤1, graph recursion limit 40. A misbehaving agent cannot loop forever and cannot multiply cost unbounded.

## RAG design

**Chunking.** One chunk per `##` section, because a section ("Root Cause", "Rollback steps") is the unit a
person cites. Long sections split on paragraphs with 200-character overlap. Each chunk begins with a
header, `[doc_id] title | type | department | section`, so a chunk that says "the pool was
exhausted" still matches "payments-ledger". This is a cheap form of contextual retrieval.

**Hybrid scoring.** Dense embeddings (bge-small, local ONNX) find meaning; BM25 finds exact tokens
(incident ids, service names, error strings). BM25 is encoded as a sparse vector (TF side on
documents, IDF side on queries, hashed token indices), so Pinecone scores both in one query. With
the query weighted `(α·dense, (1-α)·sparse)`, the dot product is `α·dense_score + (1-α)·sparse_score`.
The local store implements the identical arithmetic, so its results match Pinecone's.

The evaluation (README, Results) shows why both are needed: dense-only scored 0.00 on exact-id
questions, BM25-only 0.00 on paraphrases.

**Reranking.** A cross-encoder (ms-marco-MiniLM-L-6) rescores the top 20 hybrid candidates by reading
query and chunk together. It lifted recall@5 from 0.872 to 0.919, entirely on paraphrased questions.

**Corrective retrieval.** If the best reranked hit is weak and the supervisor added filters, the
retrieval agent retries once without them before handing evidence on.

**Attribution.** Chunk text, title, section, date and access level are stored in vector metadata, so a
hit can be cited without a second lookup. Citations are chunk ids (`[INC-2026-028#root-cause]`), which
the validator can check exactly.

## RLM (Recursive Language Model)

The research agent never puts the collection in a context window. It treats the corpus as an
environment to explore with code and sub-calls:

1. **Explore.** `catalog.overview()` gives counts by type and department and the date range. No document text is read.
2. **Plan with code.** The LLM writes a short Python plan against a small API: `overview()`,
   `list_documents(...)` (exhaustive metadata filter) and `search(...)` (hybrid). The plan returns
   doc ids and the section names worth reading. It runs in the sandbox, and the sandbox's `search`
   calls back into the async retriever via `run_coroutine_threadsafe`. A failed plan's error is fed
   back once. After that, a deterministic fallback plan runs.
3. **Code-enforced scope.** The user's date window is applied to the selected ids *after* the plan runs.
   In one test run the model "broadened" a failing plan by widening the date range, which silently
   changed the question. Code now removes out-of-window documents and says so in the activity panel.
4. **Targeted sections.** Only the chosen sections are fetched (for example Summary and Root Cause), by
   chunk id, per namespace, concurrently.
5. **Decompose and recurse.** Documents are batched. A batch over the sub-agent's character budget
   is split in half and each half goes to a deeper sub-agent. Sub-agents run concurrently under a
   semaphore. A sub-agent may return follow-up queries ("see the runbook for …"); those are searched
   and analysed one level deeper. Follow-up documents are context only and are not counted.
6. **Validate sub-agent output.** A finding about a document that was not in the batch is discarded.
   An evidence id not in the batch is replaced, and the date comes from metadata, never from the model.
7. **Aggregate in code.** Counts per root-cause category, the timeline and recurrence are computed with
   `Counter`, not by the model.
8. **Reduce.** One LLM call writes the report from the findings and the code-computed numbers. If the
   model changes a count, code overwrites it.

The recursion is visible live (batch labels like `b3.2` = batch 3, second half, depth 2) and in
LangSmith (`rlm_sub_agent` spans).

## Memory design decisions

| Layer | Stored in | Lifetime | Scope |
|---|---|---|---|
| Conversation (short-term) | LangGraph checkpointer (SQLite) | the thread; survives API restarts and approval pauses | thread, owned by one user |
| User context | Verified token (name, role, department) + `user_facts` table | long-term | user |
| Relevant historical interactions (episodic) | `interactions` table + question embedding | long-term | user, across that user's threads |

- **Why checkpointing for the conversation.** The same mechanism that persists messages also
  persists a paused human-approval turn, so "memory survives multiple turns" and "resume after
  approval" have one implementation.
- **Condensation triggers on tokens, not message count.** One huge turn (a long research answer)
  should trigger it; ten short "thanks" should not.
- **User messages are pinned.** Only assistant answers are folded into the running summary. On a
  previous agent platform, a summary-of-a-summary once claimed the user had never given an id they
  had given 30 events earlier. Pinning user text means the condenser can never rewrite what the
  user said.
- **Episodic recall has a similarity threshold (0.55)** and excludes the current thread, so it only
  surfaces genuinely related past questions from *other* sessions.
- **Strict per-user scoping.** Every memory query takes `user_id` from the verified principal. There
  is no API or code path that reads another user's memory; a test checks that a second user recalls
  nothing.
- **Thread ownership.** A thread id is bound to its creator on first use. Another user presenting the
  same id gets 404 (not 403, so existence is not confirmed).

## Model selection

| Stage | Default model | Why |
|---|---|---|
| supervisor, tool agent, research plan, memory | `openai/gpt-4o-mini` | Classification, JSON, tool calling: cheap, fast, reliable function calling |
| research map (sub-agents) | `openai/gpt-4.1-mini` | Classification quality drives the counts. 4o-mini mislabelled some incidents in testing |
| research reduce, response | `openai/gpt-4.1-mini` | What the user reads: better instruction following for citations, still low cost |
| fallback (every stage) | `google/gemini-2.5-flash` | **Different provider**, so one vendor outage cannot take out both |

- **No reasoning models by default.** Reasoning tokens are billed as output, and these stages are
  well-defined. The supervisor's plan is small and checked by code; a reasoning model would add
  cost and latency for little gain. If research quality needed it, only `research_reduce` would move.
- **Access through OpenRouter** (OpenAI-compatible), so any stage can be switched to Claude, Gemini or an
  open model with one env var (`MODELS__RESPONSE=anthropic/claude-sonnet-5`), without code changes.
- **Temperature 0** everywhere. That makes output *repeatable*, not *guaranteed* (provider batching,
  model updates), which is why tests use a scripted LLM and evals use validators rather than
  exact string matches.
- **Cost per task, not per call.** Each LLM call emits tokens and an estimated cost; the UI shows the
  total per turn. A 12-call RLM turn costs about $0.01.

## Failure handling

| Failure | Detection | Behaviour | Test |
|---|---|---|---|
| LLM primary down / 5xx / timeout | client retries, then exception | fallback model (other provider) | fault `llm_primary` |
| LLM returns HTTP 200 with empty body | `_record_usage` checks content | treated as transport failure → fallback | code path in `llm.py` |
| All LLMs down | `LLMUnavailableError` | supervisor → keyword router; response → extractive answer with citations; research → code aggregates only | `test_llm_outage_degrades…` |
| Invalid JSON / schema from LLM | Pydantic | error fed back as a user message, ≤3 attempts | live runs (null filters) |
| Vector DB down | `VectorStoreError` (per namespace) | partial results if some namespaces answer; else local BM25 keyword index; marked degraded | `test_vector_store_failure…`, `test_vector_db_outage…` |
| MCP down | `MCPUnavailableError` | tool result "ERROR …" to the agent; circuit opens after 3 failures (fail fast for 30 s) | `test_mcp_failure_opens_circuit` |
| Tool timeout | `asyncio.timeout` | "TIMEOUT …" tool result; agent continues | `test_tool_timeout…` |
| Invalid request | FastAPI/Pydantic | 422 with a readable message and `request_id` | `test_requests_without_token…` |
| Rate limited | token bucket | 429 + `Retry-After`; UI shows the wait | `test_rate_limit…` |
| Unexpected exception in a turn | runner's last-resort handler | stream ends with an `error` event, not a broken connection | `runner.py` |

Every row except the empty-body case can be forced live from the admin sidebar (`POST /admin/faults`).

## Observability

- **LangSmith**: each turn is one run named `chat_turn`, with `run_id` generated by the API (so feedback
  can be attached), tags `role:<role>` and metadata user/role/thread. LangGraph traces every node
  (the agent transitions). `@traceable` adds spans for `hybrid_search` and `fetch_sections`
  (run_type retriever), `execute_tool` and `mcp_call` (tool), and `rlm_sub_agent` (chain, one per
  recursive call). LLM calls are traced by LangChain automatically.
- **Structured logs**: JSON lines with `request_id`, `user`, `role`, `thread_id` bound once per request.
  Includes `llm_call` (stage, model, tokens, cost, latency), `tool_executed`, `security_event`,
  `turn_done`.
- **Activity stream**: the same events go to the UI over SSE, so an evaluator sees the agent's
  internals without opening LangSmith.
