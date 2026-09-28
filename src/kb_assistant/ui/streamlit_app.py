"""Streamlit UI: chat on the left, live agent activity on the right.

    uv run streamlit run src/kb_assistant/ui/streamlit_app.py

The UI is a thin client of the API: it holds a JWT, posts messages, and renders the server-sent
event stream. It has no access to the model, the documents or the tools.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import httpx
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")
TIMEOUT = httpx.Timeout(10.0, read=240.0)

st.set_page_config(page_title="CB Knowledge Assistant", layout="wide")

defaults: dict[str, Any] = {"token": None, "user": None, "thread_id": None, "history": [], "pending": None,
                            "activity": [], "flash": None}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)
S = st.session_state


# --- API helpers ---------------------------------------------------------------------------------

def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {S.token}"} if S.token else {}


def _error_text(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        return body.get("message") or body.get("detail") or resp.text
    except ValueError:
        return resp.text


def api(method: str, path: str, **kwargs) -> Any:
    resp = httpx.request(method, f"{API}{path}", headers=_headers(), timeout=TIMEOUT, **kwargs)
    if resp.status_code == 401 and S.token:
        S.token = None
        raise RuntimeError("Session expired, please log in again.")
    if resp.status_code >= 400:
        raise RuntimeError(f"{resp.status_code}: {_error_text(resp)}")
    return resp.json()


def sse(path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    with httpx.stream("POST", f"{API}{path}", json=payload, headers=_headers(), timeout=TIMEOUT) as resp:
        if resp.status_code >= 400:
            resp.read()
            yield {"type": "http_error", "status": resp.status_code, "message": _error_text(resp),
                   "retry_after": resp.headers.get("retry-after")}
            return
        for line in resp.iter_lines():
            if line.startswith("data: "):
                yield json.loads(line[6:])


# --- activity rendering -------------------------------------------------------------------------------

ICONS = {"node": "🔷", "supervisor": "🧭", "retrieval": "🔎", "tool_call": "🛠️", "tool_agent": "🤖",
         "rlm": "🌀", "memory": "🧠", "validation": "✅", "guard": "🛡️", "llm_call": "💬", "approval": "🙋",
         "error": "⚠️", "response": "✍️"}


def describe(e: dict[str, Any]) -> str | None:
    t = e.get("type")
    if t == "node":
        extra = ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("type", "node", "status"))
        return f"**{e['node']}** · {e['status']}" + (f" ({extra})" if extra else "")
    if t == "guard":
        return f"input guard: {'allowed' if e['allowed'] else '**BLOCKED** ' + ', '.join(e['categories'])}"
    if t == "supervisor":
        plan = " → ".join(s["agent"] for s in e["plan"]) or "respond directly"
        notes = f"  \n  policy: {'; '.join(e['policy_notes'])}" if e.get("policy_notes") else ""
        return (f"intent **{e['intent']}** · plan: {plan} · filters: {e.get('filters') or 'none'}  \n"
                f"  _{e.get('reasoning', '')}_{notes}")
    if t == "retrieval":
        if "hits" not in e:
            return f"retrieval: {e.get('status')}"
        top = ", ".join(h["chunk_id"] for h in e["hits"][:4])
        flags = f" · ⚠️ injection removed from {e['flagged_chunks']}" if e.get("flagged_chunks") else ""
        return (f"retrieval [{e['mode']}{' · DEGRADED' if e['degraded'] else ''}] "
                f"{len(e['hits'])} hits in {e['elapsed_ms']} ms: {top}{flags}")
    if t == "tool_agent":
        calls = ", ".join(c["name"] for c in e.get("calls", [])) or e.get("note", "")[:120]
        return f"tool agent {e['status']}: {calls}"
    if t == "tool_call":
        if e["status"] == "started":
            return f"→ `{e['tool']}` {json.dumps(e.get('args'))[:160]}"
        err = f" · {e['error']}" if e.get("error") else ""
        return f"← `{e['tool']}` **{e['status']}**{err} ({e.get('elapsed_ms', 0)} ms)"
    if t == "rlm":
        phase = e["phase"]
        if phase == "plan" and e.get("code"):
            return f"RLM plan ({e['source']}): {e.get('rationale', '')}\n```python\n{e['code']}\n```"
        details = {k: v for k, v in e.items() if k not in ("type", "phase")}
        return f"RLM **{phase}**: {json.dumps(details, default=str)[:260]}"
    if t == "memory":
        details = {k: v for k, v in e.items() if k not in ("type", "action")}
        return f"memory **{e['action']}**: {json.dumps(details)[:200]}"
    if t == "validation":
        ok = "passed" if e["ok"] else "**failed**"
        issues = "; ".join(e.get("issues") or ([e["detail"]] if e.get("detail") else []))
        warn = f" · warnings: {len(e['warnings'])}" if e.get("warnings") else ""
        red = f" · redacted: {e['redactions']}" if e.get("redactions") else ""
        return f"validation of {e['target']} {ok} {issues[:300]}{warn}{red}"
    if t == "llm_call":
        return (f"LLM {e['stage']} · {e['model']} ({e['attempt']}) · {e['input_tokens']}→{e['output_tokens']} tok"
                f" · cached {e['cached_tokens']} · ${e['cost_usd']:.5f} · {e['latency_ms']} ms")
    if t == "approval":
        return f"human {'approved' if e['approved'] else 'rejected'} {e['calls']}"
    if t == "error":
        return f"{e.get('component')}: {e.get('detail', '')[:200]} → {e.get('action', '')}"
    if t == "response":
        return f"response agent: {e['status']} (attempt {e['attempt']})"
    return None


# --- turn execution -------------------------------------------------------------------------------------

def run_turn(path: str, payload: dict[str, Any], chat_col, activity_col) -> None:
    with chat_col, st.chat_message("assistant"):
        answer_ph = st.empty()
    with activity_col:
        state_ph = st.empty()
        log_ph = st.container(height=560)
    buffer = ""
    lines: list[str] = []
    current = "starting"
    tokens = 0
    for event in sse(path, payload):
        t = event.get("type")
        if t == "http_error":
            wait = f" (retry in {event['retry_after']}s)" if event.get("retry_after") else ""
            S.flash = ("error", f"{event['message']}{wait}")
            return
        if t == "run_started":
            S.thread_id = event["thread_id"]
            continue
        if t == "token":
            buffer += event["text"]
            answer_ph.markdown(buffer + " ▌")
            continue
        if t == "token_reset":
            buffer = ""
            continue
        if t == "node":
            current = event["node"]
        if t == "llm_call":
            tokens += event["input_tokens"] + event["output_tokens"]
        if t == "approval_required":
            S.pending = event
            answer_ph.warning("Waiting for your approval (see below).")
            break
        if t == "final":
            answer_ph.markdown(event["answer"])
            S.history.append({"role": "assistant", "content": event["answer"], "citations": event["citations"],
                              "explanation": event["explanation"], "run_id": event["run_id"],
                              "usage": event["usage"], "validation": event["validation"]})
            current = "done"
        if t == "error":
            S.flash = ("error", event.get("detail", "error"))
        text = describe(event)
        if text:
            lines.append(f"{ICONS.get(t, '•')} {text}")
            with log_ph:
                st.markdown(lines[-1])
        state_ph.info(f"**Active node:** `{current}` · events: {len(lines)} · tokens: {tokens}")
    S.activity = lines


# --- sidebar ------------------------------------------------------------------------------------------------

with st.sidebar:
    st.header("CB Knowledge Assistant")
    if not S.token:
        with st.form("login"):
            username = st.text_input("Username", value="anil")
            password = st.text_input("Password", type="password", value="analyst-pass")
            if st.form_submit_button("Log in"):
                try:
                    data = api("POST", "/auth/login", json={"username": username, "password": password})
                    S.token, S.user = data["access_token"], data["user"]
                    S.history, S.thread_id, S.pending = [], None, None
                    st.rerun()
                except RuntimeError as exc:
                    st.error(str(exc))
        st.caption("Demo users: vera / viewer-pass · anil / analyst-pass · amal / admin-pass")
    else:
        u = S.user
        st.markdown(f"**{u['name']}** · `{u['role']}` · {u['department']}")
        st.caption(f"Documents: {', '.join(u['access_levels'])}")
        st.caption(f"Tools: {', '.join(u['tools'])}")
        c1, c2 = st.columns(2)
        if c1.button("New chat"):
            S.history, S.thread_id, S.pending, S.activity = [], None, None, []
            st.rerun()
        if c2.button("Log out"):
            for key, value in defaults.items():
                S[key] = value
            st.rerun()
        if S.thread_id:
            st.caption(f"thread `{S.thread_id}`")
        if u["role"] == "admin":
            st.divider()
            st.subheader("Fault injection")
            try:
                faults = api("GET", "/admin/faults")
                chosen = st.multiselect("Force failures", faults["available"], default=faults["active"])
                if st.button("Apply faults"):
                    api("POST", "/admin/faults", json={"faults": chosen})
                    st.rerun()
            except RuntimeError as exc:
                st.error(str(exc))
        st.divider()
        try:
            health = httpx.get(f"{API}/health", timeout=5).json()
            st.caption(f"vector store: {health['vector_store']['store']} ({health['vector_store']['status']}) · "
                       f"MCP circuit: {health['mcp_circuit']} · tracing: {health['tracing']}")
        except Exception:
            st.caption("API unreachable")

if S.flash:
    kind, message = S.flash
    getattr(st, kind)(message)
    S.flash = None

if not S.token:
    st.info("Log in from the sidebar to start.")
    st.stop()

chat_col, activity_col = st.columns([3, 2], gap="large")

with chat_col:
    st.subheader("Chat")
    for i, msg in enumerate(S.history):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] != "assistant":
                continue
            if msg.get("citations"):
                with st.expander(f"Sources ({len(msg['citations'])})"):
                    for c in msg["citations"]:
                        st.markdown(f"**[{c['id']}]** {c['title']}"
                                    + (f" · {c['section']} · {c.get('created_date', '')} · {c.get('access_level', '')}"
                                       if c.get("section") else ""))
                        if c.get("snippet"):
                            st.caption(c["snippet"])
            if msg.get("explanation"):
                with st.expander("How this answer was produced"):
                    st.json(msg["explanation"])
                    st.caption(f"usage: {msg.get('usage')}")
            if msg.get("run_id"):
                f1, f2, _ = st.columns([1, 1, 8])
                for col, score, label in ((f1, 1, "👍"), (f2, -1, "👎")):
                    if col.button(label, key=f"fb{score}-{i}"):
                        question = S.history[i - 1]["content"] if i else None
                        try:
                            api("POST", "/feedback", json={"thread_id": S.thread_id, "run_id": msg["run_id"],
                                                           "score": score, "question": question,
                                                           "answer": msg["content"][:8000]})
                            st.toast("Thanks, feedback recorded.")
                        except RuntimeError as exc:
                            st.toast(str(exc))

    if S.pending:
        p = S.pending
        st.warning(p.get("message", "Approval required"))
        for call in p.get("calls", []):
            st.code(f"{call['tool']}({json.dumps(call['args'], indent=1)})", language="python")
        a1, a2, _ = st.columns([1, 1, 4])
        decision = True if a1.button("Approve", type="primary") else (False if a2.button("Reject") else None)
        if decision is not None:
            S.pending = None
            S.history.append({"role": "user", "content": f"_{'Approved' if decision else 'Rejected'} the action._"})
            run_turn("/chat/resume", {"thread_id": p["thread_id"], "approved": decision}, chat_col, activity_col)
            st.rerun()

with activity_col:
    st.subheader("Agent activity")
    if S.activity:
        with st.expander("Last turn", expanded=True):
            st.markdown("\n\n".join(S.activity))

prompt = st.chat_input("Ask about policies, runbooks, incidents, services…", disabled=bool(S.pending))
if prompt:
    S.history.append({"role": "user", "content": prompt})
    with chat_col, st.chat_message("user"):
        st.markdown(prompt)
    payload = {"message": prompt, **({"thread_id": S.thread_id} if S.thread_id else {})}
    run_turn("/chat/stream", payload, chat_col, activity_col)
    st.rerun()
