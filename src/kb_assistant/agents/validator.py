"""Validator agent: checks the draft answer before it reaches the user.

Deterministic checks, in order:
  1. output guard     leaked system prompt (canary), brand rules, unknown URLs, markdown images,
                      contact-detail redaction for roles that may not see it
  2. citations exist  every [id] cited must be a chunk or tool that was actually retrieved/called
                      (catches hallucinated citations)
  3. citations present  a factual answer built on evidence must cite something
  4. grounding        numbers (dates, counts, durations) in a cited sentence must appear in a cited
                      source; low word overlap with the cited source is reported as a warning

Hard failures send the draft back to the response agent once, with the list of problems. If it still
fails, invalid citations are stripped and a visible caveat is added, or, for a leak or a brand
violation, the answer is replaced by a safe message. The user never gets an unvalidated answer.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from kb_assistant.agents.common import node_event
from kb_assistant.agents.events import emit
from kb_assistant.agents.state import AgentState, RunContext
from kb_assistant.retrieval.sparse import tokenize
from kb_assistant.security.guards import check_output
from kb_assistant.security.rbac import Permission
from kb_assistant.tools.registry import record_security_event

_ID = r"(?:tool:[a-z_]+|[A-Z]{2,5}-[\w-]+#[\w-]+)"
CITATION = re.compile(rf"\[({_ID})\]")
# Models often write [A, B] or [A; B] despite the instructions; accept it and normalise to [A][B].
_GROUPED = re.compile(rf"\[({_ID}(?:\s*[,;]\s*{_ID})+)\]")
_BRACKETED = re.compile(r"\[([^\[\]]{3,80})\]")
_NUMBER = re.compile(r"\b\d[\d,.:]*\d\b|\b\d\b")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
# Intents that make factual claims and therefore must carry citations.
_CITED_INTENTS = {"knowledge_question", "research_summary", "enterprise_lookup", "analytics"}
_NOT_FOUND = re.compile(r"couldn.t find|could not find|not (in|within) the documents|no (relevant )?documents", re.I)


def _sources(state: AgentState) -> dict[str, str]:
    sources = {e["chunk_id"]: e["text"] for e in state.get("evidence", [])}
    for r in state.get("tool_results", []):
        sources[f"tool:{r['tool']}"] = r["content"]
    research = state.get("research") or {}
    if research.get("aggregates"):
        # Counts computed by code are a legitimate source for the numbers they contain.
        sources.setdefault("tool:research_aggregates", str(research["aggregates"]))
    return sources


def validate_answer(state: AgentState, ctx: RunContext) -> dict[str, Any]:
    principal = ctx.principal
    draft = state.get("draft", "")
    sources = _sources(state)
    source_urls = set(re.findall(r"https?://[^\s)\]>\"']+", " ".join(sources.values())))
    guard = check_output(
        draft, allowed_urls=source_urls - {u for u in source_urls if "exfil" in u},
        redact_contact_details=not principal.can(Permission.MCP_READ),
    )
    text = _GROUPED.sub(lambda m: "".join(f"[{i.strip()}]" for i in re.split(r"[,;]", m.group(1))), guard.text)
    issues = list(guard.issues)
    warnings: list[str] = []

    cited = CITATION.findall(text)
    unknown = sorted({c for c in cited if c not in sources})
    # Bracketed labels that are not ids at all, e.g. [research_report] or [the runbook].
    pseudo = sorted({b for b in _BRACKETED.findall(text)
                     if not CITATION.fullmatch(f"[{b}]") and re.fullmatch(r"[\w .:-]+", b)
                     and ("_" in b or "report" in b.lower() or "source" in b.lower())})
    if pseudo:
        issues.append(f"invalid_citation_labels: {', '.join(pseudo)} (cite chunk ids, not labels)")
    if unknown:
        issues.append(f"hallucinated_citations: {', '.join(unknown)} (these ids were not retrieved)")

    intent = (state.get("decision") or {}).get("intent")
    if intent in _CITED_INTENTS and sources and not cited and not _NOT_FOUND.search(text):
        issues.append("missing_citations: factual answer without any [source] citation")

    for sentence in _SENTENCE.split(text):
        if sentence.lstrip("*_ ").lower().startswith("why this answer"):
            continue  # the provenance line names sources; it makes no factual claim to ground
        ids = [c for c in CITATION.findall(sentence) if c in sources]
        if not ids:
            continue
        claim = CITATION.sub("", sentence)
        support = " ".join(sources[i] for i in ids) + " " + sources.get("tool:research_aggregates", "")
        missing = [n for n in _NUMBER.findall(claim) if n not in support and n.replace(",", "") not in support]
        if missing:
            issues.append(f"ungrounded_numbers: {missing} in '{claim.strip()[:90]}' not found in {ids}")
        words = set(tokenize(claim))
        if len(words) >= 6:
            overlap = len(words & set(tokenize(support))) / len(words)
            if overlap < 0.25:
                warnings.append(f"weak_grounding ({overlap:.0%} overlap): '{claim.strip()[:80]}'")

    return {"ok": not issues, "issues": issues, "warnings": warnings, "redactions": guard.redactions,
            "text": text, "citations": sorted({c for c in cited if c in sources})}


def _finalize_failed(text: str, issues: list[str], brand: str) -> str:
    if any(i == "system_prompt_leak" or i.startswith("brand:") for i in issues):
        return (f"I can't provide that answer in a way that meets {brand}'s communication standards. "
                "Please rephrase your question or contact the relevant team directly.")
    cleaned = text
    for issue in issues:
        if issue.startswith("hallucinated_citations"):
            for bad in issue.split(":", 1)[1].split("(")[0].split(","):
                cleaned = cleaned.replace(f"[{bad.strip()}]", "")
    return cleaned + ("\n\n> **Note:** parts of this answer could not be fully verified against the source "
                      "documents. Please check the cited sources before acting on it.")


async def validator(state: AgentState, runtime: Runtime[RunContext]) -> dict[str, Any]:
    node_event("validator", "started")
    ctx = runtime.context
    settings = ctx.services.settings
    result = validate_answer(state, ctx)
    attempts = state.get("validation_attempts", 0) + 1
    emit({"type": "validation", "target": "answer", "ok": result["ok"], "attempt": attempts,
          "issues": result["issues"], "warnings": result["warnings"], "redactions": result["redactions"],
          "citations": result["citations"]})

    if "system_prompt_leak" in result["issues"]:
        record_security_event("system_prompt_leak_blocked", ctx.principal)

    if result["ok"] or attempts > settings.response_max_validation_retries:
        answer = result["text"] if result["ok"] else _finalize_failed(result["text"], result["issues"], settings.brand_name)
        citations = _citation_details(state, result["citations"])
        return {"answer": answer, "citations": citations, "validation_attempts": attempts,
                "validation": {k: result[k] for k in ("ok", "issues", "warnings", "redactions")},
                "messages": [AIMessage(content=answer)]}
    return {"validation_attempts": attempts, "validation_feedback": "\n".join(f"- {i}" for i in result["issues"]),
            "validation": {k: result[k] for k in ("ok", "issues", "warnings", "redactions")}}


def route_after_validation(state: AgentState) -> str:
    return "update_memory" if state.get("answer") else "response_agent"


def _citation_details(state: AgentState, ids: list[str]) -> list[dict[str, Any]]:
    by_id = {e["chunk_id"]: e for e in state.get("evidence", [])}
    out = []
    for cid in ids:
        if cid.startswith("tool:"):
            out.append({"id": cid, "doc_id": cid, "title": f"Tool result: {cid[5:]}", "section": None})
            continue
        e = by_id[cid]
        out.append({"id": cid, "doc_id": e["doc_id"], "title": e["title"], "section": e["section"],
                    "created_date": e["created_date"], "access_level": e.get("access_level"),
                    "snippet": e["text"].split("\n", 1)[-1][:300]})
    return out
