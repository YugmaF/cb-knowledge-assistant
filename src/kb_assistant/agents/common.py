"""Helpers shared by the agent nodes."""

from __future__ import annotations

from datetime import date
from typing import Any

from kb_assistant.agents.events import emit
from kb_assistant.retrieval.store import Hit


def today() -> str:
    return date.today().isoformat()


def node_event(node: str, status: str, **detail: Any) -> None:
    emit({"type": "node", "node": node, "status": status, **detail})


def hit_to_evidence(hit: Hit) -> dict[str, Any]:
    md = hit.metadata
    return {
        "chunk_id": hit.chunk_id, "doc_id": hit.doc_id, "title": md.get("title"), "section": md.get("section"),
        "document_type": md.get("document_type"), "department": md.get("department"),
        "access_level": md.get("access_level"), "created_date": md.get("created_date"),
        "text": hit.text, "score": round(hit.rerank_score if hit.rerank_score is not None else hit.score, 4),
        "flags": list(hit.flags),
    }


def merge_evidence(existing: list[dict[str, Any]], new: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    seen = {e["chunk_id"] for e in existing}
    merged = list(existing)
    for item in new:
        if item["chunk_id"] not in seen:
            merged.append(item)
            seen.add(item["chunk_id"])
    return merged[:limit]


def format_evidence(evidence: list[dict[str, Any]], max_chars: int = 1200) -> str:
    if not evidence:
        return "(no documents retrieved)"
    blocks = []
    for e in evidence:
        text = e["text"][:max_chars]
        blocks.append(
            f'<document id="{e["chunk_id"]}" title="{e["title"]}" date="{e["created_date"]}" '
            f'access="{e["access_level"]}">\n{text}\n</document>'
        )
    return "\n".join(blocks)
