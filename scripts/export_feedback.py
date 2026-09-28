"""Close the feedback loop: turn thumbs-down answers into candidate regression cases.

    uv run python scripts/export_feedback.py > data/eval/feedback_candidates.jsonl

A reviewer fills in `expected_doc_ids` for each candidate and moves it into golden.jsonl, so every
answer a user rejected becomes a test that future changes must pass.
"""

from __future__ import annotations

import asyncio
import json

from kb_assistant.agents.memory import MemoryStore
from kb_assistant.config import get_settings
from kb_assistant.retrieval.embeddings import HashingEmbedder


async def main() -> None:
    settings = get_settings()
    store = MemoryStore(settings.memory_db_path, HashingEmbedder())
    for row in await store.negative_feedback():
        print(json.dumps({"question": row["question"], "rejected_answer": (row["answer"] or "")[:500],
                          "user_comment": row["comment"], "expected_doc_ids": None, "role": None}))


if __name__ == "__main__":
    asyncio.run(main())
