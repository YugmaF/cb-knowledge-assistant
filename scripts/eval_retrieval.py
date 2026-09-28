"""Retrieval evals over data/eval/golden.jsonl. No LLM needed, so it can run on every change (CI).

    uv run python scripts/eval_retrieval.py [--k 5] [--alpha 0.6] [--no-rerank]

Reports recall@k per question and overall, and a security check: for questions with no expected
documents (e.g. a viewer asking about a restricted board paper) any hit from a document above the
role's access level is a failure. Compare runs by changing ONE variable at a time (alpha, rerank, k).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

# Hundreds of bare retrieval calls would bury the demo's chat traces; trace evals only on request.
if "--trace" not in sys.argv:
    os.environ["LANGSMITH_TRACING"] = "false"

from kb_assistant.config import PROJECT_ROOT, get_settings
from kb_assistant.container import build_services
from kb_assistant.security.auth import USERS
from kb_assistant.security.rbac import Principal

ROLE_USER = {"viewer": "vera", "analyst": "anil", "admin": "amal"}


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--trace", action="store_true", help="send the eval's retrieval calls to LangSmith")
    args = parser.parse_args()

    settings = get_settings()
    if args.alpha is not None:
        settings.hybrid_alpha = args.alpha
    services = build_services(settings, rerank=not args.no_rerank)
    rows = [json.loads(line) for line in (PROJECT_ROOT / "data" / "eval" / "golden.jsonl").read_text().splitlines() if line]

    recalls: list[float] = []
    by_kind: dict[str, list[float]] = {}
    security_failures = 0
    print(f"alpha={services.retriever.alpha} rerank={services.retriever.reranker.name} k={args.k}\n")
    for row in rows:
        u = USERS[ROLE_USER[row["role"]]]
        principal = Principal.for_role(u.user_id, u.name, u.role, u.department)
        result = await services.retriever.search(row["question"], principal, top_k=args.k)
        got = [h.doc_id for h in result.hits]
        leaked = [h.doc_id for h in result.hits if not principal.can_read(h.metadata.get("access_level", ""))]
        security_failures += bool(leaked)
        expected = set(row["expected_doc_ids"])
        if expected:
            recall = len(expected & set(got)) / len(expected)
            recalls.append(recall)
            by_kind.setdefault(row.get("kind", "standard"), []).append(recall)
            print(f"{recall:5.2f}  {row['role']:<7} {row['question'][:70]}")
        else:
            print(f"  neg  {row['role']:<7} {row['question'][:70]}  -> {'LEAK ' + str(leaked) if leaked else 'ok'}")
    mean = sum(recalls) / len(recalls)
    print(f"\nmean recall@{args.k}: {mean:.3f} over {len(recalls)} questions; access-control leaks: {security_failures}")
    print("by kind: " + ", ".join(f"{k}={sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(by_kind.items())))
    print("note: multi-document questions (e.g. 'recurring root causes') are answered by the research agent, "
          "not by top-k retrieval, so their recall@k is expected to be low.")
    return 1 if security_failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
