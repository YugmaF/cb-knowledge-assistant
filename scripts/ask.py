"""Run one or more turns through the graph from the terminal, printing the activity stream.

    uv run python scripts/ask.py --user anil "What are the recurring root causes of payment failures in the last year?"
    uv run python scripts/ask.py --user amal --approve "Set branch-teller status to operational, note: fixed"
    uv run python scripts/ask.py --user anil --faults vectordb,mcp "Who is on call for payments-ledger?"

Useful for debugging without the API/UI, and for recording what each agent did.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid

from langgraph.checkpoint.memory import InMemorySaver

from kb_assistant import faults
from kb_assistant.agents.graph import build_graph
from kb_assistant.api.runner import stream_turn
from kb_assistant.config import get_settings
from kb_assistant.container import build_services
from kb_assistant.observability import configure_logging, configure_tracing
from kb_assistant.security.auth import USERS
from kb_assistant.security.rbac import Principal

QUIET = {"token", "node_done"}


def show(event: dict) -> None:
    kind = event.get("type")
    if kind == "token":
        print(event["text"], end="", flush=True)
        return
    if kind in QUIET:
        return
    if kind == "final":
        print("\n\n=== FINAL ===\n" + event["answer"])
        print("\ncitations:", [c["id"] for c in event["citations"]])
        print("validation:", json.dumps(event["validation"]))
        print("explanation:", json.dumps(event["explanation"], indent=1)[:1500])
        print("usage:", event["usage"], f"elapsed={event['elapsed_ms']}ms")
        return
    if kind == "rlm" and event.get("code"):
        print(f"\n[rlm plan: {event.get('source')}] {event.get('rationale', '')}\n{event['code']}")
        return
    compact = {k: v for k, v in event.items() if k != "type"}
    print(f"\n[{kind}] {json.dumps(compact, default=str)[:400]}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--user", default="anil", choices=sorted(USERS))
    parser.add_argument("--approve", action="store_true", help="auto-approve admin actions")
    parser.add_argument("--faults", default="", help="comma-separated faults to inject, e.g. vectordb,mcp")
    args = parser.parse_args()

    if args.faults:
        faults.set_faults(set(args.faults.split(",")))
    settings = get_settings()
    configure_logging(settings)
    configure_tracing(settings)
    services = build_services(settings)
    graph = build_graph(InMemorySaver())
    u = USERS[args.user]
    principal = Principal.for_role(u.user_id, u.name, u.role, u.department)
    thread_id = uuid.uuid4().hex
    await services.memory.claim_thread(thread_id, principal.user_id, args.questions[0])

    for question in args.questions:
        print(f"\n\n######## {principal.name} ({principal.role.value}): {question}")
        async for event in stream_turn(graph, services, principal, thread_id, message=question):
            show(event)
            if event["type"] == "approval_required":
                print(f"\n>>> approval requested: {json.dumps(event.get('calls'))} -> "
                      f"{'APPROVE' if args.approve else 'REJECT'}")
                async for ev in stream_turn(graph, services, principal, thread_id,
                                            resume={"approved": args.approve}):
                    show(ev)


if __name__ == "__main__":
    asyncio.run(main())
