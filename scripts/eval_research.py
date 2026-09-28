"""Score the RLM research agent against structured ground truth.

    uv run python scripts/eval_research.py      # needs an LLM key; ~10 LLM calls, ~$0.015

The incident markdown and the MCP incident records (data/enterprise/incidents.json) describe the same
incidents, so the records are an answer key: which incidents are payment-related inside the window,
and each one's root_cause_category. This checks that the agent found them (recall), did not count
others (precision), and classified them correctly.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import date, timedelta

from langgraph.checkpoint.memory import InMemorySaver

from kb_assistant.agents.graph import build_graph
from kb_assistant.api.runner import stream_turn
from kb_assistant.config import PROJECT_ROOT, get_settings
from kb_assistant.container import build_services
from kb_assistant.observability import configure_logging
from kb_assistant.security.rbac import Principal

QUESTION = ("Summarize all outage reports related to payment failures during the last year and identify "
            "recurring root causes.")


async def main() -> None:
    settings = get_settings()
    settings.log_level = "WARNING"
    configure_logging(settings)
    services = build_services(settings)
    graph = build_graph(InMemorySaver())
    principal = Principal.for_role("anil", "Anil Jayasuriya", "analyst", "payments")
    thread = uuid.uuid4().hex
    await services.memory.claim_thread(thread, principal.user_id, QUESTION)
    async for event in stream_turn(graph, services, principal, thread, message=QUESTION):
        if event["type"] == "final":
            usage = event["usage"]
    state = (await graph.aget_state({"configurable": {"thread_id": thread}})).values
    research = state["research"]

    since = (date.today() - timedelta(days=365)).isoformat()
    records = json.loads((PROJECT_ROOT / "data" / "enterprise" / "incidents.json").read_text())
    truth = {r["incident_id"]: r["root_cause_category"] for r in records
             if r["payment_related"] and r["opened_at"][:10] >= since}
    counted = {doc for cat in research["aggregates"]["by_category"].values() for doc in cat["doc_ids"]}
    found = {f["doc_id"]: f["category"] for f in research["findings"] if f["doc_id"] in counted}

    tp = set(found) & set(truth)
    correct = [d for d in tp if found[d] == truth[d]]
    print(f"window since {since}: {len(truth)} payment incidents in the answer key")
    print(f"recall     {len(tp)}/{len(truth)}  missed: {sorted(set(truth) - set(found))}")
    print(f"precision  {len(tp)}/{len(found)}  extra: {sorted(set(found) - set(truth))}")
    print(f"category   {len(correct)}/{len(tp)} correct")
    for d in sorted(tp - set(correct)):
        print(f"   {d}: agent={found[d]}  truth={truth[d]}")
    print(f"sub-agent calls {research['sub_agent_calls']}, documents analysed {research['documents_analysed']}, "
          f"plan source {research['plan']['source']}, usage {usage}")


if __name__ == "__main__":
    asyncio.run(main())
