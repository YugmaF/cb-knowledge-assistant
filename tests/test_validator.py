"""Number grounding in the validator: a number in a cited sentence must appear in the cited source as a
whole number, not as a fragment of another one."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kb_assistant.agents.validator import validate_answer
from kb_assistant.security.rbac import Principal

CHUNK = "INC-2026-020#summary"


def check(draft: str, source: str) -> dict:
    state = {
        "evidence": [{"chunk_id": CHUNK, "doc_id": "INC-2026-020", "title": "t", "section": "Summary", "text": source}],
        "tool_results": [], "decision": {"intent": "knowledge_question"}, "research": None, "draft": draft,
    }
    ctx = SimpleNamespace(principal=Principal.for_role("anil", "Anil", "analyst", "payments"))
    return validate_answer(state, ctx)


def test_a_number_is_not_grounded_by_being_part_of_a_longer_one():
    result = check(f"The outage lasted 4 minutes [{CHUNK}].", "The outage lasted 42 minutes.")
    assert not result["ok"]
    assert result["issues"][0].startswith("ungrounded_numbers")


def test_a_matching_number_is_grounded():
    assert check(f"The outage lasted 42 minutes [{CHUNK}].", "The outage lasted 42 minutes.")["ok"]


@pytest.mark.parametrize("claim,source", [
    ("It affected 1200 customers", "It affected 1,200 customers."),
    ("It affected 1,200 customers", "It affected 1200 customers."),
    ("The check takes 3.5 hours", "The check takes 3.5 hours."),
    ("The job runs at 10:30", "The job runs at 10:30 daily."),
    ("Opened on 2026-09-28", "Opened on 2026-09-28."),
    ("Opened on 9 September", "Opened on 2026-09-09."),       # leading zero: 9 is the same number as 09
    ("Availability was 99.9%", "Availability was 99.9% in March."),
])
def test_equivalent_spellings_of_a_number_are_grounded(claim, source):
    assert check(f"{claim} [{CHUNK}].", source)["ok"]


def test_a_different_decimal_is_not_grounded():
    assert not check(f"The check takes 3 hours [{CHUNK}].", "The check takes 3.5 hours.")["ok"]


def test_list_numbering_is_not_treated_as_a_claim():
    draft = (f"1. The outage lasted 42 minutes [{CHUNK}].\n2. It affected the ledger [{CHUNK}].\n"
             f"- Customers were notified [{CHUNK}].")
    assert check(draft, "The outage lasted 42 minutes and affected the ledger. Customers were notified.")["ok"]
