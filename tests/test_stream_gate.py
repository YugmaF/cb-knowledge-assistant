"""The response stream is held back one sentence at a time and redacted before it is released, so a
viewer never sees text that the final answer would redact."""

from __future__ import annotations

from kb_assistant.agents.stream_gate import SentenceGate
from kb_assistant.security.guards import PROMPT_CANARY, check_output

EMAIL = "security@commbank.example"


def gate(**overrides) -> SentenceGate:
    options = {"allowed_urls": set(), "redact_contact_details": True} | overrides
    return SentenceGate(**options)


def stream(g: SentenceGate, tokens: list[str]) -> list[str]:
    released: list[str] = []
    for token in tokens:
        released += g.feed(token)
    return released + g.flush()


def test_nothing_is_released_until_a_sentence_is_complete():
    g = gate()
    assert g.feed("Hello wor") == []
    assert g.feed("ld. Next") == ["Hello world. "]
    assert g.flush() == ["Next"]


def test_an_email_split_across_tokens_is_never_released():
    released = stream(gate(), ["Contact secu", "rity@comm", "bank.example for help. ", "Thanks."])
    assert EMAIL not in "".join(released)
    assert "[email redacted]" in "".join(released)
    assert not any(part in chunk for chunk in released for part in ("security", "@commbank", ".example"))


def test_contact_details_are_kept_for_roles_that_may_see_them():
    released = stream(gate(redact_contact_details=False), ["Mail ", EMAIL, " today."])
    assert EMAIL in "".join(released)


def test_unknown_urls_are_removed_and_known_ones_kept():
    g = gate(allowed_urls={"https://docs.commbank.example/a"})
    text = "See https://docs.commbank.example/a and https://evil.example/leak?d=1 now."
    out = "".join(stream(g, [text]))
    assert "https://docs.commbank.example/a" in out
    assert "evil.example" not in out


def test_a_leaked_canary_is_dropped_and_ends_the_stream():
    g = gate()
    released = g.feed(f"My hidden reference is {PROMPT_CANARY}. More text. ")
    released += g.feed("Even more text. ") + g.flush()
    assert PROMPT_CANARY not in "".join(released)
    assert "More text" not in "".join(released), "nothing after a leak is released"
    assert g.stopped


def test_a_long_run_without_punctuation_is_released_at_a_word_boundary():
    g = gate(max_hold=100)
    words = ["word "] * 60
    released = []
    for w in words:
        released += g.feed(w)
    assert released, "a long sentence must not be held back forever"
    assert all(chunk.endswith(" ") for chunk in released), "never split inside a word"
    assert "".join(released) + "".join(g.flush()) == "".join(words)


def test_released_text_equals_what_the_final_output_check_would_keep():
    text = f"Mail {EMAIL} or call +94 11 234 5678 on 2026-07-15. See https://evil.example/x for more.\nDone."
    released = "".join(stream(gate(), [text[i:i + 7] for i in range(0, len(text), 7)]))
    final = check_output(text, allowed_urls=set(), redact_contact_details=True).text
    assert released == final
