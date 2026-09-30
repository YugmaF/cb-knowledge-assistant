"""Sentence-level hold-back for the response stream.

Tokens used to go straight to the client, so a contact detail, an unknown URL or a leaked prompt
marker was on the wire before the validator could redact it. The gate holds tokens back until a
sentence is complete, runs the same output redactions the validator applies to the final answer on
that sentence, and only then releases it. Holding back whole sentences also means an email split
across tokens ("secu" + "rity@comm" + "bank.example") is seen whole before anything is released.

What it does not do: check citations or grounding (they need the whole answer, so the validator
still replaces the draft at the end), or catch a phone number that straddles a forced split in a
very long sentence (released at a word boundary after `max_hold` characters).
"""

from __future__ import annotations

import re

from kb_assistant.security.guards import check_output

_BOUNDARY = re.compile(r"(?<=[.!?])\s|\n")


class SentenceGate:
    def __init__(self, *, allowed_urls: set[str], redact_contact_details: bool, max_hold: int = 600) -> None:
        self._allowed_urls = allowed_urls
        self._redact_contact_details = redact_contact_details
        self._max_hold = max_hold
        self._buffer = ""
        self.stopped = False  # set when the system prompt leaks: nothing further is released

    def feed(self, token: str) -> list[str]:
        """Add a token; return the sentences that are now complete and safe to show."""
        if self.stopped:
            return []
        self._buffer += token
        released: list[str] = []
        while not self.stopped and (cut := self._next_cut()) is not None:
            sentence, self._buffer = self._buffer[:cut], self._buffer[cut:]
            if safe := self._clean(sentence):
                released.append(safe)
        return released

    def flush(self) -> list[str]:
        """End of stream: release whatever is left, after the same checks."""
        if self.stopped or not self._buffer:
            return []
        rest, self._buffer = self._buffer, ""
        safe = self._clean(rest)
        return [safe] if safe else []

    def _next_cut(self) -> int | None:
        match = _BOUNDARY.search(self._buffer)
        if match:
            return match.end()
        if len(self._buffer) > self._max_hold:  # a very long sentence: release up to the last word boundary
            space = self._buffer.rfind(" ")
            if space > 0:
                return space + 1
        return None

    def _clean(self, text: str) -> str:
        result = check_output(text, allowed_urls=self._allowed_urls,
                              redact_contact_details=self._redact_contact_details)
        if "system_prompt_leak" in result.issues:
            self.stopped = True  # the validator will replace the answer; release nothing more
            return ""
        return result.text
