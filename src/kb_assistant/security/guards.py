"""Deterministic guards for the three untrusted inputs, plus the output.

    user message     -> check_user_input      (instruction override, exfiltration, tool abuse)
    retrieved text   -> sanitize_retrieved     (indirect injection hidden in documents)
    model output     -> check_output           (leaked prompt, unknown URLs, PII, brand rules)

Pattern matching is not a complete defence against prompt injection; nothing is. It is one layer.
The layers that hold even when it misses are structural: the model never holds credentials, tools
re-check permissions in code, retrieval filters are applied server-side, retrieved text is delimited
as data, and output URLs must come from a cited source. See docs/SECURITY.md.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# A random marker placed in the system prompt. If it ever appears in an answer, the prompt leaked.
PROMPT_CANARY = "cb-canary-5e1f9a"

_ZERO_WIDTH = re.compile(r"[​-‏⁠-⁤﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# (category, pattern). Categories map to the three threats the brief names.
_INPUT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("instruction_override", re.compile(
        r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|all|earlier|system)\b"
        r".{0,20}\b(instructions?|rules?|prompts?|guidelines?)", re.I)),
    ("instruction_override", re.compile(
        r"\b(you are now|act as|pretend (to be|you are)|from now on you)\b.{0,60}"
        r"\b(unrestricted|jailbroken|dan|developer mode|no (rules|restrictions|filters))", re.I)),
    ("prompt_exfiltration", re.compile(
        r"\b(reveal|show|print|repeat|output|leak|tell me)\b.{0,40}"
        r"\b(system prompt|hidden (prompt|instructions)|your (instructions|rules|prompt))", re.I)),
    ("data_exfiltration", re.compile(
        r"\b(send|post|upload|forward|exfiltrate|transmit|encode)\b.{0,60}"
        r"(https?://|\bwebhook\b|\bto (this|my|an?) (url|endpoint|server|email)\b)", re.I)),
    ("data_exfiltration", re.compile(r"!\[[^\]]*\]\(\s*https?://", re.I)),
    ("data_exfiltration", re.compile(
        r"\b(list|dump|export|give me)\b.{0,30}\b(all|every|entire)\b.{0,30}"
        r"\b(restricted|confidential|secret|passwords?|credentials?|api keys?)\b", re.I)),
    ("tool_abuse", re.compile(
        r"(\bimport\s+(os|sys|subprocess|socket|shutil)\b|__import__|\beval\s*\(|\bexec\s*\("
        r"|os\.system|subprocess\.|rm\s+-rf|/etc/passwd)", re.I)),
    ("tool_abuse", re.compile(
        r"\b(bypass|skip|disable|circumvent|escalate)\b.{0,40}"
        r"\b(authori[sz]ation|permissions?|rbac|role|guardrails?|approval|rate limit)", re.I)),
    ("tool_abuse", re.compile(
        r"\b(i am|i'm|treat me as|give me)\b.{0,20}\b(an? )?(admin|administrator|root|superuser)\b", re.I)),
]

# Retrieved documents: the same override / exfiltration signals, plus text addressed to the AI.
_RETRIEVED_RULES: list[re.Pattern[str]] = [
    _INPUT_RULES[0][1],
    _INPUT_RULES[1][1],
    _INPUT_RULES[2][1],
    re.compile(r"\b(note|notice|message|instruction)s? (to|for) (the )?(ai|assistant|llm|model|chatbot)s?\b", re.I),
    re.compile(r"\binclude (this|the following) (link|url)\b", re.I),
    re.compile(r"https?://\S*(exfil|collect|steal|leak)\S*", re.I),
]

_URL = re.compile(r"https?://[^\s)\]>\"']+", re.I)
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<![\w-])(\+?\d[\d\s()-]{7,}\d)(?![\w-])")
_DATE_LIKE = re.compile(r"\d{4}-\d{2}-\d{2}(\s*(to|-)\s*\d{4}-\d{2}-\d{2})?")

# Brand rules for an assistant that speaks as the bank. These are phrasings the bank must not use
# in its own voice, whatever the model was asked.
_BRAND_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("financial_advice", re.compile(
        r"\b(you should|i recommend|we recommend)\b.{0,40}\b(invest|buy|sell)\b.{0,30}"
        r"\b(shares?|stocks?|bonds?|crypto)", re.I)),
    ("guarantee", re.compile(r"\b(guarantee[sd]?|risk[- ]free|100% (safe|secure))\b", re.I)),
    ("disparagement", re.compile(
        r"\b(commercial bank|the bank|our bank)\b.{0,40}\b(is|are)\b.{0,15}"
        r"\b(incompetent|terrible|a joke|failing|insolvent|going bankrupt)\b", re.I)),
    ("speculation", re.compile(r"\b(the bank|commercial bank)\b.{0,30}\b(will|might|could) (collapse|fail|go bankrupt)\b", re.I)),
]


def normalize(text: str) -> str:
    """NFKC folds look-alike Unicode (fullwidth letters etc.); zero-width characters are a common
    way to split a keyword so a filter misses it."""
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return _CONTROL.sub("", text)


@dataclass
class GuardResult:
    allowed: bool
    text: str
    categories: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def check_user_input(message: str, max_chars: int) -> GuardResult:
    text = normalize(message).strip()
    if not text:
        return GuardResult(False, text, ["invalid_request"], ["empty message"])
    if len(text) > max_chars:
        return GuardResult(False, text, ["invalid_request"], [f"message longer than {max_chars} characters"])

    categories: list[str] = []
    reasons: list[str] = []
    for category, pattern in _INPUT_RULES:
        match = pattern.search(text)
        if match:
            categories.append(category)
            reasons.append(f"{category}: '{match.group(0)[:60]}'")
    return GuardResult(not categories, text, sorted(set(categories)), reasons)


@dataclass
class SanitizedText:
    text: str
    flagged: bool
    findings: list[str]


def sanitize_retrieved(text: str) -> SanitizedText:
    """Neutralise instructions hidden in a retrieved document. The chunk is kept (the rest of it
    may be legitimate evidence) but the suspicious sentence is replaced, and the chunk is flagged
    so the activity panel and the trace show that an injection was found."""
    clean = normalize(text)
    findings: list[str] = []
    sentences = re.split(r"(?<=[.!?:])\s+", clean)
    kept: list[str] = []
    for sentence in sentences:
        hit = next((p for p in _RETRIEVED_RULES if p.search(sentence)), None)
        if hit:
            findings.append(sentence[:80])
            kept.append("[removed: text addressed to AI systems]")
        else:
            kept.append(sentence)
    return SanitizedText(" ".join(kept), bool(findings), findings)


@dataclass
class OutputCheck:
    text: str
    issues: list[str] = field(default_factory=list)   # must be fixed (retry or block)
    redactions: list[str] = field(default_factory=list)  # fixed in place, reported only


def check_output(answer: str, *, allowed_urls: set[str], redact_contact_details: bool) -> OutputCheck:
    result = OutputCheck(answer)
    if PROMPT_CANARY in answer:
        result.issues.append("system_prompt_leak")
        return result

    for category, pattern in _BRAND_RULES:
        if pattern.search(answer):
            result.issues.append(f"brand:{category}")

    # Markdown images render automatically in many clients, which turns them into a zero-click
    # exfiltration channel. The assistant never needs them.
    if _MD_IMAGE.search(answer):
        result.text = _MD_IMAGE.sub("[image removed]", result.text)
        result.redactions.append("markdown_image")

    def _url(match: re.Match[str]) -> str:
        url = match.group(0).rstrip(".,;")
        if url in allowed_urls:
            return match.group(0)
        result.redactions.append(f"url:{url[:60]}")
        return "[link removed]"

    result.text = _URL.sub(_url, result.text)

    if redact_contact_details:
        if _EMAIL.search(result.text):
            result.text = _EMAIL.sub("[email redacted]", result.text)
            result.redactions.append("email")
        def _phone(match: re.Match[str]) -> str:
            candidate = match.group(0)
            # Dates and ranges of numbers look like phone numbers to a regex; require 9+ digits.
            if _DATE_LIKE.fullmatch(candidate.strip()) or sum(c.isdigit() for c in candidate) < 9:
                return candidate
            result.redactions.append("phone")
            return "[phone redacted]"

        result.text = _PHONE.sub(_phone, result.text)
    return result


REFUSALS: dict[str, str] = {
    "instruction_override": (
        "I can't change how I operate or set aside my guidelines. I'm happy to help with questions "
        "about {brand}'s internal documents, runbooks, incidents and services."
    ),
    "prompt_exfiltration": (
        "I can't share my internal configuration. Ask me about {brand}'s policies, architecture, "
        "runbooks or incidents instead."
    ),
    "data_exfiltration": (
        "I can't send information to external destinations or bulk-export sensitive records. "
        "I can answer specific questions within your access level."
    ),
    "tool_abuse": (
        "I can't run that request. Tool access follows your role and can't be changed from the chat."
    ),
    "invalid_request": "I couldn't process that message: {reason}.",
}


def refusal_for(result: GuardResult, brand: str) -> str:
    category = result.categories[0] if result.categories else "invalid_request"
    template = REFUSALS.get(category, REFUSALS["invalid_request"])
    reason = result.reasons[0] if result.reasons else "unknown"
    return template.format(brand=brand, reason=reason)
