from __future__ import annotations

import re
from dataclasses import dataclass


PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b(?:\+?\d[\d -]{8,}\d)\b"),
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|system) instructions", re.I),
    re.compile(r"reveal (the )?(system prompt|hidden prompt|chain of thought)", re.I),
    re.compile(r"developer message|system message", re.I),
]

JAILBREAK_PATTERNS = [
    re.compile(r"act as (dan|do anything now)", re.I),
    re.compile(r"bypass safety|disable safety|override safety", re.I),
]

UNSAFE_ENGINEERING_PATTERNS = [
    re.compile(r"bypass (interlock|guard|lockout|tagout)", re.I),
    re.compile(r"run .* without (guard|inspection|lockout|tagout)", re.I),
    re.compile(r"ignore (oem|manufacturer|safety) procedure", re.I),
]


@dataclass
class SafetyDecision:
    allowed: bool
    reasons: list[str]
    sanitized_text: str


class SafetyLayer:
    def inspect_text(self, text: str) -> SafetyDecision:
        reasons: list[str] = []
        sanitized = text
        for pattern in PII_PATTERNS:
            if pattern.search(sanitized):
                reasons.append("Potential PII detected and redacted.")
                sanitized = pattern.sub("[REDACTED]", sanitized)
        for pattern in PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                reasons.append("Prompt-injection attempt detected.")
        for pattern in JAILBREAK_PATTERNS:
            if pattern.search(text):
                reasons.append("Jailbreak attempt detected.")
        for pattern in UNSAFE_ENGINEERING_PATTERNS:
            if pattern.search(text):
                reasons.append("Unsafe engineering instruction detected.")
        blocking = any("Prompt-injection" in r or "Jailbreak" in r for r in reasons)
        return SafetyDecision(allowed=not blocking, reasons=reasons, sanitized_text=sanitized)

    def filter_engineering_advice(self, recommendations: list[str]) -> list[str]:
        safe: list[str] = []
        for rec in recommendations:
            if any(pattern.search(rec) for pattern in UNSAFE_ENGINEERING_PATTERNS):
                safe.append("Escalate to qualified maintenance personnel and follow site/OEM safety procedures.")
            else:
                safe.append(rec)
        return safe

    def disclaimer(self) -> str:
        return (
            "Decision-support only. Not final engineering certification or maintenance authorization. "
            "Follow site procedures, lockout/tagout, qualified inspection requirements, and OEM documentation."
        )


safety_layer = SafetyLayer()

