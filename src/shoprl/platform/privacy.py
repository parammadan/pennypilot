"""PennyData privacy stage — PII scrubbing at the ingestion boundary.

The store's data is synthetic today, so this stage should find NOTHING —
which is exactly why it runs on every customer-side text anyway: the
pipeline must be safe for real traffic BEFORE real traffic exists, and a
nonzero redaction count on synthetic data is itself a data-quality alarm.

Scope (deterministic, pattern-based): emails, phone numbers, credit-card-
like digit runs (Luhn-checked), SSN-shaped ids, street addresses. Names/
free-text PII would need an NER model — documented as the production
evolution, not faked here.

Redactions REPLACE in place with typed tags (e.g. <EMAIL>) and are COUNTED;
counts surface in ingest stats, dataset manifests, and data-quality checks.
"""
from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")),
    ("SSN", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("ADDRESS", re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:St|Ave|Rd|Blvd|Lane|Ln|Drive|Dr)\b")),
]
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alt = not alt
    return total % 10 == 0


def scrub(text: str) -> tuple[str, dict[str, int]]:
    """Redact PII in `text`; returns (clean_text, counts_by_type)."""
    counts: dict[str, int] = {}
    if not text:
        return text, counts
    for name, pat in _PATTERNS:
        text, n = pat.subn(f"<{name}>", text)
        if n:
            counts[name] = counts.get(name, 0) + n
    def card_sub(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            counts["CARD"] = counts.get("CARD", 0) + 1
            return "<CARD>"
        return m.group(0)
    text = _CARD.sub(card_sub, text)
    return text, counts
