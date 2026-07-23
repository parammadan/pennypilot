"""Bilingual constraint extraction: user words (EN/ES/mixed) → structured facts.

Every constraint must survive regardless of input language — budget, currency,
party size, must-haves, owned items, removals, permission holds. The extractor
returns only what THIS message states; merging across turns is the dialogue
state's job (state/dialogue.py). Product names/brands/SKUs are never translated
or normalized — they pass through as written.
"""
from __future__ import annotations

import re
from pydantic import BaseModel, Field

# Bilingual number words (digits handled separately).
_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
    "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9, "diez": 10,
}
_N = r"(\d+|" + "|".join(_NUM) + r")"


def _num(tok: str) -> int:
    return int(tok) if tok.isdigit() else _NUM[tok.lower()]


# Canonical category ← bilingual surface forms. Small on purpose; grows with
# the catalog. Detection never rewrites the surface form the user used.
_CATEGORIES = {
    "sunscreen": ["sunscreen", "sun screen", "protector solar", "bloqueador"],
    "laptop": ["laptop", "laptops", "portátil", "portatil", "computadora", "notebook"],
    "umbrella": ["umbrella", "umbrellas", "sombrilla", "paraguas"],
    "towel": ["towel", "towels", "toalla", "toallas"],
    "snacks": ["snacks", "snack", "botanas"],
    "cooler": ["cooler", "hielera"],
}
_SURFACE_TO_CAT = {s: c for c, forms in _CATEGORIES.items() for s in forms}
_CAT_RE = re.compile(
    "(" + "|".join(sorted(_SURFACE_TO_CAT, key=len, reverse=True)) + ")",
    re.IGNORECASE)

_BUDGET = [
    re.compile(r"\$\s*(\d+(?:\.\d+)?)"),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:dollars|dólares|dolares|bucks|usd)\b", re.I),
    re.compile(r"(?:budget|presupuesto)\s+(?:is|es|of|de)?\s*\$?\s*(\d+(?:\.\d+)?)", re.I),
    re.compile(r"(?:under|less than|menos de|máximo|maximo)\s+\$?\s*(\d+(?:\.\d+)?)", re.I),
]
_CHILDREN = re.compile(_N + r"\s+(?:niñ[oa]s|kids|children|hij[oa]s)", re.I)
_PEOPLE = re.compile(
    r"(?:somos\s+" + _N + r"|" + _N + r"\s+(?:people|persons|personas|adults|adultos)"
    r"|party of\s+" + _N + r")", re.I)
_SPF = re.compile(r"spf\s*(\d+)", re.I)
_RAM = re.compile(_N + r"\s*(?:gb|gigs?)\s*(?:of\s+)?(?:ram|memory|memoria)?", re.I)
_BATTERY = re.compile(_N + r"\+?\s*(?:hours?|hrs|horas)\s*(?:of\s+)?(?:battery|batería|bateria)?", re.I)
_OWNED = re.compile(
    r"(?:ya\s+ten(?:go|emos)|(?:i|we)\s+already\s+(?:have|got|own))\s+"
    r"(?:an?\s+|the\s+|una?\s+|el\s+|la\s+)?([\w\sáéíóúüñ-]+?)(?=[.,;!?]|$)", re.I)
_FORBIDDEN = re.compile(
    r"(?:no\s+(?:quiero|necesito)|(?:i\s+)?(?:don'?t|do\s+not)\s+(?:want|need))\s+"
    r"(?:an?\s+|the\s+|una?\s+|el\s+|la\s+)?([\w\sáéíóúüñ-]+?)(?=[.,;!?]|$)", re.I)
_REMOVE = re.compile(
    r"(?:remove|quita|quitar|drop|take\s+off)\s+"
    r"(?:the\s+|an?\s+|el\s+|la\s+|una?\s+)?([\w\sáéíóúüñ-]+?)(?=[.,;!?]|$)", re.I)
_BRANDS = ("acer", "dell", "lenovo", "hp", "asus", "apple", "framework",
            "razer")
_BRAND_RE = re.compile(r"\b(" + "|".join(_BRANDS) + r")\b", re.IGNORECASE)
_HOLD = re.compile(
    r"(?:do\s+not|don'?t)\s+add\s+anything|no\s+agregues\s+nada|not?\s+yet\b"
    r"|todavía\s+no|todavia\s+no", re.I)
# Price FLOORS ("at least $2500", "2500 minimum", "mínimo 2500") are OUTSIDE
# the domain model — budget is strictly a maximum and the objective is
# cheapest-valid. Coercing a floor into the budget slot silently inverts the
# user's intent (live-demo finding, docs CHALLENGES #31), so floors are
# detected, EXCLUDED from budget extraction, and surfaced as an unsupported
# note. "minimun" is kept: the misspelling is common enough to have been the
# first observed failure. Unit guards stop "at least 32 GB" (a real, supported
# min_ram/battery floor) from matching.
_UNITS = r"(?:gb|gigs?|hours?|hrs|horas|lbs?|libras?|kg)"
_PRICE_FLOOR = re.compile(
    r"(?:minimum|minimun|m[íi]nimo|at\s+least|más\s+de|mas\s+de|more\s+than"
    r"|over|encima\s+de|starting\s+(?:at|from)|desde)\s+(?:of\s+)?"
    r"\$?\s*(\d{3,}(?:\.\d+)?)\b(?!\s*" + _UNITS + r")"
    r"|\$?\s*(\d{3,}(?:\.\d+)?)\s*(?:dollars|d[óo]lares|usd|bucks)?\s+"
    r"(?:minimum|minimun|m[íi]nimo|or\s+more|o\s+m[áa]s|\+)(?!\s*" + _UNITS + r")",
    re.I)


class ExtractedInfo(BaseModel):
    """Facts stated by ONE user message (unset = not mentioned)."""
    budget_total: float | None = None
    currency: str | None = None
    number_of_children: int | None = None
    number_of_people: int | None = None
    required_categories: list[str] = Field(default_factory=list)
    hard_constraints: dict[str, float | str] = Field(default_factory=dict)
    owned_items: list[str] = Field(default_factory=list)
    forbidden_items: list[str] = Field(default_factory=list)
    removed_items: list[str] = Field(default_factory=list)
    hold_permission: bool = False
    unsupported_notes: list[str] = Field(default_factory=list)


def _canon_item(phrase: str) -> str:
    """Map a noun phrase to its canonical category when known, else keep the
    cleaned surface phrase (never translated)."""
    p = phrase.strip().lower()
    m = _CAT_RE.search(p)
    return _SURFACE_TO_CAT[m.group(1).lower()] if m else p


def extract_info(text: str) -> ExtractedInfo:
    t = text or ""
    out = ExtractedInfo()

    floor_spans = []
    for m in _PRICE_FLOOR.finditer(t):
        floor_spans.append(m.span())
        amount = next(g for g in m.groups() if g)
        out.unsupported_notes.append(
            f"price minimum ${float(amount):g} requested — unsupported: budget "
            "is a MAXIMUM and the store finds the cheapest option that fits")

    for pat in _BUDGET:
        for m in pat.finditer(t):
            # a number inside a floor phrase is NOT a budget ceiling
            if any(a <= m.start(1) < b for a, b in floor_spans):
                continue
            out.budget_total = float(m.group(1))
            out.currency = "USD"
            break
        if out.budget_total is not None:
            break

    m = _CHILDREN.search(t)
    if m:
        out.number_of_children = _num(m.group(1))
    m = _PEOPLE.search(t)
    if m:
        out.number_of_people = _num(next(g for g in m.groups() if g))

    # Negated/owned/removed phrases must not ALSO register as required
    # categories, so collect their spans first and skip categories inside them.
    negated_spans: list[tuple[int, int]] = []
    for pat, dest in ((_OWNED, out.owned_items),
                      (_FORBIDDEN, out.forbidden_items),
                      (_REMOVE, out.removed_items)):
        for m in pat.finditer(t):
            dest.append(_canon_item(m.group(1)))
            negated_spans.append(m.span())

    for m in _CAT_RE.finditer(t):
        if any(a <= m.start() < b for a, b in negated_spans):
            continue
        cat = _SURFACE_TO_CAT[m.group(1).lower()]
        if cat not in out.required_categories:
            out.required_categories.append(cat)

    m = _SPF.search(t)
    if m:
        out.hard_constraints["spf_minimum"] = float(m.group(1))
    m = _BRAND_RE.search(t)
    if m and not any(m.start() >= a and m.start() < b for a, b in negated_spans):
        out.hard_constraints["brand"] = m.group(1).capitalize().replace("Hp", "HP")
    m = _RAM.search(t)
    if m:
        out.hard_constraints["min_ram"] = float(_num(m.group(1)))
    m = _BATTERY.search(t)
    if m:
        out.hard_constraints["min_battery"] = float(_num(m.group(1)))

    out.hold_permission = bool(_HOLD.search(t))
    return out


def english_gloss(info: ExtractedInfo) -> str:
    """Compact English interpretation of one message (translation_mode). Only
    states extracted facts — never a free translation, so it cannot invent."""
    parts: list[str] = []
    if info.number_of_children is not None:
        parts.append(f"{info.number_of_children} children")
    if info.number_of_people is not None:
        parts.append(f"{info.number_of_people} people")
    if info.budget_total is not None:
        parts.append(f"budget ${info.budget_total:g} {info.currency or 'USD'}")
    if info.required_categories:
        parts.append("needs " + ", ".join(info.required_categories))
    for k, v in info.hard_constraints.items():
        parts.append(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}")
    if info.owned_items:
        parts.append("already has " + ", ".join(info.owned_items))
    if info.forbidden_items:
        parts.append("does not want " + ", ".join(info.forbidden_items))
    if info.removed_items:
        parts.append("remove " + ", ".join(info.removed_items))
    if info.hold_permission:
        parts.append("do not add to cart yet")
    for note in info.unsupported_notes:
        parts.append(f"UNSUPPORTED: {note}")
    return "[interpreted: " + "; ".join(parts) + "]" if parts else ""
