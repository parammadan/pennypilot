"""Language detection for English / Spanish / code-switched (Spanglish) turns.

Marker-word counting over two small function-word lexicons. Function words are
the right evidence: content words (product names, brands, specs) are shared
across languages and must not influence detection — "SPF 50" is not English or
Spanish. A turn with markers from both lexicons is code-switched.
"""
from __future__ import annotations

import re
from pydantic import BaseModel

_ES = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "que",
    "y", "o", "con", "sin", "para", "por", "es", "son", "está", "esta", "necesito",
    "quiero", "tengo", "voy", "vamos", "somos", "pero", "también", "más", "menos",
    "mi", "mis", "tu", "su", "no", "sí", "ya", "muy", "todo", "nada", "algo",
    "niños", "niñas", "hijos", "hijas", "presupuesto", "dólares", "gracias",
    "hola", "ayuda", "comprar", "busco", "nuevo", "nueva", "cuánto", "cuesta",
    "dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez",
    "quita", "quiero", "necesita", "agregues", "todavía", "playa", "fiesta",
}
_EN = {
    "the", "a", "an", "of", "and", "or", "with", "without", "for", "to", "is",
    "are", "i", "we", "my", "our", "your", "need", "want", "have", "looking",
    "buy", "get", "under", "budget", "please", "help", "new", "but", "also",
    "not", "don't", "dont", "already", "remove", "add", "yet", "that", "this",
    "it", "them", "me", "you", "what", "how", "much", "going", "kids", "children",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
}

_WORD = re.compile(r"[a-záéíóúüñ']+", re.IGNORECASE)


class DetectedLanguage(BaseModel):
    languages: list[str]          # subset of ["english", "spanish"], detection order
    code_switched: bool
    primary: str                  # majority-marker language ("english" on ties)


def detect_language(text: str) -> DetectedLanguage:
    words = [w.lower() for w in _WORD.findall(text or "")]
    es = sum(w in _ES for w in words)
    en = sum(w in _EN for w in words)
    langs: list[str] = []
    if en:
        langs.append("english")
    if es:
        langs.append("spanish")
    if not langs:
        langs = ["english"]  # no markers (bare SKUs/numbers): default english
    return DetectedLanguage(
        languages=langs,
        code_switched=(en > 0 and es > 0),
        primary="spanish" if es > en else "english",
    )
