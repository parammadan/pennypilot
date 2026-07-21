"""Multilingual conversation layer (v2): detection + constraint extraction.

Deterministic and rule-based on purpose — the language layer feeds the
STRUCTURED dialogue state, so its behaviour must be unit-testable on CPU with
no model in the loop. Multilingual understanding is judged on preserved intent
and constraints, never on literal translation quality; product names, brands,
model numbers, and technical attributes pass through untranslated.
"""
from shoprl.lang.detect import DetectedLanguage, detect_language
from shoprl.lang.extract import ExtractedInfo, english_gloss, extract_info

__all__ = ["DetectedLanguage", "detect_language",
           "ExtractedInfo", "extract_info", "english_gloss"]
