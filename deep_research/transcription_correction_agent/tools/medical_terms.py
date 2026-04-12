"""
Medical Term Linking Tool
=========================
Loads the multilingual medical glossary and provides:

1. load_glossary()     — returns the full glossary as a dict
2. find_matches()      — fuzzy-matches text against all known terms
3. link_terms()        — called after retranscription to annotate corrections
4. build_llm_hint()    — formats glossary entries as an LLM context hint

The glossary (data/medical_glossary.json) covers 30 terms across 8 languages
and includes common ASR phonetic mishearings so the LLM can recognize
mangled versions of medical words.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_GLOSSARY_PATH = Path(__file__).parent.parent / "data" / "medical_glossary.json"

# Cache so we only read JSON once per process
_GLOSSARY_CACHE: dict | None = None


# ---------------------------------------------------------------------------
# Glossary loading
# ---------------------------------------------------------------------------

def load_glossary(path: str | Path | None = None) -> dict:
    """Return the parsed medical glossary dict (cached after first load)."""
    global _GLOSSARY_CACHE
    if _GLOSSARY_CACHE is not None:
        return _GLOSSARY_CACHE

    p = Path(path) if path else _GLOSSARY_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    _GLOSSARY_CACHE = raw
    return raw


def get_terms(glossary: dict) -> list[dict]:
    """Return the list of term dicts from a loaded glossary."""
    return glossary.get("terms", [])


# ---------------------------------------------------------------------------
# Term matching
# ---------------------------------------------------------------------------

def find_matches(
    text: str,
    glossary: dict,
    min_word_length: int = 4,
) -> list[dict]:
    """
    Find all glossary terms whose canonical name, translations, or phonetic
    variants appear in *text*.

    Returns a list of matching term dicts (subset of glossary["terms"]).
    """
    text_lower = text.lower()
    matches: list[dict] = []
    seen: set[str] = set()

    for term in get_terms(glossary):
        term_en = term["en"].lower()
        if term_en in seen:
            continue

        # Build all surface forms to search for
        surface_forms: list[str] = [term_en]
        # Translations
        for lang in ("es", "fr", "de", "pt"):
            val = term.get(lang, "")
            if val:
                surface_forms.append(val.lower())
        # Phonetic variants
        for variant in term.get("phonetic_variants", []):
            surface_forms.append(variant.lower())

        for form in surface_forms:
            if len(form) >= min_word_length and re.search(
                r"\b" + re.escape(form) + r"\b", text_lower
            ):
                matches.append(term)
                seen.add(term_en)
                break

    return matches


def link_terms(
    text: str,
    glossary: dict,
) -> list[dict]:
    """
    Return a list of LinkedMedicalTerm dicts for all terms found in *text*.
    Each dict contains: raw_text, canonical_en, icd_hint, translations,
    phonetic_variants, specialty.
    """
    matched = find_matches(text, glossary)
    linked: list[dict] = []

    for term in matched:
        linked.append({
            "raw_text": _extract_raw_form(text, term),
            "canonical_en": term["en"],
            "icd_hint": term.get("icd_hint") or "",
            "translations": {
                lang: term.get(lang, "")
                for lang in ("es", "fr", "de", "pt", "ar", "zh_pinyin", "ja_romaji")
            },
            "phonetic_variants": term.get("phonetic_variants", []),
            "specialty": term.get("specialty", "general"),
        })

    return linked


def _extract_raw_form(text: str, term: dict) -> str:
    """Find the actual surface form of this term in *text* (best effort)."""
    text_lower = text.lower()
    candidates = [term["en"]] + term.get("phonetic_variants", [])
    for c in candidates:
        m = re.search(re.escape(c.lower()), text_lower)
        if m:
            return text[m.start():m.end()]
    return term["en"]


# ---------------------------------------------------------------------------
# LLM context hint builder
# ---------------------------------------------------------------------------

def build_llm_hint(
    specialty: str,
    glossary: dict,
    max_terms: int = 8,
) -> str:
    """
    Build a compact multilingual reference string for *specialty* to include
    in an LLM prompt, so the model can recognise and correctly transcribe
    medical terms.

    Example output:
      "myocardial infarction (ES: infarto de miocardio, DE: Herzinfarkt,
       ASR variants: MI, my cardiac infarction) | atrial fibrillation ..."
    """
    terms = [t for t in get_terms(glossary) if t.get("specialty") == specialty]
    if not terms:
        # Fall back to all terms
        terms = get_terms(glossary)

    terms = terms[:max_terms]
    snippets: list[str] = []

    for t in terms:
        parts = [t["en"]]
        translations = []
        for lang in ("es", "fr", "de", "pt"):
            val = t.get(lang, "")
            if val and val.lower() != t["en"].lower():
                translations.append(f"{lang.upper()}: {val}")
        if translations:
            parts.append(f"({', '.join(translations)})")
        variants = t.get("phonetic_variants", [])[:3]
        if variants:
            parts.append(f"ASR: {', '.join(variants)}")
        snippets.append(" ".join(parts))

    return " | ".join(snippets)


# ---------------------------------------------------------------------------
# Context-aware medical candidate detection (rule-based pre-filter)
# ---------------------------------------------------------------------------

MEDICAL_KEYWORD_PATTERNS = [
    # Drug suffixes
    r"\b\w+(cillin|mycin|mab|nib|statin|pril|sartan|olol|azole|oxacin|prazole)\b",
    # Procedural suffixes
    r"\b\w+(scopy|gram|graphy|ectomy|plasty|otomy|ostomy)\b",
    # Disease/condition suffixes
    r"\b\w+(itis|emia|osis|pathy|algia|trophy|plasia|oma)\b",
    # Latin prefixes common in medicine
    r"\b(cardio|neuro|hepato|nephro|pulmon|gastro|onco|hemo|hemato|ortho)\w+\b",
    # Common abbreviations
    r"\b(MI|ECG|EKG|ICU|IV|BP|HR|RR|O2|SpO2|CBC|BMP|PT|INR|MRI|CT|PET)\b",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in MEDICAL_KEYWORD_PATTERNS]


def has_medical_keywords(text: str) -> bool:
    """Quick heuristic: does *text* contain patterns that suggest medical content?"""
    return any(p.search(text) for p in _COMPILED_PATTERNS)
