"""
Whisper prompt builder — converts stored medical terms into a string
that can be passed as Whisper's ``initial_prompt`` (or ``prefix``) parameter.

Why this matters
----------------
Whisper is a fixed model that cannot be fine-tuned at inference time.
However, it *does* accept an ``initial_prompt`` that biases its language
model toward the vocabulary you provide. The decoder treats the prompt as
prior context, so spelling Whisper's vocabulary toward medical terms
significantly improves recognition accuracy for domain-specific content.

Technical constraints
---------------------
- Whisper's ``initial_prompt`` is fed through the same tokeniser as the
  audio. The effective budget is **roughly 224 tokens** (~180 English words).
- Exceeding this silently truncates the prompt; the function therefore caps
  output conservatively at ``MAX_PROMPT_WORDS`` (default 160 words).
- The optimal format is a natural-looking sentence fragment, not a bare list.

Output example
--------------
    Medical consultation in cardiology. Terms: atrial fibrillation (AF,
    a-fib), myocardial infarction (MI, heart attack), echocardiogram (ECHO),
    lisinopril 10 mg, metformin, hypertension (hipertensión). Patient
    history discussion.

Usage
-----
    from deep_research.medical_asr_context_system.context_store.retriever import (
        build_whisper_prompt,
        format_terms_for_llm_hint,
    )

    prompt = build_whisper_prompt(retrieved_terms, language="es", specialty="cardiology")
    whisper_result = model.transcribe(audio_path, initial_prompt=prompt)
"""

from __future__ import annotations

from typing import Any

# Whisper's practical token budget for initial_prompt is ~224 tokens.
# 160 words is a safe upper bound before we risk silent truncation.
MAX_PROMPT_WORDS = 160

# Languages that Whisper recognises by BCP-47 code.
_LANG_LABELS: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic",
    "hi": "Hindi",
}

# Map BCP-47 code → glossary translation field key
_LANG_TO_FIELD: dict[str, str] = {
    "es": "es",
    "fr": "fr",
    "de": "de",
    "pt": "pt",
    "ar": "ar",
    "zh": "zh_pinyin",
    "ja": "ja_romaji",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pick_translation(term: dict[str, Any], language: str) -> str:
    """Return the translation for ``language`` if present in the term dict."""
    field = _LANG_TO_FIELD.get(language, "")
    return term.get(field, "") or ""


def _term_snippet(term: dict[str, Any], language: str, max_variants: int = 2) -> str:
    """
    Format one term as: ``canonical (VARIANT1, VARIANT2, translation)``

    Keeps the snippet short so we can pack more terms within the token budget.
    """
    canonical = term.get("canonical_en", "").strip()
    if not canonical:
        return ""

    extras: list[str] = []

    # Include up to max_variants phonetic variants (ASR mishearings)
    raw_variants = term.get("phonetic_variants", "")
    if isinstance(raw_variants, str):
        variants = [v.strip() for v in raw_variants.split("|") if v.strip()]
    else:
        variants = list(raw_variants)
    extras.extend(variants[:max_variants])

    # Include native-language translation if the session is not English
    if language != "en":
        translation = _pick_translation(term, language)
        if translation and translation.lower() != canonical.lower():
            extras.append(translation)

    if extras:
        return f"{canonical} ({', '.join(extras)})"
    return canonical


def _word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_whisper_prompt(
    retrieved_terms: list[dict[str, Any]],
    *,
    language: str = "en",
    specialty: str = "general",
    max_words: int = MAX_PROMPT_WORDS,
) -> str:
    """
    Build a Whisper ``initial_prompt`` string from retrieved context-store terms.

    The output is a natural sentence so Whisper's language model sees it as
    prior text rather than a raw word list.

    Parameters
    ----------
    retrieved_terms : list of term metadata dicts from ``TermContextStore.query()``
    language        : BCP-47 session language — used to pick translations
    specialty       : human-readable label used in the opening sentence
    max_words       : safety cap; default 160 ≈ ~200 Whisper tokens

    Returns
    -------
    str : ready-to-use initial_prompt (empty string if no terms provided)
    """
    if not retrieved_terms:
        return ""

    lang_label = _LANG_LABELS.get(language, language.upper())
    header = f"Medical consultation in {specialty}."
    term_prefix = "Terms:"

    # Sort: specialty-matched terms first, then by frequency descending
    sorted_terms = sorted(
        retrieved_terms,
        key=lambda t: (
            0 if t.get("specialty") == specialty else 1,
            -int(t.get("frequency", 1)),
        ),
    )

    snippets: list[str] = []
    current_words = _word_count(header) + _word_count(term_prefix)

    for term in sorted_terms:
        snippet = _term_snippet(term, language)
        if not snippet:
            continue
        candidate_words = _word_count(snippet) + 2  # +2 for separator overhead
        if current_words + candidate_words > max_words:
            break
        snippets.append(snippet)
        current_words += candidate_words

    if not snippets:
        return header

    # Trailing context hint helps Whisper's language model stay in domain
    footer = f"Patient history in {lang_label}."

    prompt = f"{header} {term_prefix} {', '.join(snippets)}. {footer}"
    return prompt


def format_terms_for_llm_hint(
    retrieved_terms: list[dict[str, Any]],
    specialty: str,
    max_terms: int = 12,
) -> str:
    """
    Format retrieved terms as a compact reference string for injection into
    a GPT-4.1 extraction prompt.

    Uses the same pipe-separated format as ``build_llm_hint()`` in
    ``transcription_correction_agent/tools/medical_terms.py`` so the two
    code paths stay consistent.

    Example output::

        atrial fibrillation (ES: fibrilación auricular, DE: Vorhofflimmern,
        ASR: a fib, afib) | myocardial infarction (ES: infarto de miocardio,
        ASR: MI, heart attack) | ...
    """
    # Filter to specialty first; fall back to all terms
    relevant = [t for t in retrieved_terms if t.get("specialty") == specialty]
    if not relevant:
        relevant = list(retrieved_terms)
    relevant = relevant[:max_terms]

    snippets: list[str] = []
    for term in relevant:
        canonical = term.get("canonical_en", "").strip()
        if not canonical:
            continue

        parts = [canonical]
        translations = []
        for lang, field in _LANG_TO_FIELD.items():
            if lang in ("zh", "ja", "ar"):  # skip romanised forms in hint
                continue
            val = term.get(field, "")
            if val and val.lower() != canonical.lower():
                translations.append(f"{lang.upper()}: {val}")
        if translations:
            parts.append(f"({', '.join(translations)})")

        raw_variants = term.get("phonetic_variants", "")
        if isinstance(raw_variants, str):
            variants = [v.strip() for v in raw_variants.split("|") if v.strip()][:3]
        else:
            variants = list(raw_variants)[:3]
        if variants:
            parts.append(f"ASR: {', '.join(variants)}")

        snippets.append(" ".join(parts))

    return " | ".join(snippets)
