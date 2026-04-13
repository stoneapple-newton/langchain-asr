"""
Context store update nodes — validate extracted terms and persist them.

Two public nodes are exported:

``validate_terms_node``
    Receives the accumulated ``extracted_terms`` list (merged from all
    extract_worker results via operator.add), deduplicates, normalises the
    schema to match the store, and writes the result to ``validated_terms``.

``update_store_node``
    Takes ``validated_terms`` and upserts each one into the ``TermContextStore``.
    Existing terms have their ``frequency`` counter incremented; new terms are
    inserted. Records ``terms_added`` and ``terms_updated`` in state.

``annotate_transcript_node``
    Adds a ``medical_terms`` array to each segment in ``raw_transcript`` where
    a matched term was found. Writes the annotated dict to ``corrected_transcript``.

After ``update_store_node`` runs, the next session's ``retrieve_context_node``
will find the new terms and include them in the Whisper initial_prompt —
completing the feedback loop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.transcription_correction_agent.tools.medical_terms import find_matches, load_glossary

from ..context_store.store import TermContextStore
from ..state import MedicalASRContextState
from .extraction import deduplicate_extracted_terms

# ---------------------------------------------------------------------------
# Schema normalisation
# ---------------------------------------------------------------------------

_CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


def _normalise_to_store_schema(term: dict[str, Any], session_id: str) -> dict[str, Any]:
    """
    Convert an ``ExtractedMedicalTerm`` / ``ExtractedMedicine`` dict (as
    produced by the LLM extraction chain) to the store's upsert schema.
    """
    canonical = (
        term.get("canonical")
        or term.get("canonical_en")
        or term.get("raw_text")
        or ""
    ).strip()

    return {
        "canonical_en": canonical,
        "specialty": term.get("specialty", "general"),
        "category": term.get("category", "other"),
        "icd_code": term.get("icd_hint", "") or term.get("icd_code", ""),
        "phonetic_variants": "",       # not derived at extraction time
        "language_coverage": "",       # will be enriched on next glossary link
        "frequency": 1,
        "last_session": session_id,
        "source": "extracted",
    }


def _term_is_already_in_glossary(canonical: str) -> bool:
    """
    Quick check: does the existing static glossary already know this term?
    If so, we still upsert (to increment frequency) but we mark it as
    source='glossary' to avoid inflating the extracted count.
    """
    try:
        glossary = load_glossary()
        matches = find_matches(canonical, glossary)
        return len(matches) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Node: validate_terms
# ---------------------------------------------------------------------------

def validate_terms_node(state: MedicalASRContextState) -> dict[str, Any]:
    """
    Deduplicate and normalise extracted terms ready for store upsert.

    Filters out:
    - Terms with empty canonical forms
    - Duplicates (keeps highest-confidence version)
    - Very short strings (<= 2 characters) that are likely abbreviation noise

    Writes ``validated_terms`` and a log entry.
    """
    raw_extracted: list[dict] = state.get("extracted_terms") or []

    # Deduplicate and filter by confidence (keep "low" and above — let the
    # store decide whether to surface them)
    deduped = deduplicate_extracted_terms(raw_extracted, min_confidence="low")

    validated: list[dict] = []
    for term in deduped:
        canonical = (
            term.get("canonical") or term.get("canonical_en") or ""
        ).strip()
        if not canonical or len(canonical) <= 2:
            continue
        validated.append(term)

    return {
        "validated_terms": validated,
        "stage_log": [
            f"Validated {len(validated)} unique term(s) "
            f"from {len(raw_extracted)} extracted (before dedup)"
        ],
    }


# ---------------------------------------------------------------------------
# Node: update_store
# ---------------------------------------------------------------------------

def update_store_node(state: MedicalASRContextState) -> dict[str, Any]:
    """
    Upsert all validated terms into the ``TermContextStore``.

    New terms are inserted; existing terms have their frequency incremented.
    The store is loaded from the default persist dir (configurable via the
    MEDICAL_CONTEXT_STORE_DIR env var — see graph/pipeline.py).

    Returns ``terms_added`` and ``terms_updated`` for the summary log.
    """
    validated: list[dict] = state.get("validated_terms") or []
    session_id: str = state.get("session_id", "")

    if not validated:
        return {
            "terms_added": 0,
            "terms_updated": 0,
            "stage_log": ["No validated terms to store"],
        }

    store = TermContextStore()
    added = updated = 0

    for term in validated:
        normalised = _normalise_to_store_schema(term, session_id)
        result = store.upsert_term(normalised, session_id=session_id)
        if result == "added":
            added += 1
        elif result == "updated":
            updated += 1

    return {
        "terms_added": added,
        "terms_updated": updated,
        "stage_log": [
            f"Context store: +{added} new term(s), {updated} updated — "
            f"total in store: {store.count()}"
        ],
    }


# ---------------------------------------------------------------------------
# Node: annotate_transcript
# ---------------------------------------------------------------------------

def annotate_transcript_node(state: MedicalASRContextState) -> dict[str, Any]:
    """
    Annotate the raw transcript with detected medical terms.

    For each segment, attaches a ``medical_terms`` list containing the
    canonical names of any validated terms found in that segment's text.
    Also promotes the corrected canonical form where an ASR error was detected
    (``raw_text != canonical``).

    Writes the annotated dict to ``corrected_transcript``.
    """
    raw: dict[str, Any] = state.get("raw_transcript") or {}
    validated: list[dict] = state.get("validated_terms") or []

    if not raw:
        return {
            "corrected_transcript": {},
            "stage_log": ["No transcript to annotate"],
        }

    # Build a lookup: canonical_en → term for quick scanning
    term_lookup: dict[str, dict] = {}
    for t in validated:
        canonical = (t.get("canonical") or t.get("canonical_en") or "").strip().lower()
        raw_text = (t.get("raw_text") or "").strip().lower()
        if canonical:
            term_lookup[canonical] = t
        if raw_text and raw_text != canonical:
            term_lookup[raw_text] = t

    import copy
    annotated = copy.deepcopy(raw)
    segments = annotated.get("segments", [])
    total_annotations = 0

    for seg in segments:
        text_lower = (seg.get("text") or "").lower()
        matched_terms: list[str] = []

        for surface_form, term in term_lookup.items():
            if surface_form and surface_form in text_lower:
                canonical = (
                    term.get("canonical") or term.get("canonical_en") or surface_form
                )
                if canonical not in matched_terms:
                    matched_terms.append(canonical)

        if matched_terms:
            seg["medical_terms"] = matched_terms
            total_annotations += len(matched_terms)

    # Attach session-level summary
    annotated["medical_context"] = {
        "session_id": state.get("session_id", ""),
        "language": state.get("language", "en"),
        "specialty": state.get("specialty", "general"),
        "terms_extracted": len(validated),
        "terms_added_to_store": state.get("terms_added", 0),
        "terms_updated_in_store": state.get("terms_updated", 0),
        "whisper_prompt_used": state.get("whisper_prompt", ""),
    }

    return {
        "corrected_transcript": annotated,
        "stage_log": [f"Annotated transcript: {total_annotations} term mention(s) across {len(segments)} segment(s)"],
    }
