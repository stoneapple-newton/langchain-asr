"""
State schema for the Transcription Correction Agent.

All nodes read from and write partial updates to this TypedDict.
Fields with Annotated[list, operator.add] accumulate across Send fan-out workers.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Sub-schemas (stored as plain dicts inside state lists)
# ---------------------------------------------------------------------------

class ErrorCandidate(TypedDict):
    """A segment flagged as a possible transcription error."""
    segment_id: str
    start: float
    end: float
    text: str
    speaker: str | None
    issue_type: str   # "low_confidence" | "incoherence" | "repetition" | "garbled"
    reason: str       # Short explanation for the flag
    context: str      # Surrounding text to give the LLM context


class MedicalCandidate(TypedDict):
    """A segment that may contain a medical term."""
    segment_id: str
    start: float
    end: float
    text: str
    speaker: str | None
    suspected_term: str   # What the LLM thinks the term might be
    specialty: str        # "cardiology" | "oncology" | "neurology" | "general" | …
    context: str


class LinkedMedicalTerm(TypedDict):
    """A resolved medical term with multilingual annotations."""
    raw_text: str          # How the ASR/LLM heard it
    canonical_en: str      # Canonical English term
    icd_hint: str          # ICD-10 code hint if known (e.g. "I21" for STEMI)
    translations: dict[str, str]   # {"es": "infarto", "fr": "infarctus", ...}
    phonetic_variants: list[str]   # Common ASR mishearings
    specialty: str


class CorrectionRecord(TypedDict):
    """One completed correction applied to a segment."""
    segment_id: str
    original_text: str
    corrected_text: str
    method: str          # "faster_whisper" | "multimodal_llm" | "llm_text" | "unchanged"
    correction_type: str # "error" | "medical_term"
    confidence: float    # 0.0–1.0
    medical_terms: list[str]
    linked_terms: list[LinkedMedicalTerm]


# ---------------------------------------------------------------------------
# Main agent state
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    transcript_path: str       # Path to WhisperX-like JSON transcript
    audio_path: str            # Path to the source audio file
    # "faster_whisper" | "multimodal_llm" | "auto" (tries fw first, falls back to llm)
    retranscription_method: str

    # ── Loaded data ────────────────────────────────────────────────────────
    segments: list[dict]       # Working copy of transcript segments (dicts)
    medical_glossary: dict     # Loaded from data/medical_glossary.json

    # ── Detection results ──────────────────────────────────────────────────
    error_candidates: list[dict]    # List[ErrorCandidate]
    medical_candidates: list[dict]  # List[MedicalCandidate]

    # ── Corrections (accumulated via Send fan-out) ─────────────────────────
    corrections: Annotated[list[dict], operator.add]  # List[CorrectionRecord]

    # ── Final output ───────────────────────────────────────────────────────
    corrected_segments: list[dict]
    stage_log: Annotated[list[str], operator.add]
    output_paths: list[str]


# ---------------------------------------------------------------------------
# Worker sub-states (used with Send API)
# ---------------------------------------------------------------------------

class ErrorWorkerState(TypedDict):
    """Passed to each error-correction worker via Send."""
    candidate: dict          # ErrorCandidate
    audio_path: str
    retranscription_method: str
    segments: list[dict]     # Full segments list (read-only, for context)
    medical_glossary: dict


class MedicalWorkerState(TypedDict):
    """Passed to each medical-term worker via Send."""
    candidate: dict          # MedicalCandidate
    audio_path: str
    retranscription_method: str
    segments: list[dict]
    medical_glossary: dict
