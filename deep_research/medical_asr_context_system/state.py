"""
LangGraph state schema for the Medical ASR Context System.

The pipeline flows:
  retrieve_context → transcribe → fan_out_segments → extract_worker(*)
  → validate_terms → update_store → annotate_transcript → save_output

The context store is queried *before* Whisper so that known medical terms
can be injected into Whisper's ``initial_prompt`` parameter, biasing the
decoder toward the correct vocabulary for the current specialty/language.

After transcription, GPT-4.1 extracts newly encountered terms. These are
validated and written back into the Chroma store, enriching future sessions.
"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict


class MedicalASRContextState(TypedDict):
    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    audio_path: str              # Path to the audio file to transcribe
    transcript_path: str         # Output path for the WhisperX JSON (or existing)
    language: str                # BCP-47 code: "en", "es", "fr", "de", "pt", "zh", "ja", "ar", "hi"
    specialty: str               # Medical specialty: "cardiology", "oncology", "general", …
    session_id: str              # Unique session identifier for provenance tracking

    # ------------------------------------------------------------------
    # Context retrieval (step 1 — happens before Whisper runs)
    # ------------------------------------------------------------------
    retrieved_terms: list[dict]  # Term dicts from the Chroma context store
    whisper_prompt: str          # Formatted string injected as Whisper initial_prompt

    # ------------------------------------------------------------------
    # Raw transcription output (step 2)
    # ------------------------------------------------------------------
    raw_transcript: dict         # WhisperX-format JSON dict produced by Whisper

    # ------------------------------------------------------------------
    # Term extraction (step 3 — parallel fan-out across transcript chunks)
    # operator.add accumulates results from all extract_worker invocations
    # ------------------------------------------------------------------
    extracted_terms: Annotated[list[dict], operator.add]

    # ------------------------------------------------------------------
    # Validation (step 4)
    # ------------------------------------------------------------------
    validated_terms: list[dict]  # Deduplicated, standardised, confidence-filtered

    # ------------------------------------------------------------------
    # Context store update (step 5)
    # ------------------------------------------------------------------
    terms_added: int             # New terms inserted into Chroma
    terms_updated: int           # Existing terms whose frequency was incremented

    # ------------------------------------------------------------------
    # Output (step 6)
    # ------------------------------------------------------------------
    corrected_transcript: dict   # Transcript dict annotated with medical term metadata
    output_paths: list[str]      # Paths of files written by save_output
    stage_log: Annotated[list[str], operator.add]  # Accumulated log lines


class ExtractWorkerState(TypedDict):
    """State passed to each parallel extract_worker via Send()."""
    chunk_text: str              # Rendered text of the segment chunk
    chunk_index: int             # Position in the chunk list (for logging)
    language: str
    specialty: str
    session_id: str
    glossary_hint: str           # Pre-built LLM hint string from build_llm_hint()
