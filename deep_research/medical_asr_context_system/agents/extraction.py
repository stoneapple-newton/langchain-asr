"""
GPT-4.1 medical term extraction — LangGraph nodes.

Two public nodes are exported:

``fan_out_segments_node``
    Reads ``raw_transcript`` from state, chunks the segments, and returns a
    list of ``Send("extract_worker", ...)`` objects so LangGraph fans out
    extraction across all chunks in parallel.

``extract_worker_node``
    Processes a single chunk. Uses GPT-4.1 with ``with_structured_output``
    to produce a ``SegmentExtractionResult`` (existing Pydantic schema from
    ``medical_term_extraction/shared/extraction_models.py``). The glossary
    hint built from the context store is injected into the system prompt so
    the model can recognise known terms even when mangled by ASR.

    Returns ``{"extracted_terms": [list of term dicts]}`` which LangGraph
    accumulates across all workers via ``operator.add`` (see state.py).

Design
------
- Reuses ``SegmentExtractionResult``, ``ExtractedMedicalTerm``, ``ExtractedMedicine``
  from the existing extraction models — no new Pydantic schemas.
- Reuses the chunking helper from stage_01_extraction_basics.
- The LLM profile is ``"gpt41"``; falls back to ``"asr_v2"`` (Ollama) if the
  profile is not configured, so the pipeline runs locally too.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.prompts import ChatPromptTemplate
from langgraph.types import Send

from config import create_chat_model
from deep_research.medical_term_extraction.shared.extraction_models import (
    ExtractedMedicalTerm,
    ExtractedMedicine,
    SegmentExtractionResult,
)

from ..state import ExtractWorkerState, MedicalASRContextState

# ---------------------------------------------------------------------------
# LLM factory (lazy so import works without API keys set)
# ---------------------------------------------------------------------------

_llm_cache: Any = None


def _get_llm() -> Any:
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache
    try:
        _llm_cache = create_chat_model("gpt41", temperature=0, max_tokens=2048)
    except Exception:
        # Fallback to local model when OpenAI is not configured
        _llm_cache = create_chat_model("asr_v2", temperature=0, max_tokens=1024)
    return _llm_cache


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a clinical language expert specialising in multilingual medical \
transcription analysis. Language of this session: {language}. \
Medical specialty: {specialty}.

Given a chunk of an ASR transcript, extract two types of items:

1. **Medical terms** — conditions, symptoms, clinical findings, procedures,
   anatomical references, measurements, diagnoses.
2. **Medicine names** — drug names, medications, supplements, dosages.

ASR context (important):
- The transcript was produced by automatic speech recognition and may contain
  spelling errors or split compound words, e.g. "a trial fibrillation" instead
  of "atrial fibrillation", "lissanopril" instead of "lisinopril".
- Correct obvious ASR errors in the ``canonical`` field; preserve the original
  spoken text in ``raw_text``.
- Use "low" confidence when the term appears garbled or ambiguous.
- The text may be in {language} or mixed with other languages; extract terms
  regardless of the surface language.

Known terms reference (from the session context store):
{glossary_hint}

Return ONLY valid JSON matching the schema below, with no extra prose.
{format_instructions}"""

# Build the prompt template once at module level
from langchain_core.output_parsers import JsonOutputParser as _JsonOutputParser

_parser = _JsonOutputParser(pydantic_object=SegmentExtractionResult)

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "Transcript chunk:\n\n{chunk_text}\n\nExtract all medical terms and medicine names."),
    ]
).partial(format_instructions=_parser.get_format_instructions())


# ---------------------------------------------------------------------------
# Chunk helpers (mirrors stage_01_extraction_basics/_chunk_segments)
# ---------------------------------------------------------------------------

def _segment_text(seg: dict[str, Any]) -> str:
    speaker = seg.get("speaker") or "UNKNOWN"
    start = seg.get("start", 0.0)
    end = seg.get("end", 0.0)
    text = seg.get("text", "")
    return f"[{start:07.2f}-{end:07.2f}] {speaker}: {text}"


def _chunk_raw_transcript(
    raw_transcript: dict[str, Any],
    max_chars: int = 800,
) -> list[tuple[int, str]]:
    """
    Split raw WhisperX transcript segments into text chunks.

    Returns a list of ``(chunk_index, chunk_text)`` pairs.
    """
    segments: list[dict] = raw_transcript.get("segments", [])
    chunks: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_chars = 0
    chunk_idx = 0

    for seg in segments:
        line = _segment_text(seg)
        size = len(line) + 1
        if current_lines and current_chars + size > max_chars:
            chunks.append((chunk_idx, "\n".join(current_lines)))
            chunk_idx += 1
            current_lines = []
            current_chars = 0
        current_lines.append(line)
        current_chars += size

    if current_lines:
        chunks.append((chunk_idx, "\n".join(current_lines)))

    return chunks


# ---------------------------------------------------------------------------
# Node: fan_out_segments
# ---------------------------------------------------------------------------

def fan_out_segments_node(state: MedicalASRContextState) -> list[Send]:
    """
    Routing function for ``add_conditional_edges("transcribe", ...)``.

    Reads ``raw_transcript`` from state, chunks the segments, and returns
    one ``Send("extract_worker", ...)`` per chunk so LangGraph dispatches
    them all in parallel.

    If the transcript has no segments, routes directly to ``validate_terms``
    so the rest of the pipeline continues cleanly with an empty term list.

    Note: this function is used as a *routing function*, not as a registered
    node. LangGraph calls it after ``transcribe`` completes.
    """
    raw = state.get("raw_transcript") or {}
    chunks = _chunk_raw_transcript(raw)

    if not chunks:
        # No segments — skip straight to validate_terms
        return [Send("validate_terms", dict(state))]

    # Build the LLM hint from the whisper_prompt already in state.
    # The whisper_prompt was built by retrieve_context_node for Whisper, but
    # it also serves as a compact term reference for the GPT-4.1 system prompt.
    glossary_hint = state.get("whisper_prompt", "")

    worker_base: dict[str, Any] = {
        "language": state.get("language", "en"),
        "specialty": state.get("specialty", "general"),
        "session_id": state.get("session_id", ""),
        "glossary_hint": glossary_hint,
    }

    return [
        Send("extract_worker", {
            **worker_base,
            "chunk_index": chunk_index,
            "chunk_text": chunk_text,
        })
        for chunk_index, chunk_text in chunks
    ]


# ---------------------------------------------------------------------------
# Node: extract_worker (one per chunk, runs in parallel)
# ---------------------------------------------------------------------------

def extract_worker_node(state: ExtractWorkerState) -> dict[str, Any]:
    """
    Extract medical terms from a single transcript chunk using GPT-4.1.

    Returns ``{"extracted_terms": [...], "stage_log": [...]}`` — LangGraph
    accumulates both lists across all workers via ``operator.add``.
    """
    chunk_text = state["chunk_text"]
    chunk_index = state.get("chunk_index", 0)
    language = state.get("language", "en")
    specialty = state.get("specialty", "general")
    glossary_hint = state.get("glossary_hint", "")

    llm = _get_llm()
    chain = _prompt | llm | _parser

    try:
        raw = chain.invoke({
            "chunk_text": chunk_text,
            "language": language,
            "specialty": specialty,
            "glossary_hint": glossary_hint or "(none — use your medical knowledge)",
        })
        result = SegmentExtractionResult(**raw) if isinstance(raw, dict) else raw

        terms: list[dict] = [t.model_dump() for t in result.medical_terms]
        medicines: list[dict] = [
            {**m.model_dump(), "category": "medication"} for m in result.medicine_names
        ]
        all_extracted = terms + medicines

        log_msg = (
            f"Chunk {chunk_index}: extracted "
            f"{len(terms)} term(s) + {len(medicines)} medicine(s)"
        )
        return {"extracted_terms": all_extracted, "stage_log": [log_msg]}

    except Exception as exc:
        return {
            "extracted_terms": [],
            "stage_log": [f"Chunk {chunk_index}: extraction failed — {exc}"],
        }


# ---------------------------------------------------------------------------
# Deduplication helper (used by validate_terms node in agents/update.py)
# ---------------------------------------------------------------------------

def deduplicate_extracted_terms(
    extracted: list[dict[str, Any]],
    min_confidence: str = "low",
) -> list[dict[str, Any]]:
    """
    Deduplicate and filter a flat list of extracted term dicts.

    Parameters
    ----------
    extracted       : combined output from all extract_worker invocations
    min_confidence  : "high" | "medium" | "low" — discard terms below this level

    Returns
    -------
    list[dict] — unique, canonical-keyed terms sorted by confidence
    """
    _conf_rank = {"high": 2, "medium": 1, "low": 0}
    min_rank = _conf_rank.get(min_confidence, 0)

    seen: dict[str, dict] = {}
    for term in extracted:
        key = (term.get("canonical") or term.get("canonical_en") or "").lower().strip()
        if not key:
            continue
        conf = term.get("confidence", "low")
        rank = _conf_rank.get(conf, 0)
        if rank < min_rank:
            continue
        # Keep the version with the highest confidence
        if key not in seen or _conf_rank.get(seen[key].get("confidence", "low"), 0) < rank:
            seen[key] = term

    return sorted(seen.values(), key=lambda t: -_conf_rank.get(t.get("confidence", "low"), 0))
