"""
LangGraph workflow for end-to-end ASR medical verification.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import create_chat_model

from .shared.audio_utils import extract_audio_segment
from .shared.detection import identify_error_candidates
from .shared.medical_glossary import build_llm_hint, link_medical_terms, load_medical_glossary
from .shared.retranscription import retranscribe_segment
from .shared.transcript_utils import (
    analyze_transcript,
    apply_corrections,
    build_context_text,
    load_transcript,
    save_document,
    write_correction_sidecar,
)
from .state import AgentState, WorkerState


def _get_llm():
    return create_chat_model("asr_v2", temperature=0, max_tokens=256)


def _candidate_to_dict(candidate: Any) -> dict[str, Any]:
    if hasattr(candidate, "__dict__"):
        return dict(candidate.__dict__)
    return dict(candidate)


def _should_apply(original_text: str, corrected_text: str, confidence: float, linked_terms: list[dict[str, Any]]) -> bool:
    original = " ".join(original_text.lower().split())
    corrected = " ".join(corrected_text.lower().split())
    if not corrected or corrected == original:
        return False
    if linked_terms:
        return confidence >= 0.65
    return confidence >= 0.72


def load_node(state: AgentState) -> dict[str, Any]:
    doc = load_transcript(state["transcript_path"], state["audio_path"] or None)
    glossary = load_medical_glossary(state["glossary_path"] or None)
    stats = analyze_transcript(doc)
    return {
        "transcript_doc": doc,
        "glossary": glossary,
        "corrections": [],
        "stage_log": [
            f"Loaded {stats['segment_count']} segments",
            f"Low-confidence rate: {stats['low_confidence_rate']:.1%}",
        ],
        "output_paths": [],
    }


def detect_candidates_node(state: AgentState) -> dict[str, Any]:
    doc = state["transcript_doc"]
    result = identify_error_candidates(
        doc,
        state["glossary"],
        confidence_threshold=state["confidence_threshold"],
    )
    error_candidates: list[dict[str, Any]] = []
    for item in result["all_errors"]:
        candidate = _candidate_to_dict(item)
        candidate["segment_id"] = doc.segments[item.segment_idx].segment_id
        candidate["text"] = doc.segments[item.segment_idx].text
        error_candidates.append(candidate)
    return {
        "error_candidates": error_candidates,
        "medical_candidates": [_candidate_to_dict(item) for item in result["medical_candidates"]],
        "stage_log": [
            f"Detection flagged {len(result['all_errors'])} error candidates",
            f"Detection flagged {len(result['medical_candidates'])} medical candidates",
        ],
    }


def fan_out_candidates(state: AgentState) -> list[Send]:
    sends: list[Send] = []
    worker_base = {
        "audio_path": state["audio_path"],
        "retranscription_method": state["retranscription_method"],
        "confidence_threshold": state["confidence_threshold"],
        "transcript_doc": state["transcript_doc"],
        "glossary": state["glossary"],
    }
    for candidate in state.get("error_candidates", []):
        sends.append(Send("error_worker", {**worker_base, "candidate": candidate}))
    for candidate in state.get("medical_candidates", []):
        sends.append(Send("medical_worker", {**worker_base, "candidate": candidate}))
    if not sends:
        return [Send("synthesize", {})]
    return sends


def _build_correction_record(
    *,
    candidate: dict[str, Any],
    correction_type: str,
    corrected_text: str,
    confidence: float,
    method: str,
    linked_terms: list[dict[str, Any]],
    issue_reason: str,
) -> dict[str, Any]:
    accepted = _should_apply(candidate["text"], corrected_text, confidence, linked_terms)
    return {
        "segment_id": candidate["segment_id"],
        "segment_idx": candidate["segment_idx"],
        "original_text": candidate["text"],
        "corrected_text": corrected_text,
        "confidence": confidence,
        "method": method,
        "correction_type": correction_type,
        "medical_terms": [item["canonical_en"] for item in linked_terms],
        "linked_terms": linked_terms,
        "issue_reason": issue_reason,
        "accepted": accepted,
    }


def error_worker_node(state: WorkerState) -> dict[str, Any]:
    candidate = state["candidate"]
    doc = state["transcript_doc"]
    context = build_context_text(doc, candidate["segment_idx"])
    corrected_text = candidate["text"]
    confidence = 0.0
    method = "unchanged"
    if state["audio_path"]:
        try:
            wav_path = extract_audio_segment(
                state["audio_path"],
                candidate["start_time"],
                candidate["end_time"],
                padding=0.4,
            )
            corrected_text, confidence, method = retranscribe_segment(
                wav_path,
                method=state["retranscription_method"],
                context_text=context,
                original_text=candidate["text"],
                issue_reason=f"ASR error candidate: {candidate['error_type']}",
                llm=_get_llm(),
            )
        except Exception:
            corrected_text, confidence, method = retranscribe_segment(
                "missing.wav",
                method="none",
                context_text=context,
                original_text=candidate["text"],
                issue_reason=f"ASR error candidate: {candidate['error_type']}",
                llm=_get_llm(),
            )
    else:
        corrected_text, confidence, method = retranscribe_segment(
            "missing.wav",
            method="none",
            context_text=context,
            original_text=candidate["text"],
            issue_reason=f"ASR error candidate: {candidate['error_type']}",
            llm=_get_llm(),
        )
    linked_terms = link_medical_terms(corrected_text, state["glossary"])
    record = _build_correction_record(
        candidate=candidate,
        correction_type="error",
        corrected_text=corrected_text,
        confidence=confidence,
        method=method,
        linked_terms=linked_terms,
        issue_reason=f"ASR error candidate: {candidate['error_type']}",
    )
    return {"corrections": [record]}


def medical_worker_node(state: WorkerState) -> dict[str, Any]:
    candidate = state["candidate"]
    doc = state["transcript_doc"]
    context = build_context_text(doc, candidate["segment_idx"])
    specialty_hint = build_llm_hint(candidate.get("specialty", "general"), state["glossary"], max_terms=6)
    if state["audio_path"]:
        try:
            wav_path = extract_audio_segment(
                state["audio_path"],
                candidate["start_time"],
                candidate["end_time"],
                padding=0.5,
            )
            corrected_text, confidence, method = retranscribe_segment(
                wav_path,
                method=state["retranscription_method"],
                context_text=context,
                original_text=candidate["text"],
                issue_reason=candidate["reason"],
                specialty_hint=specialty_hint,
                llm=_get_llm(),
            )
        except Exception:
            corrected_text, confidence, method = retranscribe_segment(
                "missing.wav",
                method="none",
                context_text=context,
                original_text=candidate["text"],
                issue_reason=candidate["reason"],
                specialty_hint=specialty_hint,
                llm=_get_llm(),
            )
    else:
        corrected_text, confidence, method = retranscribe_segment(
            "missing.wav",
            method="none",
            context_text=context,
            original_text=candidate["text"],
            issue_reason=candidate["reason"],
            specialty_hint=specialty_hint,
            llm=_get_llm(),
        )
    linked_terms = link_medical_terms(
        corrected_text,
        state["glossary"],
        specialty=candidate.get("specialty", "general"),
    )
    record = _build_correction_record(
        candidate=candidate,
        correction_type="medical_term",
        corrected_text=corrected_text,
        confidence=confidence,
        method=method,
        linked_terms=linked_terms,
        issue_reason=candidate["reason"],
    )
    return {"corrections": [record]}


def synthesize_node(state: AgentState) -> dict[str, Any]:
    corrected_doc, applied = apply_corrections(
        state["transcript_doc"],
        state.get("corrections", []),
        min_confidence=state["confidence_threshold"],
    )
    return {
        "corrected_doc": corrected_doc,
        "stage_log": [f"Applied {len(applied)} accepted corrections"],
    }


def save_node(state: AgentState) -> dict[str, Any]:
    corrected_doc = state.get("corrected_doc") or state["transcript_doc"]
    transcript_path = Path(state["transcript_path"])
    save_document(corrected_doc, transcript_path)
    linked_terms: list[dict[str, Any]] = []
    for correction in state.get("corrections", []):
        linked_terms.extend(correction.get("linked_terms", []))
    unique_terms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in linked_terms:
        term_id = item.get("term_id", item.get("canonical_en", ""))
        if term_id and term_id not in seen:
            seen.add(term_id)
            unique_terms.append(item)
    sidecar = write_correction_sidecar(
        transcript_path,
        corrections=state.get("corrections", []),
        stage_log=state.get("stage_log", []),
        linked_terms=unique_terms,
    )
    return {
        "output_paths": [str(transcript_path), str(sidecar)],
        "stage_log": [f"Saved corrected transcript in place: {transcript_path.name}", f"Saved audit sidecar: {sidecar.name}"],
    }


def build_asr_medical_verification_graph():
    builder = StateGraph(AgentState)
    builder.add_node("load", load_node)
    builder.add_node("detect_candidates", detect_candidates_node)
    builder.add_node("error_worker", error_worker_node)
    builder.add_node("medical_worker", medical_worker_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("save", save_node)
    builder.add_edge(START, "load")
    builder.add_edge("load", "detect_candidates")
    builder.add_conditional_edges(
        "detect_candidates",
        fan_out_candidates,
        ["error_worker", "medical_worker", "synthesize"],
    )
    builder.add_edge("error_worker", "synthesize")
    builder.add_edge("medical_worker", "synthesize")
    builder.add_edge("synthesize", "save")
    builder.add_edge("save", END)
    return builder.compile()


def run_asr_medical_verification(
    transcript_path: str | Path,
    *,
    audio_path: str | Path | None = None,
    glossary_path: str | Path | None = None,
    retranscription_method: str = "auto",
    confidence_threshold: float = 0.72,
) -> dict[str, Any]:
    graph = build_asr_medical_verification_graph()
    return graph.invoke(
        {
            "transcript_path": str(transcript_path),
            "audio_path": str(audio_path or ""),
            "glossary_path": str(glossary_path or ""),
            "retranscription_method": retranscription_method,
            "confidence_threshold": confidence_threshold,
            "transcript_doc": None,
            "glossary": None,
            "error_candidates": [],
            "medical_candidates": [],
            "corrections": [],
            "corrected_doc": None,
            "stage_log": [],
            "output_paths": [],
        }
    )
