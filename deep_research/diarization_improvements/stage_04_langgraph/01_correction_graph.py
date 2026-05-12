"""
Stage 4: LangGraph Diarization Correction Pipeline
===================================================
CONCEPT: A StateGraph with typed state and conditional routing handles the
full correction workflow: detect → classify → correct → evaluate → finalize.

The graph has two routing decisions:
  1. route_by_defect: after classification, dispatch to the right correction node.
  2. route_quality_gate: after evaluation, either finalize or retry (max 3 attempts).

Rule-based nodes (head/tail) never call an LLM; only the run-on node does.

Run this file:
  uv run deep_research/diarization_improvements/stage_04_langgraph/01_correction_graph.py
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, structured_output_chain
from deep_research.diarization_improvements.shared.diarization_utils import (
    DIARIZATION_PROFILE,
    apply_head_attached_fix,
    apply_tail_attached_fix,
    dataset_path,
    detect_defect_type,
    evaluate_correction,
    format_defect_descriptions,
    load_dataset,
    render_segment_block,
    scan_embedded_labels,
    summarize_results,
)

VARIANT_NAME = "langgraph_pipeline"
MAX_ATTEMPTS = 3
QUALITY_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CorrectionState(TypedDict):
    # Input
    transcript: dict
    ground_truth: dict | None

    # Analysis
    detected_defect: str
    embedded_labels: list[dict]

    # Working data
    corrected_segments: list[dict]
    correction_notes: list[str]

    # Quality gate
    evaluation: dict
    quality_score: float
    attempt: int

    # Final output
    result: dict


# ---------------------------------------------------------------------------
# LLM output model for classify node
# ---------------------------------------------------------------------------

class DefectClassification(BaseModel):
    defect_type: str = Field(
        description="One of: run_on, head_attached, tail_attached, unknown"
    )
    reasoning: str = Field(default="")


# ---------------------------------------------------------------------------
# Node: detect
# ---------------------------------------------------------------------------

def detect_node(state: CorrectionState) -> dict:
    """Heuristic defect detection — no LLM."""
    transcript = state["transcript"]
    defect = detect_defect_type(transcript)
    labels = scan_embedded_labels(transcript)
    return {
        "detected_defect": defect,
        "embedded_labels": labels,
        "correction_notes": [f"Heuristic detected: {defect}"],
    }


# ---------------------------------------------------------------------------
# Node: classify (LLM validation)
# ---------------------------------------------------------------------------

_classify_parser = JsonOutputParser()
_classify_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a diarization quality analyst.\n"
            + format_defect_descriptions()
            + "\n\nGiven a summary of a transcript's anomalies, confirm or correct the defect "
            "classification. Return JSON only.\n{format_instructions}",
        ),
        (
            "human",
            "Heuristic classification: {heuristic_defect}\n\n"
            "Embedded label hits: {label_hits}\n\n"
            "First 5 segments:\n{segment_preview}",
        ),
    ]
).partial(format_instructions=_classify_parser.get_format_instructions())


def classify_node(state: CorrectionState) -> dict:
    """Optional LLM validation of the heuristic defect classification."""
    llm = create_chat_model(DIARIZATION_PROFILE, temperature=0, max_tokens=4096)
    chain = structured_output_chain(llm, _classify_prompt, DefectClassification)

    segments = state["transcript"].get("segments", [])
    raw = chain.invoke(
        {
            "heuristic_defect": state["detected_defect"],
            "label_hits": json.dumps(state["embedded_labels"][:5]),
            "segment_preview": render_segment_block(segments, limit=5),
        }
    )

    confirmed_defect = raw.get("defect_type", state["detected_defect"])
    notes = list(state.get("correction_notes", []))
    notes.append(f"LLM confirmed: {confirmed_defect} (was: {state['detected_defect']})")

    return {
        "detected_defect": confirmed_defect,
        "correction_notes": notes,
    }


# ---------------------------------------------------------------------------
# Node: correct_head_attached (rule-based)
# ---------------------------------------------------------------------------

def correct_head_attached_node(state: CorrectionState) -> dict:
    corrected = apply_head_attached_fix(state["transcript"])
    notes = list(state.get("correction_notes", []))
    notes.append("Applied head-attached label strip (rule-based)")
    return {
        "corrected_segments": corrected["segments"],
        "correction_notes": notes,
        "attempt": state.get("attempt", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node: correct_tail_attached (rule-based)
# ---------------------------------------------------------------------------

def correct_tail_attached_node(state: CorrectionState) -> dict:
    corrected = apply_tail_attached_fix(state["transcript"])
    notes = list(state.get("correction_notes", []))
    notes.append("Applied tail-attached label strip (rule-based)")
    return {
        "corrected_segments": corrected["segments"],
        "correction_notes": notes,
        "attempt": state.get("attempt", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node: correct_run_on (LLM)
# ---------------------------------------------------------------------------

class RunOnSplit(BaseModel):
    corrected_segments: list[dict] = Field(
        description="Two segments produced by splitting the merged segment."
    )
    split_point_word: str = Field(default="")
    new_speaker: str = Field(default="")


_run_on_parser = JsonOutputParser()
_run_on_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a diarization correction specialist.\n"
            "A run-on defect has merged two speakers into one segment.\n"
            "Split the merged segment into exactly two segments at the natural speaker boundary.\n"
            "Do NOT add or invent words. Return valid JSON only.\n{format_instructions}",
        ),
        (
            "human",
            "Known speakers: {known_speakers}\n\n"
            "Context:\n{context}\n\n"
            "Merged segment (index {idx}):\n"
            "  Speaker: {speaker}\n"
            "  Text: {text}\n"
            "  Start: {start}s  End: {end}s\n"
            "  Words: {words}\n\n"
            "Return two corrected segments as a JSON list.",
        ),
    ]
).partial(format_instructions=_run_on_parser.get_format_instructions())


def correct_run_on_node(state: CorrectionState) -> dict:
    transcript = state["transcript"]
    segments = transcript.get("segments", [])
    known_speakers = sorted({s.get("speaker", "") for s in segments if s.get("speaker")})

    # Find target: empty-speaker segment or longest segment
    target_idx = next(
        (i for i, s in enumerate(segments) if not s.get("speaker")),
        max(range(len(segments)), key=lambda i: len(segments[i].get("words", [])), default=0),
    )
    target = segments[target_idx]
    context_segs = segments[max(0, target_idx - 2): target_idx] + segments[target_idx + 1: target_idx + 3]

    llm = create_chat_model(DIARIZATION_PROFILE, temperature=0, max_tokens=4096)
    chain = structured_output_chain(llm, _run_on_prompt, RunOnSplit)

    raw = chain.invoke(
        {
            "known_speakers": ", ".join(known_speakers) or "unknown",
            "context": render_segment_block(context_segs),
            "idx": target_idx,
            "speaker": target.get("speaker") or "(none)",
            "text": target.get("text", ""),
            "start": target.get("start", 0.0),
            "end": target.get("end", 0.0),
            "words": json.dumps(target.get("words", [])[:20]),
        }
    )

    corrected_segs = raw.get("corrected_segments", [target])
    new_segments = segments[:target_idx] + corrected_segs + segments[target_idx + 1:]

    notes = list(state.get("correction_notes", []))
    notes.append(f"Run-on split at word '{raw.get('split_point_word', '')}', attempt {state.get('attempt', 0) + 1}")

    return {
        "corrected_segments": new_segments,
        "correction_notes": notes,
        "attempt": state.get("attempt", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node: evaluate
# ---------------------------------------------------------------------------

def evaluate_node(state: CorrectionState) -> dict:
    ground_truth = state.get("ground_truth")
    corrected_segs = state.get("corrected_segments", state["transcript"].get("segments", []))

    if ground_truth:
        example = {
            "id": "graph_eval",
            "defect_type": state["detected_defect"],
            "ground_truth_transcript": ground_truth,
        }
        prediction = {
            "segments": corrected_segs,
            "defect_type_detected": state["detected_defect"],
            "variant": VARIANT_NAME,
        }
        evaluation = evaluate_correction(example, prediction)
        quality_score = evaluation.get("f1", 0.0)
    else:
        # No ground truth — use a heuristic quality score based on structural checks
        has_speakers = all(seg.get("speaker") for seg in corrected_segs)
        evaluation = {"no_ground_truth": True, "has_speakers": has_speakers}
        quality_score = 0.9 if has_speakers else 0.5

    return {
        "evaluation": evaluation,
        "quality_score": quality_score,
    }


# ---------------------------------------------------------------------------
# Node: finalize
# ---------------------------------------------------------------------------

def finalize_node(state: CorrectionState) -> dict:
    corrected_segs = state.get("corrected_segments", state["transcript"].get("segments", []))
    result = {
        "segments": corrected_segs,
        "defect_type_detected": state.get("detected_defect", "unknown"),
        "notes": state.get("correction_notes", []),
        "variant": VARIANT_NAME,
        "attempt": state.get("attempt", 0),
        "quality_score": state.get("quality_score", 0.0),
        "evaluation": state.get("evaluation", {}),
    }
    return {"result": result}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_by_defect(state: CorrectionState) -> str:
    defect = state.get("detected_defect", "unknown")
    if defect == "run_on":
        return "correct_run_on"
    elif defect == "head_attached":
        return "correct_head_attached"
    elif defect == "tail_attached":
        return "correct_tail_attached"
    else:
        return "finalize"


def route_quality_gate(state: CorrectionState) -> str:
    score = state.get("quality_score", 0.0)
    attempt = state.get("attempt", 0)
    if score >= QUALITY_THRESHOLD or attempt >= MAX_ATTEMPTS:
        return "finalize"
    # Retry the same correction node
    defect = state.get("detected_defect", "unknown")
    if defect == "run_on":
        return "correct_run_on"
    elif defect == "head_attached":
        return "correct_head_attached"
    elif defect == "tail_attached":
        return "correct_tail_attached"
    return "finalize"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    builder = StateGraph(CorrectionState)

    builder.add_node("detect", detect_node)
    builder.add_node("classify", classify_node)
    builder.add_node("correct_run_on", correct_run_on_node)
    builder.add_node("correct_head_attached", correct_head_attached_node)
    builder.add_node("correct_tail_attached", correct_tail_attached_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "detect")
    builder.add_edge("detect", "classify")
    builder.add_conditional_edges("classify", route_by_defect)
    builder.add_edge("correct_run_on", "evaluate")
    builder.add_edge("correct_head_attached", "evaluate")
    builder.add_edge("correct_tail_attached", "evaluate")
    builder.add_conditional_edges("evaluate", route_quality_gate)
    builder.add_edge("finalize", END)

    return builder.compile()


def correct_transcript(transcript: dict, ground_truth: dict | None = None) -> dict:
    """Run the correction graph on a transcript. Returns a CorrectionPrediction dict."""
    graph = build_graph()
    initial_state: CorrectionState = {
        "transcript": transcript,
        "ground_truth": ground_truth,
        "detected_defect": "unknown",
        "embedded_labels": [],
        "corrected_segments": [],
        "correction_notes": [],
        "evaluation": {},
        "quality_score": 0.0,
        "attempt": 0,
        "result": {},
    }
    final_state = graph.invoke(initial_state)
    return final_state["result"]


if __name__ == "__main__":
    dataset = load_dataset(dataset_path())
    example = dataset[0]

    result = correct_transcript(example.input_transcript, example.ground_truth_transcript)

    print("=" * 60)
    print("  LANGGRAPH DIARIZATION PIPELINE")
    print("=" * 60)
    print(f"Example ID:      {example.id}  ({example.defect_type})")
    print(f"Input segments:  {len(example.input_transcript['segments'])}")
    print(f"Output segments: {len(result['segments'])}")
    print(f"Defect detected: {result['defect_type_detected']}")
    print(f"Attempts:        {result['attempt']}")
    print(f"Quality score:   {result['quality_score']:.3f}")
    print(f"Notes:")
    for note in result["notes"]:
        print(f"  • {note}")
