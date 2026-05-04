"""
Stage 3: LangChain Tools Diarization Agent
==========================================
CONCEPT: A LangChain agent with @tool decorators decides which correction tool
to call based on the detected defect type. Rule-based tools handle head/tail;
an LLM-backed tool handles the harder run-on case.

Run this file:
  uv run deep_research/diarization_improvements/stage_03_langchain_tools/01_tool_agent.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from langchain.agents import create_agent
from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from deep_research.diarization_improvements.shared.diarization_utils import (
    CorrectionPrediction,
    apply_head_attached_fix,
    apply_tail_attached_fix,
    dataset_path,
    detect_defect_type,
    evaluate_correction,
    format_defect_descriptions,
    load_dataset,
    render_segment_block,
    scan_embedded_labels,
)

VARIANT_NAME = "langchain_tools"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def tool_detect_defect_type(transcript_json: str) -> str:
    """Classify the primary diarization defect in a WhisperX transcript JSON.
    Returns one of: run_on, head_attached, tail_attached, unknown."""
    try:
        transcript = json.loads(transcript_json)
        return detect_defect_type(transcript)
    except Exception as exc:
        return f"error: {exc}"


@tool
def tool_scan_embedded_labels(transcript_json: str) -> str:
    """Scan all segment text fields for embedded SPEAKER_XX patterns.
    Returns a JSON list of hits with keys: segment_index, position, matched_label."""
    try:
        transcript = json.loads(transcript_json)
        hits = scan_embedded_labels(transcript)
        return json.dumps(hits, indent=2)
    except Exception as exc:
        return f"error: {exc}"


@tool
def tool_get_known_speakers(transcript_json: str) -> str:
    """Extract all unique speaker labels from a transcript's segments.
    Returns a JSON list of speaker label strings."""
    try:
        transcript = json.loads(transcript_json)
        speakers = sorted({
            seg.get("speaker", "")
            for seg in transcript.get("segments", [])
            if seg.get("speaker")
        })
        return json.dumps(speakers)
    except Exception as exc:
        return f"error: {exc}"


@tool
def tool_fix_head_attached(transcript_json: str) -> str:
    """Remove SPEAKER_XX: prefixes from segment text fields and move each label
    to the segment's speaker field. Rule-based — no LLM needed.
    Returns the corrected transcript as JSON."""
    try:
        transcript = json.loads(transcript_json)
        corrected = apply_head_attached_fix(transcript)
        return json.dumps(corrected)
    except Exception as exc:
        return f"error: {exc}"


@tool
def tool_fix_tail_attached(transcript_json: str) -> str:
    """Remove SPEAKER_XX suffixes from segment text fields and move each label
    to the segment's speaker field. Rule-based — no LLM needed.
    Returns the corrected transcript as JSON."""
    try:
        transcript = json.loads(transcript_json)
        corrected = apply_tail_attached_fix(transcript)
        return json.dumps(corrected)
    except Exception as exc:
        return f"error: {exc}"


@tool
def tool_fix_run_on(transcript_json: str) -> str:
    """Split a run-on segment where two speakers' speech was merged.
    Uses an LLM to find the natural speaker boundary and produce two segments.
    Returns the corrected transcript as JSON."""
    try:
        from deep_research.diarization_improvements.stage_02_one_shot.fix_run_on import fix_run_on
        transcript = json.loads(transcript_json)
        result = fix_run_on(transcript)
        # Return the corrected transcript (reassembled)
        corrected_transcript = dict(transcript)
        corrected_transcript["segments"] = result["segments"]
        return json.dumps(corrected_transcript)
    except ImportError:
        # Fallback: use the same logic inline via stage_02
        import importlib.util
        stage02_path = Path(__file__).parent.parent / "stage_02_one_shot" / "01_fix_run_on.py"
        spec = importlib.util.spec_from_file_location("_fix_run_on", stage02_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        transcript = json.loads(transcript_json)
        result = mod.fix_run_on(transcript)
        corrected = dict(transcript)
        corrected["segments"] = result["segments"]
        return json.dumps(corrected)
    except Exception as exc:
        return f"error: {exc}"


@tool
def tool_evaluate_result(corrected_json: str, ground_truth_json: str, defect_type: str = "unknown") -> str:
    """Score a corrected transcript against the ground truth.
    Returns evaluation metrics as JSON including speaker_accuracy, f1, text_preserved."""
    try:
        corrected = json.loads(corrected_json)
        ground_truth = json.loads(ground_truth_json)
        example = {
            "id": "agent_eval",
            "defect_type": defect_type,
            "ground_truth_transcript": ground_truth,
        }
        prediction = {
            "segments": corrected.get("segments", []),
            "defect_type_detected": defect_type,
            "variant": VARIANT_NAME,
        }
        metrics = evaluate_correction(example, prediction)
        return json.dumps(metrics, indent=2)
    except Exception as exc:
        return f"error: {exc}"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    tool_detect_defect_type,
    tool_scan_embedded_labels,
    tool_get_known_speakers,
    tool_fix_head_attached,
    tool_fix_tail_attached,
    tool_fix_run_on,
    tool_evaluate_result,
]


def build_agent():
    return create_agent(
        model=create_chat_model(temperature=0, max_tokens=4096),
        tools=ALL_TOOLS,
        system_prompt=(
            "You are a diarization correction specialist.\n\n"
            + format_defect_descriptions()
            + "\n\n"
            "Workflow:\n"
            "1. Call tool_detect_defect_type to classify the problem.\n"
            "2. Call tool_scan_embedded_labels if the defect is head_attached or tail_attached.\n"
            "3. Call the appropriate fix tool: tool_fix_head_attached, tool_fix_tail_attached, "
            "   or tool_fix_run_on.\n"
            "4. Return the corrected segments as a JSON list.\n\n"
            "Always set defect_type_detected in your final response."
        ),
        response_format=CorrectionPrediction,
    )


def correct_transcript(transcript: dict, ground_truth: dict | None = None) -> dict:
    """
    Run the LangChain tools agent on a transcript and return a correction dict.
    """
    agent = build_agent()
    transcript_json = json.dumps(transcript)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Fix the diarization defect in this transcript:\n{transcript_json}"
                    ),
                }
            ]
        },
        config={"recursion_limit": 12},
    )

    structured: CorrectionPrediction = result.get("structured_response") or CorrectionPrediction()
    structured_dict = structured.model_dump() if hasattr(structured, "model_dump") else dict(structured)

    # If agent returned empty segments, fall back to input segments
    if not structured_dict.get("segments"):
        structured_dict["segments"] = transcript.get("segments", [])

    structured_dict["variant"] = VARIANT_NAME
    return structured_dict


if __name__ == "__main__":
    dataset = load_dataset(dataset_path())
    example = dataset[0]

    result = correct_transcript(example.input_transcript, example.ground_truth_transcript)

    print("=" * 60)
    print("  LANGCHAIN TOOLS DIARIZATION AGENT")
    print("=" * 60)
    print(f"Example ID:      {example.id}  ({example.defect_type})")
    print(f"Input segments:  {len(example.input_transcript['segments'])}")
    print(f"Output segments: {len(result['segments'])}")
    print(f"Defect detected: {result['defect_type_detected']}")
    print(f"Notes: {result['notes']}")
