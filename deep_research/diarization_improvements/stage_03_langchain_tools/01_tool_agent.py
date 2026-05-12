"""
Stage 3: LangChain Tools Diarization Agent
==========================================
CONCEPT: A LangChain tool-calling agent with @tool decorators decides which
correction to apply based on the detected defect type. Rule-based tools handle
head/tail; an LLM-backed tool handles the harder run-on case.

Improvement over one-shot: the agent actively inspects the transcript, picks the
right tool, and produces a structured final answer — rather than firing a single
fixed prompt regardless of the defect type.

Run this file:
  uv run deep_research/diarization_improvements/stage_03_langchain_tools/01_tool_agent.py
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
import sys

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from deep_research.diarization_improvements.shared.diarization_utils import (
    DIARIZATION_PROFILE,
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
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a diarization correction specialist.\n\n"
    + format_defect_descriptions()
    + "\n\n"
    "Workflow:\n"
    "1. Call tool_detect_defect_type to classify the problem.\n"
    "2. Call tool_scan_embedded_labels if the defect is head_attached or tail_attached.\n"
    "3. Call the appropriate fix tool: tool_fix_head_attached, tool_fix_tail_attached, "
    "or tool_fix_run_on.\n"
    "4. Return a JSON object as your final answer with these keys:\n"
    "   corrected_segments (list of segment dicts), defect_type_detected (string), "
    "notes (list of strings).\n\n"
    "Always set defect_type_detected in your final response."
)


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
def tool_evaluate_result(
    corrected_json: str,
    ground_truth_json: str,
    defect_type: str = "unknown",
) -> str:
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


def build_agent() -> AgentExecutor:
    llm = create_chat_model(DIARIZATION_PROFILE, temperature=0, max_tokens=4096)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=ALL_TOOLS, max_iterations=12, verbose=False)


def correct_transcript(transcript: dict, ground_truth: dict | None = None) -> dict:
    """Run the LangChain tools agent on a transcript and return a correction dict."""
    executor = build_agent()
    result = executor.invoke({
        "input": (
            f"Fix the diarization defect in this transcript:\n"
            f"{json.dumps(transcript)}"
        ),
    })

    output = result.get("output", "")

    # The agent is instructed to end with a JSON object. Parse it.
    try:
        match = re.search(r"\{.*\}", output, re.DOTALL)
        payload = json.loads(match.group(0)) if match else {}
    except Exception:
        payload = {}

    segments = payload.get("corrected_segments", transcript.get("segments", []))
    # If corrected_segments came from a tool result embedded in the output, fall back.
    if not segments:
        segments = transcript.get("segments", [])

    return {
        "segments": segments,
        "defect_type_detected": payload.get("defect_type_detected", "unknown"),
        "notes": payload.get("notes", [output[:200]] if output else []),
        "variant": VARIANT_NAME,
    }


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
