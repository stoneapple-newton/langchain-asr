"""
Stage 5: Deep Agents Diarization Swarm
=======================================
CONCEPT: A supervisor delegates to 4 specialist subagents. Each specialist
has a focused role — detection, run-on repair, label extraction, or evaluation.
The supervisor routes work based on the detected defect type.

Install deepagents first:
  uv add deepagents

Run this file:
  uv run deep_research/diarization_improvements/stage_05_deep_agents/01_diarization_swarm.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import sys
from typing import Any

from langchain_core.tools import tool

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from deep_research.diarization_improvements.shared.diarization_utils import (
    apply_head_attached_fix,
    apply_tail_attached_fix,
    dataset_path,
    detect_defect_type,
    evaluate_correction,
    format_defect_descriptions,
    load_dataset,
    scan_embedded_labels,
)

VARIANT_NAME = "deep_agents_swarm"

# ---------------------------------------------------------------------------
# Shared Tools
# ---------------------------------------------------------------------------

@tool
def inspect_transcript(transcript_json: str) -> str:
    """Return an overview of a WhisperX transcript: segment count, speakers,
    duration, and any segments with missing speaker fields."""
    try:
        t = json.loads(transcript_json)
        segments = t.get("segments", [])
        speakers = {s.get("speaker", "") for s in segments}
        missing = [i for i, s in enumerate(segments) if not s.get("speaker")]
        return json.dumps({
            "segment_count": len(segments),
            "unique_speakers": sorted(sp for sp in speakers if sp),
            "duration": t.get("duration", 0.0),
            "segments_missing_speaker": missing,
            "language": t.get("language", "en"),
        }, indent=2)
    except Exception as exc:
        return f"error: {exc}"


@tool
def scan_embedded_labels_tool(transcript_json: str) -> str:
    """Find SPEAKER_XX patterns embedded in segment text fields.
    Returns a JSON list with segment_index, position (head/tail), and matched_label."""
    try:
        t = json.loads(transcript_json)
        hits = scan_embedded_labels(t)
        return json.dumps(hits, indent=2)
    except Exception as exc:
        return f"error: {exc}"


@tool
def check_timing_boundaries(transcript_json: str) -> str:
    """Identify consecutive segments with gap < 0.15s and different speakers.
    These are run-on candidates. Returns a JSON list of suspect pairs."""
    try:
        t = json.loads(transcript_json)
        segments = t.get("segments", [])
        candidates = []
        for i in range(len(segments) - 1):
            cur = segments[i]
            nxt = segments[i + 1]
            gap = nxt.get("start", 0.0) - cur.get("end", 0.0)
            if gap < 0.15 and cur.get("speaker") != nxt.get("speaker"):
                candidates.append({
                    "segment_a": i,
                    "speaker_a": cur.get("speaker"),
                    "text_a": cur.get("text", "")[:60],
                    "segment_b": i + 1,
                    "speaker_b": nxt.get("speaker"),
                    "text_b": nxt.get("text", "")[:60],
                    "gap_seconds": round(gap, 3),
                })
        return json.dumps(candidates, indent=2)
    except Exception as exc:
        return f"error: {exc}"


@tool
def apply_label_strip(transcript_json: str, position: str) -> str:
    """Remove embedded speaker labels from transcript text fields.
    position must be 'head' or 'tail'. Returns the cleaned transcript as JSON."""
    try:
        t = json.loads(transcript_json)
        if position == "head":
            result = apply_head_attached_fix(t)
        elif position == "tail":
            result = apply_tail_attached_fix(t)
        else:
            return f"error: position must be 'head' or 'tail', got '{position}'"
        return json.dumps(result)
    except Exception as exc:
        return f"error: {exc}"


@tool
def split_segment(transcript_json: str, segment_index: int, split_word: str) -> str:
    """Split one segment at the word boundary where split_word ends speaker A's turn.
    All words up to and including split_word go to segment A; the rest go to segment B.
    segment_index is the 0-based index into segments[].
    Returns the updated transcript as JSON."""
    try:
        t = json.loads(transcript_json)
        segments = list(t.get("segments", []))
        if segment_index >= len(segments):
            return f"error: segment_index {segment_index} out of range (len={len(segments)})"

        target = segments[segment_index]
        words = target.get("words", [])

        # Find the split point: last occurrence of split_word
        split_i = -1
        for j, w in enumerate(words):
            if w.get("word", "").lower().rstrip(".,!?") == split_word.lower().rstrip(".,!?"):
                split_i = j

        if split_i < 0:
            return f"error: split_word '{split_word}' not found in segment {segment_index}"

        words_a = words[:split_i + 1]
        words_b = words[split_i + 1:]

        if not words_a or not words_b:
            return f"error: split produces an empty segment (split_i={split_i})"

        seg_a = {
            "start": target["start"],
            "end": words_a[-1]["end"],
            "text": " ".join(w["word"] for w in words_a),
            "speaker": target.get("speaker", ""),
            "words": words_a,
        }

        # Try to infer speaker B from the word-level speaker field
        speaker_b_candidates = [w.get("speaker", "") for w in words_b if w.get("speaker")]
        speaker_b = speaker_b_candidates[0] if speaker_b_candidates else ""

        seg_b = {
            "start": words_b[0]["start"],
            "end": target["end"],
            "text": " ".join(w["word"] for w in words_b),
            "speaker": speaker_b,
            "words": words_b,
        }

        new_segments = segments[:segment_index] + [seg_a, seg_b] + segments[segment_index + 1:]
        result = dict(t)
        result["segments"] = new_segments
        return json.dumps(result)
    except Exception as exc:
        return f"error: {exc}"


@tool
def score_correction(corrected_json: str, ground_truth_json: str, defect_type: str = "unknown") -> str:
    """Score a corrected transcript against ground truth.
    Returns evaluation metrics as JSON: speaker_accuracy, f1, text_preserved, etc."""
    try:
        corrected = json.loads(corrected_json)
        ground_truth = json.loads(ground_truth_json)
        example = {
            "id": "swarm_eval",
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
# Shared tools list
# ---------------------------------------------------------------------------

SHARED_TOOLS = [
    inspect_transcript,
    scan_embedded_labels_tool,
    check_timing_boundaries,
    apply_label_strip,
    split_segment,
    score_correction,
]


# ---------------------------------------------------------------------------
# Agent builder
# ---------------------------------------------------------------------------

def build_diarization_swarm():
    try:
        from deepagents import create_deep_agent
        from deepagents.backends import FilesystemBackend
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as exc:
        raise RuntimeError(
            "deepagents is not installed. Install it with `uv add deepagents`."
        ) from exc

    return create_deep_agent(
        name="diarization-swarm",
        model=create_chat_model(temperature=0, max_tokens=1024),
        tools=SHARED_TOOLS,
        system_prompt=(
            "You are a diarization correction supervisor.\n\n"
            + format_defect_descriptions()
            + "\n\n"
            "Workflow:\n"
            "1. Delegate to detector_agent to classify the defect type.\n"
            "2. Based on the defect:\n"
            "   - head_attached or tail_attached → delegate to label_extractor_agent\n"
            "   - run_on → delegate to run_on_repair_agent\n"
            "3. Delegate to evaluator_agent to score the result.\n"
            "4. Return the corrected transcript segments and quality score.\n\n"
            "Final answer must include:\n"
            "  - corrected_segments: list of segment dicts\n"
            "  - defect_type_detected: string\n"
            "  - notes: list of strings\n"
        ),
        subagents=[
            {
                "name": "detector_agent",
                "description": "Classify which diarization defect type is present in the transcript.",
                "system_prompt": (
                    "Analyze the transcript and classify which diarization defect is present.\n"
                    "Use inspect_transcript and scan_embedded_labels_tool first.\n"
                    "Then use check_timing_boundaries to check for run-on candidates.\n"
                    "Return: defect_type (run_on|head_attached|tail_attached|unknown) and reasoning."
                ),
                "tools": [inspect_transcript, scan_embedded_labels_tool, check_timing_boundaries],
            },
            {
                "name": "run_on_repair_agent",
                "description": "Repair run-on segments by splitting them at the speaker boundary.",
                "system_prompt": (
                    "You repair run-on diarization defects.\n"
                    "A run-on is when two speakers' speech is merged into one segment.\n"
                    "Use check_timing_boundaries to find run-on candidates.\n"
                    "Use split_segment to split merged segments at the natural boundary.\n"
                    "Return the repaired transcript JSON."
                ),
                "tools": [check_timing_boundaries, split_segment, inspect_transcript],
            },
            {
                "name": "label_extractor_agent",
                "description": "Remove speaker labels incorrectly embedded in segment text fields.",
                "system_prompt": (
                    "You remove speaker labels from transcript text fields.\n"
                    "Use scan_embedded_labels_tool to find labels.\n"
                    "Use apply_label_strip with position='head' for SPEAKER_XX: prefixes.\n"
                    "Use apply_label_strip with position='tail' for SPEAKER_XX suffixes.\n"
                    "Return the cleaned transcript JSON."
                ),
                "tools": [scan_embedded_labels_tool, apply_label_strip, inspect_transcript],
            },
            {
                "name": "evaluator_agent",
                "description": "Score a corrected transcript against ground truth and recommend next steps.",
                "system_prompt": (
                    "You assess diarization correction quality.\n"
                    "Use score_correction to compute metrics.\n"
                    "Return a quality assessment with: f1_score, speaker_accuracy, "
                    "text_preserved, spurious_labels_removed, and recommended_next_step."
                ),
                "tools": [score_correction, inspect_transcript],
            },
        ],
        backend=FilesystemBackend(root_dir=str(REPO_ROOT), virtual_mode=True),
        checkpointer=MemorySaver(),
    )


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

def _extract_json_block(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        joined = "".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        joined = str(content)

    match = re.search(r"\{.*\}", joined, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse JSON from swarm output: {joined[:200]}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def correct_transcript(transcript: dict, ground_truth: dict | None = None) -> dict:
    """Run the diarization swarm on a transcript and return a correction dict."""
    agent = build_diarization_swarm()
    transcript_json = json.dumps(transcript)
    gt_json = json.dumps(ground_truth) if ground_truth else "null"

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Fix the diarization defect in this transcript:\n{transcript_json}\n\n"
                        f"Ground truth (for evaluation only, do not use for correction):\n{gt_json}"
                    ),
                }
            ]
        },
        config={"configurable": {"thread_id": "diarization-swarm-demo"}},
    )

    last_message = result["messages"][-1]
    payload = _extract_json_block(getattr(last_message, "content", last_message))

    return {
        "segments": payload.get("corrected_segments", transcript.get("segments", [])),
        "defect_type_detected": payload.get("defect_type_detected", "unknown"),
        "notes": payload.get("notes", []),
        "variant": VARIANT_NAME,
    }


if __name__ == "__main__":
    dataset = load_dataset(dataset_path())
    example = dataset[0]

    try:
        result = correct_transcript(example.input_transcript, example.ground_truth_transcript)
    except RuntimeError as exc:
        print(exc)
    else:
        print("=" * 60)
        print("  DEEP AGENTS DIARIZATION SWARM")
        print("=" * 60)
        print(f"Example ID:      {example.id}  ({example.defect_type})")
        print(f"Input segments:  {len(example.input_transcript['segments'])}")
        print(f"Output segments: {len(result['segments'])}")
        print(f"Defect detected: {result['defect_type_detected']}")
        print(f"Notes: {result['notes']}")
