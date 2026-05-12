"""
Stage 3b: Chain-of-Thought Diarization Correction
==================================================
CONCEPT: A single LLM call with a structured chain-of-thought prompt handles
all three defect types (run_on, head_attached, tail_attached) in one pass.

Unlike Stage 2 (three separate one-shot files, each targeting one defect type),
CoT uses a four-step reasoning prompt that instructs the model to:
  1. Classify the defect.
  2. Locate the affected segments.
  3. Reason about the precise correction.
  4. Apply it across all segments.

Unlike Stage 3 (tool-calling ReAct loop), CoT is still a single LLM call.
The improvement comes purely from structured reasoning, not tool use.

Why CoT improves over one-shot:
  - Forces explicit defect classification before correction.
  - The model surfaces its intermediate reasoning in the response, making
    errors easier to diagnose.
  - Unified across all three defect types — no routing logic needed.
  - Particularly benefits run_on cases where context reasoning is critical.

Run this file:
  uv run deep_research/diarization_improvements/stage_03b_chain_of_thought/01_fix_all_cot.py
"""

from __future__ import annotations

from pathlib import Path
import sys

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model, structured_output_chain
from deep_research.diarization_improvements.shared.diarization_utils import (
    DIARIZATION_PROFILE,
    dataset_path,
    detect_defect_type,
    format_defect_descriptions,
    load_dataset,
    render_segment_block,
)

VARIANT_NAME = "cot_all_types"


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

class CoTCorrectionResult(BaseModel):
    reasoning_steps: list[str] = Field(
        description=(
            "Step-by-step reasoning: one entry each for CLASSIFY, LOCATE, "
            "REASON, and CORRECT."
        )
    )
    defect_type_detected: str = Field(
        description="One of: run_on, head_attached, tail_attached, unknown"
    )
    corrected_segments: list[dict] = Field(
        description=(
            "All segments from the transcript with defects corrected. "
            "Return every segment, not just the changed ones."
        )
    )
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_parser = JsonOutputParser()

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a diarization correction specialist.\n\n"
            "{defect_descriptions}\n\n"
            "Think through the problem step by step before correcting:\n\n"
            "STEP 1 — CLASSIFY: Examine the segments. Which defect type is present?\n"
            "  (run_on | head_attached | tail_attached | unknown)\n\n"
            "STEP 2 — LOCATE: Which specific segment indices are affected and why?\n\n"
            "STEP 3 — REASON: What precise change is needed?\n"
            "  • head_attached: strip 'SPEAKER_XX:' from the START of text; "
            "move the label to the speaker field.\n"
            "  • tail_attached: strip 'SPEAKER_XX' from the END of text; "
            "move the label to the speaker field.\n"
            "  • run_on: split the merged segment into exactly two segments "
            "at the natural speaker boundary. Do NOT change any words.\n\n"
            "STEP 4 — CORRECT: Apply the fix and return ALL segments "
            "(both corrected and unchanged).\n\n"
            "Rules:\n"
            "- Do NOT add, remove, or alter any spoken words.\n"
            "- Preserve all start/end timestamps exactly.\n"
            "- Each segment must have: start, end, text, speaker (string, not empty).\n"
            "- Return valid JSON only.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Known speakers in this transcript: {known_speakers}\n\n"
            "Heuristic defect guess: {heuristic_defect}\n\n"
            "Transcript segments:\n{segment_block}\n\n"
            "Follow the four steps above. Your reasoning_steps list must have "
            "exactly 4 entries — one per step.",
        ),
    ]
).partial(defect_descriptions=format_defect_descriptions())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_known_speakers(transcript: dict) -> list[str]:
    return sorted({
        seg.get("speaker", "")
        for seg in transcript.get("segments", [])
        if seg.get("speaker")
    })


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def fix_all_cot(transcript: dict) -> dict:
    """
    Single CoT LLM call that classifies and corrects any diarization defect.
    Returns a CorrectionPrediction-compatible dict.
    """
    segments = transcript.get("segments", [])
    known_speakers = _get_known_speakers(transcript)
    heuristic = detect_defect_type(transcript)

    prompt_with_format = _prompt.partial(
        format_instructions=_parser.get_format_instructions()
    )

    llm = create_chat_model(DIARIZATION_PROFILE, temperature=0, max_tokens=4096)
    chain = structured_output_chain(llm, prompt_with_format, CoTCorrectionResult)

    raw = chain.invoke(
        {
            "known_speakers": ", ".join(known_speakers) or "unknown",
            "heuristic_defect": heuristic,
            "segment_block": render_segment_block(segments, limit=len(segments)),
        }
    )

    corrected_segs = raw.get("corrected_segments") or segments
    reasoning = raw.get("reasoning_steps", [])
    detected = raw.get("defect_type_detected", heuristic)
    notes = raw.get("notes", []) + [f"CoT reasoning steps: {len(reasoning)}"]

    return {
        "segments": corrected_segs,
        "defect_type_detected": detected,
        "notes": notes,
        "variant": VARIANT_NAME,
    }


# Alias for benchmark runner
def correct_transcript(transcript: dict, ground_truth: dict | None = None) -> dict:
    return fix_all_cot(transcript)


if __name__ == "__main__":
    from deep_research.diarization_improvements.shared.diarization_utils import (
        evaluate_correction,
    )

    dataset = load_dataset(dataset_path())
    example = dataset[0]

    result = correct_transcript(example.input_transcript, example.ground_truth_transcript)
    metrics = evaluate_correction(example, result)

    print("=" * 60)
    print("  CHAIN-OF-THOUGHT DIARIZATION CORRECTION")
    print("=" * 60)
    print(f"Example ID:      {example.id}  ({example.defect_type})")
    print(f"Input segments:  {len(example.input_transcript['segments'])}")
    print(f"Output segments: {len(result['segments'])}")
    print(f"Defect detected: {result['defect_type_detected']}")
    print(f"Speaker accuracy: {metrics['speaker_accuracy']:.3f}")
    print(f"F1:               {metrics['f1']:.3f}")
    print()
    print("Reasoning steps:")
    for i, step in enumerate(result.get("notes", [])[:4], 1):
        print(f"  {i}. {step}")
