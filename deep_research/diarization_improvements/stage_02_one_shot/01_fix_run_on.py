"""
Stage 2a: One-Shot Run-On Correction
=====================================
CONCEPT: A single LLM prompt with surrounding context asks the model to split
a merged segment back into two speaker turns.

Run this file:
  uv run deep_research/diarization_improvements/stage_02_one_shot/01_fix_run_on.py
"""

from __future__ import annotations

import json
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
    CorrectionPrediction,
    dataset_path,
    format_defect_descriptions,
    load_dataset,
    render_segment_block,
)

VARIANT_NAME = "one_shot_run_on"


# ---------------------------------------------------------------------------
# Output model for the LLM
# ---------------------------------------------------------------------------

class RunOnCorrectionResult(BaseModel):
    corrected_segments: list[dict] = Field(
        description="The two segments after splitting at the speaker boundary."
    )
    split_point_word: str = Field(
        description="The last word of the first speaker before the split."
    )
    new_speaker: str = Field(
        description="The speaker label assigned to the second (split-off) segment."
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
            "Your task: a run_on defect has occurred — two speakers' speech was merged "
            "into a single segment. The segment's speaker field belongs to the first speaker, "
            "but part of the text is actually spoken by a different speaker.\n\n"
            "Split the merged segment into exactly two segments:\n"
            "  1. Segment A: the first speaker's text (keep the original speaker label).\n"
            "  2. Segment B: the second speaker's text (assign a plausible new speaker label).\n\n"
            "Rules:\n"
            "- Do NOT add or remove words from the original text.\n"
            "- Choose the most natural conversational split point.\n"
            "- Preserve start/end timestamps from the words array where possible.\n"
            "- Return valid JSON only.\n\n"
            "{format_instructions}",
        ),
        (
            "human",
            "Known speakers in this transcript: {known_speakers}\n\n"
            "Context (surrounding segments):\n{context_block}\n\n"
            "Target merged segment (index {target_index}):\n"
            "  Speaker: {target_speaker}\n"
            "  Text: {target_text}\n"
            "  Start: {target_start}s  End: {target_end}s\n"
            "  Words: {target_words}\n\n"
            "Split this segment into two. Return corrected_segments as a list of two dicts "
            "with keys: start, end, text, speaker.",
        ),
    ]
).partial(
    defect_descriptions=format_defect_descriptions(),
    format_instructions=_parser.get_format_instructions(),
)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def _get_known_speakers(transcript: dict) -> list[str]:
    speakers = {seg.get("speaker", "") for seg in transcript.get("segments", [])}
    return sorted(s for s in speakers if s)


def fix_run_on(transcript: dict) -> dict:
    """
    Find the most likely run-on segment and split it using one LLM call.
    Returns a CorrectionPrediction-compatible dict.
    """
    segments = transcript.get("segments", [])
    known_speakers = _get_known_speakers(transcript)

    # Find the candidate: a segment whose speaker field is empty OR the longest segment
    # with a very short gap to the next
    target_idx = None
    for i, seg in enumerate(segments):
        if not seg.get("speaker"):
            target_idx = i
            break

    # Fallback: pick the longest segment (most likely to contain two speakers)
    if target_idx is None:
        target_idx = max(range(len(segments)), key=lambda i: len(segments[i].get("words", [])))

    target = segments[target_idx]
    context_segs = (
        segments[max(0, target_idx - 2): target_idx]
        + segments[target_idx + 1: target_idx + 3]
    )

    llm = create_chat_model(DIARIZATION_PROFILE, temperature=0, max_tokens=4096)
    chain = structured_output_chain(llm, _prompt, RunOnCorrectionResult)

    raw = chain.invoke(
        {
            "known_speakers": ", ".join(known_speakers) if known_speakers else "unknown",
            "context_block": render_segment_block(context_segs),
            "target_index": target_idx,
            "target_speaker": target.get("speaker") or "(none)",
            "target_text": target.get("text", ""),
            "target_start": target.get("start", 0.0),
            "target_end": target.get("end", 0.0),
            "target_words": json.dumps(target.get("words", [])[:20]),  # limit to 20 words
        }
    )

    # Rebuild full segment list: replace target with the two corrected segments
    corrected_segs = raw.get("corrected_segments", [target])
    new_segments = segments[:target_idx] + corrected_segs + segments[target_idx + 1:]

    return {
        "segments": new_segments,
        "defect_type_detected": "run_on",
        "notes": raw.get("notes", []) + [f"split at: '{raw.get('split_point_word', '')}'"],
        "variant": VARIANT_NAME,
    }


# Alias for benchmark runner
def correct_transcript(transcript: dict, ground_truth: dict | None = None) -> dict:
    return fix_run_on(transcript)


if __name__ == "__main__":
    dataset = load_dataset(dataset_path())
    run_on_examples = [ex for ex in dataset if ex.defect_type == "run_on"]
    example = run_on_examples[0]

    result = fix_run_on(example.input_transcript)

    print("=" * 60)
    print("  ONE-SHOT RUN-ON CORRECTION")
    print("=" * 60)
    print(f"Input segments:  {len(example.input_transcript['segments'])}")
    print(f"Output segments: {len(result['segments'])}")
    print(f"GT segments:     {len(example.ground_truth_transcript['segments'])}")
    print(f"Notes: {result['notes']}")
