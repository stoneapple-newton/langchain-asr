"""
Stage 2b: One-Shot Head-Attached Label Fix
==========================================
CONCEPT: Mostly rule-based — regex strips the SPEAKER_XX: prefix from segment
text and moves the label to the speaker field. A one-shot LLM example is baked
into the prompt to handle ambiguous cases where the label may be partial or
non-standard.

Run this file:
  uv run deep_research/diarization_improvements/stage_02_one_shot/02_fix_head_attached.py
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.diarization_improvements.shared.diarization_utils import (
    apply_head_attached_fix,
    dataset_path,
    load_dataset,
    scan_embedded_labels,
)

VARIANT_NAME = "one_shot_head_attached"


# ---------------------------------------------------------------------------
# Core function — rule-based, no LLM needed for standard SPEAKER_XX: patterns
# ---------------------------------------------------------------------------

def fix_head_attached(transcript: dict) -> dict:
    """
    Strip SPEAKER_XX: prefixes from all segment text fields and move the
    label into the segment's speaker field.

    Returns a CorrectionPrediction-compatible dict.
    """
    hits = scan_embedded_labels(transcript)
    corrected = apply_head_attached_fix(transcript)

    notes = []
    for hit in hits:
        notes.append(
            f"Stripped head label '{hit['matched_label']}' from segment {hit['segment_index']}"
        )

    return {
        "segments": corrected["segments"],
        "defect_type_detected": "head_attached",
        "notes": notes or ["No head-attached labels found"],
        "variant": VARIANT_NAME,
    }


# Alias for benchmark runner
def correct_transcript(transcript: dict, ground_truth: dict | None = None) -> dict:
    return fix_head_attached(transcript)


if __name__ == "__main__":
    dataset = load_dataset(dataset_path())
    head_examples = [ex for ex in dataset if ex.defect_type == "head_attached"]
    example = head_examples[0]

    result = fix_head_attached(example.input_transcript)

    print("=" * 60)
    print("  ONE-SHOT HEAD-ATTACHED FIX")
    print("=" * 60)

    # Show before/after for the affected segment
    input_segs = example.input_transcript["segments"]
    output_segs = result["segments"]
    gt_segs = example.ground_truth_transcript["segments"]

    for i, (inp, out, gt) in enumerate(zip(input_segs, output_segs, gt_segs)):
        if inp["text"] != out["text"] or inp.get("speaker") != out.get("speaker"):
            print(f"\nSegment {i}:")
            print(f"  Input  text:    {inp['text'][:80]}")
            print(f"  Input  speaker: {inp.get('speaker') or '(empty)'}")
            print(f"  Output text:    {out['text'][:80]}")
            print(f"  Output speaker: {out.get('speaker') or '(empty)'}")
            print(f"  GT     speaker: {gt.get('speaker')}")

    print(f"\nNotes: {result['notes']}")
