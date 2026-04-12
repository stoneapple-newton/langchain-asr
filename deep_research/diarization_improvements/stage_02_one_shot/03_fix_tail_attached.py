"""
Stage 2c: One-Shot Tail-Attached Label Fix
==========================================
CONCEPT: Mostly rule-based — regex strips the SPEAKER_XX suffix from segment
text and moves the label to the speaker field.

Run this file:
  uv run deep_research/diarization_improvements/stage_02_one_shot/03_fix_tail_attached.py
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.diarization_improvements.shared.diarization_utils import (
    apply_tail_attached_fix,
    dataset_path,
    load_dataset,
    scan_embedded_labels,
)

VARIANT_NAME = "one_shot_tail_attached"


# ---------------------------------------------------------------------------
# Core function — rule-based, no LLM needed for standard SPEAKER_XX suffixes
# ---------------------------------------------------------------------------

def fix_tail_attached(transcript: dict) -> dict:
    """
    Strip SPEAKER_XX suffixes from all segment text fields and move the
    label into the segment's speaker field.

    Returns a CorrectionPrediction-compatible dict.
    """
    hits = scan_embedded_labels(transcript)
    corrected = apply_tail_attached_fix(transcript)

    notes = []
    for hit in hits:
        if hit["position"] == "tail":
            notes.append(
                f"Stripped tail label '{hit['matched_label']}' from segment {hit['segment_index']}"
            )

    return {
        "segments": corrected["segments"],
        "defect_type_detected": "tail_attached",
        "notes": notes or ["No tail-attached labels found"],
        "variant": VARIANT_NAME,
    }


# Alias for benchmark runner
def correct_transcript(transcript: dict, ground_truth: dict | None = None) -> dict:
    return fix_tail_attached(transcript)


if __name__ == "__main__":
    dataset = load_dataset(dataset_path())
    tail_examples = [ex for ex in dataset if ex.defect_type == "tail_attached"]
    example = tail_examples[0]

    result = fix_tail_attached(example.input_transcript)

    print("=" * 60)
    print("  ONE-SHOT TAIL-ATTACHED FIX")
    print("=" * 60)

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
