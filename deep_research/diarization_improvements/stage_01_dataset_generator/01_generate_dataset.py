"""
Stage 1: Diarization Dataset Generator
=======================================
CONCEPT: Programmatically create a synthetic WhisperX dataset with three
types of diarization defects injected into clean source segments.

Defect types:
  run_on       — Two consecutive different-speaker segments merged into one.
  head_attached — SPEAKER_XX: prefix embedded in segment text; speaker field cleared.
  tail_attached — SPEAKER_XX suffix embedded at end of segment text; speaker field cleared.

The generator is deterministic (seeded RNG) and requires no LLM.

Run this file:
  uv run deep_research/diarization_improvements/stage_01_dataset_generator/01_generate_dataset.py
"""

from __future__ import annotations

import copy
import json
import random
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_PATH = REPO_ROOT / "deep_research" / "asr" / "sample_data" / "sample_transcript.json"
OUTPUT_PATH = REPO_ROOT / "deep_research" / "diarization_improvements" / "dataset" / "diarization_dataset.json"

SEED = 42
EXAMPLES_PER_TYPE = 10  # 10 × 3 = 30 total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transcript(segments: list[dict], source_meta: dict) -> dict:
    """Wrap segments in a WhisperX-compatible transcript dict."""
    duration = segments[-1]["end"] if segments else 0.0
    return {
        "meeting_metadata": source_meta.get("meeting_metadata", {"title": "Synthetic Meeting", "date": "2024-01-01"}),
        "duration": round(duration, 2),
        "language": source_meta.get("language", "en"),
        "segments": segments,
    }


def _context_window(segments: list[dict], idx: int, before: int = 2, after: int = 2) -> list[dict]:
    """Return up to `before` segments before idx and `after` segments after idx."""
    start = max(0, idx - before)
    end = min(len(segments), idx + after + 1)
    return segments[start:end]


# ---------------------------------------------------------------------------
# Defect injection
# ---------------------------------------------------------------------------

def inject_run_on(segments: list[dict], idx_a: int, idx_b: int) -> tuple[list[dict], list[dict]]:
    """
    Merge segment idx_b into idx_a (run-on defect).

    Returns (defective_segments, ground_truth_segments).
    The defective output has one merged segment where the ground truth has two.
    """
    segs = copy.deepcopy(segments)
    seg_a = segs[idx_a]
    seg_b = segs[idx_b]

    # Ground truth: the two clean segments side by side
    ground_truth = copy.deepcopy(segs)

    # Build merged (defective) segment
    merged = {
        "start": seg_a["start"],
        "end": seg_b["end"],
        # Concatenate text, stripping trailing punctuation from seg_a
        "text": seg_a["text"].rstrip(".!?,;") + " " + seg_b["text"].lstrip(),
        "speaker": seg_a["speaker"],   # only speaker A's label remains
        "words": seg_a.get("words", []) + seg_b.get("words", []),
    }

    # Build defective segments list: replace idx_a and idx_b with merged
    defective = segs[:idx_a] + [merged] + segs[idx_b + 1:]

    return defective, ground_truth


def inject_head_attached(segments: list[dict], idx: int) -> tuple[list[dict], list[dict]]:
    """
    Embed a speaker label at the start of segment idx's text (head_attached defect).
    The segment's own speaker field is set to a neighbouring speaker (or empty).

    Returns (defective_segments, ground_truth_segments).
    """
    segs = copy.deepcopy(segments)
    ground_truth = copy.deepcopy(segs)

    seg = segs[idx]
    original_speaker = seg["speaker"]
    label_to_embed = original_speaker

    # Inject: prefix the label into text, clear the speaker field
    seg["text"] = f"{label_to_embed}: {seg['text'].lstrip()}"
    seg["speaker"] = ""

    defective = segs
    return defective, ground_truth


def inject_tail_attached(segments: list[dict], idx: int) -> tuple[list[dict], list[dict]]:
    """
    Append a speaker label to the end of segment idx's text (tail_attached defect).
    The segment's own speaker field is cleared.

    Returns (defective_segments, ground_truth_segments).
    """
    segs = copy.deepcopy(segments)
    ground_truth = copy.deepcopy(segs)

    seg = segs[idx]
    original_speaker = seg["speaker"]

    # Inject: append label, clear speaker field
    seg["text"] = seg["text"].rstrip() + f" {original_speaker}"
    seg["speaker"] = ""

    defective = segs
    return defective, ground_truth


# ---------------------------------------------------------------------------
# Candidate pool builder
# ---------------------------------------------------------------------------

def _build_candidate_pool(segments: list[dict]) -> dict:
    """
    Build pools of segment indices suitable for each defect type.

    For run_on: need two adjacent segments with different speakers, each ≥4 words.
    For head/tail: need a segment ≥4 words with a known speaker.
    """
    run_on_pairs = []
    for i in range(len(segments) - 1):
        seg_a = segments[i]
        seg_b = segments[i + 1]
        if (
            seg_a.get("speaker")
            and seg_b.get("speaker")
            and seg_a["speaker"] != seg_b["speaker"]
            and len(seg_a.get("words", [])) >= 4
            and len(seg_b.get("words", [])) >= 4
        ):
            run_on_pairs.append((i, i + 1))

    label_indices = [
        i for i, seg in enumerate(segments)
        if seg.get("speaker") and len(seg.get("words", [])) >= 4
    ]

    return {
        "run_on_pairs": run_on_pairs,
        "label_indices": label_indices,
    }


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_dataset(
    source_path: Path = SOURCE_PATH,
    output_path: Path = OUTPUT_PATH,
    seed: int = SEED,
    n_per_type: int = EXAMPLES_PER_TYPE,
) -> list[dict]:
    rng = random.Random(seed)

    raw = json.loads(source_path.read_text(encoding="utf-8"))
    segments = raw["segments"]
    source_meta = {k: v for k, v in raw.items() if k != "segments"}

    pool = _build_candidate_pool(segments)
    examples = []

    # --- run_on examples ---
    pairs = pool["run_on_pairs"]
    chosen_pairs: list[tuple[int, int]] = []
    # Allow reuse if not enough unique pairs (with different offsets to vary text)
    for i in range(n_per_type):
        pair = pairs[rng.randint(0, len(pairs) - 1)]
        chosen_pairs.append(pair)

    for i, (idx_a, idx_b) in enumerate(chosen_pairs):
        defective_segs, gt_segs = inject_run_on(segments, idx_a, idx_b)
        examples.append({
            "id": f"run_on_{i:02d}",
            "defect_type": "run_on",
            "input_transcript": _make_transcript(defective_segs, source_meta),
            "ground_truth_transcript": _make_transcript(gt_segs, source_meta),
            "metadata": {
                "seed": seed,
                "injected_at_indices": [idx_a, idx_b],
                "speakers_involved": [
                    segments[idx_a]["speaker"],
                    segments[idx_b]["speaker"],
                ],
                "description": (
                    f"SPEAKER_{segments[idx_b]['speaker']} utterance merged into "
                    f"preceding {segments[idx_a]['speaker']} segment"
                ),
            },
        })

    # --- head_attached examples ---
    label_indices = pool["label_indices"]
    chosen_head: list[int] = []
    for i in range(n_per_type):
        idx = label_indices[rng.randint(0, len(label_indices) - 1)]
        chosen_head.append(idx)

    for i, idx in enumerate(chosen_head):
        defective_segs, gt_segs = inject_head_attached(segments, idx)
        examples.append({
            "id": f"head_attached_{i:02d}",
            "defect_type": "head_attached",
            "input_transcript": _make_transcript(defective_segs, source_meta),
            "ground_truth_transcript": _make_transcript(gt_segs, source_meta),
            "metadata": {
                "seed": seed,
                "injected_at_index": idx,
                "original_speaker": segments[idx]["speaker"],
                "description": (
                    f"Speaker label '{segments[idx]['speaker']}' embedded as prefix "
                    f"in segment {idx} text"
                ),
            },
        })

    # --- tail_attached examples ---
    chosen_tail: list[int] = []
    for i in range(n_per_type):
        idx = label_indices[rng.randint(0, len(label_indices) - 1)]
        chosen_tail.append(idx)

    for i, idx in enumerate(chosen_tail):
        defective_segs, gt_segs = inject_tail_attached(segments, idx)
        examples.append({
            "id": f"tail_attached_{i:02d}",
            "defect_type": "tail_attached",
            "input_transcript": _make_transcript(defective_segs, source_meta),
            "ground_truth_transcript": _make_transcript(gt_segs, source_meta),
            "metadata": {
                "seed": seed,
                "injected_at_index": idx,
                "original_speaker": segments[idx]["speaker"],
                "description": (
                    f"Speaker label '{segments[idx]['speaker']}' appended as suffix "
                    f"in segment {idx} text"
                ),
            },
        })

    return examples


def save_dataset(examples: list[dict], output_path: Path = OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "seed": SEED,
        "total_examples": len(examples),
        "examples": examples,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    examples = generate_dataset()
    save_dataset(examples)

    by_type: dict[str, int] = {}
    for ex in examples:
        by_type[ex["defect_type"]] = by_type.get(ex["defect_type"], 0) + 1

    print(f"Generated {len(examples)} examples: " + ", ".join(f"{v} {k}" for k, v in by_type.items()))
    print(f"Saved: {OUTPUT_PATH}")
