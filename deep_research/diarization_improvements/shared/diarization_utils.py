"""
Shared utilities for the diarization improvements series.

Provides data models, defect detection heuristics, evaluation metrics,
and prompt helpers used by all stages (one-shot → LangChain → LangGraph → Deep Agents).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def dataset_path() -> Path:
    return Path(__file__).resolve().parents[1] / "dataset" / "diarization_dataset.json"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class WhisperWord(BaseModel):
    word: str
    start: float
    end: float
    score: float
    speaker: str = ""


class WhisperSegment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str = ""
    words: list[WhisperWord] = Field(default_factory=list)


class WhisperTranscript(BaseModel):
    meeting_metadata: dict = Field(default_factory=dict)
    duration: float = 0.0
    language: str = "en"
    segments: list[WhisperSegment] = Field(default_factory=list)


class DiarizationExample(BaseModel):
    id: str
    defect_type: str          # "run_on" | "head_attached" | "tail_attached"
    input_transcript: dict    # raw WhisperX dict (not validated to keep benchmark loading fast)
    ground_truth_transcript: dict
    metadata: dict = Field(default_factory=dict)


class CorrectionPrediction(BaseModel):
    segments: list[dict] = Field(default_factory=list)
    defect_type_detected: str = "unknown"
    notes: list[str] = Field(default_factory=list)
    variant: str = "unknown"


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str | Path | None = None) -> list[DiarizationExample]:
    resolved = Path(path) if path else dataset_path()
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    return [DiarizationExample(**ex) for ex in raw["examples"]]


# ---------------------------------------------------------------------------
# Defect detection heuristics
# ---------------------------------------------------------------------------

# Regex patterns for embedded speaker labels
_HEAD_PATTERN = re.compile(r"^\s*SPEAKER_\d+\s*:", re.IGNORECASE)
_TAIL_PATTERN = re.compile(r"\bSPEAKER_\d+\s*$", re.IGNORECASE)
_ANY_LABEL_PATTERN = re.compile(r"\bSPEAKER_\d+\b", re.IGNORECASE)

# Threshold below which a timing gap is "suspiciously fast" (seconds)
_FAST_SWITCH_GAP = 0.15


def detect_defect_type(transcript: dict) -> str:
    """
    Heuristic classification of the primary diarization defect in a transcript.

    Returns one of: "head_attached", "tail_attached", "run_on", "unknown".

    Priority order: head_attached > tail_attached > run_on > unknown.
    Only one defect type is returned per call (the most likely dominant one).
    """
    segments = transcript.get("segments", [])

    for seg in segments:
        text = seg.get("text", "")
        if _HEAD_PATTERN.search(text):
            return "head_attached"

    for seg in segments:
        text = seg.get("text", "")
        if _TAIL_PATTERN.search(text):
            return "tail_attached"

    # Run-on detection: adjacent segments with different speakers but near-zero gap
    for i in range(len(segments) - 1):
        cur = segments[i]
        nxt = segments[i + 1]
        cur_spk = cur.get("speaker", "")
        nxt_spk = nxt.get("speaker", "")
        gap = nxt.get("start", 0.0) - cur.get("end", 0.0)
        if cur_spk and nxt_spk and cur_spk == nxt_spk:
            # Same speaker adjacent — not a run-on (could be fragmentation)
            continue
        # Different (or missing) speakers with tiny gap → run-on candidate
        if gap < _FAST_SWITCH_GAP and _ANY_LABEL_PATTERN.search(cur.get("text", "") + nxt.get("text", "")):
            return "run_on"

    # Broader run-on: a segment whose speaker field is empty/None but text has content
    for seg in segments:
        if not seg.get("speaker") and seg.get("text", "").strip():
            return "run_on"

    return "unknown"


def scan_embedded_labels(transcript: dict) -> list[dict]:
    """
    Scan all segment text fields for embedded SPEAKER_XX patterns.
    Returns a list of hits: {segment_index, text, position, matched_label}.
    """
    hits = []
    for i, seg in enumerate(transcript.get("segments", [])):
        text = seg.get("text", "")
        head_match = _HEAD_PATTERN.search(text)
        if head_match:
            hits.append({
                "segment_index": i,
                "text": text,
                "position": "head",
                "matched_label": head_match.group(0).strip().rstrip(":").strip(),
            })
            continue
        tail_match = _TAIL_PATTERN.search(text)
        if tail_match:
            hits.append({
                "segment_index": i,
                "text": text,
                "position": "tail",
                "matched_label": tail_match.group(0).strip(),
            })
    return hits


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def render_segment_block(segments: list[dict], limit: int = 10) -> str:
    """Format a segment list as a numbered text block for LLM prompts."""
    lines = []
    for i, seg in enumerate(segments[:limit]):
        spk = seg.get("speaker") or "(no label)"
        text = seg.get("text", "").strip()
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        lines.append(f"[{i}] {spk} ({start:.1f}s–{end:.1f}s): {text}")
    return "\n".join(lines)


def format_defect_descriptions() -> str:
    """Return a prompt-ready description of all three defect types."""
    return (
        "Diarization defect types:\n"
        "1. run_on      — Two speakers' speech was merged into one segment. "
        "The segment.speaker field belongs to speaker A, but the text also contains "
        "speech from speaker B without any boundary.\n"
        "2. head_attached — A speaker label (e.g. 'SPEAKER_01:') was incorrectly "
        "embedded at the START of the segment's text field. The segment.speaker field "
        "may be empty or wrong.\n"
        "3. tail_attached — A speaker label (e.g. 'SPEAKER_02') was incorrectly "
        "appended to the END of the segment's text field."
    )


# ---------------------------------------------------------------------------
# Rule-based correction helpers (used by stage_02 and stage_03)
# ---------------------------------------------------------------------------

def strip_head_label(text: str) -> tuple[str, str]:
    """
    Remove a SPEAKER_XX: prefix from text.
    Returns (cleaned_text, extracted_label).  label is '' if none found.
    """
    m = re.match(r"^\s*(SPEAKER_\d+)\s*:\s*", text, re.IGNORECASE)
    if m:
        label = m.group(1).upper()
        cleaned = text[m.end():]
        return cleaned, label
    return text, ""


def strip_tail_label(text: str) -> tuple[str, str]:
    """
    Remove a SPEAKER_XX suffix from text.
    Returns (cleaned_text, extracted_label).  label is '' if none found.
    """
    m = re.search(r"\s+(SPEAKER_\d+)\s*$", text, re.IGNORECASE)
    if m:
        label = m.group(1).upper()
        cleaned = text[: m.start()]
        return cleaned, label
    return text, ""


def apply_head_attached_fix(transcript: dict) -> dict:
    """
    Rule-based fix for head_attached defects.
    Strips SPEAKER_XX: prefixes from all segment text fields and moves the
    label into the segment's speaker field (if speaker field is empty/wrong).
    Returns a new transcript dict (does not mutate the input).
    """
    import copy
    result = copy.deepcopy(transcript)
    for seg in result.get("segments", []):
        text = seg.get("text", "")
        cleaned, label = strip_head_label(text)
        if label:
            seg["text"] = cleaned
            if not seg.get("speaker"):
                seg["speaker"] = label
    return result


def apply_tail_attached_fix(transcript: dict) -> dict:
    """
    Rule-based fix for tail_attached defects.
    Strips SPEAKER_XX suffixes from all segment text fields and moves the
    label into the segment's speaker field (if speaker field is empty/wrong).
    Returns a new transcript dict (does not mutate the input).
    """
    import copy
    result = copy.deepcopy(transcript)
    for seg in result.get("segments", []):
        text = seg.get("text", "")
        cleaned, label = strip_tail_label(text)
        if label:
            seg["text"] = cleaned
            if not seg.get("speaker"):
                seg["speaker"] = label
    return result


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_correction(
    example: DiarizationExample | dict,
    prediction: CorrectionPrediction | dict,
) -> dict[str, Any]:
    """
    Evaluate a diarization correction against ground truth.

    Metrics:
      - defect_detected_correctly: bool
      - speaker_accuracy: fraction of segments with the correct speaker label
      - text_preserved: no hallucinated words (predicted words ⊆ ground truth words)
      - spurious_labels_removed: no SPEAKER_XX patterns remain in text fields
      - precision / recall / f1 over (segment_index, speaker) pairs
    """
    if isinstance(example, dict):
        ex_id = example.get("id", "?")
        ex_defect = example.get("defect_type", "unknown")
        gt_segs = example.get("ground_truth_transcript", {}).get("segments", [])
    else:
        ex_id = example.id
        ex_defect = example.defect_type
        gt_segs = example.ground_truth_transcript.get("segments", [])

    if isinstance(prediction, dict):
        pred_segs = prediction.get("segments", [])
        detected = prediction.get("defect_type_detected", "unknown")
        variant = prediction.get("variant", "unknown")
    else:
        pred_segs = prediction.segments
        detected = prediction.defect_type_detected
        variant = prediction.variant

    # --- defect detection ---
    defect_detected_correctly = detected == ex_defect

    # --- align by segment index (pad shorter with None) ---
    n = max(len(gt_segs), len(pred_segs))
    pairs = [
        (gt_segs[i] if i < len(gt_segs) else None,
         pred_segs[i] if i < len(pred_segs) else None)
        for i in range(n)
    ]

    # --- speaker accuracy ---
    correct_speakers = sum(
        1
        for gt, pr in pairs
        if gt and pr and (gt.get("speaker") or "").upper() == (pr.get("speaker") or "").upper()
    )
    segments_total = max(len(gt_segs), 1)
    speaker_accuracy = correct_speakers / segments_total

    # --- text preservation (no hallucination) ---
    gt_words: set[str] = set()
    for seg in gt_segs:
        for w in seg.get("text", "").lower().split():
            gt_words.add(re.sub(r"[^a-z0-9']", "", w))
    pred_words: set[str] = set()
    for seg in pred_segs:
        for w in seg.get("text", "").lower().split():
            pred_words.add(re.sub(r"[^a-z0-9']", "", w))

    extra_words = pred_words - gt_words - {""}
    text_preserved = len(extra_words) == 0

    # --- spurious labels removed ---
    spurious_labels_removed = not any(
        _ANY_LABEL_PATTERN.search(seg.get("text", ""))
        for seg in pred_segs
    )

    # --- P/R/F1 over (segment_index, speaker) pairs ---
    gt_pairs = {
        (i, (seg.get("speaker") or "").upper())
        for i, seg in enumerate(gt_segs)
        if seg.get("speaker")
    }
    pred_pairs = {
        (i, (seg.get("speaker") or "").upper())
        for i, seg in enumerate(pred_segs)
        if seg.get("speaker")
    }

    tp = len(gt_pairs & pred_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "id": ex_id,
        "variant": variant,
        "defect_type": ex_defect,
        "defect_detected_correctly": defect_detected_correctly,
        "segments_correct": correct_speakers,
        "segments_total": segments_total,
        "speaker_accuracy": round(speaker_accuracy, 3),
        "text_preserved": text_preserved,
        "spurious_labels_removed": spurious_labels_removed,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate evaluation metrics across multiple examples."""
    if not results:
        return {
            "cases": 0,
            "speaker_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "defect_detection_rate": 0.0,
            "text_preserved_rate": 0.0,
            "spurious_labels_removed_rate": 0.0,
        }

    n = len(results)
    tp = sum(r["segments_correct"] for r in results)
    total_segs = sum(r["segments_total"] for r in results)

    # Micro-average P/R/F1
    all_tp = sum(
        int(r["precision"] * (r["segments_total"]) + 0.5)  # approx from stored values
        for r in results
    )
    precision = sum(r["precision"] for r in results) / n
    recall = sum(r["recall"] for r in results) / n
    f1 = sum(r["f1"] for r in results) / n

    return {
        "cases": n,
        "speaker_accuracy": round(tp / max(total_segs, 1), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "defect_detection_rate": round(
            sum(1 for r in results if r["defect_detected_correctly"]) / n, 3
        ),
        "text_preserved_rate": round(
            sum(1 for r in results if r["text_preserved"]) / n, 3
        ),
        "spurious_labels_removed_rate": round(
            sum(1 for r in results if r["spurious_labels_removed"]) / n, 3
        ),
    }
