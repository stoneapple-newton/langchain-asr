"""Tests for shared/diarization_utils.py — no LLM required."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.diarization_improvements.shared.diarization_utils import (
    CorrectionPrediction,
    DiarizationExample,
    detect_defect_type,
    evaluate_correction,
    format_defect_descriptions,
    render_segment_block,
    scan_embedded_labels,
    strip_head_label,
    strip_tail_label,
    summarize_results,
)
from conftest import (
    CLEAN_SEGMENTS,
    HEAD_ATTACHED_TRANSCRIPT,
    RUN_ON_TRANSCRIPT,
    TAIL_ATTACHED_TRANSCRIPT,
    FAKE_TRANSCRIPT,
)


# ---------------------------------------------------------------------------
# detect_defect_type
# ---------------------------------------------------------------------------

class TestDetectDefectType:
    def test_detects_head_attached(self):
        assert detect_defect_type(HEAD_ATTACHED_TRANSCRIPT) == "head_attached"

    def test_detects_tail_attached(self):
        assert detect_defect_type(TAIL_ATTACHED_TRANSCRIPT) == "tail_attached"

    def test_detects_run_on_via_empty_speaker(self):
        # A segment with no speaker field is treated as a run-on candidate
        t = {
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "hello world", "speaker": ""},
            ]
        }
        assert detect_defect_type(t) == "run_on"

    def test_clean_transcript_returns_unknown(self):
        assert detect_defect_type(FAKE_TRANSCRIPT) == "unknown"

    def test_empty_transcript(self):
        assert detect_defect_type({"segments": []}) == "unknown"

    def test_head_takes_priority_over_tail(self):
        # Both head and tail patterns — head should win
        t = {
            "segments": [
                {"start": 0.0, "end": 5.0,
                 "text": "SPEAKER_00: hello SPEAKER_01", "speaker": ""},
            ]
        }
        assert detect_defect_type(t) == "head_attached"


# ---------------------------------------------------------------------------
# scan_embedded_labels
# ---------------------------------------------------------------------------

class TestScanEmbeddedLabels:
    def test_finds_head_label(self):
        hits = scan_embedded_labels(HEAD_ATTACHED_TRANSCRIPT)
        assert len(hits) >= 1
        assert hits[0]["position"] == "head"

    def test_finds_tail_label(self):
        hits = scan_embedded_labels(TAIL_ATTACHED_TRANSCRIPT)
        assert len(hits) >= 1
        assert hits[0]["position"] == "tail"

    def test_clean_returns_empty(self):
        assert scan_embedded_labels(FAKE_TRANSCRIPT) == []


# ---------------------------------------------------------------------------
# strip_head_label / strip_tail_label
# ---------------------------------------------------------------------------

class TestLabelStripping:
    def test_strip_head_basic(self):
        cleaned, label = strip_head_label("SPEAKER_01: hello world")
        assert label == "SPEAKER_01"
        assert "SPEAKER_01" not in cleaned
        assert "hello world" in cleaned

    def test_strip_head_with_whitespace(self):
        cleaned, label = strip_head_label("  SPEAKER_02:   text here")
        assert label == "SPEAKER_02"

    def test_strip_head_no_match(self):
        cleaned, label = strip_head_label("hello world")
        assert label == ""
        assert cleaned == "hello world"

    def test_strip_tail_basic(self):
        cleaned, label = strip_tail_label("hello world SPEAKER_00")
        assert label == "SPEAKER_00"
        assert "SPEAKER_00" not in cleaned
        assert "hello world" in cleaned

    def test_strip_tail_no_match(self):
        cleaned, label = strip_tail_label("hello world")
        assert label == ""
        assert cleaned == "hello world"


# ---------------------------------------------------------------------------
# evaluate_correction
# ---------------------------------------------------------------------------

def _make_example(defect_type: str, input_segs: list[dict], gt_segs: list[dict]) -> DiarizationExample:
    return DiarizationExample(
        id="test_00",
        defect_type=defect_type,
        input_transcript={"segments": input_segs},
        ground_truth_transcript={"segments": gt_segs},
    )


class TestEvaluateCorrection:
    def test_perfect_correction_f1_one(self):
        gt_segs = [
            {"speaker": "SPEAKER_00", "text": "hello"},
            {"speaker": "SPEAKER_01", "text": "world"},
        ]
        example = _make_example("run_on", [], gt_segs)
        prediction = CorrectionPrediction(
            segments=gt_segs,
            defect_type_detected="run_on",
            variant="test",
        )
        result = evaluate_correction(example, prediction)
        assert result["f1"] == 1.0
        assert result["speaker_accuracy"] == 1.0
        assert result["defect_detected_correctly"] is True

    def test_wrong_speaker_lowers_accuracy(self):
        gt_segs = [
            {"speaker": "SPEAKER_00", "text": "hello"},
            {"speaker": "SPEAKER_01", "text": "world"},
        ]
        example = _make_example("run_on", [], gt_segs)
        pred_segs = [
            {"speaker": "SPEAKER_00", "text": "hello"},
            {"speaker": "SPEAKER_00", "text": "world"},  # wrong
        ]
        prediction = CorrectionPrediction(
            segments=pred_segs,
            defect_type_detected="run_on",
            variant="test",
        )
        result = evaluate_correction(example, prediction)
        assert result["speaker_accuracy"] < 1.0
        assert result["f1"] < 1.0

    def test_text_preserved_true_for_subset(self):
        gt_segs = [{"speaker": "SPEAKER_00", "text": "hello world foo"}]
        example = _make_example("head_attached", [], gt_segs)
        prediction = CorrectionPrediction(
            segments=[{"speaker": "SPEAKER_00", "text": "hello world"}],
            defect_type_detected="head_attached",
            variant="test",
        )
        result = evaluate_correction(example, prediction)
        assert result["text_preserved"] is True

    def test_text_preserved_false_for_hallucination(self):
        gt_segs = [{"speaker": "SPEAKER_00", "text": "hello world"}]
        example = _make_example("head_attached", [], gt_segs)
        prediction = CorrectionPrediction(
            segments=[{"speaker": "SPEAKER_00", "text": "hello world totally_new_word_xyz"}],
            defect_type_detected="head_attached",
            variant="test",
        )
        result = evaluate_correction(example, prediction)
        assert result["text_preserved"] is False

    def test_spurious_labels_removed_true(self):
        gt_segs = [{"speaker": "SPEAKER_00", "text": "hello"}]
        example = _make_example("tail_attached", [], gt_segs)
        prediction = CorrectionPrediction(
            segments=[{"speaker": "SPEAKER_00", "text": "hello"}],
            defect_type_detected="tail_attached",
            variant="test",
        )
        result = evaluate_correction(example, prediction)
        assert result["spurious_labels_removed"] is True

    def test_spurious_labels_removed_false_when_label_remains(self):
        gt_segs = [{"speaker": "SPEAKER_00", "text": "hello"}]
        example = _make_example("tail_attached", [], gt_segs)
        prediction = CorrectionPrediction(
            segments=[{"speaker": "SPEAKER_00", "text": "hello SPEAKER_00"}],
            defect_type_detected="tail_attached",
            variant="test",
        )
        result = evaluate_correction(example, prediction)
        assert result["spurious_labels_removed"] is False

    def test_wrong_defect_type_detected(self):
        gt_segs = [{"speaker": "SPEAKER_00", "text": "hello"}]
        example = _make_example("run_on", [], gt_segs)
        prediction = CorrectionPrediction(
            segments=gt_segs,
            defect_type_detected="head_attached",
            variant="test",
        )
        result = evaluate_correction(example, prediction)
        assert result["defect_detected_correctly"] is False


# ---------------------------------------------------------------------------
# summarize_results
# ---------------------------------------------------------------------------

class TestSummarizeResults:
    def test_empty_returns_zeros(self):
        s = summarize_results([])
        assert s["cases"] == 0
        assert s["f1"] == 0.0

    def test_perfect_results(self):
        r = {
            "segments_correct": 2,
            "segments_total": 2,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "defect_detected_correctly": True,
            "text_preserved": True,
            "spurious_labels_removed": True,
        }
        s = summarize_results([r, r])
        assert s["cases"] == 2
        assert s["f1"] == 1.0
        assert s["defect_detection_rate"] == 1.0
        assert s["text_preserved_rate"] == 1.0


# ---------------------------------------------------------------------------
# prompt helpers
# ---------------------------------------------------------------------------

class TestPromptHelpers:
    def test_render_segment_block(self):
        block = render_segment_block(CLEAN_SEGMENTS, limit=3)
        assert "[0]" in block
        assert "[1]" in block
        assert "[2]" in block
        assert "[3]" not in block

    def test_format_defect_descriptions(self):
        desc = format_defect_descriptions()
        assert "run_on" in desc
        assert "head_attached" in desc
        assert "tail_attached" in desc
