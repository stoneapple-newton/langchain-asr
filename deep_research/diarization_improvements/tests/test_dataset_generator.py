"""Tests for stage_01_dataset_generator — no LLM required."""
import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import CLEAN_SEGMENTS, load_diarization_module

# Load the generator module (filename starts with digit, so use importlib helper)
_gen = load_diarization_module("stage_01_dataset_generator/01_generate_dataset.py")
inject_run_on = _gen.inject_run_on
inject_head_attached = _gen.inject_head_attached
inject_tail_attached = _gen.inject_tail_attached
generate_dataset = _gen.generate_dataset


# ---------------------------------------------------------------------------
# inject_run_on
# ---------------------------------------------------------------------------

class TestInjectRunOn:
    def test_reduces_segment_count_by_one(self):
        defective, gt = inject_run_on(CLEAN_SEGMENTS, 0, 1)
        assert len(defective) == len(CLEAN_SEGMENTS) - 1
        assert len(gt) == len(CLEAN_SEGMENTS)

    def test_merged_text_contains_both_utterances(self):
        defective, gt = inject_run_on(CLEAN_SEGMENTS, 0, 1)
        merged_text = defective[0]["text"]
        assert "Alright" in merged_text or "let" in merged_text
        assert "agenda" in merged_text or "roadmap" in merged_text

    def test_merged_speaker_is_first_speaker(self):
        defective, gt = inject_run_on(CLEAN_SEGMENTS, 0, 1)
        assert defective[0]["speaker"] == CLEAN_SEGMENTS[0]["speaker"]

    def test_word_count_preserved(self):
        defective, gt = inject_run_on(CLEAN_SEGMENTS, 1, 2)
        orig_words = len(CLEAN_SEGMENTS[1].get("words", [])) + len(CLEAN_SEGMENTS[2].get("words", []))
        merged_words = len(defective[1].get("words", []))
        assert merged_words == orig_words

    def test_timing_spans_both_segments(self):
        defective, gt = inject_run_on(CLEAN_SEGMENTS, 0, 1)
        assert defective[0]["start"] == CLEAN_SEGMENTS[0]["start"]
        assert defective[0]["end"] == CLEAN_SEGMENTS[1]["end"]

    def test_ground_truth_unchanged(self):
        defective, gt = inject_run_on(CLEAN_SEGMENTS, 0, 1)
        assert gt[0]["speaker"] == CLEAN_SEGMENTS[0]["speaker"]
        assert gt[1]["speaker"] == CLEAN_SEGMENTS[1]["speaker"]
        assert len(gt) == len(CLEAN_SEGMENTS)

    def test_does_not_mutate_input(self):
        original = copy.deepcopy(CLEAN_SEGMENTS)
        inject_run_on(CLEAN_SEGMENTS, 0, 1)
        assert CLEAN_SEGMENTS[0]["text"] == original[0]["text"]


# ---------------------------------------------------------------------------
# inject_head_attached
# ---------------------------------------------------------------------------

class TestInjectHeadAttached:
    def test_text_starts_with_speaker_label(self):
        defective, gt = inject_head_attached(CLEAN_SEGMENTS, 2)
        seg = defective[2]
        assert "SPEAKER_" in seg["text"][:20]

    def test_label_colon_separator(self):
        defective, gt = inject_head_attached(CLEAN_SEGMENTS, 2)
        seg = defective[2]
        assert ":" in seg["text"]

    def test_speaker_field_cleared(self):
        defective, gt = inject_head_attached(CLEAN_SEGMENTS, 2)
        assert defective[2]["speaker"] == ""

    def test_other_segments_unchanged(self):
        defective, gt = inject_head_attached(CLEAN_SEGMENTS, 2)
        for i, (d, c) in enumerate(zip(defective, CLEAN_SEGMENTS)):
            if i != 2:
                assert d["speaker"] == c["speaker"]

    def test_ground_truth_speaker_preserved(self):
        defective, gt = inject_head_attached(CLEAN_SEGMENTS, 2)
        assert gt[2]["speaker"] == CLEAN_SEGMENTS[2]["speaker"]

    def test_does_not_mutate_input(self):
        original_text = CLEAN_SEGMENTS[2]["text"]
        inject_head_attached(CLEAN_SEGMENTS, 2)
        assert CLEAN_SEGMENTS[2]["text"] == original_text


# ---------------------------------------------------------------------------
# inject_tail_attached
# ---------------------------------------------------------------------------

class TestInjectTailAttached:
    def test_text_ends_with_speaker_label(self):
        import re
        defective, gt = inject_tail_attached(CLEAN_SEGMENTS, 4)
        seg = defective[4]
        assert re.search(r"SPEAKER_\d+\s*$", seg["text"])

    def test_speaker_field_cleared(self):
        defective, gt = inject_tail_attached(CLEAN_SEGMENTS, 4)
        assert defective[4]["speaker"] == ""

    def test_original_words_still_present(self):
        defective, gt = inject_tail_attached(CLEAN_SEGMENTS, 4)
        seg_text = defective[4]["text"]
        # The start of the text should still contain original content
        assert "DevOps" in seg_text or "needs" in seg_text

    def test_ground_truth_speaker_preserved(self):
        defective, gt = inject_tail_attached(CLEAN_SEGMENTS, 4)
        assert gt[4]["speaker"] == CLEAN_SEGMENTS[4]["speaker"]

    def test_does_not_mutate_input(self):
        original_text = CLEAN_SEGMENTS[4]["text"]
        inject_tail_attached(CLEAN_SEGMENTS, 4)
        assert CLEAN_SEGMENTS[4]["text"] == original_text


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------

class TestGenerateDataset:
    def test_produces_thirty_examples(self):
        examples = generate_dataset()
        assert len(examples) == 30

    def test_ten_of_each_type(self):
        examples = generate_dataset()
        by_type: dict[str, int] = {}
        for ex in examples:
            by_type[ex["defect_type"]] = by_type.get(ex["defect_type"], 0) + 1
        assert by_type.get("run_on", 0) == 10
        assert by_type.get("head_attached", 0) == 10
        assert by_type.get("tail_attached", 0) == 10

    def test_deterministic_same_seed(self):
        examples_a = generate_dataset(seed=99)
        examples_b = generate_dataset(seed=99)
        for a, b in zip(examples_a, examples_b):
            assert a["id"] == b["id"]
            assert a["metadata"] == b["metadata"]

    def test_different_seed_produces_different_examples(self):
        examples_a = generate_dataset(seed=1)
        examples_b = generate_dataset(seed=2)
        diffs = [
            i for i, (a, b) in enumerate(zip(examples_a, examples_b))
            if a["metadata"] != b["metadata"]
        ]
        assert len(diffs) > 0

    def test_each_example_has_required_fields(self):
        for ex in generate_dataset():
            assert "id" in ex
            assert "defect_type" in ex
            assert "input_transcript" in ex
            assert "ground_truth_transcript" in ex
            assert "metadata" in ex
            assert "segments" in ex["input_transcript"]
            assert "segments" in ex["ground_truth_transcript"]
