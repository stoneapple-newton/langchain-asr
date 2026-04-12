"""Tests for stage_02 one-shot scripts — no live LLM required."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import (
    HEAD_ATTACHED_TRANSCRIPT,
    TAIL_ATTACHED_TRANSCRIPT,
    FAKE_TRANSCRIPT,
    load_diarization_module,
    stage2_head_mod,
    stage2_tail_mod,
)


# ---------------------------------------------------------------------------
# head_attached script — rule-based, no LLM call needed
# ---------------------------------------------------------------------------

class TestFixHeadAttached:
    def test_strips_speaker_label_from_text(self, stage2_head_mod):
        result = stage2_head_mod.fix_head_attached(HEAD_ATTACHED_TRANSCRIPT)
        for seg in result["segments"]:
            text = seg.get("text", "")
            # No SPEAKER_XX: prefix should remain
            assert not text.lstrip().startswith("SPEAKER_"), \
                f"Label still in text: {text[:60]}"

    def test_sets_speaker_field(self, stage2_head_mod):
        result = stage2_head_mod.fix_head_attached(HEAD_ATTACHED_TRANSCRIPT)
        # All segments should have a non-empty speaker field now
        for seg in result["segments"]:
            # The originally cleared segment should now have a speaker
            assert seg.get("speaker") is not None

    def test_correct_variant_name(self, stage2_head_mod):
        result = stage2_head_mod.fix_head_attached(HEAD_ATTACHED_TRANSCRIPT)
        assert result["variant"] == "one_shot_head_attached"

    def test_defect_detected(self, stage2_head_mod):
        result = stage2_head_mod.fix_head_attached(HEAD_ATTACHED_TRANSCRIPT)
        assert result["defect_type_detected"] == "head_attached"

    def test_segment_count_unchanged(self, stage2_head_mod):
        result = stage2_head_mod.fix_head_attached(HEAD_ATTACHED_TRANSCRIPT)
        assert len(result["segments"]) == len(HEAD_ATTACHED_TRANSCRIPT["segments"])

    def test_notes_non_empty(self, stage2_head_mod):
        result = stage2_head_mod.fix_head_attached(HEAD_ATTACHED_TRANSCRIPT)
        assert isinstance(result["notes"], list)
        assert len(result["notes"]) > 0

    def test_clean_transcript_no_change(self, stage2_head_mod):
        result = stage2_head_mod.fix_head_attached(FAKE_TRANSCRIPT)
        # Nothing to strip — segments should be unchanged
        for orig, fixed in zip(FAKE_TRANSCRIPT["segments"], result["segments"]):
            assert orig["text"] == fixed["text"]


# ---------------------------------------------------------------------------
# tail_attached script — rule-based, no LLM call needed
# ---------------------------------------------------------------------------

class TestFixTailAttached:
    def test_strips_speaker_label_from_text(self, stage2_tail_mod):
        import re
        result = stage2_tail_mod.fix_tail_attached(TAIL_ATTACHED_TRANSCRIPT)
        for seg in result["segments"]:
            text = seg.get("text", "")
            assert not re.search(r"SPEAKER_\d+\s*$", text), \
                f"Label still at end of text: {text[:60]}"

    def test_sets_speaker_field(self, stage2_tail_mod):
        result = stage2_tail_mod.fix_tail_attached(TAIL_ATTACHED_TRANSCRIPT)
        for seg in result["segments"]:
            assert seg.get("speaker") is not None

    def test_correct_variant_name(self, stage2_tail_mod):
        result = stage2_tail_mod.fix_tail_attached(TAIL_ATTACHED_TRANSCRIPT)
        assert result["variant"] == "one_shot_tail_attached"

    def test_defect_detected(self, stage2_tail_mod):
        result = stage2_tail_mod.fix_tail_attached(TAIL_ATTACHED_TRANSCRIPT)
        assert result["defect_type_detected"] == "tail_attached"

    def test_segment_count_unchanged(self, stage2_tail_mod):
        result = stage2_tail_mod.fix_tail_attached(TAIL_ATTACHED_TRANSCRIPT)
        assert len(result["segments"]) == len(TAIL_ATTACHED_TRANSCRIPT["segments"])

    def test_clean_transcript_no_change(self, stage2_tail_mod):
        result = stage2_tail_mod.fix_tail_attached(FAKE_TRANSCRIPT)
        for orig, fixed in zip(FAKE_TRANSCRIPT["segments"], result["segments"]):
            assert orig["text"] == fixed["text"]


# ---------------------------------------------------------------------------
# run_on script — test non-LLM parts only (prompt construction, helpers)
# ---------------------------------------------------------------------------

class TestFixRunOnHelpers:
    def test_get_known_speakers(self, stage2_run_on_mod):
        speakers = stage2_run_on_mod._get_known_speakers(FAKE_TRANSCRIPT)
        assert "SPEAKER_00" in speakers
        assert "SPEAKER_01" in speakers
        assert "SPEAKER_02" in speakers

    def test_get_known_speakers_empty(self, stage2_run_on_mod):
        speakers = stage2_run_on_mod._get_known_speakers({"segments": []})
        assert speakers == []

    def test_correct_transcript_alias_exists(self, stage2_run_on_mod):
        assert hasattr(stage2_run_on_mod, "correct_transcript")
        assert callable(stage2_run_on_mod.correct_transcript)
