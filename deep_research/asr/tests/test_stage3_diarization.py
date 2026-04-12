"""Tests for stage_03_diarization/02_diarization_correction.py"""

import copy
import pytest
from .conftest import MINIMAL_SEGMENTS


# ---------------------------------------------------------------------------
# render_context_window
# ---------------------------------------------------------------------------

def test_render_context_window_center_marked(stage3_diarize_mod):
    result = stage3_diarize_mod.render_context_window(MINIMAL_SEGMENTS, center_idx=2, window=1)
    lines = result.split("\n")
    # Center line (idx=2) should start with ">>>"
    center_lines = [l for l in lines if ">>>" in l]
    assert len(center_lines) == 1
    assert "[02]" in center_lines[0]


def test_render_context_window_non_center_marked(stage3_diarize_mod):
    result = stage3_diarize_mod.render_context_window(MINIMAL_SEGMENTS, center_idx=2, window=1)
    lines = result.split("\n")
    non_center = [l for l in lines if ">>>" not in l]
    # All non-center lines start with "   "
    for line in non_center:
        assert line.startswith("   ")


def test_render_context_window_uses_name_map(stage3_diarize_mod):
    name_map = {"SPEAKER_01": "Bob Nakamura"}
    result = stage3_diarize_mod.render_context_window(
        MINIMAL_SEGMENTS, center_idx=2, window=0, name_map=name_map
    )
    assert "Bob Nakamura" in result


def test_render_context_window_clamps_start(stage3_diarize_mod):
    # center_idx=0, window=3 — should not go negative
    result = stage3_diarize_mod.render_context_window(MINIMAL_SEGMENTS, center_idx=0, window=3)
    assert result.count("[00]") == 1


def test_render_context_window_clamps_end(stage3_diarize_mod):
    last = len(MINIMAL_SEGMENTS) - 1
    result = stage3_diarize_mod.render_context_window(MINIMAL_SEGMENTS, center_idx=last, window=3)
    # Should not crash; last index is the center
    assert f"[0{last}]" in result


# ---------------------------------------------------------------------------
# find_suspicious_segments
# ---------------------------------------------------------------------------

def test_find_suspicious_segments_fast_switch(stage3_diarize_mod):
    # Seg 2 (SPEAKER_01) starts 0.02s after seg 1 (SPEAKER_00) ends → gap < 0.15 → suspicious
    result = stage3_diarize_mod.find_suspicious_segments(MINIMAL_SEGMENTS)
    assert 2 in result


def test_find_suspicious_segments_short_segment(stage3_diarize_mod):
    # Seg 3 has 2 words and is not first/last → suspicious
    result = stage3_diarize_mod.find_suspicious_segments(MINIMAL_SEGMENTS)
    assert 3 in result


def test_find_suspicious_segments_returns_list(stage3_diarize_mod):
    result = stage3_diarize_mod.find_suspicious_segments(MINIMAL_SEGMENTS)
    assert isinstance(result, list)


def test_find_suspicious_segments_no_duplicates(stage3_diarize_mod):
    result = stage3_diarize_mod.find_suspicious_segments(MINIMAL_SEGMENTS)
    # set() is applied internally, so no dupes
    assert len(result) == len(set(result))


# ---------------------------------------------------------------------------
# merge_backchannels
# ---------------------------------------------------------------------------

def test_merge_backchannels_same_speaker(stage3_diarize_mod):
    # Build segments where a short 2-word segment has the SAME speaker as prev → should merge
    segs = [
        {"start": 0.0, "end": 5.0, "text": " Hello there", "speaker": "SPEAKER_00",
         "words": [
             {"word": "Hello", "start": 0.0, "end": 0.5, "score": 0.99, "speaker": "SPEAKER_00"},
             {"word": "there", "start": 0.6, "end": 1.0, "score": 0.99, "speaker": "SPEAKER_00"},
         ]},
        {"start": 5.1, "end": 6.0, "text": " yeah", "speaker": "SPEAKER_00",
         "words": [
             {"word": "yeah", "start": 5.1, "end": 5.5, "score": 0.95, "speaker": "SPEAKER_00"},
         ]},
        {"start": 6.5, "end": 10.0, "text": " That is correct.", "speaker": "SPEAKER_01",
         "words": [
             {"word": "That", "start": 6.5, "end": 6.8, "score": 0.99, "speaker": "SPEAKER_01"},
             {"word": "is", "start": 6.9, "end": 7.0, "score": 0.99, "speaker": "SPEAKER_01"},
             {"word": "correct", "start": 7.1, "end": 7.6, "score": 0.99, "speaker": "SPEAKER_01"},
         ]},
    ]
    result = stage3_diarize_mod.merge_backchannels(segs, max_words=2, max_duration=2.0)
    assert len(result) == 2  # seg 1 merged into seg 0


def test_merge_backchannels_different_speaker_not_merged(stage3_diarize_mod):
    # MINIMAL_SEGMENTS seg 3 (SPEAKER_02, 2 words) follows seg 2 (SPEAKER_01) → different speaker → NOT merged
    result = stage3_diarize_mod.merge_backchannels(MINIMAL_SEGMENTS, max_words=2, max_duration=2.0)
    assert len(result) == len(MINIMAL_SEGMENTS)


def test_merge_backchannels_does_not_mutate_input(stage3_diarize_mod):
    original = copy.deepcopy(MINIMAL_SEGMENTS)
    stage3_diarize_mod.merge_backchannels(MINIMAL_SEGMENTS, max_words=2, max_duration=2.0)
    assert MINIMAL_SEGMENTS[0]["text"] == original[0]["text"]
