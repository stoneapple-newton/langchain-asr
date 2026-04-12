"""Tests for stage_03_diarization/01_speaker_analysis.py"""

import copy
import pytest
from .conftest import MINIMAL_SEGMENTS


# ---------------------------------------------------------------------------
# build_turns
# ---------------------------------------------------------------------------

def test_build_turns_count(stage3_analysis_mod):
    # MINIMAL_SEGMENTS: S00, S00, S01, S02, S01 → merges S00+S00 → 4 turns
    turns = stage3_analysis_mod.build_turns(MINIMAL_SEGMENTS)
    assert len(turns) == 4


def test_build_turns_speakers(stage3_analysis_mod):
    turns = stage3_analysis_mod.build_turns(MINIMAL_SEGMENTS)
    speakers = [t.speaker for t in turns]
    assert speakers == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_01"]


def test_build_turns_merged_duration(stage3_analysis_mod):
    turns = stage3_analysis_mod.build_turns(MINIMAL_SEGMENTS)
    # First turn merges segs 0 and 1: start=0.0, end=10.0
    assert turns[0].start == pytest.approx(0.0)
    assert turns[0].end == pytest.approx(10.0)


def test_build_turns_empty(stage3_analysis_mod):
    turns = stage3_analysis_mod.build_turns([])
    assert turns == []


def test_build_turns_single(stage3_analysis_mod):
    segs = [MINIMAL_SEGMENTS[0]]
    turns = stage3_analysis_mod.build_turns(segs)
    assert len(turns) == 1
    assert turns[0].speaker == "SPEAKER_00"


# ---------------------------------------------------------------------------
# compute_speaker_stats
# ---------------------------------------------------------------------------

def test_compute_speaker_stats_keys(stage3_analysis_mod):
    turns = stage3_analysis_mod.build_turns(MINIMAL_SEGMENTS)
    stats = stage3_analysis_mod.compute_speaker_stats(turns)
    assert set(stats.keys()) == {"SPEAKER_00", "SPEAKER_01", "SPEAKER_02"}


def test_compute_speaker_stats_turn_count(stage3_analysis_mod):
    turns = stage3_analysis_mod.build_turns(MINIMAL_SEGMENTS)
    stats = stage3_analysis_mod.compute_speaker_stats(turns)
    # SPEAKER_00 appears in 1 merged turn, SPEAKER_01 in 2 turns, SPEAKER_02 in 1
    assert stats["SPEAKER_00"].turn_count == 1
    assert stats["SPEAKER_01"].turn_count == 2
    assert stats["SPEAKER_02"].turn_count == 1


# ---------------------------------------------------------------------------
# build_transition_matrix
# ---------------------------------------------------------------------------

def test_build_transition_matrix_entries(stage3_analysis_mod):
    turns = stage3_analysis_mod.build_turns(MINIMAL_SEGMENTS)
    matrix = stage3_analysis_mod.build_transition_matrix(turns)
    # Transitions: S00→S01, S01→S02, S02→S01 = 3 entries, each count=1
    assert sum(matrix.values()) == 3


def test_build_transition_matrix_no_self_loop(stage3_analysis_mod):
    turns = stage3_analysis_mod.build_turns(MINIMAL_SEGMENTS)
    matrix = stage3_analysis_mod.build_transition_matrix(turns)
    for (a, b), count in matrix.items():
        assert a != b, f"Self-loop found: {a} → {b}"


# ---------------------------------------------------------------------------
# detect_anomalies
# ---------------------------------------------------------------------------

def test_detect_anomalies_fast_switch(stage3_analysis_mod):
    turns = stage3_analysis_mod.build_turns(MINIMAL_SEGMENTS)
    stats = stage3_analysis_mod.compute_speaker_stats(turns)
    anomalies = stage3_analysis_mod.detect_anomalies(MINIMAL_SEGMENTS, turns, stats)
    fast_switch = [a for a in anomalies if a.type == "FAST_SWITCH"]
    # Seg 2 starts at 10.02, seg 1 ends at 10.0 → gap=0.02 < 0.1
    assert len(fast_switch) >= 1
    assert fast_switch[0].segment_idx == 2


def test_detect_anomalies_short_turn(stage3_analysis_mod):
    turns = stage3_analysis_mod.build_turns(MINIMAL_SEGMENTS)
    stats = stage3_analysis_mod.compute_speaker_stats(turns)
    anomalies = stage3_analysis_mod.detect_anomalies(MINIMAL_SEGMENTS, turns, stats)
    short = [a for a in anomalies if a.type == "SHORT_TURN"]
    # SPEAKER_02 turn: 1.0s, 2 words → SHORT_TURN
    assert len(short) >= 1


def test_detect_anomalies_no_same_speaker_adjacent(stage3_analysis_mod):
    turns = stage3_analysis_mod.build_turns(MINIMAL_SEGMENTS)
    stats = stage3_analysis_mod.compute_speaker_stats(turns)
    anomalies = stage3_analysis_mod.detect_anomalies(MINIMAL_SEGMENTS, turns, stats)
    same_adj = [a for a in anomalies if a.type == "SAME_SPEAKER_ADJACENT"]
    assert len(same_adj) == 0
