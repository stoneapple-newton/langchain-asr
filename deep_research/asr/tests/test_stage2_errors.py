"""Tests for stage_02_llm_enhancement/03_error_correction.py"""

import copy
import pytest
from .conftest import MINIMAL_SEGMENTS


# ---------------------------------------------------------------------------
# find_error_candidates
# ---------------------------------------------------------------------------

def test_finds_low_conf_words(stage2_errors_mod):
    # MINIMAL_SEGMENTS has: authentication(0.68), JWT(0.71) both < 0.75 default
    candidates = stage2_errors_mod.find_error_candidates(MINIMAL_SEGMENTS)
    assert len(candidates) == 2


def test_candidate_word_values(stage2_errors_mod):
    candidates = stage2_errors_mod.find_error_candidates(MINIMAL_SEGMENTS)
    words = {c.word for c in candidates}
    assert "authentication" in words
    assert "JWT" in words


def test_finds_nothing_above_threshold(stage2_errors_mod):
    # Set threshold so low that nothing qualifies
    candidates = stage2_errors_mod.find_error_candidates(MINIMAL_SEGMENTS, threshold=0.5)
    assert len(candidates) == 0


def test_candidate_context_populated(stage2_errors_mod):
    candidates = stage2_errors_mod.find_error_candidates(MINIMAL_SEGMENTS)
    # authentication is at word_idx=2 in seg 2 — should have left_context words
    auth_candidates = [c for c in candidates if c.word == "authentication"]
    assert len(auth_candidates) == 1
    c = auth_candidates[0]
    assert c.segment_idx == 2
    assert c.left_context != ""  # has "yeah the" before


def test_candidate_segment_idx_correct(stage2_errors_mod):
    candidates = stage2_errors_mod.find_error_candidates(MINIMAL_SEGMENTS)
    jwt_cands = [c for c in candidates if c.word == "JWT"]
    assert len(jwt_cands) == 1
    assert jwt_cands[0].segment_idx == 4


# ---------------------------------------------------------------------------
# apply_corrections — pure dict manipulation, no LLM
# ---------------------------------------------------------------------------

def _make_correction(original, corrected, confidence=0.9):
    return {"original": original, "corrected": corrected, "confidence": confidence, "reason": "test"}


def test_apply_corrections_updates_word(stage2_errors_mod):
    candidates = stage2_errors_mod.find_error_candidates(MINIMAL_SEGMENTS)
    corrections = [_make_correction(c.word, c.word.upper(), confidence=0.95) for c in candidates]
    result = stage2_errors_mod.apply_corrections(MINIMAL_SEGMENTS, candidates, corrections)
    # JWT in seg 4 word_idx=2 should be updated
    jwt_cands = [c for c in candidates if c.word == "JWT"]
    if jwt_cands:
        c = jwt_cands[0]
        assert result[c.segment_idx]["words"][c.word_idx]["word"] == "JWT"


def test_apply_corrections_does_not_mutate_input(stage2_errors_mod):
    original = copy.deepcopy(MINIMAL_SEGMENTS)
    candidates = stage2_errors_mod.find_error_candidates(MINIMAL_SEGMENTS)
    corrections = [_make_correction(c.word, "replacement", confidence=0.95) for c in candidates]
    stage2_errors_mod.apply_corrections(MINIMAL_SEGMENTS, candidates, corrections)
    # Original should be unchanged
    for i, seg in enumerate(MINIMAL_SEGMENTS):
        assert seg["text"] == original[i]["text"]


def test_apply_corrections_skips_low_confidence(stage2_errors_mod):
    original = copy.deepcopy(MINIMAL_SEGMENTS)
    candidates = stage2_errors_mod.find_error_candidates(MINIMAL_SEGMENTS)
    corrections = [_make_correction(c.word, "CHANGED", confidence=0.5) for c in candidates]
    result = stage2_errors_mod.apply_corrections(MINIMAL_SEGMENTS, candidates, corrections, min_confidence=0.7)
    # Nothing should be changed since confidence=0.5 < min_confidence=0.7
    for i, seg in enumerate(result):
        assert seg["text"] == original[i]["text"]


def test_apply_corrections_skips_unchanged_word(stage2_errors_mod):
    original = copy.deepcopy(MINIMAL_SEGMENTS)
    candidates = stage2_errors_mod.find_error_candidates(MINIMAL_SEGMENTS)
    # Return same word (no change)
    corrections = [_make_correction(c.word, c.word, confidence=0.99) for c in candidates]
    result = stage2_errors_mod.apply_corrections(MINIMAL_SEGMENTS, candidates, corrections)
    for i, seg in enumerate(result):
        assert seg["text"] == original[i]["text"]
