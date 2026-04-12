"""Tests for stage_04_langgraph_pipeline/02_quality_loop.py"""

import copy
import pytest
from .conftest import MINIMAL_SEGMENTS


# ---------------------------------------------------------------------------
# compute_quality
# ---------------------------------------------------------------------------

def test_compute_quality_returns_keys(stage4_loop_mod):
    q = stage4_loop_mod.compute_quality(MINIMAL_SEGMENTS)
    assert "avg_confidence" in q
    assert "filler_rate" in q
    assert "punctuation_score" in q
    assert "capitalisation_score" in q


def test_compute_quality_filler_rate_stage4(stage4_loop_mod):
    # Stage 4 FILLERS = {"uh", "um", "you know", "i mean", "like"} — only uh+um from MINIMAL_SEGMENTS
    q = stage4_loop_mod.compute_quality(MINIMAL_SEGMENTS)
    assert q["filler_rate"] == pytest.approx(2 / 30, abs=0.005)


def test_compute_quality_punctuation(stage4_loop_mod):
    # Stage 4 checks for [.!?]$ (no trailing space); segs 0 and 4 → 2/5 = 0.4
    q = stage4_loop_mod.compute_quality(MINIMAL_SEGMENTS)
    assert q["punctuation_score"] == pytest.approx(0.4, abs=0.005)


def test_compute_quality_capitalisation(stage4_loop_mod):
    q = stage4_loop_mod.compute_quality(MINIMAL_SEGMENTS)
    assert q["capitalisation_score"] == pytest.approx(0.4, abs=0.005)


def test_compute_quality_empty(stage4_loop_mod):
    q = stage4_loop_mod.compute_quality([])
    assert q["filler_rate"] == 0.0
    assert q["punctuation_score"] == 0.0


# ---------------------------------------------------------------------------
# check_thresholds
# ---------------------------------------------------------------------------

def test_check_thresholds_all_failing(stage4_loop_mod):
    q = {"punctuation_score": 0.0, "filler_rate": 0.9, "capitalisation_score": 0.0}
    failing = stage4_loop_mod.check_thresholds(q)
    assert "punctuation" in failing
    assert "fillers" in failing
    assert "capitalisation" in failing


def test_check_thresholds_all_passing(stage4_loop_mod):
    q = {"punctuation_score": 0.80, "filler_rate": 0.02, "capitalisation_score": 0.80}
    failing = stage4_loop_mod.check_thresholds(q)
    assert len(failing) == 0


def test_check_thresholds_boundary_exact(stage4_loop_mod):
    # Exactly at threshold: punct=0.70 → passes (strict <); filler=0.04 → passes (not >)
    q = {"punctuation_score": 0.70, "filler_rate": 0.04, "capitalisation_score": 0.70}
    failing = stage4_loop_mod.check_thresholds(q)
    assert len(failing) == 0


# ---------------------------------------------------------------------------
# _apply_block_result
# ---------------------------------------------------------------------------

def test_apply_block_result_updates_text(stage4_loop_mod):
    segs = copy.deepcopy(MINIMAL_SEGMENTS)
    raw_result = "[0] Corrected first segment.\n[4] Updated fifth segment."
    result = stage4_loop_mod._apply_block_result(segs, raw_result)
    assert result[0]["text"] == " Corrected first segment."
    assert result[4]["text"] == " Updated fifth segment."


def test_apply_block_result_ignores_out_of_range(stage4_loop_mod):
    segs = copy.deepcopy(MINIMAL_SEGMENTS)
    raw_result = "[99] This index does not exist"
    # Should not raise
    result = stage4_loop_mod._apply_block_result(segs, raw_result)
    assert len(result) == len(MINIMAL_SEGMENTS)


def test_apply_block_result_ignores_non_prefixed_lines(stage4_loop_mod):
    segs = copy.deepcopy(MINIMAL_SEGMENTS)
    original_text = segs[0]["text"]
    raw_result = "Some random line without prefix"
    result = stage4_loop_mod._apply_block_result(segs, raw_result)
    assert result[0]["text"] == original_text


# ---------------------------------------------------------------------------
# should_continue
# ---------------------------------------------------------------------------

def test_should_continue_when_done(stage4_loop_mod):
    state = {"done": True, "failing_dimensions": []}
    assert stage4_loop_mod.should_continue(state) == "report"


def test_should_continue_when_not_done(stage4_loop_mod):
    state = {"done": False, "failing_dimensions": ["punctuation"]}
    assert stage4_loop_mod.should_continue(state) == "improve"
