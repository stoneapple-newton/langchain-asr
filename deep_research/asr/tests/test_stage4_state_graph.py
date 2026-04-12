"""Tests for stage_04_langgraph_pipeline/01_asr_state_graph.py"""

import copy
import pytest
from .conftest import MINIMAL_SEGMENTS


# ---------------------------------------------------------------------------
# compute_quality (stage 4 file 1 variant — no capitalisation_score)
# ---------------------------------------------------------------------------

def test_compute_quality_returns_required_keys(stage4_graph_mod):
    q = stage4_graph_mod.compute_quality(MINIMAL_SEGMENTS)
    assert "avg_confidence" in q
    assert "filler_rate" in q
    assert "punctuation_score" in q
    assert "total_segments" in q
    assert "total_words" in q


def test_compute_quality_total_segments(stage4_graph_mod):
    q = stage4_graph_mod.compute_quality(MINIMAL_SEGMENTS)
    assert q["total_segments"] == 5


def test_compute_quality_total_words(stage4_graph_mod):
    q = stage4_graph_mod.compute_quality(MINIMAL_SEGMENTS)
    assert q["total_words"] == 30


def test_compute_quality_filler_rate_stage4_graph(stage4_graph_mod):
    # Same FILLERS as stage4/02: {"uh", "um", "you know", "i mean", "like"} → 2/30
    q = stage4_graph_mod.compute_quality(MINIMAL_SEGMENTS)
    assert q["filler_rate"] == pytest.approx(2 / 30, abs=0.005)


def test_compute_quality_empty(stage4_graph_mod):
    q = stage4_graph_mod.compute_quality([])
    assert q["total_words"] == 0
    assert q["total_segments"] == 0


# ---------------------------------------------------------------------------
# rule-based cleanup via CONTRACTION_MAP (tested through rule_based_cleanup if present,
# otherwise just confirm the map is accessible)
# ---------------------------------------------------------------------------

def test_contraction_map_exists(stage4_graph_mod):
    assert hasattr(stage4_graph_mod, "CONTRACTION_MAP")
    assert isinstance(stage4_graph_mod.CONTRACTION_MAP, dict)


def test_contraction_map_has_arent(stage4_graph_mod):
    # Key pattern matches "arent"
    import re
    cmap = stage4_graph_mod.CONTRACTION_MAP
    matched = any(
        re.search(pat, "arent", re.IGNORECASE) for pat in cmap
    )
    assert matched
