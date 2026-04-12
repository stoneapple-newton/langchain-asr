"""Tests for stage_04_langgraph — no live LLM required."""
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import (
    CLEAN_SEGMENTS,
    FAKE_TRANSCRIPT,
    HEAD_ATTACHED_TRANSCRIPT,
    RUN_ON_TRANSCRIPT,
    TAIL_ATTACHED_TRANSCRIPT,
    stage4_graph_mod,
)


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_build_graph_returns_object(self, stage4_graph_mod):
        graph = stage4_graph_mod.build_graph()
        assert graph is not None

    def test_graph_has_invoke(self, stage4_graph_mod):
        graph = stage4_graph_mod.build_graph()
        assert hasattr(graph, "invoke")


# ---------------------------------------------------------------------------
# Routing functions (pure logic, no LLM)
# ---------------------------------------------------------------------------

class TestRouteByDefect:
    def test_run_on_routes_to_correct_run_on(self, stage4_graph_mod):
        state = {"detected_defect": "run_on"}
        assert stage4_graph_mod.route_by_defect(state) == "correct_run_on"

    def test_head_attached_routes_correctly(self, stage4_graph_mod):
        state = {"detected_defect": "head_attached"}
        assert stage4_graph_mod.route_by_defect(state) == "correct_head_attached"

    def test_tail_attached_routes_correctly(self, stage4_graph_mod):
        state = {"detected_defect": "tail_attached"}
        assert stage4_graph_mod.route_by_defect(state) == "correct_tail_attached"

    def test_unknown_routes_to_finalize(self, stage4_graph_mod):
        state = {"detected_defect": "unknown"}
        assert stage4_graph_mod.route_by_defect(state) == "finalize"

    def test_missing_defect_routes_to_finalize(self, stage4_graph_mod):
        state = {}
        assert stage4_graph_mod.route_by_defect(state) == "finalize"


class TestRouteQualityGate:
    def test_high_score_goes_to_finalize(self, stage4_graph_mod):
        state = {"quality_score": 0.95, "attempt": 1, "detected_defect": "run_on"}
        assert stage4_graph_mod.route_quality_gate(state) == "finalize"

    def test_exactly_threshold_goes_to_finalize(self, stage4_graph_mod):
        state = {"quality_score": 0.85, "attempt": 1, "detected_defect": "run_on"}
        assert stage4_graph_mod.route_quality_gate(state) == "finalize"

    def test_low_score_retries(self, stage4_graph_mod):
        state = {"quality_score": 0.5, "attempt": 1, "detected_defect": "run_on"}
        route = stage4_graph_mod.route_quality_gate(state)
        assert route == "correct_run_on"

    def test_max_attempts_forces_finalize(self, stage4_graph_mod):
        state = {"quality_score": 0.1, "attempt": 3, "detected_defect": "run_on"}
        assert stage4_graph_mod.route_quality_gate(state) == "finalize"

    def test_max_attempts_head_attached(self, stage4_graph_mod):
        state = {"quality_score": 0.0, "attempt": 3, "detected_defect": "head_attached"}
        assert stage4_graph_mod.route_quality_gate(state) == "finalize"


# ---------------------------------------------------------------------------
# Rule-based nodes (no LLM calls)
# ---------------------------------------------------------------------------

class TestHeadAttachedNode:
    def test_strips_head_labels(self, stage4_graph_mod):
        import re
        state = {
            "transcript": HEAD_ATTACHED_TRANSCRIPT,
            "correction_notes": [],
            "attempt": 0,
        }
        updates = stage4_graph_mod.correct_head_attached_node(state)
        for seg in updates["corrected_segments"]:
            assert not re.match(r"^\s*SPEAKER_\d+\s*:", seg.get("text", ""))

    def test_increments_attempt(self, stage4_graph_mod):
        state = {
            "transcript": HEAD_ATTACHED_TRANSCRIPT,
            "correction_notes": [],
            "attempt": 0,
        }
        updates = stage4_graph_mod.correct_head_attached_node(state)
        assert updates["attempt"] == 1


class TestTailAttachedNode:
    def test_strips_tail_labels(self, stage4_graph_mod):
        import re
        state = {
            "transcript": TAIL_ATTACHED_TRANSCRIPT,
            "correction_notes": [],
            "attempt": 0,
        }
        updates = stage4_graph_mod.correct_tail_attached_node(state)
        for seg in updates["corrected_segments"]:
            assert not re.search(r"SPEAKER_\d+\s*$", seg.get("text", ""))

    def test_increments_attempt(self, stage4_graph_mod):
        state = {
            "transcript": TAIL_ATTACHED_TRANSCRIPT,
            "correction_notes": [],
            "attempt": 0,
        }
        updates = stage4_graph_mod.correct_tail_attached_node(state)
        assert updates["attempt"] == 1


# ---------------------------------------------------------------------------
# detect_node (no LLM)
# ---------------------------------------------------------------------------

class TestDetectNode:
    def test_detects_head_attached(self, stage4_graph_mod):
        state = {"transcript": HEAD_ATTACHED_TRANSCRIPT}
        updates = stage4_graph_mod.detect_node(state)
        assert updates["detected_defect"] == "head_attached"

    def test_detects_tail_attached(self, stage4_graph_mod):
        state = {"transcript": TAIL_ATTACHED_TRANSCRIPT}
        updates = stage4_graph_mod.detect_node(state)
        assert updates["detected_defect"] == "tail_attached"

    def test_clean_is_unknown(self, stage4_graph_mod):
        state = {"transcript": FAKE_TRANSCRIPT}
        updates = stage4_graph_mod.detect_node(state)
        assert updates["detected_defect"] == "unknown"


# ---------------------------------------------------------------------------
# evaluate_node (no ground truth path)
# ---------------------------------------------------------------------------

class TestEvaluateNode:
    def test_no_ground_truth_gives_heuristic_score(self, stage4_graph_mod):
        state = {
            "ground_truth": None,
            "detected_defect": "head_attached",
            "corrected_segments": CLEAN_SEGMENTS,
            "transcript": FAKE_TRANSCRIPT,
        }
        updates = stage4_graph_mod.evaluate_node(state)
        assert "quality_score" in updates
        assert 0.0 <= updates["quality_score"] <= 1.0

    def test_with_ground_truth_returns_f1(self, stage4_graph_mod):
        # Perfect prediction → f1 = 1.0
        gt = FAKE_TRANSCRIPT
        state = {
            "ground_truth": gt,
            "detected_defect": "head_attached",
            "corrected_segments": CLEAN_SEGMENTS,
            "transcript": FAKE_TRANSCRIPT,
        }
        updates = stage4_graph_mod.evaluate_node(state)
        assert updates["quality_score"] == 1.0


# ---------------------------------------------------------------------------
# correct_transcript public API (signature checks only — graph is mocked in tests)
# ---------------------------------------------------------------------------

class TestCorrectTranscriptApi:
    def test_correct_transcript_is_callable(self, stage4_graph_mod):
        assert callable(stage4_graph_mod.correct_transcript)

    def test_correct_transcript_accepts_two_args(self, stage4_graph_mod):
        import inspect
        sig = inspect.signature(stage4_graph_mod.correct_transcript)
        params = list(sig.parameters)
        assert "transcript" in params
        assert "ground_truth" in params

    def test_variant_name_constant(self, stage4_graph_mod):
        assert stage4_graph_mod.VARIANT_NAME == "langgraph_pipeline"
