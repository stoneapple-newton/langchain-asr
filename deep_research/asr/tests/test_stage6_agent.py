"""Tests for stage_06_production_agent/02_llm_guided_agent.py"""

import copy
import pytest
from .conftest import MINIMAL_SEGMENTS


# ---------------------------------------------------------------------------
# _rule_clean
# ---------------------------------------------------------------------------

def test_rule_clean_removes_uh(stage6_agent_mod):
    result = stage6_agent_mod._rule_clean("uh we need to fix this")
    assert "uh" not in result.lower()


def test_rule_clean_removes_um(stage6_agent_mod):
    result = stage6_agent_mod._rule_clean("um the service is down")
    assert "um" not in result.lower()


def test_rule_clean_removes_you_know(stage6_agent_mod):
    result = stage6_agent_mod._rule_clean("you know it takes time")
    assert "you know" not in result.lower()


def test_rule_clean_contraction_arent(stage6_agent_mod):
    result = stage6_agent_mod._rule_clean("arent we done")
    assert "aren't" in result


def test_rule_clean_contraction_lets(stage6_agent_mod):
    result = stage6_agent_mod._rule_clean("lets go ahead")
    assert "let's" in result


def test_rule_clean_collapses_spaces(stage6_agent_mod):
    result = stage6_agent_mod._rule_clean("we   need  to   fix")
    assert "  " not in result


def test_rule_clean_strips_whitespace(stage6_agent_mod):
    result = stage6_agent_mod._rule_clean("  hello world  ")
    assert result == result.strip()


# ---------------------------------------------------------------------------
# _fmt_srt
# ---------------------------------------------------------------------------

def test_fmt_srt_zero(stage6_agent_mod):
    assert stage6_agent_mod._fmt_srt(0.0) == "00:00:00,000"


def test_fmt_srt_one_hour(stage6_agent_mod):
    assert stage6_agent_mod._fmt_srt(3600.0) == "01:00:00,000"


def test_fmt_srt_fractional_seconds(stage6_agent_mod):
    # 1.5 seconds = 00:00:01,500
    assert stage6_agent_mod._fmt_srt(1.5) == "00:00:01,500"


def test_fmt_srt_milliseconds(stage6_agent_mod):
    # 61.25 = 00:01:01,250
    assert stage6_agent_mod._fmt_srt(61.25) == "00:01:01,250"


def test_fmt_srt_returns_string(stage6_agent_mod):
    result = stage6_agent_mod._fmt_srt(42.0)
    assert isinstance(result, str)
    assert "," in result  # SRT uses comma for ms separator


# ---------------------------------------------------------------------------
# _apply_block_output
# ---------------------------------------------------------------------------

def test_apply_block_output_updates_segment(stage6_agent_mod):
    segs = copy.deepcopy(MINIMAL_SEGMENTS)
    raw = "[0] Alright, let's get started.\n[4] The JWT library needs updating."
    result = stage6_agent_mod._apply_block_output(segs, raw)
    assert result[0]["text"] == " Alright, let's get started."
    assert result[4]["text"] == " The JWT library needs updating."


def test_apply_block_output_ignores_out_of_range(stage6_agent_mod):
    segs = copy.deepcopy(MINIMAL_SEGMENTS)
    raw = "[99] Should be ignored"
    result = stage6_agent_mod._apply_block_output(segs, raw)
    assert len(result) == len(MINIMAL_SEGMENTS)


def test_apply_block_output_ignores_empty_text(stage6_agent_mod):
    segs = copy.deepcopy(MINIMAL_SEGMENTS)
    original_text = segs[0]["text"]
    # "[0] " with empty stripped text → should NOT update
    raw = "[0] "
    result = stage6_agent_mod._apply_block_output(segs, raw)
    assert result[0]["text"] == original_text


def test_apply_block_output_ignores_non_prefixed(stage6_agent_mod):
    segs = copy.deepcopy(MINIMAL_SEGMENTS)
    original_text = segs[0]["text"]
    raw = "No prefix line here"
    result = stage6_agent_mod._apply_block_output(segs, raw)
    assert result[0]["text"] == original_text


# ---------------------------------------------------------------------------
# supervisor_router
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("action,expected_node", [
    ("continue_cleaning", "cleaner"),
    ("run_technical", "technical"),
    ("run_fluency", "fluency"),
    ("run_diarize", "diarize"),
    ("export", "export"),
    ("export (forced)", "export"),
])
def test_supervisor_router_all_actions(stage6_agent_mod, action, expected_node):
    state = {"supervisor_history": [action]}
    assert stage6_agent_mod.supervisor_router(state) == expected_node


def test_supervisor_router_empty_history(stage6_agent_mod):
    state = {"supervisor_history": []}
    # No history → default to cleaner
    assert stage6_agent_mod.supervisor_router(state) == "cleaner"


def test_supervisor_router_unknown_action(stage6_agent_mod):
    state = {"supervisor_history": ["unknown_action"]}
    # Unknown action → default to export (safe fallback)
    assert stage6_agent_mod.supervisor_router(state) == "export"
