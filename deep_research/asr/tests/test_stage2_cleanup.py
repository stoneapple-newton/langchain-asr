"""Tests for stage_02_llm_enhancement/01_punctuation_cleanup.py — rule_based_cleanup"""

import pytest


# ---------------------------------------------------------------------------
# rule_based_cleanup — pure regex, no LLM
# ---------------------------------------------------------------------------

def test_removes_uh(stage2_cleanup_mod):
    result = stage2_cleanup_mod.rule_based_cleanup("uh we should start")
    assert "uh" not in result.lower()


def test_removes_um(stage2_cleanup_mod):
    result = stage2_cleanup_mod.rule_based_cleanup("um the service is down")
    assert "um" not in result.lower()


def test_removes_you_know(stage2_cleanup_mod):
    result = stage2_cleanup_mod.rule_based_cleanup("you know it takes time")
    assert "you know" not in result.lower()


def test_collapses_stutter(stage2_cleanup_mod):
    result = stage2_cleanup_mod.rule_based_cleanup("the the service is broken")
    assert result.lower().count("the") == 1


@pytest.mark.parametrize("raw,expected", [
    ("arent", "aren't"),
    ("cant", "can't"),
    ("dont", "don't"),
    ("lets", "let's"),
    ("itll", "it'll"),
    ("im", "I'm"),
    ("ive", "I've"),
    ("weve", "we've"),
    ("thats", "that's"),
    ("its", "it's"),
])
def test_contraction_expansion(stage2_cleanup_mod, raw, expected):
    result = stage2_cleanup_mod.rule_based_cleanup(raw)
    assert result == expected


def test_collapses_multiple_spaces(stage2_cleanup_mod):
    result = stage2_cleanup_mod.rule_based_cleanup("we   need  to   fix   this")
    assert "  " not in result


def test_strips_leading_trailing_whitespace(stage2_cleanup_mod):
    result = stage2_cleanup_mod.rule_based_cleanup("  hello world  ")
    assert result == result.strip()


def test_returns_string(stage2_cleanup_mod):
    result = stage2_cleanup_mod.rule_based_cleanup("some text")
    assert isinstance(result, str)
