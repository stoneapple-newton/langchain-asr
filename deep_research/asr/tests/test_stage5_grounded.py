"""Tests for stage_05_rag_context/03_grounded_correction.py — grade_and_filter"""

import pytest


# ---------------------------------------------------------------------------
# grade_and_filter — uses mocked grade_chain
# ---------------------------------------------------------------------------

def _make_doc(doc_id, content="some content"):
    """Create a minimal Document-like object with metadata id."""
    class FakeDoc:
        def __init__(self, id_, text):
            self.metadata = {"id": id_}
            self.page_content = text
    return FakeDoc(doc_id, content)


def test_grade_and_filter_keeps_relevant(stage5_grounded_mod, mocker):
    docs = [_make_doc("term_jwt"), _make_doc("term_devops")]
    mocker.patch.object(
        stage5_grounded_mod.grade_chain,
        "invoke",
        return_value={
            "grades": [
                {"doc_id": "term_jwt", "relevance": 0.9, "reason": "relevant"},
                {"doc_id": "term_devops", "relevance": 0.8, "reason": "relevant"},
            ]
        },
    )
    result = stage5_grounded_mod.grade_and_filter("jwt token invalidation", docs)
    assert len(result) == 2


def test_grade_and_filter_removes_irrelevant(stage5_grounded_mod, mocker):
    docs = [_make_doc("term_jwt"), _make_doc("participant_carol")]
    mocker.patch.object(
        stage5_grounded_mod.grade_chain,
        "invoke",
        return_value={
            "grades": [
                {"doc_id": "term_jwt", "relevance": 0.9, "reason": "relevant"},
                {"doc_id": "participant_carol", "relevance": 0.2, "reason": "irrelevant"},
            ]
        },
    )
    result = stage5_grounded_mod.grade_and_filter("jwt token invalidation", docs)
    assert len(result) == 1
    assert result[0].metadata["id"] == "term_jwt"


def test_grade_and_filter_threshold_boundary(stage5_grounded_mod, mocker):
    # Exactly at threshold (0.5) should be kept
    docs = [_make_doc("term_jwt")]
    mocker.patch.object(
        stage5_grounded_mod.grade_chain,
        "invoke",
        return_value={
            "grades": [
                {"doc_id": "term_jwt", "relevance": 0.5, "reason": "borderline"},
            ]
        },
    )
    result = stage5_grounded_mod.grade_and_filter("jwt token", docs)
    assert len(result) == 1


def test_grade_and_filter_empty_docs(stage5_grounded_mod, mocker):
    result = stage5_grounded_mod.grade_and_filter("some text", [])
    assert result == []


def test_grade_and_filter_fallback_on_exception(stage5_grounded_mod, mocker):
    docs = [_make_doc("term_jwt"), _make_doc("term_devops")]
    mocker.patch.object(
        stage5_grounded_mod.grade_chain,
        "invoke",
        side_effect=Exception("LLM error"),
    )
    # Should return all docs unfiltered on exception
    result = stage5_grounded_mod.grade_and_filter("jwt token invalidation", docs)
    assert len(result) == 2


def test_grade_and_filter_missing_doc_id(stage5_grounded_mod, mocker):
    # Doc has no id in grades → grade_map lookup returns 0.0 → filtered out
    docs = [_make_doc("term_jwt")]
    mocker.patch.object(
        stage5_grounded_mod.grade_chain,
        "invoke",
        return_value={
            "grades": [
                {"doc_id": "unknown_id", "relevance": 0.9, "reason": "unknown"},
            ]
        },
    )
    result = stage5_grounded_mod.grade_and_filter("some text", docs)
    assert len(result) == 0
