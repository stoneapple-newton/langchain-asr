"""Tests for action_item_utils deterministic helpers."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.asr_action_items.shared.action_item_utils import (
    ActionItem,
    CandidateItem,
    detect_candidates,
    format_candidates_for_llm,
    render_action_items_report,
)


@dataclass
class FakeSegment:
    segment_id: str
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def seg(segment_id: str, text: str, speaker: str = "SPEAKER_00") -> FakeSegment:
    return FakeSegment(segment_id, 0.0, 1.0, text, speaker)


class TestDetectCandidates:
    def test_task_verb_detected(self):
        segments = [seg("0", "I will send the report by Friday")]
        candidates = detect_candidates(segments)
        assert len(candidates) == 1
        assert "task" in candidates[0].candidate_types

    def test_decision_phrase_detected(self):
        segments = [seg("0", "we have decided to postpone the launch")]
        candidates = detect_candidates(segments)
        assert len(candidates) == 1
        assert "decision" in candidates[0].candidate_types

    def test_question_phrase_detected(self):
        segments = [seg("0", "who will own the deployment process")]
        candidates = detect_candidates(segments)
        assert len(candidates) == 1
        assert "open_question" in candidates[0].candidate_types

    def test_no_signal_excluded(self):
        segments = [seg("0", "the weather is nice today")]
        candidates = detect_candidates(segments)
        assert len(candidates) == 0

    def test_sorted_by_match_count_descending(self):
        segments = [
            seg("0", "will send and check and follow up and review"),
            seg("1", "can you do that"),
        ]
        candidates = detect_candidates(segments)
        assert candidates[0].segment_id == "0"
        assert candidates[0].match_count >= candidates[1].match_count

    def test_multiple_types_in_one_segment(self):
        segments = [seg("0", "we agreed to send the report can you confirm")]
        candidates = detect_candidates(segments)
        assert len(candidates) == 1
        types = candidates[0].candidate_types
        assert "task" in types or "decision" in types

    def test_speaker_preserved(self):
        segments = [seg("0", "I will review the code", speaker="SPEAKER_42")]
        candidates = detect_candidates(segments)
        assert candidates[0].speaker == "SPEAKER_42"

    def test_empty_segments(self):
        candidates = detect_candidates([])
        assert candidates == []

    def test_case_insensitive(self):
        segments = [seg("0", "WILL SEND THE REPORT")]
        candidates = detect_candidates(segments)
        assert len(candidates) == 1

    def test_match_count_reflects_all_patterns(self):
        segments = [seg("0", "will should can you please follow up")]
        candidates = detect_candidates(segments)
        assert candidates[0].match_count >= 3


class TestFormatCandidatesForLlm:
    def test_returns_string(self):
        candidates = [
            CandidateItem("0", "AGENT", 0.0, 1.0, "will send report", ["task"], 1)
        ]
        result = format_candidates_for_llm(candidates)
        assert isinstance(result, str)
        assert "seg=0" in result
        assert "will send report" in result

    def test_empty_candidates(self):
        result = format_candidates_for_llm([])
        assert "(no candidates" in result.lower()

    def test_multiple_candidates_all_present(self):
        candidates = [
            CandidateItem("0", "A", 0.0, 1.0, "first task", ["task"], 1),
            CandidateItem("1", "B", 1.0, 2.0, "second decision", ["decision"], 1),
        ]
        result = format_candidates_for_llm(candidates)
        assert "seg=0" in result
        assert "seg=1" in result


class TestRenderActionItemsReport:
    def _make_item(self, item_id: str, item_type: str, text: str) -> ActionItem:
        return ActionItem(
            item_id=item_id,
            item_type=item_type,
            text=text,
            raw_segment_text="",
            segment_id="0",
            owner=None,
            due_context=None,
            confidence=0.9,
        )

    def test_empty_items(self):
        report = render_action_items_report([])
        assert "No action items" in report

    def test_task_section_present(self):
        items = [self._make_item("AI-001", "task", "send the report")]
        report = render_action_items_report(items)
        assert "## Tasks" in report
        assert "send the report" in report

    def test_decision_section_present(self):
        items = [self._make_item("AI-001", "decision", "we go with option A")]
        report = render_action_items_report(items)
        assert "## Decisions" in report

    def test_owner_shown(self):
        item = ActionItem(
            item_id="AI-001",
            item_type="task",
            text="do the thing",
            raw_segment_text="",
            segment_id="0",
            owner="Alice",
            due_context="by Friday",
            confidence=0.95,
        )
        report = render_action_items_report([item])
        assert "Alice" in report
        assert "by Friday" in report

    def test_multiple_types_all_sections(self):
        items = [
            self._make_item("AI-001", "task", "send docs"),
            self._make_item("AI-002", "decision", "go with plan B"),
            self._make_item("AI-003", "open_question", "who owns QA"),
        ]
        report = render_action_items_report(items)
        assert "Tasks" in report
        assert "Decisions" in report
        assert "Open Questions" in report

    def test_sections_omitted_when_empty(self):
        items = [self._make_item("AI-001", "task", "send docs")]
        report = render_action_items_report(items)
        assert "Decisions" not in report
        assert "Open Questions" not in report
