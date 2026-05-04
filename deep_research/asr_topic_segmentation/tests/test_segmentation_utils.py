"""Tests for segmentation_utils deterministic helpers."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.asr_topic_segmentation.shared.segmentation_utils import (
    TopicSegment,
    detect_boundary_signals,
    group_into_topic_segments,
    render_segmentation_report,
)


@dataclass
class FakeSegment:
    """Minimal stand-in for TranscriptSegment."""

    segment_id: str
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def make_segs(*specs: tuple) -> list[FakeSegment]:
    """Build segments from (id, start, end, text, speaker) tuples."""
    return [FakeSegment(str(s[0]), s[1], s[2], s[3], s[4] if len(s) > 4 else None) for s in specs]


class TestDetectBoundarySignals:
    def test_returns_one_signal_per_segment(self):
        segs = make_segs(
            (0, 0.0, 2.0, "hello world", "A"),
            (1, 2.1, 4.0, "another topic here", "B"),
        )
        signals = detect_boundary_signals(segs)
        assert len(signals) == 2

    def test_first_segment_gets_zero_score_when_no_shift(self):
        segs = make_segs((0, 0.0, 2.0, "hello", "A"))
        signals = detect_boundary_signals(segs)
        assert signals[0].signal_score == 0.0

    def test_shift_phrase_raises_score(self):
        segs = make_segs(
            (0, 0.0, 2.0, "we covered the release", "A"),
            (1, 2.1, 5.0, "moving on let us talk about the bug", "B"),
        )
        signals = detect_boundary_signals(segs)
        assert signals[1].has_shift_phrase is True
        assert signals[1].signal_score >= 0.5

    def test_long_gap_raises_score(self):
        segs = make_segs(
            (0, 0.0, 2.0, "first topic", "A"),
            (1, 10.0, 12.0, "second topic after long pause", "A"),
        )
        signals = detect_boundary_signals(segs)
        assert signals[1].gap_seconds >= 2.0
        assert signals[1].signal_score >= 0.2

    def test_speaker_change_raises_score(self):
        segs = make_segs(
            (0, 0.0, 2.0, "i was speaking", "A"),
            (1, 2.1, 4.0, "now i am speaking", "B"),
        )
        signals = detect_boundary_signals(segs)
        assert signals[1].speaker_changed is True
        assert signals[1].signal_score >= 0.2

    def test_question_mark_raises_score(self):
        segs = make_segs(
            (0, 0.0, 2.0, "some statement", "A"),
            (1, 2.1, 4.0, "can we move on?", "A"),
        )
        signals = detect_boundary_signals(segs)
        assert signals[1].has_question is True
        assert signals[1].signal_score >= 0.1

    def test_score_capped_at_one(self):
        segs = make_segs(
            (0, 0.0, 2.0, "topic one", "A"),
            (1, 20.0, 22.0, "moving on by the way switching to next topic?", "B"),
        )
        signals = detect_boundary_signals(segs)
        assert signals[1].signal_score <= 1.0

    def test_empty_segments(self):
        signals = detect_boundary_signals([])
        assert signals == []


class TestGroupIntoTopicSegments:
    def test_single_topic_when_no_high_signals(self):
        segs = make_segs(
            (0, 0.0, 2.0, "hello", "A"),
            (1, 2.1, 4.0, "world", "A"),
        )
        signals = detect_boundary_signals(segs)
        topics = group_into_topic_segments(segs, signals, threshold=0.5)
        assert len(topics) == 1
        assert topics[0].topic_id == "topic_01"

    def test_split_at_high_signal(self):
        segs = make_segs(
            (0, 0.0, 5.0, "release discussion", "A"),
            (1, 5.1, 10.0, "moving on let us talk about the bugs", "B"),
            (2, 10.1, 15.0, "payment gateway error", "A"),
        )
        signals = detect_boundary_signals(segs)
        topics = group_into_topic_segments(segs, signals, threshold=0.5)
        assert len(topics) == 2

    def test_topic_ids_sequential(self):
        segs = make_segs(
            (0, 0.0, 5.0, "topic one", "A"),
            (1, 10.0, 15.0, "moving on next agenda item", "B"),
            (2, 20.0, 25.0, "another topic by the way", "A"),
        )
        signals = detect_boundary_signals(segs)
        topics = group_into_topic_segments(segs, signals, threshold=0.3)
        ids = [t.topic_id for t in topics]
        assert ids == sorted(ids)

    def test_segment_ids_covered(self):
        segs = make_segs(
            (0, 0.0, 3.0, "first part", "A"),
            (1, 3.1, 6.0, "second part moving on", "B"),
            (2, 6.1, 9.0, "third part", "A"),
        )
        signals = detect_boundary_signals(segs)
        topics = group_into_topic_segments(segs, signals, threshold=0.5)
        all_ids = [sid for t in topics for sid in t.segment_ids]
        assert sorted(all_ids) == ["0", "1", "2"]

    def test_empty_segments(self):
        topics = group_into_topic_segments([], [], threshold=0.5)
        assert topics == []

    def test_time_ranges_correct(self):
        segs = make_segs(
            (0, 1.0, 5.0, "start", "A"),
            (1, 8.0, 12.0, "moving on to next item", "B"),
        )
        signals = detect_boundary_signals(segs)
        topics = group_into_topic_segments(segs, signals, threshold=0.5)
        assert topics[0].start_time == 1.0
        assert topics[-1].end_time == 12.0


class TestRenderSegmentationReport:
    def test_renders_without_error(self):
        topics = [
            TopicSegment("topic_01", "Sprint Review", 0.0, 20.0, ["0", "1"], "Team reviewed sprint output."),
            TopicSegment("topic_02", "Bug Triage", 23.0, 50.0, ["2", "3"], "Payment gateway issue discussed."),
        ]
        report = render_segmentation_report(topics)
        assert "# Topic Segmentation Report" in report
        assert "Sprint Review" in report
        assert "Bug Triage" in report

    def test_summary_included_when_present(self):
        topics = [TopicSegment("topic_01", "Release", 0.0, 10.0, ["0"], "Release reviewed.")]
        report = render_segmentation_report(topics)
        assert "Release reviewed." in report

    def test_empty_topics(self):
        report = render_segmentation_report([])
        assert "# Topic Segmentation Report" in report
