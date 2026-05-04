"""Tests for sentiment_utils deterministic helpers."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.asr_sentiment_analysis.shared.sentiment_utils import (
    SegmentSentimentLabel,
    aggregate_speaker_profiles,
    render_sentiment_report,
    score_segment_naive,
)


def make_label(segment_id: str, speaker: str, text: str) -> SegmentSentimentLabel:
    return score_segment_naive(
        segment_id=segment_id,
        speaker=speaker,
        start=0.0,
        end=1.0,
        text=text,
    )


class TestScoreSegmentNaive:
    def test_positive_text(self):
        lbl = make_label("0", "AGENT", "this is great and wonderful thank you")
        assert lbl.naive_label == "positive"
        assert lbl.positive_count >= 2

    def test_negative_text(self):
        lbl = make_label("1", "CUSTOMER", "this is terrible and awful I am frustrated")
        assert lbl.naive_label == "negative"
        assert lbl.negative_count >= 2

    def test_mixed_text(self):
        lbl = make_label("2", "SPEAKER_00", "great service but the issue is still broken")
        assert lbl.naive_label == "mixed"
        assert lbl.positive_count > 0
        assert lbl.negative_count > 0

    def test_neutral_text(self):
        lbl = make_label("3", "AGENT", "please hold while I check your account")
        assert lbl.naive_label == "neutral"

    def test_filler_count(self):
        lbl = make_label("4", "SPEAKER_01", "um uh you know I mean like the service")
        assert lbl.filler_count >= 4

    def test_speaker_preserved(self):
        lbl = make_label("5", "SPEAKER_42", "good morning")
        assert lbl.speaker == "SPEAKER_42"
        assert lbl.segment_id == "5"

    def test_empty_text(self):
        lbl = make_label("6", None, "")
        assert lbl.naive_label == "neutral"
        assert lbl.positive_count == 0
        assert lbl.negative_count == 0

    def test_case_insensitive(self):
        lbl_lower = make_label("7", "A", "GREAT SERVICE THANK YOU")
        assert lbl_lower.naive_label == "positive"


class TestAggregateProfiles:
    def test_dominant_positive(self):
        labels = [
            make_label("0", "AGENT", "great wonderful perfect"),
            make_label("1", "AGENT", "excellent thank you"),
            make_label("2", "AGENT", "I am so happy"),
        ]
        profiles = aggregate_speaker_profiles(labels)
        assert "AGENT" in profiles
        assert profiles["AGENT"].dominant_label == "positive"
        assert profiles["AGENT"].segment_count == 3

    def test_dominant_negative(self):
        labels = [
            make_label("0", "CUSTOMER", "terrible awful broken"),
            make_label("1", "CUSTOMER", "I am frustrated and upset"),
        ]
        profiles = aggregate_speaker_profiles(labels)
        assert profiles["CUSTOMER"].dominant_label == "negative"

    def test_unknown_speaker_grouped(self):
        labels = [
            make_label("0", None, "good morning"),
            make_label("1", None, "have a nice day"),
        ]
        profiles = aggregate_speaker_profiles(labels)
        assert "UNKNOWN" in profiles
        assert profiles["UNKNOWN"].segment_count == 2

    def test_multiple_speakers(self):
        labels = [
            make_label("0", "AGENT", "great to help you"),
            make_label("1", "CUSTOMER", "this is terrible"),
        ]
        profiles = aggregate_speaker_profiles(labels)
        assert "AGENT" in profiles
        assert "CUSTOMER" in profiles
        assert profiles["AGENT"].dominant_label == "positive"
        assert profiles["CUSTOMER"].dominant_label == "negative"

    def test_empty_labels(self):
        profiles = aggregate_speaker_profiles([])
        assert profiles == {}

    def test_positive_ratio_calculation(self):
        labels = [
            make_label("0", "S", "great"),
            make_label("1", "S", "great"),
            make_label("2", "S", "terrible"),
        ]
        profiles = aggregate_speaker_profiles(labels)
        assert abs(profiles["S"].positive_ratio - 2 / 3) < 0.01
        assert abs(profiles["S"].negative_ratio - 1 / 3) < 0.01


class TestRenderSentimentReport:
    def test_renders_without_error(self):
        labels = [
            make_label("0", "AGENT", "great service"),
            make_label("1", "CUSTOMER", "terrible experience"),
        ]
        profiles = aggregate_speaker_profiles(labels)
        report = render_sentiment_report(labels, profiles)
        assert "# Sentiment Analysis Report" in report
        assert "AGENT" in report
        assert "CUSTOMER" in report

    def test_report_contains_labels(self):
        labels = [make_label("0", "AGENT", "thank you great")]
        profiles = aggregate_speaker_profiles(labels)
        report = render_sentiment_report(labels, profiles)
        assert "POSITIVE" in report

    def test_empty_labels_renders(self):
        report = render_sentiment_report([], {})
        assert "# Sentiment Analysis Report" in report
