"""Tests for stage_01_basics/02_quality_metrics.py"""

import copy
import pytest
from .conftest import MINIMAL_SEGMENTS


# ---------------------------------------------------------------------------
# Individual metric function tests
# ---------------------------------------------------------------------------

def test_avg_confidence(stage1_metrics_mod):
    # 30 words; see MINIMAL_SEGMENTS in conftest for score breakdown
    result = stage1_metrics_mod.avg_confidence(MINIMAL_SEGMENTS)
    assert result == pytest.approx(0.928, abs=0.01)


def test_filler_rate_stage1(stage1_metrics_mod):
    # Stage 1 FILLERS includes: uh, um, so, right, you know, i mean, like, basically, literally, actually
    # In MINIMAL_SEGMENTS: "so" (seg1), "uh" (seg1), "um" (seg1), "right" (seg3) = 4 fillers / 30 words
    result = stage1_metrics_mod.filler_rate(MINIMAL_SEGMENTS)
    assert result == pytest.approx(4 / 30, abs=0.005)


def test_punctuation_score(stage1_metrics_mod):
    # Segs 0 and 4 end with period → 2/5 = 0.4
    result = stage1_metrics_mod.punctuation_score(MINIMAL_SEGMENTS)
    assert result == pytest.approx(0.4, abs=0.005)


def test_capitalisation_score(stage1_metrics_mod):
    # Segs 0 ("Alright...") and 4 ("The the...") start with capital → 2/5 = 0.4
    result = stage1_metrics_mod.capitalisation_score(MINIMAL_SEGMENTS)
    assert result == pytest.approx(0.4, abs=0.005)


def test_diarization_coverage_full(stage1_metrics_mod):
    result = stage1_metrics_mod.diarization_coverage(MINIMAL_SEGMENTS)
    assert result == pytest.approx(1.0, abs=0.001)


def test_diarization_coverage_partial(stage1_metrics_mod):
    segs = copy.deepcopy(MINIMAL_SEGMENTS)
    segs[0]["speaker"] = None
    segs[1]["speaker"] = ""
    result = stage1_metrics_mod.diarization_coverage(segs)
    assert result == pytest.approx(3 / 5, abs=0.005)


def test_speaker_count(stage1_metrics_mod):
    result = stage1_metrics_mod.speaker_count(MINIMAL_SEGMENTS)
    assert result == 3


def test_low_confidence_word_rate(stage1_metrics_mod):
    # threshold = 0.75: authentication(0.68), JWT(0.71) = 2 low-conf words / 30 total
    result = stage1_metrics_mod.low_confidence_word_rate(MINIMAL_SEGMENTS)
    assert result == pytest.approx(2 / 30, abs=0.005)


def test_duplicate_word_rate_nonzero(stage1_metrics_mod):
    # Seg 4: "The the" → at least 1 duplicate pair
    result = stage1_metrics_mod.duplicate_word_rate(MINIMAL_SEGMENTS)
    assert result > 0.0


def test_avg_segment_duration(stage1_metrics_mod):
    # Durations: 4.5, 5.4, 5.98, 1.0, 6.5 → avg ≈ 4.676
    result = stage1_metrics_mod.avg_segment_duration(MINIMAL_SEGMENTS)
    assert result == pytest.approx(4.676, abs=0.05)


def test_suspicious_speaker_switches(stage1_metrics_mod):
    # Seg 1 (SPEAKER_00, end=10.0) → Seg 2 (SPEAKER_01, start=10.02): gap=0.02 < 0.05 default
    result = stage1_metrics_mod.suspicious_speaker_switches(MINIMAL_SEGMENTS)
    assert len(result) == 1
    assert result[0]["gap_ms"] == 20


def test_suspicious_speaker_switches_no_fast(stage1_metrics_mod):
    import copy
    segs = copy.deepcopy(MINIMAL_SEGMENTS)
    # Widen the gap so no fast switch
    segs[2]["start"] = 10.5
    result = stage1_metrics_mod.suspicious_speaker_switches(segs)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# QualityReport tests
# ---------------------------------------------------------------------------

def _make_report(mod, segs):
    """Build a QualityReport from segments using the module's functions."""
    return mod.QualityReport(
        avg_confidence=mod.avg_confidence(segs),
        filler_rate=mod.filler_rate(segs),
        punctuation_score=mod.punctuation_score(segs),
        capitalisation_score=mod.capitalisation_score(segs),
        diarization_coverage=mod.diarization_coverage(segs),
        duplicate_word_rate=mod.duplicate_word_rate(segs),
        low_confidence_rate=mod.low_confidence_word_rate(segs),
        num_speakers=mod.speaker_count(segs),
        avg_segment_duration=mod.avg_segment_duration(segs),
        speaking_time=mod.speaking_time_per_speaker(segs),
        suspicious_switches=mod.suspicious_speaker_switches(segs),
        total_segments=len(segs),
        total_words=sum(len(s.get("words", [])) for s in segs),
    )


def test_quality_report_overall_score_range(stage1_metrics_mod):
    report = _make_report(stage1_metrics_mod, MINIMAL_SEGMENTS)
    assert 0.0 <= report.overall_score <= 1.0


def test_quality_report_grade_a(stage1_metrics_mod):
    """All perfect fields → overall_score = 1.0 → grade A."""
    report = stage1_metrics_mod.QualityReport(
        avg_confidence=1.0, filler_rate=0.0, punctuation_score=1.0,
        capitalisation_score=1.0, diarization_coverage=1.0, duplicate_word_rate=0.0,
        low_confidence_rate=0.0, num_speakers=2, avg_segment_duration=5.0,
        speaking_time={}, suspicious_switches=[], total_segments=10, total_words=80,
    )
    # overall = 0.25*1+0.20*1+0.15*1+0.10*1+0.15*1+0.15*1 = 1.0 → A
    assert report.grade.startswith("A")


def test_quality_report_grade_b(stage1_metrics_mod):
    """Mid-range fields → grade B (≥0.70)."""
    # overall = 0.25*0.5 + 0.20*(1-0) + 0.15*0.5 + 0.10*0.5 + 0.15*1.0 + 0.15*1.0
    #         = 0.125 + 0.20 + 0.075 + 0.05 + 0.15 + 0.15 = 0.75 → B
    report = stage1_metrics_mod.QualityReport(
        avg_confidence=0.5, filler_rate=0.0, punctuation_score=0.5,
        capitalisation_score=0.5, diarization_coverage=1.0, duplicate_word_rate=0.0,
        low_confidence_rate=0.1, num_speakers=2, avg_segment_duration=3.0,
        speaking_time={}, suspicious_switches=[], total_segments=5, total_words=30,
    )
    assert report.grade.startswith("B")


def test_quality_report_grade_d(stage1_metrics_mod):
    """Very poor fields → grade D (< 0.50)."""
    # overall = 0.25*0 + 0.20*(1-1.0) + 0.15*0 + 0.10*0 + 0.15*0 + 0.15*(1-1.0)
    #         = 0 + 0 + 0 + 0 + 0 + 0 = 0.0 → D
    report = stage1_metrics_mod.QualityReport(
        avg_confidence=0.0, filler_rate=0.2, punctuation_score=0.0,
        capitalisation_score=0.0, diarization_coverage=0.0, duplicate_word_rate=0.1,
        low_confidence_rate=0.5, num_speakers=1, avg_segment_duration=1.0,
        speaking_time={}, suspicious_switches=[], total_segments=5, total_words=30,
    )
    assert report.grade.startswith("D")


def test_quality_report_issues_returns_list(stage1_metrics_mod):
    report = _make_report(stage1_metrics_mod, MINIMAL_SEGMENTS)
    issues = report.issues()
    assert isinstance(issues, list)
