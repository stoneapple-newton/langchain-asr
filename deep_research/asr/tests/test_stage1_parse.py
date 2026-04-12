"""Tests for stage_01_basics/01_load_and_parse.py"""

import copy
import pytest
from .conftest import MINIMAL_SEGMENTS


# ---------------------------------------------------------------------------
# Word tests
# ---------------------------------------------------------------------------

def test_word_low_confidence_boundary_above(stage1_parse_mod):
    Word = stage1_parse_mod.Word
    w = Word(word="test", start=0.0, end=1.0, score=0.75, speaker="SPEAKER_00")
    assert w.is_low_confidence is False


def test_word_low_confidence_boundary_below(stage1_parse_mod):
    Word = stage1_parse_mod.Word
    w = Word(word="test", start=0.0, end=1.0, score=0.74, speaker="SPEAKER_00")
    assert w.is_low_confidence is True


@pytest.mark.parametrize("word", ["uh", "um", "like", "you know", "i mean"])
def test_word_is_filler_true(stage1_parse_mod, word):
    Word = stage1_parse_mod.Word
    w = Word(word=word, start=0.0, end=0.5, score=0.9, speaker="SPEAKER_00")
    assert w.is_filler is True


def test_word_is_filler_false(stage1_parse_mod):
    Word = stage1_parse_mod.Word
    w = Word(word="roadmap", start=0.0, end=0.5, score=0.9, speaker="SPEAKER_00")
    assert w.is_filler is False


# ---------------------------------------------------------------------------
# Segment tests
# ---------------------------------------------------------------------------

def test_segment_avg_confidence_no_words(stage1_parse_mod):
    Segment = stage1_parse_mod.Segment
    seg = Segment(start=0.0, end=1.0, text="hello", speaker="SPEAKER_00", words=[])
    assert seg.avg_confidence == 0.0


def test_segment_avg_confidence_with_words(stage1_parse_mod):
    Segment = stage1_parse_mod.Segment
    Word = stage1_parse_mod.Word
    words = [
        Word(word="hello", start=0.0, end=0.5, score=0.8, speaker="SPEAKER_00"),
        Word(word="world", start=0.6, end=1.0, score=1.0, speaker="SPEAKER_00"),
    ]
    seg = Segment(start=0.0, end=1.0, text="hello world", speaker="SPEAKER_00", words=words)
    assert seg.avg_confidence == pytest.approx(0.9, abs=0.001)


# ---------------------------------------------------------------------------
# parse_transcript tests
# ---------------------------------------------------------------------------

def _raw(segs):
    return {"segments": segs, "language": "en", "duration": 30.0}


def test_parse_transcript_segment_count(stage1_parse_mod):
    transcript = stage1_parse_mod.parse_transcript(_raw(MINIMAL_SEGMENTS))
    assert len(transcript.segments) == 5


def test_parse_transcript_preserves_speaker(stage1_parse_mod):
    transcript = stage1_parse_mod.parse_transcript(_raw(MINIMAL_SEGMENTS))
    assert transcript.segments[2].speaker == "SPEAKER_01"


def test_parse_transcript_word_count(stage1_parse_mod):
    transcript = stage1_parse_mod.parse_transcript(_raw(MINIMAL_SEGMENTS))
    # Seg 0 has 4 words
    assert len(transcript.segments[0].words) == 4


def test_parse_transcript_empty(stage1_parse_mod):
    transcript = stage1_parse_mod.parse_transcript(_raw([]))
    assert len(transcript.segments) == 0
