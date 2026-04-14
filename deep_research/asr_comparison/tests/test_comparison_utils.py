"""
Tests for deep_research/asr_comparison/shared/comparison_utils.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ASR_V2_ROOT = Path(__file__).resolve().parents[2] / "asr-v2"
REPO_ROOT = Path(__file__).resolve().parents[3]

for p in (REPO_ROOT, ASR_V2_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

# Both modules live in asr-v2/shared/
from shared.transcript_utils import TranscriptDocument, TranscriptSegment
from shared.comparison_utils import (
    AlignedPair,
    DiffCategory,
    _levenshtein,
    _normalise,
    _only_filler_diff,
    _only_readability_diff,
    align_segments,
    categorise_all,
    categorise_pair,
    compute_document_summary,
    compute_wer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(*segs: tuple) -> TranscriptDocument:
    """Create a minimal TranscriptDocument from (start, end, text, speaker?) tuples."""
    segments = []
    for i, seg in enumerate(segs):
        if len(seg) == 3:
            start, end, text = seg
            speaker = None
        else:
            start, end, text, speaker = seg
        segments.append(TranscriptSegment(
            segment_id=str(i),
            start=start,
            end=end,
            text=text,
            speaker=speaker,
        ))
    return TranscriptDocument(
        source_path="synthetic.json",
        language="en",
        raw_data={},
        segments=segments,
    )


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def test_normalise_strips_punctuation_and_lowercases():
    assert _normalise("Hello, World!") == "hello world"
    assert _normalise("  multiple   spaces  ") == "multiple spaces"


def test_only_readability_diff_detects_punct_change():
    assert _only_readability_diff("hello world", "Hello, World!") is True
    assert _only_readability_diff("hello world", "hello earth") is False


def test_only_filler_diff_detects_filler_tokens():
    assert _only_filler_diff(["um", "sure"], ["sure"]) is True
    assert _only_filler_diff(["sure"], ["certain"]) is False
    assert _only_filler_diff(["hello", "there"], ["hello", "there"]) is False


# ---------------------------------------------------------------------------
# WER / Levenshtein
# ---------------------------------------------------------------------------

def test_levenshtein_identical_lists():
    assert _levenshtein(["a", "b", "c"], ["a", "b", "c"]) == (0, 0, 0)


def test_levenshtein_one_substitution():
    s, i, d = _levenshtein(["a", "b", "c"], ["a", "x", "c"])
    assert s == 1 and i == 0 and d == 0


def test_levenshtein_one_deletion():
    s, i, d = _levenshtein(["a", "b", "c"], ["a", "c"])
    assert d == 1 and i == 0 and s == 0


def test_levenshtein_one_insertion():
    s, i, d = _levenshtein(["a", "c"], ["a", "b", "c"])
    assert i == 1 and d == 0 and s == 0


def test_compute_wer_perfect():
    assert compute_wer(["hello", "world"], ["hello", "world"]) == 0.0


def test_compute_wer_all_wrong():
    # 2 substitutions / 2 ref words = 1.0
    assert compute_wer(["hello", "world"], ["foo", "bar"]) == 1.0


def test_compute_wer_empty_ref():
    assert compute_wer([], []) == 0.0
    assert compute_wer([], ["extra"]) == 1.0


# ---------------------------------------------------------------------------
# Segment alignment
# ---------------------------------------------------------------------------

def test_align_overlapping_segments():
    doc_a = _doc((0.0, 2.0, "hello there", "SPK_A"))
    doc_b = _doc((0.5, 2.5, "hello there", "SPK_A"))
    pairs = align_segments(doc_a, doc_b)
    assert len(pairs) == 1
    assert pairs[0].hyp_text == "hello there"


def test_align_no_overlap_becomes_deletion_and_insertion():
    doc_a = _doc((0.0, 1.0, "early segment", "SPK_A"))
    doc_b = _doc((5.0, 6.0, "late segment", "SPK_B"))
    pairs = align_segments(doc_a, doc_b)
    categories = {p.category for p in pairs}
    assert DiffCategory.DELETION in categories
    assert DiffCategory.INSERTION in categories


def test_align_multiple_segments_greedy_best_match():
    doc_a = _doc(
        (0.0, 2.0, "first", "SPK_A"),
        (3.0, 5.0, "second", "SPK_A"),
    )
    doc_b = _doc(
        (0.2, 2.2, "first", "SPK_A"),
        (3.1, 4.9, "second", "SPK_A"),
    )
    pairs = align_segments(doc_a, doc_b)
    assert len(pairs) == 2
    assert all(p.category != DiffCategory.DELETION for p in pairs)
    assert all(p.hyp_text is not None for p in pairs)


def test_align_uses_sample_files():
    """Integration test: load both sample files and verify at least one match."""
    SAMPLE_A = ROOT / "sample_data" / "transcript_a.json"
    SAMPLE_B = ROOT / "sample_data" / "transcript_b.json"
    from shared.transcript_utils import load_transcript
    doc_a = load_transcript(SAMPLE_A)
    doc_b = load_transcript(SAMPLE_B)
    pairs = align_segments(doc_a, doc_b)
    assert len(pairs) >= len(doc_a.segments)
    matched = [p for p in pairs if p.hyp_text is not None]
    assert len(matched) > 0


# ---------------------------------------------------------------------------
# Pair categorisation
# ---------------------------------------------------------------------------

def _make_pair(ref: str, hyp: str, ref_spk: str = "SPK_A", hyp_spk: str = "SPK_A") -> AlignedPair:
    return AlignedPair(
        pair_id="test",
        ref_start=0.0, ref_end=1.0,
        ref_text=ref, ref_speaker=ref_spk,
        hyp_text=hyp, hyp_speaker=hyp_spk,
    )


def test_categorise_exact_match():
    pair = categorise_pair(_make_pair("hello world", "hello world"))
    assert pair.category == DiffCategory.MATCH
    assert pair.wer == 0.0


def test_categorise_readability_punct_caps():
    pair = categorise_pair(_make_pair("hello world", "Hello, World!"))
    assert pair.category == DiffCategory.READABILITY
    assert pair.wer == 0.0


def test_categorise_filler_word_removal():
    pair = categorise_pair(_make_pair("um sure let's go", "sure let's go"))
    assert pair.category == DiffCategory.FILLER_WORD


def test_categorise_speaker_mismatch_same_text():
    pair = categorise_pair(_make_pair("hello world", "hello world", "SPK_A", "SPK_B"))
    assert pair.category == DiffCategory.SPEAKER_MISMATCH


def test_categorise_substitution_low_wer():
    # "weather" → "whether": 1 substitution / 10 words = 0.1 WER
    pair = categorise_pair(_make_pair(
        "great the weather looks good for the demo on friday",
        "great the whether looks good for the demo on friday",
    ))
    assert pair.category == DiffCategory.SUBSTITUTION
    assert pair.wer > 0


def test_categorise_deletion():
    pair = AlignedPair(
        pair_id="del_0",
        ref_start=0.0, ref_end=1.0,
        ref_text="deleted segment", ref_speaker="SPK_A",
        hyp_text=None, hyp_speaker=None,
        category=DiffCategory.DELETION,
    )
    result = categorise_pair(pair)
    assert result.category == DiffCategory.DELETION


def test_categorise_insertion():
    pair = AlignedPair(
        pair_id="ins_0",
        ref_start=0.0, ref_end=1.0,
        ref_text="", ref_speaker=None,
        hyp_text="extra segment", hyp_speaker="SPK_B",
        category=DiffCategory.INSERTION,
    )
    result = categorise_pair(pair)
    assert result.category == DiffCategory.INSERTION


def test_categorise_all_processes_list():
    pairs = [
        _make_pair("hello world", "hello world"),
        _make_pair("um sure", "sure"),
        _make_pair("great weather", "great whether"),
    ]
    # Wrap in AlignedPair list (already are from _make_pair)
    diffs = categorise_all(pairs)
    assert len(diffs) == 3
    assert diffs[0].category == DiffCategory.MATCH
    assert diffs[1].category == DiffCategory.FILLER_WORD


# ---------------------------------------------------------------------------
# Document summary
# ---------------------------------------------------------------------------

def test_compute_document_summary_counts_categories():
    pairs = [
        AlignedPair("p0", 0.0, 1.0, "hello", "SPK_A", "hello", "SPK_A",
                    category=DiffCategory.MATCH),
        AlignedPair("p1", 1.0, 2.0, "weather good", "SPK_A", "whether good", "SPK_A",
                    category=DiffCategory.SUBSTITUTION, wer=0.5),
        AlignedPair("p2", 2.0, 3.0, "um sure", "SPK_A", "sure", "SPK_A",
                    category=DiffCategory.FILLER_WORD, wer=0.5),
    ]
    summary = compute_document_summary(pairs)
    assert summary["category_counts"]["match"] == 1
    assert summary["category_counts"]["substitution"] == 1
    assert summary["category_counts"]["filler_word"] == 1
    assert summary["total_pairs"] == 3
    assert isinstance(summary["overall_wer"], float)


def test_compute_document_summary_overall_wer_with_deletion():
    pairs = [
        AlignedPair("d0", 0.0, 1.0, "hello world", "SPK_A", None, None,
                    category=DiffCategory.DELETION),
    ]
    summary = compute_document_summary(pairs)
    # 2 deleted / 2 ref words = 1.0
    assert summary["overall_wer"] == 1.0
    assert summary["word_errors"]["deletions"] == 2
