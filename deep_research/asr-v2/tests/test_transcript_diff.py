from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.transcript_diff import build_comparison_report, format_report_markdown


REFERENCE_PATH = ROOT / "sample_data" / "meeting_sample.json"
CANDIDATE_PATH = ROOT / "sample_data" / "meeting_sample_variant.json"


def test_build_comparison_report_categorizes_expected_changes():
    report = build_comparison_report(REFERENCE_PATH, CANDIDATE_PATH)
    counts = report["totals"]["by_category"]

    assert counts["punctuation_or_case"] == 1
    assert counts["timing_shift"] == 1
    assert counts["wording_change"] == 2
    assert counts["deletion"] == 1
    assert counts["speaker_change"] == 1
    assert counts["insertion"] == 1


def test_markdown_formatter_hides_unchanged_by_default():
    report = build_comparison_report(REFERENCE_PATH, CANDIDATE_PATH)
    rendered = format_report_markdown(report)

    assert "**unchanged**" not in rendered
    assert "**speaker_change**" in rendered
    assert "Segment `6`" in rendered
