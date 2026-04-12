from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.transcript_utils import (
    TranscriptDocument,
    TranscriptSegment,
    WordToken,
    analyze_transcript,
    improve_readability,
    load_transcript,
    render_markdown,
    repair_diarization,
    save_enhanced_outputs,
)


SAMPLE_PATH = ROOT / "sample_data" / "meeting_sample.json"


def test_loader_accepts_sample_fixture():
    doc = load_transcript(SAMPLE_PATH)
    assert doc.language == "en"
    assert doc.metadata["job_name"] == "demo_team_sync"
    assert len(doc.segments) == 6


def test_loader_derives_missing_segment_speakers_from_words():
    doc = load_transcript(SAMPLE_PATH)
    assert doc.segments[1].speaker == "SPEAKER_01"
    assert doc.segments[5].speaker == "SPEAKER_01"


def test_analyze_transcript_reports_expected_metrics():
    stats = analyze_transcript(load_transcript(SAMPLE_PATH))
    assert stats == {
        "segment_count": 6,
        "speaker_count": 2,
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "missing_speaker_segments": 0,
        "single_word_segments": 2,
        "speaker_changes": 3,
        "duration_seconds": 10.7,
    }


def test_repair_diarization_fills_missing_speakers_and_merges_fragments():
    repaired = repair_diarization(load_transcript(SAMPLE_PATH))
    stats = analyze_transcript(repaired)
    assert stats["missing_speaker_segments"] == 0
    assert stats["segment_count"] == 4
    assert repaired.segments[1].speaker == "SPEAKER_01"
    assert repaired.segments[1].text == "sure we fixed the payment bug and i think the dashboard issue too but"


def test_repair_diarization_smooths_isolated_speaker_flip_without_merging_neighbors():
    synthetic = TranscriptDocument(
        source_path="synthetic.json",
        language="en",
        raw_data={},
        segments=[
            TranscriptSegment("0", 0.0, 1.0, "hello there", "SPEAKER_00"),
            TranscriptSegment(
                "1",
                1.2,
                1.6,
                "yes",
                "SPEAKER_01",
                words=[WordToken("yes", 1.2, 1.6, "SPEAKER_01")],
            ),
            TranscriptSegment("2", 2.8, 4.0, "continue please", "SPEAKER_00"),
        ],
    )

    repaired = repair_diarization(synthetic, max_gap_seconds=0.1)
    assert repaired.segments[1].speaker == "SPEAKER_00"
    assert repaired.segments[1].metadata["speaker_smoothed"] is True
    assert len(repaired.segments) == 3


def test_improve_readability_updates_text_but_preserves_structure():
    repaired = repair_diarization(load_transcript(SAMPLE_PATH))
    improved = improve_readability(repaired)
    assert len(improved.segments) == len(repaired.segments)
    assert improved.segments[0].speaker == repaired.segments[0].speaker
    assert improved.segments[0].start == repaired.segments[0].start
    assert improved.segments[0].text == "Okay lets get started with the release update."


def test_render_markdown_includes_heading_and_speaker_lines():
    markdown = render_markdown(improve_readability(repair_diarization(load_transcript(SAMPLE_PATH))))
    assert "# Enhanced Transcript" in markdown
    assert "**SPEAKER_00**" in markdown
    assert "**SPEAKER_01**" in markdown
    assert "meeting_sample.json" in markdown


def test_save_outputs_writes_json_and_markdown(tmp_path):
    doc = improve_readability(repair_diarization(load_transcript(SAMPLE_PATH)))
    paths = save_enhanced_outputs(doc, SAMPLE_PATH, tmp_path)
    json_path = Path(paths["json_path"])
    markdown_path = Path(paths["markdown_path"])
    assert json_path.exists()
    assert markdown_path.exists()
    assert '"speaker": "SPEAKER_00"' in json_path.read_text(encoding="utf-8")
    assert "# Enhanced Transcript" in markdown_path.read_text(encoding="utf-8")
