from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.pipelines import build_diarization_cleanup_graph, run_rule_based_cleanup


SAMPLE_PATH = ROOT / "sample_data" / "meeting_sample.json"


def test_diarization_cleanup_graph_returns_expected_before_and_after_metrics():
    app = build_diarization_cleanup_graph()
    result = app.invoke({"input_path": str(SAMPLE_PATH)})
    assert result["analysis_before"]["segment_count"] == 6
    assert result["analysis_after"]["segment_count"] == 4
    assert result["analysis_after"]["missing_speaker_segments"] == 0


def test_rule_based_cleanup_pipeline_saves_under_expected_directory(tmp_path):
    result = run_rule_based_cleanup(str(SAMPLE_PATH), str(tmp_path))
    json_path = Path(result["output_paths"]["json_path"])
    markdown_path = Path(result["output_paths"]["markdown_path"])

    assert json_path.parent.name == SAMPLE_PATH.stem
    assert markdown_path.parent == json_path.parent
    assert result["analysis_before"]["segment_count"] == 6
    assert result["analysis_after"]["segment_count"] == 4
