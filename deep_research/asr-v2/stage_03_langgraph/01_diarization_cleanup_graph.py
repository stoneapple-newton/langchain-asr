"""
Stage 3, File 1: Diarization Cleanup Graph
==========================================
CONCEPT: Turn the cleanup pipeline into explicit LangGraph nodes and state.

Run this file:
  uv run deep_research/asr-v2/stage_03_langgraph/01_diarization_cleanup_graph.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.pipelines import build_diarization_cleanup_graph


app = build_diarization_cleanup_graph()

sample_path = ROOT / "sample_data" / "meeting_sample.json"
result = app.invoke({"input_path": str(sample_path)})

print("=" * 60)
print("  Diarization cleanup graph")
print("=" * 60)
print(f"Missing speaker rows: {result['analysis_before']['missing_speaker_segments']} -> {result['analysis_after']['missing_speaker_segments']}")
print(f"Segments:             {result['analysis_before']['segment_count']} -> {result['analysis_after']['segment_count']}")
