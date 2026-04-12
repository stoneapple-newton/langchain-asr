"""
Stage 2, File 1: Rule-based Cleanup Tools
=========================================
CONCEPT: Fix obvious ASR / diarization problems deterministically before using
an LLM. Fast, inspectable heuristics should handle the low-risk repairs.

Run this file:
  uv run deep_research/asr-v2/stage_02_tools/01_rule_based_cleanup.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.pipelines import run_rule_based_cleanup


sample_path = ROOT / "sample_data" / "meeting_sample.json"
output_dir = ROOT / "outputs"

result = run_rule_based_cleanup(str(sample_path), str(output_dir))
repaired_doc = result["repaired_doc"]
before = result["analysis_before"]
after = result["analysis_after"]
paths = result["output_paths"]

print("=" * 60)
print("  Rule-based transcript cleanup")
print("=" * 60)
print(f"Missing speaker rows: {before['missing_speaker_segments']} -> {after['missing_speaker_segments']}")
print(f"Segments:             {before['segment_count']} -> {after['segment_count']}")
print(f"Markdown output:      {paths['markdown_path']}")
print(f"JSON output:          {paths['json_path']}")

print("\nCleaned preview:")
for segment in repaired_doc.segments[:5]:
    print(
        f"  [{segment.start:05.2f}-{segment.end:05.2f}] "
        f"{segment.speaker or 'UNKNOWN'}: {segment.text}"
    )
