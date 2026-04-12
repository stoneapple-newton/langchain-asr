"""
Stage 5, File 1: Batch ASR Quality Runner
=========================================
CONCEPT: Process a directory of WhisperX-like JSON files using the same repair
pipeline as the final agent stage.

Run this file:
  uv run deep_research/asr-v2/stage_05_production/01_batch_asr_quality_runner.py
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.transcript_utils import improve_readability, load_transcript, repair_diarization, save_enhanced_outputs


parser = argparse.ArgumentParser(description="Batch-process WhisperX-like JSON transcripts")
parser.add_argument(
    "--input-dir",
    default=str(ROOT / "sample_data"),
    help="Directory containing WhisperX-like JSON files",
)
parser.add_argument(
    "--output-dir",
    default=str(ROOT / "outputs"),
    help="Directory for enhanced JSON and markdown outputs",
)
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
files = sorted(input_dir.glob("*.json"))

print("=" * 60)
print("  Batch ASR quality runner")
print("=" * 60)
print(f"Input dir:  {input_dir}")
print(f"Output dir: {output_dir}")
print(f"Files:      {len(files)}")

for path in files:
    doc = improve_readability(repair_diarization(load_transcript(path)))
    paths = save_enhanced_outputs(doc, path, output_dir)
    print(f"  - {path.name} -> {paths['json_path']}")
