"""
Stage 1, File 1: Inspect a WhisperX-like JSON Transcript
=========================================================
CONCEPT: Before building an ASR agent, understand the raw transcript shape.

This file loads a WhisperX-like JSON export and answers:
  - How many segments do we have?
  - How many speakers appear?
  - Where are speaker labels missing?
  - What obvious diarization problems should we repair first?

Run this file:
  uv run deep_research/asr-v2/stage_01_basics/01_inspect_whisperx_json.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.transcript_utils import analyze_transcript, load_transcript


sample_path = ROOT / "sample_data" / "meeting_sample.json"
doc = load_transcript(sample_path)
stats = analyze_transcript(doc)

print("=" * 60)
print("  WhisperX-like transcript inspection")
print("=" * 60)
print(f"Source file:          {sample_path.name}")
print(f"Segments:             {stats['segment_count']}")
print(f"Speakers:             {stats['speaker_count']} -> {stats['speakers']}")
print(f"Missing speaker rows: {stats['missing_speaker_segments']}")
print(f"Single-word rows:     {stats['single_word_segments']}")
print(f"Speaker changes:      {stats['speaker_changes']}")
print(f"Duration (seconds):   {stats['duration_seconds']}")

print("\nPreview of raw segments:")
for segment in doc.segments[:5]:
    print(
        f"  [{segment.start:05.2f}-{segment.end:05.2f}] "
        f"{segment.speaker or 'UNKNOWN'}: {segment.text}"
    )

print("\nPotential cleanup opportunities:")
if stats["missing_speaker_segments"]:
    print("  - Some segments are missing speaker labels")
if stats["single_word_segments"]:
    print("  - Some short fragments may be merge candidates")
if stats["speaker_changes"] > stats["segment_count"] // 2:
    print("  - Speaker changes are frequent, which often means unstable diarization")
if not stats["missing_speaker_segments"] and not stats["single_word_segments"]:
    print("  - No obvious structural issues in this sample")
