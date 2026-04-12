"""
Stage 1, File 2: Build a Canonical Transcript Object
====================================================
CONCEPT: Normalize WhisperX-like JSON into a stable internal structure.

Agents work better when the input format is predictable. This file shows how
we map raw JSON into a TranscriptDocument with TranscriptSegment / WordToken.

Run this file:
  uv run deep_research/asr-v2/stage_01_basics/02_build_canonical_transcript.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.transcript_utils import dataclass_dump, document_to_plain_text, load_transcript


sample_path = ROOT / "sample_data" / "meeting_sample.json"
doc = load_transcript(sample_path)

print("=" * 60)
print("  Canonical transcript object")
print("=" * 60)
print(f"Language: {doc.language}")
print(f"Segments: {len(doc.segments)}")
print(f"Metadata keys: {sorted(doc.metadata.keys())}")

first_segment = dataclass_dump(doc)["segments"][0]
print("\nFirst normalized segment:")
for key, value in first_segment.items():
    if key == "words":
        print(f"  {key}: {len(value)} word token(s)")
    else:
        print(f"  {key}: {value}")

print("\nReadable preview:")
print(document_to_plain_text(doc))
