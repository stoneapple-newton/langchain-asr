"""
Stage 2: extract audio clips for flagged spans.

Run:
  uv run deep_research/asr_medical_verification/stage_02_audio_extraction/01_extract_audio_segments.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_research.asr_medical_verification.shared.audio_utils import extract_audio_segment
from deep_research.asr_medical_verification.shared.detection import identify_error_candidates
from deep_research.asr_medical_verification.shared.medical_glossary import load_medical_glossary
from deep_research.asr_medical_verification.shared.transcript_utils import load_transcript


HERE = Path(__file__).resolve().parent
SAMPLE_TRANSCRIPT = HERE.parent / "data" / "sample_transcript.json"
OUTPUT_DIR = HERE / "clips"


def extract_candidate_segments(
    transcript_path: str | Path,
    *,
    audio_path: str | Path,
    output_dir: str | Path = OUTPUT_DIR,
) -> list[dict]:
    doc = load_transcript(transcript_path, audio_path)
    glossary = load_medical_glossary()
    report = identify_error_candidates(doc, glossary)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []
    for idx, candidate in enumerate(report["all_errors"][:10]):
        segment_id = doc.segments[candidate.segment_idx].segment_id
        output_path = output_dir / f"candidate_{idx:02d}_{segment_id}.wav"
        clip_path = extract_audio_segment(
            audio_path,
            candidate.start_time,
            candidate.end_time,
            output_path=output_path,
            padding=0.4,
        )
        clips.append(
            {
                "segment_id": segment_id,
                "segment_idx": candidate.segment_idx,
                "word": candidate.word,
                "error_type": candidate.error_type,
                "clip_path": str(clip_path),
            }
        )
    return clips


def main() -> None:
    print("Stage 2 expects a real audio file. Import `extract_candidate_segments(...)` in your own run.")
    print(f"Sample transcript: {SAMPLE_TRANSCRIPT}")


if __name__ == "__main__":
    main()
