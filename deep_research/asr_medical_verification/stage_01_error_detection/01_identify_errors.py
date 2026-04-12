"""
Stage 1: identify suspicious transcript spans and medical-term candidates.

Run:
  uv run deep_research/asr_medical_verification/stage_01_error_detection/01_identify_errors.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_research.asr_medical_verification.shared.detection import identify_error_candidates
from deep_research.asr_medical_verification.shared.medical_glossary import load_medical_glossary
from deep_research.asr_medical_verification.shared.transcript_utils import analyze_transcript, load_transcript


HERE = Path(__file__).resolve().parent
SAMPLE_TRANSCRIPT = HERE.parent / "data" / "sample_transcript.json"
SAMPLE_GLOSSARY = HERE.parent / "data" / "medical_glossary.json"


def identify_errors(
    transcript_path: str | Path,
    *,
    audio_path: str | Path | None = None,
    glossary_path: str | Path | None = None,
    confidence_threshold: float = 0.75,
) -> dict:
    doc = load_transcript(transcript_path, audio_path)
    glossary = load_medical_glossary(glossary_path)
    stats = analyze_transcript(doc)
    results = identify_error_candidates(doc, glossary, confidence_threshold=confidence_threshold)
    return {
        "transcript_path": str(transcript_path),
        "audio_path": doc.audio_path,
        "statistics": stats,
        "error_summary": {
            "low_confidence_count": len(results["low_confidence"]),
            "homophone_count": len(results["homophones"]),
            "unknown_term_count": len(results["unknown_terms"]),
            "repetition_count": len(results["repetitions"]),
            "medical_candidate_count": len(results["medical_candidates"]),
            "total_unique_errors": len(results["all_errors"]),
        },
        "candidates": [
            {
                "segment_idx": item.segment_idx,
                "segment_id": doc.segments[item.segment_idx].segment_id,
                "word_idx": item.word_idx,
                "word": item.word,
                "score": item.score,
                "speaker": item.speaker,
                "segment_text": item.segment_text,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "error_type": item.error_type,
                "context": f"{item.left_context} [{item.word}] {item.right_context}".strip(),
                "text": doc.segments[item.segment_idx].text,
            }
            for item in results["all_errors"]
        ],
        "medical_candidates": [
            {
                "segment_idx": item.segment_idx,
                "segment_id": item.segment_id,
                "start_time": item.start_time,
                "end_time": item.end_time,
                "text": item.text,
                "speaker": item.speaker,
                "suspected_term": item.suspected_term,
                "specialty": item.specialty,
                "reason": item.reason,
                "score": item.score,
            }
            for item in results["medical_candidates"]
        ],
    }


def main() -> None:
    report = identify_errors(SAMPLE_TRANSCRIPT, glossary_path=SAMPLE_GLOSSARY)
    output_path = HERE / "error_report.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["error_summary"], indent=2))
    print(f"\nSaved report to {output_path}")


if __name__ == "__main__":
    main()
