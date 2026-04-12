"""
Stage 3: verify a suspicious span with audio or text fallback.

Run:
  uv run deep_research/asr_medical_verification/stage_03_transcription_verification/01_verify_transcription.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_research.asr_medical_verification.shared.medical_glossary import build_llm_hint, link_medical_terms, load_medical_glossary
from deep_research.asr_medical_verification.shared.retranscription import retranscribe_segment
from deep_research.asr_medical_verification.shared.transcript_utils import build_context_text, load_transcript


def verify_segment(
    transcript_path: str | Path,
    *,
    segment_idx: int,
    audio_clip_path: str | Path | None = None,
    retranscription_method: str = "auto",
    glossary_path: str | Path | None = None,
) -> dict:
    doc = load_transcript(transcript_path)
    glossary = load_medical_glossary(glossary_path)
    segment = doc.segments[segment_idx]
    context = build_context_text(doc, segment_idx)
    specialty_hint = build_llm_hint("general", glossary, max_terms=6)
    if audio_clip_path is None:
        corrected_text, confidence, method = retranscribe_segment(
            "missing.wav",
            method="none",
            context_text=context,
            original_text=segment.text,
            issue_reason="Manual verification request",
            specialty_hint=specialty_hint,
        )
    else:
        corrected_text, confidence, method = retranscribe_segment(
            audio_clip_path,
            method=retranscription_method,
            context_text=context,
            original_text=segment.text,
            issue_reason="Manual verification request",
            specialty_hint=specialty_hint,
        )
    return {
        "segment_id": segment.segment_id,
        "original_text": segment.text,
        "corrected_text": corrected_text,
        "confidence": confidence,
        "method": method,
        "linked_terms": link_medical_terms(corrected_text, glossary),
    }


def main() -> None:
    sample = {
        "note": "Import `verify_segment(...)` with a real audio clip or use the stage-5 workflow.",
    }
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()
