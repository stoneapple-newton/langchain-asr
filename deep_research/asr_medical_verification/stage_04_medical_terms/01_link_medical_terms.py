"""
Stage 4: link multilingual medical terms from transcript context.

Run:
  uv run deep_research/asr_medical_verification/stage_04_medical_terms/01_link_medical_terms.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deep_research.asr_medical_verification.shared.medical_glossary import build_llm_hint, link_medical_terms, load_medical_glossary
from deep_research.asr_medical_verification.shared.transcript_utils import load_transcript


HERE = Path(__file__).resolve().parent
SAMPLE_TRANSCRIPT = HERE.parent / "data" / "sample_transcript.json"


def identify_and_link_terms(
    transcript_path: str | Path,
    *,
    glossary_path: str | Path | None = None,
) -> dict:
    doc = load_transcript(transcript_path)
    glossary = load_medical_glossary(glossary_path)
    results = []
    for segment in doc.segments:
        linked = link_medical_terms(segment.text, glossary)
        if linked:
            results.append(
                {
                    "segment_id": segment.segment_id,
                    "text": segment.text,
                    "linked_terms": linked,
                }
            )
    return {
        "segments_with_terms": results,
        "llm_hint_preview": build_llm_hint("general", glossary, max_terms=5),
    }


def main() -> None:
    result = identify_and_link_terms(SAMPLE_TRANSCRIPT)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
