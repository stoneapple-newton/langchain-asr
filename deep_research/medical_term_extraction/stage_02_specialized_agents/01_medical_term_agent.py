"""
Stage 2, File 1: Specialised Medical Term Extraction Agent
===========================================================
CONCEPT: A dedicated agent that focuses exclusively on medical terminology.
Using a narrow, expert system prompt for clinical terms yields fewer false
positives and better category/specialty attribution than the combined prompt
used in Stage 1.

New patterns introduced:
  - Separate specialist agent (single responsibility)
  - Glossary-hint injection: the system prompt includes canonical forms from
    the existing MultilingualMedicalGlossary so the LLM knows which terms
    are in scope for this recording's specialty
  - Batch processing with .batch() for parallel chunk calls
  - Speaker-aware context: terms are attributed to the speaking role

Run this file:
  uv run deep_research/medical_term_extraction/stage_02_specialized_agents/01_medical_term_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_research.asr_medical_verification.shared.transcript_utils import (
    TranscriptDocument,
    load_transcript,
)
from deep_research.medical_term_extraction.shared.agents import MedicalTermAgent
from deep_research.medical_term_extraction.shared.extraction_models import (
    ExtractedMedicalTerm,
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    sample_path = (
        REPO_ROOT
        / "deep_research"
        / "asr_medical_verification"
        / "data"
        / "sample_transcript.json"
    )

    print("=" * 60)
    print("Stage 2a: Specialised Medical Term Agent")
    print("=" * 60)

    doc = load_transcript(sample_path)
    print(f"Loaded {len(doc.segments)} segments (language: {doc.language})\n")

    agent = MedicalTermAgent(specialty="cardiology")
    terms = agent.run(doc)

    print(f"\nExtracted {len(terms)} unique medical term(s):\n")
    for term in terms:
        icd = f"  ICD: {term.icd_hint}" if term.icd_hint else ""
        print(
            f"  [{term.confidence:6}] {term.canonical:<32} "
            f"[{term.category}/{term.specialty}]{icd}"
        )
        if term.raw_text.lower() != term.canonical.lower():
            print(f"           spoken as: \"{term.raw_text}\"")
        if term.context:
            print(f"           context:   \"{term.context[:80]}\"")

    out_path = Path(__file__).parent / "medical_terms.json"
    out_path.write_text(
        json.dumps([t.model_dump() for t in terms], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
