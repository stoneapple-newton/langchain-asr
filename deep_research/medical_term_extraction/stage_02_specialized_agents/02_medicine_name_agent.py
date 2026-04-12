"""
Stage 2, File 2: Specialised Medicine Name Extraction Agent
============================================================
CONCEPT: A dedicated agent focused solely on identifying drug/medicine names,
dosages, routes, and indications from medical transcripts.

Why separate from the medical term agent?
  - Drug names have distinctive ASR failure modes (homophones, phonetic
    spelling: "lissanopril" → "lisinopril", "metphormin" → "metformin").
  - Medicines need richer structured metadata: dosage, route, frequency,
    and the clinical indication — information a combined term extractor
    tends to miss or conflate.

New patterns introduced:
  - Few-shot examples in the system prompt to anchor the LLM on ASR-error
    correction for drug names
  - Post-processing step to phonetically cluster duplicate entries
    (same canonical but different ASR spellings → keep highest confidence)

Run this file:
  uv run deep_research/medical_term_extraction/stage_02_specialized_agents/02_medicine_name_agent.py
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
from deep_research.medical_term_extraction.shared.agents import MedicineNameAgent
from deep_research.medical_term_extraction.shared.extraction_models import (
    ExtractedMedicine,
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
    print("Stage 2b: Specialised Medicine Name Agent")
    print("=" * 60)

    doc = load_transcript(sample_path)
    print(f"Loaded {len(doc.segments)} segments (language: {doc.language})\n")

    agent = MedicineNameAgent()
    medicines = agent.run(doc)

    print(f"\nExtracted {len(medicines)} unique medicine(s):\n")
    for med in medicines:
        header = f"  [{med.confidence:6}] {med.canonical:<28}"
        meta_parts = []
        if med.dosage:
            meta_parts.append(f"dose={med.dosage}")
        if med.route:
            meta_parts.append(f"route={med.route}")
        if med.frequency:
            meta_parts.append(f"freq={med.frequency}")
        if med.indication:
            meta_parts.append(f"for={med.indication}")
        print(header + ("  " + "  ".join(meta_parts) if meta_parts else ""))
        if med.raw_text.lower() != med.canonical.lower():
            print(f"           spoken as: \"{med.raw_text}\"")
        if med.brand_name:
            print(f"           brand:     {med.brand_name}")
        if med.context:
            print(f"           context:   \"{med.context[:80]}\"")

    out_path = Path(__file__).parent / "medicine_names.json"
    out_path.write_text(
        json.dumps([m.model_dump() for m in medicines], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
