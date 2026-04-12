"""
Stage 1, File 1: Medical Term Extraction with a Single LLM Chain
=================================================================
CONCEPT: Load a WhisperX-style transcript, build a plain-text rendering,
then ask the LLM to extract medical terms and medicine names in a single
structured pass.

Key patterns used:
  - Shared transcript loader from asr_medical_verification
  - ChatPromptTemplate + JsonOutputParser with Pydantic schema
  - LCEL chain  (prompt | llm | parser)
  - Transcript chunking to stay within context limits

Run this file:
  uv run deep_research/medical_term_extraction/stage_01_extraction_basics/01_extract_with_llm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from deep_research.asr_medical_verification.shared.transcript_utils import (
    TranscriptDocument,
    TranscriptSegment,
    load_transcript,
)
from deep_research.medical_term_extraction.shared.extraction_models import (
    ExtractedMedicalTerm,
    ExtractedMedicine,
    SegmentExtractionResult,
)

# ---------------------------------------------------------------------------
# LLM + chain
# ---------------------------------------------------------------------------

llm = create_chat_model("asr_v2", temperature=0, max_tokens=1024)

parser = JsonOutputParser(pydantic_object=SegmentExtractionResult)

SYSTEM_PROMPT = """\
You are a clinical language expert specialising in medical transcription analysis.
Given a chunk of a medical audio transcript, extract two types of items:

1. **Medical terms** – conditions, symptoms, clinical findings, procedures,
   anatomical references, measurements, and diagnoses.
2. **Medicine names** – drug names, medications, supplements, and dosages
   mentioned by any speaker.

Important ASR context:
- The transcript was produced by automatic speech recognition and may contain
  spelling mistakes or split compound words (e.g. "a trial fibrillation"
  instead of "atrial fibrillation", "lissanopril" instead of "lisinopril").
- Correct obvious ASR errors in the `canonical` field while preserving the
  original spoken text in `raw_text`.
- Use "low" confidence when the term appears garbled or ambiguous.

Return ONLY valid JSON matching the schema below, with no extra prose.
{format_instructions}"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Transcript chunk:\n\n{chunk_text}\n\nExtract all medical terms and medicine names.",
        ),
    ]
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_segments(
    segments: list[TranscriptSegment], max_chars: int = 800
) -> list[list[TranscriptSegment]]:
    """Split segments into chunks that fit within the LLM context budget."""
    chunks: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_chars = 0
    for seg in segments:
        size = len(seg.text) + 30
        if current and current_chars + size > max_chars:
            chunks.append(current)
            current = []
            current_chars = 0
        current.append(seg)
        current_chars += size
    if current:
        chunks.append(current)
    return chunks


def _render_chunk(segments: list[TranscriptSegment]) -> str:
    lines = []
    for seg in segments:
        speaker = seg.speaker or "UNKNOWN"
        lines.append(f"[{seg.start:07.2f}-{seg.end:07.2f}] {speaker}: {seg.text}")
    return "\n".join(lines)


def _deduplicate_terms(
    terms: list[ExtractedMedicalTerm],
) -> list[ExtractedMedicalTerm]:
    seen: set[str] = set()
    unique: list[ExtractedMedicalTerm] = []
    for term in terms:
        key = term.canonical.lower()
        if key not in seen:
            seen.add(key)
            unique.append(term)
    return unique


def _deduplicate_medicines(
    medicines: list[ExtractedMedicine],
) -> list[ExtractedMedicine]:
    seen: set[str] = set()
    unique: list[ExtractedMedicine] = []
    for med in medicines:
        key = med.canonical.lower()
        if key not in seen:
            seen.add(key)
            unique.append(med)
    return unique


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_from_transcript(
    doc: TranscriptDocument,
) -> tuple[list[ExtractedMedicalTerm], list[ExtractedMedicine]]:
    """Run the LLM extraction chain over every chunk of the transcript."""
    chunks = _chunk_segments(doc.segments)
    all_terms: list[ExtractedMedicalTerm] = []
    all_medicines: list[ExtractedMedicine] = []

    print(f"Processing {len(doc.segments)} segments in {len(chunks)} chunk(s)...")

    for idx, chunk in enumerate(chunks, start=1):
        chunk_text = _render_chunk(chunk)
        print(f"  Chunk {idx}/{len(chunks)} ({len(chunk)} segments)...")
        try:
            raw = chain.invoke({"chunk_text": chunk_text})
            result = SegmentExtractionResult(**raw) if isinstance(raw, dict) else raw
            all_terms.extend(result.medical_terms)
            all_medicines.extend(result.medicine_names)
        except Exception as exc:
            print(f"  [warn] chunk {idx} failed: {exc}")

    return _deduplicate_terms(all_terms), _deduplicate_medicines(all_medicines)


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
    print("Stage 1: Medical Term Extraction – LLM Chain")
    print("=" * 60)
    print(f"Transcript: {sample_path.name}")

    doc = load_transcript(sample_path)
    print(f"Loaded {len(doc.segments)} segments (language: {doc.language})\n")

    medical_terms, medicine_names = extract_from_transcript(doc)

    print("\n" + "=" * 60)
    print(f"Medical Terms Found ({len(medical_terms)})")
    print("=" * 60)
    for term in medical_terms:
        print(
            f"  [{term.confidence:6}] {term.canonical:<30} "
            f"category={term.category}  specialty={term.specialty}"
        )
        if term.raw_text.lower() != term.canonical.lower():
            print(f"           (spoken as: \"{term.raw_text}\")")

    print("\n" + "=" * 60)
    print(f"Medicine Names Found ({len(medicine_names)})")
    print("=" * 60)
    for med in medicine_names:
        parts = [f"  [{med.confidence:6}] {med.canonical:<28}"]
        if med.dosage:
            parts.append(f"dosage={med.dosage}")
        if med.route:
            parts.append(f"route={med.route}")
        if med.indication:
            parts.append(f"for={med.indication}")
        print("  ".join(parts))
        if med.raw_text.lower() != med.canonical.lower():
            print(f"           (spoken as: \"{med.raw_text}\")")

    # Save results next to this script
    output = {
        "source": str(sample_path),
        "medical_terms": [t.model_dump() for t in medical_terms],
        "medicine_names": [m.model_dump() for m in medicine_names],
    }
    out_path = Path(__file__).parent / "extraction_results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
