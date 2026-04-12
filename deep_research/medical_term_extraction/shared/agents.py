"""
Reusable agent classes for medical term and medicine name extraction.

Imported by both stage_02 demo scripts and the stage_03 LangGraph pipeline.
"""

from __future__ import annotations

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# config is on sys.path when repo root has been added (see each entry script)
from config import create_chat_model

from deep_research.asr_medical_verification.shared.medical_glossary import (
    build_llm_hint,
    load_medical_glossary,
)
from deep_research.asr_medical_verification.shared.transcript_utils import (
    TranscriptDocument,
    TranscriptSegment,
)
from deep_research.medical_term_extraction.shared.extraction_models import (
    ExtractedMedicalTerm,
    ExtractedMedicine,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}


def _chunk_segments(
    segments: list[TranscriptSegment], max_chars: int = 700
) -> list[list[TranscriptSegment]]:
    chunks: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    chars = 0
    for seg in segments:
        size = len(seg.text) + 30
        if current and chars + size > max_chars:
            chunks.append(current)
            current = []
            chars = 0
        current.append(seg)
        chars += size
    if current:
        chunks.append(current)
    return chunks


def _render_chunk(segments: list[TranscriptSegment]) -> str:
    return "\n".join(
        f"[{seg.start:07.2f}-{seg.end:07.2f}] {seg.speaker or 'UNKNOWN'}: {seg.text}"
        for seg in segments
    )


# ---------------------------------------------------------------------------
# MedicalTermAgent
# ---------------------------------------------------------------------------


class _MedicalTermList(BaseModel):
    terms: list[ExtractedMedicalTerm] = Field(default_factory=list)


_MEDICAL_TERM_SYSTEM = """\
You are a clinical NLP specialist. Your task is to identify ONLY medical
terminology in the transcript chunk below. Do NOT extract drug or medicine
names — those are handled by a separate agent.

Extract the following categories:
  • condition       – diseases, disorders, syndromes (e.g. atrial fibrillation)
  • symptom         – patient-reported or observed signs (e.g. chest pain)
  • finding         – lab/test results, objective observations
  • procedure       – surgeries, tests, exams (e.g. echocardiogram)
  • anatomy         – body parts, organs, systems (e.g. left ventricle)
  • measurement     – clinical values with units (e.g. "120 over 80", "SpO2 94%")

ASR correction rules:
  • Put the verbatim spoken text in `raw_text` and the corrected standard form
    in `canonical` (e.g. raw_text="a trial fibrillation", canonical="atrial fibrillation").
  • If the word sounds like a known medical term but is unclear, mark confidence="low".

Domain hints from the medical glossary:
{glossary_hint}

Return ONLY valid JSON — no markdown fences, no prose.
{format_instructions}"""

_medical_term_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _MEDICAL_TERM_SYSTEM),
        ("human", "Transcript chunk:\n\n{chunk_text}"),
    ]
)


class MedicalTermAgent:
    """Extract medical terms from a TranscriptDocument using a focused LLM chain."""

    def __init__(self, specialty: str = "general", max_tokens: int = 768) -> None:
        self.specialty = specialty
        glossary = load_medical_glossary()
        self._parser = JsonOutputParser(pydantic_object=_MedicalTermList)
        llm = create_chat_model("asr_v2", temperature=0, max_tokens=max_tokens)
        self._chain = (
            _medical_term_prompt.partial(
                format_instructions=self._parser.get_format_instructions(),
                glossary_hint=build_llm_hint(specialty, glossary, max_terms=10),
            )
            | llm
            | self._parser
        )

    def run(self, doc: TranscriptDocument) -> list[ExtractedMedicalTerm]:
        chunks = _chunk_segments(doc.segments)
        inputs = [{"chunk_text": _render_chunk(chunk)} for chunk in chunks]
        print(
            f"[MedicalTermAgent] {len(doc.segments)} segments → "
            f"{len(chunks)} chunk(s), specialty={self.specialty}"
        )
        try:
            batch_results = self._chain.batch(inputs, config={"max_concurrency": 4})
        except Exception as exc:
            print(f"[MedicalTermAgent] batch failed, falling back: {exc}")
            batch_results = []
            for inp in inputs:
                try:
                    batch_results.append(self._chain.invoke(inp))
                except Exception as inner:
                    print(f"[MedicalTermAgent] chunk failed: {inner}")
                    batch_results.append({"terms": []})

        all_terms: list[ExtractedMedicalTerm] = []
        for raw in batch_results:
            parsed = _MedicalTermList(**raw) if isinstance(raw, dict) else raw
            all_terms.extend(parsed.terms)
        return self._deduplicate(all_terms)

    @staticmethod
    def _deduplicate(terms: list[ExtractedMedicalTerm]) -> list[ExtractedMedicalTerm]:
        seen: set[str] = set()
        unique: list[ExtractedMedicalTerm] = []
        for term in terms:
            key = term.canonical.lower()
            if key not in seen:
                seen.add(key)
                unique.append(term)
        return unique


# ---------------------------------------------------------------------------
# MedicineNameAgent
# ---------------------------------------------------------------------------


class _MedicineList(BaseModel):
    medicines: list[ExtractedMedicine] = Field(default_factory=list)


_MEDICINE_SYSTEM = """\
You are a clinical pharmacist and NLP specialist. Your ONLY task is to extract
medicine and drug names from the transcript chunk below.

Do NOT extract generic medical conditions or procedures — focus on:
  • Prescription and over-the-counter drugs (generic and brand names)
  • Dosage amounts and units (e.g. "10 mg", "500 mcg", "2 puffs")
  • Route of administration (oral, IV, IM, topical, inhaled, sublingual, etc.)
  • Dosing frequency (once daily, BID, TID, QID, PRN, etc.)
  • The clinical indication when stated ("for hypertension", "for pain")

ASR correction rules — common drug-name mishearings:
  • "lissanopril" → lisinopril          • "metphormin" → metformin
  • "atorvastaten" → atorvastatin       • "amoxacillin" → amoxicillin
  • "warferin" → warfarin               • "hydroclorothiazide" → hydrochlorothiazide
  Put the verbatim text in `raw_text` and the corrected standard name in `canonical`.

Confidence rules:
  • "high"   – name is clear and unambiguous
  • "medium" – minor ASR distortion but recognisable
  • "low"    – phonetically similar to a known drug but unclear

Return ONLY valid JSON — no markdown fences, no prose.
{format_instructions}"""

_medicine_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _MEDICINE_SYSTEM),
        ("human", "Transcript chunk:\n\n{chunk_text}"),
    ]
)


class MedicineNameAgent:
    """Extract medicine names and structured metadata from a TranscriptDocument."""

    def __init__(self, max_tokens: int = 768) -> None:
        self._parser = JsonOutputParser(pydantic_object=_MedicineList)
        llm = create_chat_model("asr_v2", temperature=0, max_tokens=max_tokens)
        self._chain = (
            _medicine_prompt.partial(
                format_instructions=self._parser.get_format_instructions()
            )
            | llm
            | self._parser
        )

    def run(self, doc: TranscriptDocument) -> list[ExtractedMedicine]:
        chunks = _chunk_segments(doc.segments)
        inputs = [{"chunk_text": _render_chunk(chunk)} for chunk in chunks]
        print(
            f"[MedicineNameAgent] {len(doc.segments)} segments → {len(chunks)} chunk(s)"
        )
        try:
            batch_results = self._chain.batch(inputs, config={"max_concurrency": 4})
        except Exception as exc:
            print(f"[MedicineNameAgent] batch failed, falling back: {exc}")
            batch_results = []
            for inp in inputs:
                try:
                    batch_results.append(self._chain.invoke(inp))
                except Exception as inner:
                    print(f"[MedicineNameAgent] chunk failed: {inner}")
                    batch_results.append({"medicines": []})

        all_medicines: list[ExtractedMedicine] = []
        for raw in batch_results:
            parsed = _MedicineList(**raw) if isinstance(raw, dict) else raw
            all_medicines.extend(parsed.medicines)
        return self._deduplicate(all_medicines)

    @staticmethod
    def _deduplicate(medicines: list[ExtractedMedicine]) -> list[ExtractedMedicine]:
        best: dict[str, ExtractedMedicine] = {}
        for med in medicines:
            key = med.canonical.lower()
            existing = best.get(key)
            if existing is None:
                best[key] = med
            else:
                new_rank = _CONFIDENCE_RANK.get(med.confidence, 0)
                old_rank = _CONFIDENCE_RANK.get(existing.confidence, 0)
                if new_rank > old_rank or (
                    new_rank == old_rank
                    and med.dosage
                    and not existing.dosage
                ):
                    best[key] = med
        return list(best.values())
