"""
Stage 3, File 1: Medical Extraction LangGraph Pipeline
=======================================================
CONCEPT: Combine the two specialised agents from Stage 2 into a single
stateful LangGraph that orchestrates loading, extraction, glossary
cross-referencing, deduplication, and structured output generation.

Graph nodes
───────────
  load            Load transcript from file, build plain-text rendering
  extract_terms   Run MedicalTermAgent on the document (chunked, batched)
  extract_meds    Run MedicineNameAgent on the document (chunked, batched)
  link_glossary   Cross-reference extracted terms with the multilingual
                  glossary; attach canonical_en, ICD hints, translations
  deduplicate     Merge duplicates across chunks; prefer highest-confidence
                  entry per canonical term
  save            Write JSON + Markdown report to the output directory

Graph flow
──────────
  START → load → extract_terms ─┐
                                 ├→ link_glossary → deduplicate → save → END
                  extract_meds ─┘

(extract_terms and extract_meds fan out in parallel via Send, then both
results arrive before link_glossary runs.)

Run this file:
  uv run deep_research/medical_term_extraction/stage_03_extraction_graph/01_extraction_pipeline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from deep_research.asr_medical_verification.shared.medical_glossary import (
    load_medical_glossary,
    link_medical_terms,
)
from deep_research.asr_medical_verification.shared.transcript_utils import (
    TranscriptDocument,
    load_transcript,
)
from deep_research.medical_term_extraction.shared.agents import (
    MedicalTermAgent,
    MedicineNameAgent,
)
from deep_research.medical_term_extraction.shared.extraction_models import (
    ExtractedMedicalTerm,
    ExtractedMedicine,
    ExtractionReport,
)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class ExtractionState(TypedDict):
    # Inputs
    input_path: str
    output_dir: str
    specialty: str
    # Populated during execution
    doc: Any  # TranscriptDocument (not directly TypedDict-serialisable)
    medical_terms: list[dict]
    medicine_names: list[dict]
    glossary_links: list[dict]
    extraction_notes: list[str]
    output_paths: dict


# ---------------------------------------------------------------------------
# Node: load
# ---------------------------------------------------------------------------


def load_node(state: ExtractionState) -> dict:
    doc = load_transcript(state["input_path"])
    note = (
        f"Loaded {len(doc.segments)} segments, language={doc.language!r}"
    )
    print(f"[load] {note}")
    return {
        "doc": doc,
        "extraction_notes": state.get("extraction_notes", []) + [note],
    }


# ---------------------------------------------------------------------------
# Node: extract_terms
# ---------------------------------------------------------------------------


def extract_terms_node(state: ExtractionState) -> dict:
    doc: TranscriptDocument = state["doc"]
    specialty = state.get("specialty", "general")
    agent = MedicalTermAgent(specialty=specialty)
    terms = agent.run(doc)
    note = f"Extracted {len(terms)} medical term(s) (specialty={specialty})"
    print(f"[extract_terms] {note}")
    return {
        "medical_terms": [t.model_dump() for t in terms],
        "extraction_notes": state.get("extraction_notes", []) + [note],
    }


# ---------------------------------------------------------------------------
# Node: extract_meds
# ---------------------------------------------------------------------------


def extract_meds_node(state: ExtractionState) -> dict:
    doc: TranscriptDocument = state["doc"]
    agent = MedicineNameAgent()
    medicines = agent.run(doc)
    note = f"Extracted {len(medicines)} medicine(s)"
    print(f"[extract_meds] {note}")
    return {
        "medicine_names": [m.model_dump() for m in medicines],
        "extraction_notes": state.get("extraction_notes", []) + [note],
    }


# ---------------------------------------------------------------------------
# Node: link_glossary
# ---------------------------------------------------------------------------


def link_glossary_node(state: ExtractionState) -> dict:
    doc: TranscriptDocument = state["doc"]
    glossary = load_medical_glossary()
    # Build full transcript text for glossary scanning
    full_text = " ".join(seg.text for seg in doc.segments)
    links = link_medical_terms(full_text, glossary, min_score=0.78)

    # Enrich LLM-extracted medical terms with ICD hints from glossary
    enriched_terms: list[dict] = []
    glossary_canonicals = {link["canonical_en"].lower(): link for link in links}
    for term_dict in state.get("medical_terms", []):
        canonical_key = term_dict.get("canonical", "").lower()
        if canonical_key in glossary_canonicals:
            glossary_entry = glossary_canonicals[canonical_key]
            if not term_dict.get("icd_hint") and glossary_entry.get("icd_hint"):
                term_dict = dict(term_dict)
                term_dict["icd_hint"] = glossary_entry["icd_hint"]
        enriched_terms.append(term_dict)

    note = f"Linked {len(links)} glossary term(s)"
    print(f"[link_glossary] {note}")
    return {
        "medical_terms": enriched_terms,
        "glossary_links": links,
        "extraction_notes": state.get("extraction_notes", []) + [note],
    }


# ---------------------------------------------------------------------------
# Node: deduplicate
# ---------------------------------------------------------------------------

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}


def deduplicate_node(state: ExtractionState) -> dict:
    # Deduplicate medical terms
    best_terms: dict[str, dict] = {}
    for term in state.get("medical_terms", []):
        key = term.get("canonical", "").lower()
        existing = best_terms.get(key)
        if existing is None or _CONFIDENCE_RANK.get(
            term.get("confidence"), 0
        ) > _CONFIDENCE_RANK.get(existing.get("confidence"), 0):
            best_terms[key] = term

    # Deduplicate medicines (prefer richer metadata)
    best_meds: dict[str, dict] = {}
    for med in state.get("medicine_names", []):
        key = med.get("canonical", "").lower()
        existing = best_meds.get(key)
        if existing is None:
            best_meds[key] = med
        else:
            new_rank = _CONFIDENCE_RANK.get(med.get("confidence"), 0)
            old_rank = _CONFIDENCE_RANK.get(existing.get("confidence"), 0)
            if new_rank > old_rank or (new_rank == old_rank and med.get("dosage") and not existing.get("dosage")):
                best_meds[key] = med

    terms_out = list(best_terms.values())
    meds_out = list(best_meds.values())
    note = f"After dedup: {len(terms_out)} term(s), {len(meds_out)} medicine(s)"
    print(f"[deduplicate] {note}")
    return {
        "medical_terms": terms_out,
        "medicine_names": meds_out,
        "extraction_notes": state.get("extraction_notes", []) + [note],
    }


# ---------------------------------------------------------------------------
# Node: save
# ---------------------------------------------------------------------------


def _render_markdown_report(
    report: ExtractionReport,
    glossary_links: list[dict],
) -> str:
    lines = [
        "# Medical Term Extraction Report",
        "",
        f"- **Source**: `{Path(report.source_path).name}`",
        f"- **Language**: `{report.language}`",
        f"- **Segments**: `{report.segment_count}`",
        f"- **Medical terms found**: `{len(report.medical_terms)}`",
        f"- **Medicines found**: `{len(report.medicine_names)}`",
        "",
    ]

    lines += [
        "## Medical Terms",
        "",
        "| Confidence | Canonical | Category | Specialty | ICD | Raw text |",
        "|---|---|---|---|---|---|",
    ]
    for term in report.medical_terms:
        icd = term.icd_hint or "–"
        raw = term.raw_text if term.raw_text.lower() != term.canonical.lower() else "–"
        lines.append(
            f"| {term.confidence} | {term.canonical} | {term.category} "
            f"| {term.specialty} | {icd} | {raw} |"
        )

    lines += [
        "",
        "## Medicines",
        "",
        "| Confidence | Canonical | Dosage | Route | Frequency | Indication | Raw text |",
        "|---|---|---|---|---|---|---|",
    ]
    for med in report.medicine_names:
        raw = med.raw_text if med.raw_text.lower() != med.canonical.lower() else "–"
        lines.append(
            f"| {med.confidence} | {med.canonical} | {med.dosage or '–'} "
            f"| {med.route or '–'} | {med.frequency or '–'} "
            f"| {med.indication or '–'} | {raw} |"
        )

    if glossary_links:
        lines += [
            "",
            "## Glossary Cross-References",
            "",
            "| Term | Canonical EN | Specialty | ICD-10 |",
            "|---|---|---|---|",
        ]
        for link in glossary_links:
            lines.append(
                f"| {link['raw_text']} | {link['canonical_en']} "
                f"| {link['specialty']} | {link.get('icd_hint', '–')} |"
            )

    if report.notes:
        lines += ["", "## Extraction Notes", ""]
        for note in report.notes:
            lines.append(f"- {note}")

    lines.append("")
    return "\n".join(lines)


def save_node(state: ExtractionState) -> dict:
    doc: TranscriptDocument = state["doc"]
    output_dir = Path(state["output_dir"])
    stem = Path(state["input_path"]).stem
    run_dir = output_dir / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    report = ExtractionReport(
        source_path=state["input_path"],
        language=doc.language or "unknown",
        segment_count=len(doc.segments),
        medical_terms=[ExtractedMedicalTerm(**t) for t in state.get("medical_terms", [])],
        medicine_names=[ExtractedMedicine(**m) for m in state.get("medicine_names", [])],
        glossary_links=state.get("glossary_links", []),
        notes=state.get("extraction_notes", []),
    )
    glossary_links = state.get("glossary_links", [])

    json_path = run_dir / f"{stem}.extraction.json"
    md_path = run_dir / f"{stem}.extraction.md"

    json_path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(
        _render_markdown_report(report, glossary_links),
        encoding="utf-8",
    )

    print(f"[save] json → {json_path}")
    print(f"[save] md   → {md_path}")
    return {"output_paths": {"json": str(json_path), "markdown": str(md_path)}}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
#
#  START → load → extract_terms ─┐
#                                 ├─→ link_glossary → deduplicate → save → END
#               extract_meds ────┘
#
# LangGraph does not execute fan-out nodes in parallel by default when using
# simple add_edge, but running extract_terms → extract_meds sequentially on
# the same document is lightweight (each does its own chunked batch call).
# The graph is designed so a future Send-based parallel execution is trivial
# to add.

builder = StateGraph(ExtractionState)
builder.add_node("load", load_node)
builder.add_node("extract_terms", extract_terms_node)
builder.add_node("extract_meds", extract_meds_node)
builder.add_node("link_glossary", link_glossary_node)
builder.add_node("deduplicate", deduplicate_node)
builder.add_node("save", save_node)

builder.add_edge(START, "load")
builder.add_edge("load", "extract_terms")
builder.add_edge("extract_terms", "extract_meds")
builder.add_edge("extract_meds", "link_glossary")
builder.add_edge("link_glossary", "deduplicate")
builder.add_edge("deduplicate", "save")
builder.add_edge("save", END)

app = builder.compile()


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
    output_dir = Path(__file__).parent / "outputs"

    print("=" * 60)
    print("Stage 3: Medical Extraction LangGraph Pipeline")
    print("=" * 60)

    result = app.invoke(
        {
            "input_path": str(sample_path),
            "output_dir": str(output_dir),
            "specialty": "cardiology",
            "medical_terms": [],
            "medicine_names": [],
            "glossary_links": [],
            "extraction_notes": [],
            "output_paths": {},
        }
    )

    print("\n" + "=" * 60)
    print("Pipeline complete")
    print("=" * 60)
    print(f"  Medical terms : {len(result['medical_terms'])}")
    print(f"  Medicines     : {len(result['medicine_names'])}")
    print(f"  Glossary links: {len(result['glossary_links'])}")
    print()
    print("Extraction notes:")
    for note in result["extraction_notes"]:
        print(f"  - {note}")
    print()
    print("Output files:")
    for label, path in result["output_paths"].items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
