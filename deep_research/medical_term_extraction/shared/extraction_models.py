"""
Pydantic output schemas for medical term and medicine name extraction.

These models are used with JsonOutputParser / with_structured_output
throughout all stages of the extraction pipeline.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractedMedicalTerm(BaseModel):
    """A medical term identified in transcript text."""

    raw_text: str = Field(
        description="Exact phrase as spoken in the transcript, preserving ASR quirks"
    )
    canonical: str = Field(
        description="Standardised medical term (e.g. 'atrial fibrillation')"
    )
    category: Literal[
        "condition",
        "symptom",
        "procedure",
        "anatomy",
        "measurement",
        "finding",
        "other",
    ] = Field(description="Clinical category of the term")
    specialty: str = Field(
        default="general",
        description="Medical specialty most associated with this term (e.g. cardiology)",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Extraction confidence given the surrounding context"
    )
    context: str = Field(
        default="",
        description="Short surrounding sentence used as evidence",
    )
    icd_hint: str = Field(
        default="",
        description="ICD-10 code hint when clearly identifiable, otherwise empty",
    )


class ExtractedMedicine(BaseModel):
    """A medicine or drug name identified in transcript text."""

    raw_text: str = Field(
        description="Exact phrase as spoken, preserving ASR spelling errors"
    )
    canonical: str = Field(
        description="Standard generic drug name (e.g. 'lisinopril')"
    )
    brand_name: str = Field(
        default="", description="Brand name if explicitly mentioned"
    )
    dosage: str = Field(
        default="", description="Dosage amount and unit if mentioned (e.g. '10 mg')"
    )
    route: str = Field(
        default="",
        description="Route of administration: oral, IV, topical, inhaled, etc.",
    )
    frequency: str = Field(
        default="",
        description="Dosing frequency if mentioned (e.g. 'once daily', 'BID')",
    )
    indication: str = Field(
        default="",
        description="Clinical indication the medicine is prescribed for, if stated",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Extraction confidence given context and ASR quality"
    )
    context: str = Field(
        default="", description="Short surrounding sentence used as evidence"
    )


class SegmentExtractionResult(BaseModel):
    """LLM output for a single transcript chunk."""

    medical_terms: list[ExtractedMedicalTerm] = Field(default_factory=list)
    medicine_names: list[ExtractedMedicine] = Field(default_factory=list)


class ExtractionReport(BaseModel):
    """Full extraction results for one transcript file."""

    source_path: str
    language: str
    segment_count: int
    medical_terms: list[ExtractedMedicalTerm] = Field(default_factory=list)
    medicine_names: list[ExtractedMedicine] = Field(default_factory=list)
    glossary_links: list[dict] = Field(
        default_factory=list,
        description="Medical terms cross-referenced with the multilingual glossary",
    )
    notes: list[str] = Field(default_factory=list)
