"""
ASR Medical Verification - Shared Utilities
============================================
Common utilities for transcription error detection and medical term verification.
"""

from .transcript_utils import (
    WordToken,
    TranscriptSegment,
    TranscriptDocument,
    ErrorCandidate,
    load_transcript,
    analyze_transcript,
    document_to_plain_text,
    find_low_confidence_regions,
    find_homophone_candidates,
    extract_segment_context,
    save_document,
)

from .medical_glossary import (
    MedicalTerm,
    MultilingualMedicalGlossary,
    load_medical_glossary,
    detect_medical_terms,
    link_medical_terms,
    build_llm_hint,
)

from .audio_utils import (
    AudioSegment,
    AudioSegmentExtractor,
    extract_audio_segment,
    get_audio_duration,
)

__all__ = [
    # Transcript utilities
    "WordToken",
    "TranscriptSegment",
    "TranscriptDocument",
    "ErrorCandidate",
    "load_transcript",
    "analyze_transcript",
    "document_to_plain_text",
    "find_low_confidence_regions",
    "find_homophone_candidates",
    "extract_segment_context",
    "save_document",
    # Medical glossary
    "MedicalTerm",
    "MultilingualMedicalGlossary",
    "load_medical_glossary",
    "detect_medical_terms",
    "link_medical_terms",
    "build_llm_hint",
    # Audio utilities
    "AudioSegment",
    "AudioSegmentExtractor",
    "extract_audio_segment",
    "get_audio_duration",
]
