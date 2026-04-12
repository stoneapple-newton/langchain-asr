"""
ASR Medical Verification - Shared Utilities
============================================
Common utilities for transcription error detection and medical term verification.
"""

from .transcript_utils import (
    WordToken,
    TranscriptSegment,
    TranscriptDocument,
    load_transcript,
    analyze_transcript,
    document_to_plain_text,
    find_low_confidence_regions,
    extract_segment_context,
)

from .medical_glossary import (
    MedicalTerm,
    MultilingualMedicalGlossary,
    load_medical_glossary,
    detect_medical_terms,
)

from .audio_utils import (
    AudioSegmentExtractor,
    extract_audio_segment,
    get_audio_duration,
)

__all__ = [
    # Transcript utilities
    "WordToken",
    "TranscriptSegment",
    "TranscriptDocument",
    "load_transcript",
    "analyze_transcript",
    "document_to_plain_text",
    "find_low_confidence_regions",
    "extract_segment_context",
    # Medical glossary
    "MedicalTerm",
    "MultilingualMedicalGlossary",
    "load_medical_glossary",
    "detect_medical_terms",
    # Audio utilities
    "AudioSegmentExtractor",
    "extract_audio_segment",
    "get_audio_duration",
]
