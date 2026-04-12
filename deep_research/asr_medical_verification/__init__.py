"""
ASR Medical Verification
========================
A comprehensive agent system for transcription error detection and correction
with specialized medical terminology support.
"""

from .shared.transcript_utils import (
    load_transcript,
    analyze_transcript,
    save_document,
    TranscriptDocument,
    WordToken,
    TranscriptSegment,
    ErrorCandidate,
)
from .shared.medical_glossary import (
    load_medical_glossary,
    MultilingualMedicalGlossary,
    MedicalTerm,
)
from .shared.audio_utils import (
    AudioSegmentExtractor,
    extract_audio_segment,
    AudioSegment,
)

__version__ = "0.1.0"

__all__ = [
    # Transcript utilities
    "load_transcript",
    "analyze_transcript",
    "save_document",
    "TranscriptDocument",
    "WordToken",
    "TranscriptSegment",
    "ErrorCandidate",
    # Medical glossary
    "load_medical_glossary",
    "MultilingualMedicalGlossary",
    "MedicalTerm",
    # Audio utilities
    "AudioSegmentExtractor",
    "extract_audio_segment",
    "AudioSegment",
]
