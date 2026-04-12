"""
Detection helpers for suspicious ASR spans and medical terminology.
"""

from __future__ import annotations

from typing import Any

from .medical_glossary import MultilingualMedicalGlossary, detect_medical_terms
from .transcript_utils import (
    ErrorCandidate,
    MedicalCandidate,
    TranscriptDocument,
    find_homophone_candidates,
    find_low_confidence_regions,
)


COMMON_WORDS = {
    "a", "about", "after", "also", "and", "are", "as", "at", "be", "because", "been",
    "but", "by", "can", "could", "day", "did", "do", "does", "for", "from", "get",
    "go", "good", "had", "has", "have", "he", "her", "him", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "know", "like", "look", "make", "me",
    "most", "my", "new", "no", "not", "now", "of", "on", "one", "only", "or", "other",
    "our", "out", "over", "people", "right", "say", "see", "she", "so", "some", "take",
    "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "time", "to", "two", "up", "us", "use", "want", "was", "way", "we", "well", "were",
    "what", "when", "which", "who", "will", "with", "work", "would", "year", "you", "your",
}


def detect_unknown_terms(doc: TranscriptDocument) -> list[ErrorCandidate]:
    candidates: list[ErrorCandidate] = []
    for seg_idx, segment in enumerate(doc.segments):
        for word_idx, word in enumerate(segment.words):
            token = word.text.lower().strip(".,?!")
            if len(token) <= 2 or token in COMMON_WORDS:
                continue
            suspicious = (
                token.isdigit()
                or len(token) > 15
                or token.count("-") > 2
                or any(char.isupper() for char in token[1:])
            )
            if not suspicious:
                continue
            candidates.append(
                ErrorCandidate(
                    segment_idx=seg_idx,
                    word_idx=word_idx,
                    word=word.text,
                    score=word.score if word.score is not None else 0.5,
                    speaker=segment.speaker or "UNKNOWN",
                    left_context=" ".join(item.text for item in segment.words[max(0, word_idx - 3):word_idx]),
                    right_context=" ".join(item.text for item in segment.words[word_idx + 1:word_idx + 4]),
                    segment_text=segment.text,
                    start_time=word.start if word.start is not None else segment.start,
                    end_time=word.end if word.end is not None else segment.end,
                    error_type="unknown_term",
                )
            )
    return candidates


def detect_repetition_candidates(doc: TranscriptDocument) -> list[ErrorCandidate]:
    candidates: list[ErrorCandidate] = []
    for seg_idx, segment in enumerate(doc.segments):
        tokens = [word.text for word in segment.words] or segment.text.split()
        for word_idx in range(1, len(tokens)):
            current = tokens[word_idx].lower().strip(".,?!")
            previous = tokens[word_idx - 1].lower().strip(".,?!")
            if current != previous:
                continue
            candidates.append(
                ErrorCandidate(
                    segment_idx=seg_idx,
                    word_idx=word_idx,
                    word=tokens[word_idx],
                    score=0.4,
                    speaker=segment.speaker or "UNKNOWN",
                    left_context=" ".join(tokens[max(0, word_idx - 3):word_idx]),
                    right_context=" ".join(tokens[word_idx + 1:word_idx + 4]),
                    segment_text=segment.text,
                    start_time=segment.start,
                    end_time=segment.end,
                    error_type="repetition",
                )
            )
    return candidates


def detect_medical_candidates(
    doc: TranscriptDocument,
    glossary: MultilingualMedicalGlossary,
) -> list[MedicalCandidate]:
    candidates: list[MedicalCandidate] = []
    for seg_idx, segment in enumerate(doc.segments):
        matches = detect_medical_terms(segment.text, glossary, min_score=0.76)
        for matched_text, term, score in matches[:3]:
            candidates.append(
                MedicalCandidate(
                    segment_idx=seg_idx,
                    segment_id=segment.segment_id,
                    start_time=segment.start,
                    end_time=segment.end,
                    text=segment.text,
                    speaker=segment.speaker or "UNKNOWN",
                    suspected_term=matched_text,
                    specialty=term.specialty,
                    reason=f"Matched glossary term {term.canonical_en}",
                    score=score,
                )
            )
    deduped: list[MedicalCandidate] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (candidate.segment_id, candidate.suspected_term.lower())
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def identify_error_candidates(
    doc: TranscriptDocument,
    glossary: MultilingualMedicalGlossary,
    *,
    confidence_threshold: float = 0.75,
) -> dict[str, Any]:
    low_conf = find_low_confidence_regions(doc, threshold=confidence_threshold)
    homophones = find_homophone_candidates(doc)
    unknown_terms = detect_unknown_terms(doc)
    repetitions = detect_repetition_candidates(doc)
    medical_candidates = detect_medical_candidates(doc, glossary)

    all_errors = low_conf + homophones + unknown_terms + repetitions
    deduped_errors: list[ErrorCandidate] = []
    seen: set[tuple[int, int, str]] = set()
    for candidate in sorted(all_errors, key=lambda item: item.score):
        key = (candidate.segment_idx, candidate.word_idx, candidate.error_type)
        if key not in seen:
            seen.add(key)
            deduped_errors.append(candidate)

    return {
        "low_confidence": low_conf,
        "homophones": homophones,
        "unknown_terms": unknown_terms,
        "repetitions": repetitions,
        "medical_candidates": medical_candidates,
        "all_errors": deduped_errors,
    }
