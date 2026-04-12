"""
Transcript Utilities
====================
Data models and utilities for working with ASR transcripts.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class WordToken:
    """A single word/token in a transcript."""
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptSegment:
    """A segment of transcribed speech."""
    segment_id: str
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[WordToken] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptDocument:
    """Complete transcript document."""
    source_path: str
    audio_path: str | None = None  # Path to source audio file
    language: str | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)
    segments: list[TranscriptSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorCandidate:
    """A potential transcription error identified for review."""
    segment_idx: int
    word_idx: int
    word: str
    score: float
    speaker: str
    left_context: str
    right_context: str
    segment_text: str
    start_time: float
    end_time: float
    error_type: Literal["low_confidence", "medical_term", "homophone", "unknown_term"] = "low_confidence"
    suggested_correction: str | None = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _speaker_votes(words: list[WordToken]) -> str | None:
    """Determine speaker by majority vote from word-level speaker labels."""
    counts: dict[str, int] = {}
    for word in words:
        if word.speaker:
            counts[word.speaker] = counts.get(word.speaker, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _normalize_word(raw_word: dict[str, Any]) -> WordToken:
    """Convert raw word dict to WordToken."""
    return WordToken(
        text=str(raw_word.get("word") or raw_word.get("text") or "").strip(),
        start=raw_word.get("start"),
        end=raw_word.get("end"),
        speaker=raw_word.get("speaker"),
        score=raw_word.get("score") or raw_word.get("probability"),
        metadata={
            key: value
            for key, value in raw_word.items()
            if key not in {"word", "text", "start", "end", "speaker", "score", "probability"}
        },
    )


def load_transcript(
    path: str | Path,
    audio_path: str | Path | None = None,
) -> TranscriptDocument:
    """Load a transcript from JSON (WhisperX format)."""
    path = Path(path)
    raw_data = json.loads(path.read_text(encoding="utf-8"))
    raw_segments = raw_data.get("segments", [])

    segments: list[TranscriptSegment] = []
    for idx, raw_segment in enumerate(raw_segments):
        words = [_normalize_word(word) for word in raw_segment.get("words", [])]
        text = str(raw_segment.get("text") or "").strip()
        speaker = raw_segment.get("speaker") or _speaker_votes(words)
        segment_id = str(raw_segment.get("id", idx))
        start = _safe_float(raw_segment.get("start"))
        end = _safe_float(raw_segment.get("end"), default=start)
        
        segments.append(
            TranscriptSegment(
                segment_id=segment_id,
                start=start,
                end=end,
                text=text,
                speaker=speaker,
                words=words,
                metadata={
                    key: value
                    for key, value in raw_segment.items()
                    if key not in {"id", "start", "end", "text", "speaker", "words"}
                },
            )
        )

    # Try to find associated audio file
    if audio_path is None:
        # Check if transcript has audio_path in metadata
        audio_path = raw_data.get("audio_path")
        if audio_path is None:
            # Try to infer from transcript filename
            potential_audio = path.with_suffix(".wav")
            if potential_audio.exists():
                audio_path = str(potential_audio)
            else:
                potential_audio = path.with_suffix(".mp3")
                if potential_audio.exists():
                    audio_path = str(potential_audio)
                else:
                    potential_audio = path.with_suffix(".m4a")
                    if potential_audio.exists():
                        audio_path = str(potential_audio)

    return TranscriptDocument(
        source_path=str(path),
        audio_path=str(audio_path) if audio_path else None,
        language=raw_data.get("language"),
        raw_data=raw_data,
        segments=segments,
        metadata={
            key: value
            for key, value in raw_data.items()
            if key not in {"segments", "language", "audio_path"}
        },
    )


def analyze_transcript(doc: TranscriptDocument) -> dict[str, Any]:
    """Analyze transcript and return statistics."""
    missing_speakers = sum(1 for segment in doc.segments if not segment.speaker)
    single_word_segments = sum(1 for segment in doc.segments if len(segment.text.split()) <= 1)
    speaker_changes = 0
    speakers: set[str] = set()
    previous_speaker: str | None = None

    for segment in doc.segments:
        if segment.speaker:
            speakers.add(segment.speaker)
        if previous_speaker and segment.speaker and previous_speaker != segment.speaker:
            speaker_changes += 1
        if segment.speaker:
            previous_speaker = segment.speaker

    duration = 0.0
    if doc.segments:
        duration = doc.segments[-1].end - doc.segments[0].start

    # Count low-confidence words
    low_confidence_count = 0
    total_words = 0
    for segment in doc.segments:
        for word in segment.words:
            total_words += 1
            if word.score is not None and word.score < 0.75:
                low_confidence_count += 1

    return {
        "segment_count": len(doc.segments),
        "speaker_count": len(speakers),
        "speakers": sorted(speakers),
        "missing_speaker_segments": missing_speakers,
        "single_word_segments": single_word_segments,
        "speaker_changes": speaker_changes,
        "duration_seconds": round(duration, 2),
        "total_words": total_words,
        "low_confidence_words": low_confidence_count,
        "low_confidence_rate": round(low_confidence_count / max(total_words, 1), 3),
    }


def document_to_plain_text(doc: TranscriptDocument) -> str:
    """Convert document to plain text format."""
    return "\n".join(
        f"[{segment.start:07.2f}-{segment.end:07.2f}] {segment.speaker or 'UNKNOWN'}: {segment.text}"
        for segment in doc.segments
    )


def find_low_confidence_regions(
    doc: TranscriptDocument,
    threshold: float = 0.75,
    context_words: int = 3,
) -> list[ErrorCandidate]:
    """Find words with low confidence scores that may be errors."""
    candidates = []
    
    for seg_idx, segment in enumerate(doc.segments):
        words = segment.words
        for w_idx, word in enumerate(words):
            score = word.score if word.score is not None else 1.0
            if score < threshold:
                # Get context
                left = " ".join(w.text for w in words[max(0, w_idx - context_words):w_idx])
                right = " ".join(w.text for w in words[w_idx + 1:w_idx + context_words + 1])
                
                candidates.append(
                    ErrorCandidate(
                        segment_idx=seg_idx,
                        word_idx=w_idx,
                        word=word.text,
                        score=score,
                        speaker=segment.speaker or "UNKNOWN",
                        left_context=left,
                        right_context=right,
                        segment_text=segment.text,
                        start_time=word.start if word.start is not None else segment.start,
                        end_time=word.end if word.end is not None else segment.end,
                        error_type="low_confidence",
                    )
                )
    
    return candidates


def extract_segment_context(
    doc: TranscriptDocument,
    segment_idx: int,
    context_segments: int = 2,
) -> tuple[list[TranscriptSegment], TranscriptSegment, list[TranscriptSegment]]:
    """Extract a segment with surrounding context."""
    start_idx = max(0, segment_idx - context_segments)
    end_idx = min(len(doc.segments), segment_idx + context_segments + 1)
    
    before = doc.segments[start_idx:segment_idx]
    target = doc.segments[segment_idx]
    after = doc.segments[segment_idx + 1:end_idx]
    
    return before, target, after


def find_homophone_candidates(
    doc: TranscriptDocument,
    homophone_pairs: list[tuple[str, str]] | None = None,
) -> list[ErrorCandidate]:
    """Find potential homophone errors in the transcript."""
    if homophone_pairs is None:
        # Common English homophones
        homophone_pairs = [
            ("their", "there"), ("there", "their"), ("they're", "their"),
            ("to", "too"), ("too", "to"), ("two", "to"),
            ("your", "you're"), ("you're", "your"),
            ("its", "it's"), ("it's", "its"),
            ("accept", "except"), ("except", "accept"),
            ("affect", "effect"), ("effect", "affect"),
            ("than", "then"), ("then", "than"),
        ]
    
    candidates = []
    homophone_words = {pair[0] for pair in homophone_pairs}
    
    for seg_idx, segment in enumerate(doc.segments):
        words = segment.words
        for w_idx, word in enumerate(words):
            if word.text.lower().strip(".,?!") in homophone_words:
                left = " ".join(w.text for w in words[max(0, w_idx - 3):w_idx])
                right = " ".join(w.text for w in words[w_idx + 1:w_idx + 4])
                
                candidates.append(
                    ErrorCandidate(
                        segment_idx=seg_idx,
                        word_idx=w_idx,
                        word=word.text,
                        score=word.score if word.score is not None else 0.8,
                        speaker=segment.speaker or "UNKNOWN",
                        left_context=left,
                        right_context=right,
                        segment_text=segment.text,
                        start_time=word.start if word.start is not None else segment.start,
                        end_time=word.end if word.end is not None else segment.end,
                        error_type="homophone",
                    )
                )
    
    return candidates


def segment_to_dict(segment: TranscriptSegment) -> dict[str, Any]:
    """Convert segment to dictionary."""
    return {
        "id": segment.segment_id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "speaker": segment.speaker,
        "words": [
            {
                "word": word.text,
                "start": word.start,
                "end": word.end,
                "speaker": word.speaker,
                "score": word.score,
                **word.metadata,
            }
            for word in segment.words
        ],
        **segment.metadata,
    }


def document_to_dict(doc: TranscriptDocument) -> dict[str, Any]:
    """Convert document to WhisperX-like dictionary."""
    return {
        "language": doc.language,
        "audio_path": doc.audio_path,
        **doc.metadata,
        "segments": [segment_to_dict(segment) for segment in doc.segments],
    }


def save_document(doc: TranscriptDocument, output_path: str | Path) -> None:
    """Save document to JSON file."""
    output_path = Path(output_path)
    output_path.write_text(
        json.dumps(document_to_dict(doc), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
