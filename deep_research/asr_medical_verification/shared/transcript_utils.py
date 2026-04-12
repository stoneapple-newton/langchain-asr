"""
Transcript data models and correction helpers for WhisperX-like JSON.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


@dataclass
class WordToken:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptSegment:
    segment_id: str
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[WordToken] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptDocument:
    source_path: str
    audio_path: str | None = None
    language: str | None = None
    raw_data: dict[str, Any] = field(default_factory=dict)
    segments: list[TranscriptSegment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorCandidate:
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
    error_type: Literal["low_confidence", "medical_term", "homophone", "unknown_term", "repetition"] = "low_confidence"
    suggested_correction: str | None = None


@dataclass
class MedicalCandidate:
    segment_idx: int
    segment_id: str
    start_time: float
    end_time: float
    text: str
    speaker: str
    suspected_term: str
    specialty: str
    reason: str
    score: float


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _speaker_votes(words: list[WordToken]) -> str | None:
    counts: dict[str, int] = {}
    for word in words:
        if word.speaker:
            counts[word.speaker] = counts.get(word.speaker, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _normalize_word(raw_word: dict[str, Any]) -> WordToken:
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


def load_transcript(path: str | Path, audio_path: str | Path | None = None) -> TranscriptDocument:
    transcript_path = Path(path)
    raw_data = json.loads(transcript_path.read_text(encoding="utf-8"))
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

    resolved_audio = audio_path or raw_data.get("audio_path")
    if resolved_audio is None:
        for suffix in (".wav", ".mp3", ".m4a"):
            candidate = transcript_path.with_suffix(suffix)
            if candidate.exists():
                resolved_audio = str(candidate)
                break

    return TranscriptDocument(
        source_path=str(transcript_path),
        audio_path=str(resolved_audio) if resolved_audio else None,
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
    speakers = sorted({segment.speaker for segment in doc.segments if segment.speaker})
    total_words = sum(len(segment.words) or len(segment.text.split()) for segment in doc.segments)
    low_confidence_words = sum(
        1
        for segment in doc.segments
        for word in segment.words
        if word.score is not None and word.score < 0.75
    )
    return {
        "segment_count": len(doc.segments),
        "speaker_count": len(speakers),
        "speakers": speakers,
        "total_words": total_words,
        "low_confidence_words": low_confidence_words,
        "low_confidence_rate": round(low_confidence_words / max(total_words, 1), 3),
        "duration_seconds": round((doc.segments[-1].end - doc.segments[0].start), 2) if doc.segments else 0.0,
    }


def document_to_plain_text(doc: TranscriptDocument) -> str:
    return "\n".join(
        f"[{segment.start:07.2f}-{segment.end:07.2f}] {segment.speaker or 'UNKNOWN'}: {segment.text}"
        for segment in doc.segments
    )


def extract_segment_context(
    doc: TranscriptDocument,
    segment_idx: int,
    *,
    context_segments: int = 2,
) -> tuple[list[TranscriptSegment], TranscriptSegment, list[TranscriptSegment]]:
    start_idx = max(0, segment_idx - context_segments)
    end_idx = min(len(doc.segments), segment_idx + context_segments + 1)
    return (
        doc.segments[start_idx:segment_idx],
        doc.segments[segment_idx],
        doc.segments[segment_idx + 1:end_idx],
    )


def build_context_text(doc: TranscriptDocument, segment_idx: int, *, context_segments: int = 2) -> str:
    before, target, after = extract_segment_context(doc, segment_idx, context_segments=context_segments)
    lines: list[str] = []
    for segment in before:
        lines.append(f"{segment.speaker or 'UNKNOWN'}: {segment.text}")
    lines.append(f">>> {target.speaker or 'UNKNOWN'}: {target.text}")
    for segment in after:
        lines.append(f"{segment.speaker or 'UNKNOWN'}: {segment.text}")
    return "\n".join(lines)


def find_low_confidence_regions(
    doc: TranscriptDocument,
    *,
    threshold: float = 0.75,
    context_words: int = 3,
) -> list[ErrorCandidate]:
    candidates: list[ErrorCandidate] = []
    for seg_idx, segment in enumerate(doc.segments):
        for word_idx, word in enumerate(segment.words):
            score = word.score if word.score is not None else 1.0
            if score >= threshold:
                continue
            left = " ".join(item.text for item in segment.words[max(0, word_idx - context_words):word_idx])
            right = " ".join(item.text for item in segment.words[word_idx + 1:word_idx + context_words + 1])
            candidates.append(
                ErrorCandidate(
                    segment_idx=seg_idx,
                    word_idx=word_idx,
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


def find_homophone_candidates(
    doc: TranscriptDocument,
    homophone_pairs: list[tuple[str, str]] | None = None,
) -> list[ErrorCandidate]:
    if homophone_pairs is None:
        homophone_pairs = [
            ("their", "there"),
            ("there", "their"),
            ("your", "you're"),
            ("its", "it's"),
            ("than", "then"),
        ]
    lookup = {pair[0] for pair in homophone_pairs}
    candidates: list[ErrorCandidate] = []
    for seg_idx, segment in enumerate(doc.segments):
        for word_idx, word in enumerate(segment.words):
            token = word.text.lower().strip(".,?!")
            if token not in lookup:
                continue
            left = " ".join(item.text for item in segment.words[max(0, word_idx - 3):word_idx])
            right = " ".join(item.text for item in segment.words[word_idx + 1:word_idx + 4])
            candidates.append(
                ErrorCandidate(
                    segment_idx=seg_idx,
                    word_idx=word_idx,
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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", text.lower())).strip()


def apply_corrections(
    doc: TranscriptDocument,
    corrections: list[dict[str, Any]],
    *,
    min_confidence: float = 0.72,
) -> tuple[TranscriptDocument, list[dict[str, Any]]]:
    updated_segments = [
        TranscriptSegment(
            segment_id=segment.segment_id,
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=segment.speaker,
            words=list(segment.words),
            metadata=dict(segment.metadata),
        )
        for segment in doc.segments
    ]
    applied: list[dict[str, Any]] = []
    by_segment = {segment.segment_id: segment for segment in updated_segments}
    for correction in corrections:
        if not correction.get("accepted"):
            continue
        if correction.get("confidence", 0.0) < min_confidence:
            continue
        segment_id = str(correction["segment_id"])
        segment = by_segment.get(segment_id)
        if segment is None:
            continue
        corrected_text = str(correction.get("corrected_text", "")).strip()
        if not corrected_text:
            continue
        if _normalize_text(corrected_text) == _normalize_text(segment.text):
            continue
        segment.text = corrected_text
        segment.metadata.setdefault("corrections", []).append(
            {
                "method": correction.get("method", "unknown"),
                "correction_type": correction.get("correction_type", "error"),
                "confidence": correction.get("confidence", 0.0),
                "medical_terms": correction.get("medical_terms", []),
            }
        )
        applied.append(correction)
    updated_doc = TranscriptDocument(
        source_path=doc.source_path,
        audio_path=doc.audio_path,
        language=doc.language,
        raw_data=dict(doc.raw_data),
        segments=updated_segments,
        metadata=dict(doc.metadata),
    )
    return updated_doc, applied


def segment_to_dict(segment: TranscriptSegment) -> dict[str, Any]:
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
    payload = {
        "language": doc.language,
        "audio_path": doc.audio_path,
        **doc.metadata,
        "segments": [segment_to_dict(segment) for segment in doc.segments],
    }
    return {key: value for key, value in payload.items() if value is not None}


def save_document(doc: TranscriptDocument, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.write_text(json.dumps(document_to_dict(doc), indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def write_correction_sidecar(
    transcript_path: str | Path,
    *,
    corrections: list[dict[str, Any]],
    stage_log: list[str],
    linked_terms: list[dict[str, Any]] | None = None,
) -> Path:
    target = Path(transcript_path)
    sidecar = target.with_suffix(".corrections.json")
    payload = {
        "transcript_path": str(target),
        "applied_count": len([item for item in corrections if item.get("accepted")]),
        "corrections": corrections,
        "linked_terms": linked_terms or [],
        "stage_log": stage_log,
    }
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return sidecar
