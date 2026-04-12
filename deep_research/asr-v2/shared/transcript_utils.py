from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
    language: str | None
    raw_data: dict[str, Any]
    segments: list[TranscriptSegment]
    metadata: dict[str, Any] = field(default_factory=dict)


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


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", text)
    return text


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


def load_transcript(path: str | Path) -> TranscriptDocument:
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

    return TranscriptDocument(
        source_path=str(path),
        language=raw_data.get("language"),
        raw_data=raw_data,
        segments=segments,
        metadata={
            key: value
            for key, value in raw_data.items()
            if key not in {"segments", "language"}
        },
    )


def analyze_transcript(doc: TranscriptDocument) -> dict[str, Any]:
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

    return {
        "segment_count": len(doc.segments),
        "speaker_count": len(speakers),
        "speakers": sorted(speakers),
        "missing_speaker_segments": missing_speakers,
        "single_word_segments": single_word_segments,
        "speaker_changes": speaker_changes,
        "duration_seconds": round(duration, 2),
    }


def repair_diarization(doc: TranscriptDocument, max_gap_seconds: float = 0.75) -> TranscriptDocument:
    repaired_segments: list[TranscriptSegment] = []

    for index, segment in enumerate(doc.segments):
        filled_speaker = segment.speaker or _speaker_votes(segment.words)
        if filled_speaker is None:
            prev_speaker = doc.segments[index - 1].speaker if index > 0 else None
            next_speaker = doc.segments[index + 1].speaker if index + 1 < len(doc.segments) else None
            if prev_speaker and prev_speaker == next_speaker:
                filled_speaker = prev_speaker
            else:
                filled_speaker = prev_speaker or next_speaker

        repaired_segments.append(
            TranscriptSegment(
                segment_id=segment.segment_id,
                start=segment.start,
                end=segment.end,
                text=_clean_text(segment.text),
                speaker=filled_speaker,
                words=segment.words,
                metadata=dict(segment.metadata),
            )
        )

    for index in range(1, len(repaired_segments) - 1):
        previous_segment = repaired_segments[index - 1]
        current_segment = repaired_segments[index]
        next_segment = repaired_segments[index + 1]
        short_segment = (current_segment.end - current_segment.start) <= 1.2
        if (
            short_segment
            and previous_segment.speaker
            and previous_segment.speaker == next_segment.speaker
            and current_segment.speaker != previous_segment.speaker
        ):
            current_segment.speaker = previous_segment.speaker
            current_segment.metadata["speaker_smoothed"] = True

    merged_segments: list[TranscriptSegment] = []
    for segment in repaired_segments:
        if not merged_segments:
            merged_segments.append(segment)
            continue

        previous_segment = merged_segments[-1]
        gap = segment.start - previous_segment.end
        same_speaker = previous_segment.speaker and previous_segment.speaker == segment.speaker
        if same_speaker and gap <= max_gap_seconds:
            previous_segment.end = segment.end
            previous_segment.text = _clean_text(f"{previous_segment.text} {segment.text}")
            previous_segment.words.extend(segment.words)
            previous_segment.metadata.setdefault("merged_segment_ids", []).append(segment.segment_id)
        else:
            merged_segments.append(segment)

    return TranscriptDocument(
        source_path=doc.source_path,
        language=doc.language,
        raw_data=doc.raw_data,
        segments=merged_segments,
        metadata=dict(doc.metadata),
    )


def improve_readability(doc: TranscriptDocument) -> TranscriptDocument:
    improved_segments: list[TranscriptSegment] = []
    for segment in doc.segments:
        text = _clean_text(segment.text)
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        if text and text[-1].isalnum():
            text = f"{text}."
        improved_segments.append(
            TranscriptSegment(
                segment_id=segment.segment_id,
                start=segment.start,
                end=segment.end,
                text=text,
                speaker=segment.speaker,
                words=segment.words,
                metadata=dict(segment.metadata),
            )
        )

    return TranscriptDocument(
        source_path=doc.source_path,
        language=doc.language,
        raw_data=doc.raw_data,
        segments=improved_segments,
        metadata=dict(doc.metadata),
    )


def _segment_to_dict(segment: TranscriptSegment) -> dict[str, Any]:
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


def to_whisperx_like_dict(doc: TranscriptDocument) -> dict[str, Any]:
    payload = dict(doc.metadata)
    payload["language"] = doc.language
    payload["segments"] = [_segment_to_dict(segment) for segment in doc.segments]
    return payload


def render_markdown(doc: TranscriptDocument) -> str:
    lines = [
        "# Enhanced Transcript",
        "",
        f"- Source: `{Path(doc.source_path).name}`",
        f"- Language: `{doc.language or 'unknown'}`",
        f"- Segments: `{len(doc.segments)}`",
        "",
    ]
    for segment in doc.segments:
        speaker = segment.speaker or "UNKNOWN"
        lines.append(f"- [{segment.start:07.2f} - {segment.end:07.2f}] **{speaker}**: {segment.text}")
    lines.append("")
    return "\n".join(lines)


def save_enhanced_outputs(doc: TranscriptDocument, source_path: str | Path, output_dir: str | Path) -> dict[str, str]:
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    run_dir = output_dir / source_path.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    json_path = run_dir / f"{source_path.stem}.enhanced.json"
    markdown_path = run_dir / f"{source_path.stem}.transcript.md"

    json_path.write_text(
        json.dumps(to_whisperx_like_dict(doc), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(doc), encoding="utf-8")

    return {"json_path": str(json_path), "markdown_path": str(markdown_path)}


def chunk_segments(doc: TranscriptDocument, max_chars: int = 900) -> list[list[TranscriptSegment]]:
    chunks: list[list[TranscriptSegment]] = []
    current_chunk: list[TranscriptSegment] = []
    current_chars = 0

    for segment in doc.segments:
        segment_size = len(segment.text) + 24
        if current_chunk and current_chars + segment_size > max_chars:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
        current_chunk.append(segment)
        current_chars += segment_size

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def rebuild_document_from_segments(doc: TranscriptDocument, segments: list[TranscriptSegment]) -> TranscriptDocument:
    return TranscriptDocument(
        source_path=doc.source_path,
        language=doc.language,
        raw_data=doc.raw_data,
        segments=segments,
        metadata=dict(doc.metadata),
    )


def document_to_plain_text(doc: TranscriptDocument) -> str:
    return "\n".join(
        f"[{segment.start:07.2f}-{segment.end:07.2f}] {segment.speaker or 'UNKNOWN'}: {segment.text}"
        for segment in doc.segments
    )


def dataclass_dump(doc: TranscriptDocument) -> dict[str, Any]:
    return asdict(doc)
