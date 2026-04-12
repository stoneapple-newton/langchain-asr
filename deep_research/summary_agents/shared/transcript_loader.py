"""
Transcript loading utilities for ASR summarization.

Provides functions to load WhisperX-style transcript JSON and format it
for LLM consumption with various formatting options.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TranscriptSegment:
    """A single segment of a transcript with speaker and timing info."""

    start: float
    end: float
    text: str
    speaker: str | None
    words: list[dict[str, Any]]
    confidence: float  # Average word confidence

    @property
    def duration(self) -> float:
        """Duration of the segment in seconds."""
        return self.end - self.start

    @property
    def word_count(self) -> int:
        """Number of words in the segment."""
        return len(self.words)

    def format(self, include_speaker: bool = True, include_timestamp: bool = False) -> str:
        """Format segment as a string."""
        parts = []
        if include_timestamp:
            parts.append(f"[{self.start:.1f}s]")
        if include_speaker and self.speaker:
            parts.append(f"{self.speaker}:")
        parts.append(self.text)
        return " ".join(parts)


@dataclass
class MeetingTranscript:
    """A complete meeting transcript with metadata."""

    language: str
    duration: float
    segments: list[TranscriptSegment]
    raw_data: dict[str, Any]

    @property
    def speakers(self) -> set[str]:
        """Set of unique speakers in the transcript."""
        return {seg.speaker for seg in self.segments if seg.speaker}

    @property
    def total_words(self) -> int:
        """Total word count across all segments."""
        return sum(seg.word_count for seg in self.segments)

    @property
    def average_confidence(self) -> float:
        """Average confidence score across all words."""
        if not self.segments:
            return 0.0
        confidences = [seg.confidence for seg in self.segments]
        return sum(confidences) / len(confidences)

    def get_segments_by_speaker(self, speaker: str) -> list[TranscriptSegment]:
        """Get all segments for a specific speaker."""
        return [seg for seg in self.segments if seg.speaker == speaker]

    def get_low_confidence_segments(self, threshold: float = 0.8) -> list[TranscriptSegment]:
        """Get segments with average confidence below threshold."""
        return [seg for seg in self.segments if seg.confidence < threshold]


def _resolve_path(path: str | Path) -> Path:
    """Resolve a path, handling relative paths from repo root."""
    resolved = Path(path)
    if not resolved.is_absolute():
        # Try relative to current working directory first
        if resolved.exists():
            return resolved
        # Fall back to repo root
        repo_root = Path(__file__).resolve().parents[3]
        resolved = repo_root / resolved
    return resolved


def load_transcript(path: str | Path) -> MeetingTranscript:
    """
    Load a WhisperX-style transcript JSON file.

    Args:
        path: Path to the transcript JSON file

    Returns:
        MeetingTranscript object with parsed segments
    """
    resolved = _resolve_path(path)
    raw_data = json.loads(resolved.read_text(encoding="utf-8"))

    segments = []
    for seg_data in raw_data.get("segments", []):
        words = seg_data.get("words", [])

        # Calculate average confidence from word-level scores
        if words:
            confidences = [float(w.get("score", 1.0)) for w in words]
            avg_confidence = sum(confidences) / len(confidences)
        else:
            avg_confidence = 1.0

        segment = TranscriptSegment(
            start=float(seg_data.get("start", 0)),
            end=float(seg_data.get("end", 0)),
            text=seg_data.get("text", "").strip(),
            speaker=seg_data.get("speaker"),
            words=words,
            confidence=round(avg_confidence, 3),
        )
        segments.append(segment)

    return MeetingTranscript(
        language=raw_data.get("language", "unknown"),
        duration=raw_data.get("duration", 0),
        segments=segments,
        raw_data=raw_data,
    )


def format_transcript_for_llm(
    transcript: MeetingTranscript,
    format_type: str = "speaker_turns",
    include_confidence: bool = False,
    max_length: int | None = None,
) -> str:
    """
    Format a transcript for LLM consumption.

    Args:
        transcript: The meeting transcript to format
        format_type: One of "speaker_turns", "paragraph", "dialogue"
        include_confidence: Whether to include confidence annotations
        max_length: Maximum character length (will truncate with warning)

    Returns:
        Formatted transcript string
    """
    lines = []

    if format_type == "speaker_turns":
        # Format: SPEAKER_00: This is what they said
        for seg in transcript.segments:
            prefix = f"{seg.speaker}:" if seg.speaker else "UNKNOWN:"
            text = seg.text
            if include_confidence and seg.confidence < 0.9:
                text += f" [confidence: {seg.confidence}]"
            lines.append(f"{prefix} {text}")

    elif format_type == "paragraph":
        # Format: Paragraphs grouped by speaker
        current_speaker = None
        current_text = []

        for seg in transcript.segments:
            if seg.speaker != current_speaker and current_text:
                speaker_label = current_speaker or "UNKNOWN"
                lines.append(f"{speaker_label}: {' '.join(current_text)}\n")
                current_text = []
            current_speaker = seg.speaker
            current_text.append(seg.text)

        if current_text:
            speaker_label = current_speaker or "UNKNOWN"
            lines.append(f"{speaker_label}: {' '.join(current_text)}")

    elif format_type == "dialogue":
        # Format: Script-like with timestamps
        for seg in transcript.segments:
            timestamp = f"[{int(seg.start // 60):02d}:{int(seg.start % 60):02d}]"
            speaker = seg.speaker or "UNKNOWN"
            lines.append(f"{timestamp} {speaker}: {seg.text}")

    else:
        raise ValueError(f"Unknown format_type: {format_type}")

    result = "\n".join(lines)

    # Apply length limit if specified
    if max_length and len(result) > max_length:
        truncated = result[:max_length]
        # Try to cut at a newline
        last_newline = truncated.rfind("\n")
        if last_newline > max_length * 0.8:
            truncated = truncated[:last_newline]
        result = truncated + "\n\n[TRANSCRIPT TRUNCATED DUE TO LENGTH]"

    return result


def get_sample_transcript_path() -> Path:
    """Get the path to the sample meeting transcript."""
    return Path(__file__).parent / "sample_data" / "meeting_sample.json"


if __name__ == "__main__":
    # Demo usage
    sample_path = get_sample_transcript_path()
    print(f"Loading sample transcript: {sample_path}")

    transcript = load_transcript(sample_path)
    print(f"\nMeeting Duration: {transcript.duration:.1f}s")
    print(f"Speakers: {transcript.speakers}")
    print(f"Total Words: {transcript.total_words}")
    print(f"Avg Confidence: {transcript.average_confidence:.3f}")

    print("\n" + "=" * 60)
    print("Speaker Turns Format:")
    print("=" * 60)
    print(format_transcript_for_llm(transcript, format_type="speaker_turns")[:800])

    print("\n" + "=" * 60)
    print("Paragraph Format:")
    print("=" * 60)
    print(format_transcript_for_llm(transcript, format_type="paragraph")[:800])
