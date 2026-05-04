from __future__ import annotations

import re
from dataclasses import dataclass, field

_SHIFT_RE = re.compile(
    r"\b(next|moving on|another thing|let'?s talk about|let me bring up|"
    r"on a different note|switching to|before i forget|also wanted to|"
    r"by the way|so about|regarding|the other thing|one more thing|"
    r"on that note|speaking of|circling back|actually|while we're at it)\b",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"\?")
_SHORT_SEGMENT_THRESHOLD = 1.5  # seconds
_LONG_GAP_THRESHOLD = 2.0  # seconds


@dataclass
class TopicBoundarySignal:
    segment_id: str
    index: int
    start: float
    has_shift_phrase: bool
    has_question: bool
    speaker_changed: bool
    gap_seconds: float
    signal_score: float  # 0.0 – 1.0


@dataclass
class TopicSegment:
    topic_id: str
    title: str
    start_time: float
    end_time: float
    segment_ids: list[str] = field(default_factory=list)
    summary: str = ""


def detect_boundary_signals(segments: list) -> list[TopicBoundarySignal]:
    """Score each segment as a potential topic boundary. Always one result per segment."""
    signals: list[TopicBoundarySignal] = []
    prev_speaker: str | None = None
    prev_end: float = 0.0

    for idx, seg in enumerate(segments):
        gap = max(0.0, seg.start - prev_end)
        speaker_changed = (
            idx > 0
            and prev_speaker is not None
            and seg.speaker is not None
            and prev_speaker != seg.speaker
        )
        has_shift = bool(_SHIFT_RE.search(seg.text))
        has_question = bool(_QUESTION_RE.search(seg.text))

        score = 0.0
        if has_shift:
            score += 0.5
        if speaker_changed:
            score += 0.2
        if gap > _LONG_GAP_THRESHOLD:
            score += 0.2
        if has_question:
            score += 0.1

        signals.append(
            TopicBoundarySignal(
                segment_id=seg.segment_id,
                index=idx,
                start=seg.start,
                has_shift_phrase=has_shift,
                has_question=has_question,
                speaker_changed=speaker_changed,
                gap_seconds=round(gap, 2),
                signal_score=round(min(score, 1.0), 3),
            )
        )

        if seg.speaker:
            prev_speaker = seg.speaker
        prev_end = seg.end

    return signals


def group_into_topic_segments(
    segments: list,
    signals: list[TopicBoundarySignal],
    threshold: float = 0.5,
) -> list[TopicSegment]:
    """Split segments at high-score boundaries into TopicSegment groups."""
    if not segments:
        return []

    split_indices: set[int] = {0}
    for signal in signals:
        if signal.index > 0 and signal.signal_score >= threshold:
            split_indices.add(signal.index)

    sorted_splits = sorted(split_indices)
    topic_segments: list[TopicSegment] = []

    for topic_idx, split_start in enumerate(sorted_splits):
        split_end = (
            sorted_splits[topic_idx + 1]
            if topic_idx + 1 < len(sorted_splits)
            else len(segments)
        )
        chunk = segments[split_start:split_end]
        if not chunk:
            continue
        topic_segments.append(
            TopicSegment(
                topic_id=f"topic_{topic_idx + 1:02d}",
                title=f"Topic {topic_idx + 1}",
                start_time=chunk[0].start,
                end_time=chunk[-1].end,
                segment_ids=[s.segment_id for s in chunk],
            )
        )

    return topic_segments


def format_topic_for_llm(topic: TopicSegment, segments: list) -> str:
    seg_map = {s.segment_id: s for s in segments}
    lines = [f"[{topic.topic_id}]"]
    for seg_id in topic.segment_ids:
        seg = seg_map.get(seg_id)
        if seg:
            lines.append(
                f"  [{seg.start:.1f}s] {seg.speaker or 'UNKNOWN'}: {seg.text}"
            )
    return "\n".join(lines)


def render_segmentation_report(topic_segments: list[TopicSegment]) -> str:
    lines = ["# Topic Segmentation Report", ""]
    for topic in topic_segments:
        duration = topic.end_time - topic.start_time
        lines.append(f"## {topic.topic_id}: {topic.title}")
        lines.append(
            f"- Time: {topic.start_time:.1f}s – {topic.end_time:.1f}s "
            f"({duration:.1f}s, {len(topic.segment_ids)} segments)"
        )
        if topic.summary:
            lines.append(f"- Summary: {topic.summary}")
        lines.append("")
    return "\n".join(lines)
