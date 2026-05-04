from __future__ import annotations

import re
from dataclasses import dataclass, field

_POSITIVE_WORDS: frozenset[str] = frozenset({
    "great", "good", "excellent", "perfect", "happy", "pleased", "satisfied",
    "wonderful", "fantastic", "appreciate", "thanks", "thank", "helpful",
    "resolved", "fixed", "solved", "awesome", "love", "glad", "sure",
    "absolutely", "certainly", "definitely", "outstanding", "brilliant",
    "delighted", "thrilled", "impressed", "smooth", "easy",
})
_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "bad", "terrible", "awful", "horrible", "frustrated", "angry", "upset",
    "disappointed", "unacceptable", "broken", "wrong", "failed", "issue",
    "problem", "error", "never", "impossible", "ridiculous", "waste", "useless",
    "complaint", "complained", "unhappy", "dissatisfied", "annoyed", "furious",
    "outraged", "disgusted", "pathetic", "disaster", "worse", "worst",
    "incompetent", "useless", "rude", "unresponsive",
})
_FILLER_RE = re.compile(
    r"\b(um|uh|er|ah|like|you know|i mean|basically|literally|sort of|kind of)\b",
    re.IGNORECASE,
)


@dataclass
class SegmentSentimentLabel:
    segment_id: str
    speaker: str | None
    start: float
    end: float
    text: str
    positive_count: int
    negative_count: int
    filler_count: int
    naive_label: str  # "positive" | "negative" | "neutral" | "mixed"


@dataclass
class SpeakerSentimentProfile:
    speaker: str
    segment_count: int
    positive_ratio: float
    negative_ratio: float
    dominant_label: str
    total_fillers: int


def score_segment_naive(
    segment_id: str,
    speaker: str | None,
    start: float,
    end: float,
    text: str,
) -> SegmentSentimentLabel:
    tokens = re.findall(r"\b\w+\b", text.lower())
    positive = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    negative = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    fillers = len(_FILLER_RE.findall(text))

    if positive > 0 and negative > 0:
        label = "mixed"
    elif positive > negative:
        label = "positive"
    elif negative > positive:
        label = "negative"
    else:
        label = "neutral"

    return SegmentSentimentLabel(
        segment_id=segment_id,
        speaker=speaker,
        start=start,
        end=end,
        text=text,
        positive_count=positive,
        negative_count=negative,
        filler_count=fillers,
        naive_label=label,
    )


def aggregate_speaker_profiles(
    labels: list[SegmentSentimentLabel],
) -> dict[str, SpeakerSentimentProfile]:
    by_speaker: dict[str, list[SegmentSentimentLabel]] = {}
    for lbl in labels:
        key = lbl.speaker or "UNKNOWN"
        by_speaker.setdefault(key, []).append(lbl)

    profiles: dict[str, SpeakerSentimentProfile] = {}
    for speaker, segs in by_speaker.items():
        n = len(segs)
        pos = sum(1 for s in segs if s.naive_label == "positive")
        neg = sum(1 for s in segs if s.naive_label == "negative")
        fillers = sum(s.filler_count for s in segs)

        if pos > neg and pos > 0:
            dominant = "positive"
        elif neg > pos and neg > 0:
            dominant = "negative"
        elif pos > 0 and neg > 0:
            dominant = "mixed"
        else:
            dominant = "neutral"

        profiles[speaker] = SpeakerSentimentProfile(
            speaker=speaker,
            segment_count=n,
            positive_ratio=round(pos / n, 3) if n else 0.0,
            negative_ratio=round(neg / n, 3) if n else 0.0,
            dominant_label=dominant,
            total_fillers=fillers,
        )

    return profiles


def format_labels_for_llm(labels: list[SegmentSentimentLabel]) -> str:
    lines: list[str] = []
    for lbl in labels:
        lines.append(
            f"[seg={lbl.segment_id} speaker={lbl.speaker or 'UNKNOWN'} "
            f"naive={lbl.naive_label} t={lbl.start:.1f}s]"
        )
        lines.append(f'  "{lbl.text}"')
    return "\n".join(lines)


def render_sentiment_report(
    labels: list[SegmentSentimentLabel],
    profiles: dict[str, SpeakerSentimentProfile],
) -> str:
    lines = ["# Sentiment Analysis Report", ""]
    lines += ["## Per-Speaker Profiles", ""]
    for speaker, profile in sorted(profiles.items()):
        lines.append(
            f"**{speaker}**: {profile.dominant_label.upper()} "
            f"(+{profile.positive_ratio:.0%} / -{profile.negative_ratio:.0%}, "
            f"fillers={profile.total_fillers}, segments={profile.segment_count})"
        )

    lines += ["", "## Segment-Level Labels", ""]
    for lbl in labels:
        lines.append(
            f"- [{lbl.start:07.2f}–{lbl.end:07.2f}] "
            f"**{lbl.speaker or 'UNKNOWN'}** [{lbl.naive_label.upper()}]: {lbl.text}"
        )
    lines.append("")
    return "\n".join(lines)
