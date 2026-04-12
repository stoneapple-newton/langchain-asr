"""
ASR Stage 3, File 1: Speaker Turn Analysis
============================================
CONCEPT: Analysing diarization patterns to detect where speaker labels are wrong.

Diarization errors come in several flavours:
  1. Missed turn  — a long segment actually has two speakers mid-way
  2. Wrong label  — SPEAKER_00 labelled as SPEAKER_01 for a few words
  3. Fragmentation — one speaker's turn split into many tiny segments
  4. Overlap confusion — speaker overlap attributed entirely to the wrong person

This file analyses the raw diarization without calling an LLM yet, building
a statistical picture of each speaker's behaviour and flagging anomalies.

Key metrics per speaker:
  • Average turn duration
  • Turn-taking patterns (who speaks after whom)
  • Vocabulary profile (what words/topics each speaker uses most)
  • Segment size distribution (are there suspiciously short micro-turns?)

Run this file:
  uv run deep_research/asr/stage_03_diarization/01_speaker_analysis.py
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass, field

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)
segments = raw["segments"]


# ---------------------------------------------------------------------------
# 1. Turn-taking sequence
# ---------------------------------------------------------------------------
# A "turn" is a maximal run of consecutive segments from the same speaker.
# Analysing the turn sequence reveals conversation dynamics and anomalies.

@dataclass
class Turn:
    speaker: str
    start: float
    end: float
    segments: list[int]   # indices into the segments list
    text: str

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


def build_turns(segments: list[dict]) -> list[Turn]:
    """Collapse consecutive same-speaker segments into turns."""
    turns: list[Turn] = []
    for i, seg in enumerate(segments):
        spk = seg.get("speaker", "UNKNOWN")
        if turns and turns[-1].speaker == spk:
            turns[-1].end = seg["end"]
            turns[-1].segments.append(i)
            turns[-1].text += " " + seg["text"].strip()
        else:
            turns.append(Turn(
                speaker=spk,
                start=seg["start"],
                end=seg["end"],
                segments=[i],
                text=seg["text"].strip(),
            ))
    return turns


turns = build_turns(segments)

print("=== 1. Turn-taking sequence ===")
print(f"  Total turns : {len(turns)}")
print()
for i, turn in enumerate(turns):
    bar = "█" * min(int(turn.duration), 30)
    print(f"  Turn {i:02d}  [{turn.speaker}]  {turn.duration:5.1f}s  {bar}")
    print(f"          '{turn.text[:70]}...' " if len(turn.text) > 70 else f"          '{turn.text}'")
print()


# ---------------------------------------------------------------------------
# 2. Per-speaker statistics
# ---------------------------------------------------------------------------

@dataclass
class SpeakerStats:
    speaker: str
    turn_count: int = 0
    total_duration: float = 0.0
    turn_durations: list[float] = field(default_factory=list)
    word_counts: list[int] = field(default_factory=list)
    all_text: str = ""

    @property
    def avg_turn_duration(self) -> float:
        return round(self.total_duration / max(self.turn_count, 1), 2)

    @property
    def avg_word_count(self) -> float:
        return round(sum(self.word_counts) / max(len(self.word_counts), 1), 1)

    @property
    def short_turns(self) -> int:
        """Turns under 2 seconds — often backchannels or diarization errors."""
        return sum(1 for d in self.turn_durations if d < 2.0)

    def top_words(self, n: int = 8) -> list[tuple[str, int]]:
        STOPWORDS = {"the", "a", "an", "is", "are", "was", "we", "i", "to",
                     "of", "and", "it", "in", "that", "for", "this", "you",
                     "so", "be", "on", "have", "do", "with", "by", "at"}
        words = re.findall(r"\b[a-z]{3,}\b", self.all_text.lower())
        filtered = [w for w in words if w not in STOPWORDS]
        return Counter(filtered).most_common(n)


def compute_speaker_stats(turns: list[Turn]) -> dict[str, SpeakerStats]:
    stats: dict[str, SpeakerStats] = {}
    for turn in turns:
        spk = turn.speaker
        if spk not in stats:
            stats[spk] = SpeakerStats(speaker=spk)
        s = stats[spk]
        s.turn_count += 1
        s.total_duration += turn.duration
        s.turn_durations.append(turn.duration)
        s.word_counts.append(turn.word_count)
        s.all_text += " " + turn.text
    return stats


speaker_stats = compute_speaker_stats(turns)

print("=== 2. Per-speaker statistics ===")
for spk, s in sorted(speaker_stats.items()):
    print(f"\n  {spk}")
    print(f"    Turns          : {s.turn_count}")
    print(f"    Total speaking : {s.total_duration:.1f}s")
    print(f"    Avg turn dur   : {s.avg_turn_duration:.1f}s")
    print(f"    Avg words/turn : {s.avg_word_count:.0f}")
    print(f"    Short turns    : {s.short_turns} (< 2s)")
    print(f"    Top words      : {[w for w, _ in s.top_words(6)]}")
print()


# ---------------------------------------------------------------------------
# 3. Turn-taking transition matrix
# ---------------------------------------------------------------------------
# Who speaks after whom? Unusual transitions may indicate diarization errors.
# E.g., if SPEAKER_00 almost never speaks right after SPEAKER_00 except in
# a few places, those exceptions are worth investigating.

def build_transition_matrix(turns: list[Turn]) -> dict[tuple[str, str], int]:
    transitions: dict[tuple[str, str], int] = Counter()
    for i in range(1, len(turns)):
        prev_spk = turns[i-1].speaker
        curr_spk = turns[i].speaker
        transitions[(prev_spk, curr_spk)] += 1
    return dict(transitions)


transitions = build_transition_matrix(turns)
speakers = sorted(speaker_stats.keys())

print("=== 3. Turn transition matrix ===")
print(f"  {'':15}", end="")
for spk in speakers:
    print(f"  {spk:12}", end="")
print()
for from_spk in speakers:
    print(f"  {from_spk:<15}", end="")
    for to_spk in speakers:
        count = transitions.get((from_spk, to_spk), 0)
        print(f"  {count:<12}", end="")
    print()
print()
print("  (Diagonal = same speaker back-to-back, should be 0 after merging turns)")
print()


# ---------------------------------------------------------------------------
# 4. Anomaly detection
# ---------------------------------------------------------------------------

@dataclass
class DiarizationAnomaly:
    type: str
    segment_idx: int | None
    turn_idx: int | None
    description: str
    severity: str   # "low", "medium", "high"


def detect_anomalies(
    segments: list[dict],
    turns: list[Turn],
    speaker_stats: dict[str, SpeakerStats],
) -> list[DiarizationAnomaly]:
    anomalies = []

    # Anomaly 1: Very short isolated turns (< 1.5s, single word/segment)
    for i, turn in enumerate(turns):
        if turn.duration < 1.5 and turn.word_count <= 2:
            anomalies.append(DiarizationAnomaly(
                type="SHORT_TURN",
                segment_idx=turn.segments[0] if turn.segments else None,
                turn_idx=i,
                description=f"{turn.speaker} has {turn.word_count}-word turn ({turn.duration}s): '{turn.text}'",
                severity="low",
            ))

    # Anomaly 2: Same speaker in consecutive turns (after merging) — shouldn't exist
    for i in range(1, len(turns)):
        if turns[i].speaker == turns[i-1].speaker:
            anomalies.append(DiarizationAnomaly(
                type="SAME_SPEAKER_ADJACENT",
                segment_idx=None,
                turn_idx=i,
                description=f"{turns[i].speaker} has two adjacent turns — possible missed merge",
                severity="medium",
            ))

    # Anomaly 3: Suspiciously fast switches (gap < 100ms)
    for i in range(1, len(segments)):
        prev, curr = segments[i-1], segments[i]
        gap = curr["start"] - prev["end"]
        if (prev.get("speaker") != curr.get("speaker") and gap < 0.1):
            anomalies.append(DiarizationAnomaly(
                type="FAST_SWITCH",
                segment_idx=i,
                turn_idx=None,
                description=(f"{prev.get('speaker')} → {curr.get('speaker')} "
                             f"in {gap*1000:.0f}ms: '...{prev['text'].strip()[-30:]}' / "
                             f"'{curr['text'].strip()[:30]}...'"),
                severity="high",
            ))

    # Anomaly 4: Speaker dominates unusually short segments
    for spk, s in speaker_stats.items():
        short_frac = s.short_turns / max(s.turn_count, 1)
        if short_frac > 0.5 and s.turn_count > 2:
            anomalies.append(DiarizationAnomaly(
                type="HIGH_SHORT_TURN_RATE",
                segment_idx=None,
                turn_idx=None,
                description=f"{spk} has {short_frac:.0%} short turns — possible over-segmentation",
                severity="medium",
            ))

    return anomalies


anomalies = detect_anomalies(segments, turns, speaker_stats)

print("=== 4. Diarization anomalies ===")
if not anomalies:
    print("  No anomalies detected")
else:
    by_severity = {"high": [], "medium": [], "low": []}
    for a in anomalies:
        by_severity[a.severity].append(a)

    for sev in ["high", "medium", "low"]:
        if by_severity[sev]:
            print(f"\n  [{sev.upper()}]")
            for a in by_severity[sev]:
                print(f"    [{a.type}] {a.description}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Build turns (merged speaker runs) before analysing — raw segments are too granular
# ✅ Transition matrix reveals conversation dynamics and unexpected same-speaker pairs
# ✅ Short isolated turns < 1.5s are the first diarization red flag to investigate
# ✅ Fast speaker switches (< 100ms gap) are almost certainly diarization errors
# ✅ Vocab profiling per speaker confirms role inference from Stage 2
# ✅ Collect all anomalies into a list — the LangGraph agent uses it as a work queue
