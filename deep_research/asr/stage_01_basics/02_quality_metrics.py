"""
ASR Stage 1, File 2: Quality Metrics & Issue Detection
========================================================
CONCEPT: Quantify the quality of an ASR transcript before fixing it.

You need a baseline score before improving anything — otherwise you can't
tell if your pipeline actually helped. This file defines measurable quality
dimensions and scores each one.

Quality dimensions:
  1. Confidence score  — average word-level ASR confidence
  2. Filler rate       — percentage of words that are filler words
  3. Punctuation score — presence/absence of sentence-ending punctuation
  4. Diarization       — speaker coverage, label consistency, suspicious gaps
  5. Readability       — average sentence length, capitalisation, repetition

The output of this file is a QualityReport object that every later stage
can use as a baseline and as a pass/fail gate for the improvement pipeline.

Run this file:
  uv run deep_research/asr/stage_01_basics/02_quality_metrics.py
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"

with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)

# Reuse the parsing helpers from file 01
# (In a real project these would be in a shared utils module)

FILLERS = {"uh", "um", "like", "you know", "i mean", "right", "basically", "literally", "actually", "so"}


def _clean(word: str) -> str:
    return word.lower().strip(".,?!'\"")


# ---------------------------------------------------------------------------
# 1. Per-dimension metric functions
# ---------------------------------------------------------------------------

def avg_confidence(segments: list[dict]) -> float:
    """Mean word-level confidence across the transcript."""
    scores = [w["score"] for s in segments for w in s.get("words", []) if "score" in w]
    return round(sum(scores) / max(len(scores), 1), 3)


def filler_rate(segments: list[dict]) -> float:
    """Fraction of words that are fillers (0–1)."""
    words = [w["word"] for s in segments for w in s.get("words", [])]
    filler_count = sum(1 for w in words if _clean(w) in FILLERS)
    return round(filler_count / max(len(words), 1), 3)


def punctuation_score(segments: list[dict]) -> float:
    """
    Score 0–1 for punctuation coverage.
    Checks what fraction of segment texts end with sentence-final punctuation.
    """
    ending_punct = re.compile(r"[.!?]\s*$")
    texts = [s["text"].strip() for s in segments]
    has_punct = sum(1 for t in texts if ending_punct.search(t))
    return round(has_punct / max(len(texts), 1), 3)


def capitalisation_score(segments: list[dict]) -> float:
    """
    Fraction of segments whose text starts with a capital letter.
    WhisperX outputs lowercase — 0.0 means nothing is capitalised.
    """
    texts = [s["text"].strip() for s in segments if s["text"].strip()]
    capped = sum(1 for t in texts if t[0].isupper())
    return round(capped / max(len(texts), 1), 3)


def diarization_coverage(segments: list[dict]) -> float:
    """Fraction of segments that have a non-null speaker label."""
    total = len(segments)
    labelled = sum(1 for s in segments if s.get("speaker"))
    return round(labelled / max(total, 1), 3)


def speaker_count(segments: list[dict]) -> int:
    return len({s["speaker"] for s in segments if s.get("speaker")})


def duplicate_word_rate(segments: list[dict]) -> float:
    """Fraction of word pairs that are immediate duplicates (stutters)."""
    pairs = 0
    dupes = 0
    for seg in segments:
        words = seg.get("words", [])
        for i in range(1, len(words)):
            pairs += 1
            if _clean(words[i]["word"]) == _clean(words[i-1]["word"]):
                dupes += 1
    return round(dupes / max(pairs, 1), 3)


def avg_segment_duration(segments: list[dict]) -> float:
    durations = [s["end"] - s["start"] for s in segments]
    return round(sum(durations) / max(len(durations), 1), 2)


def speaking_time_per_speaker(segments: list[dict]) -> dict[str, float]:
    """Total speaking time in seconds for each speaker."""
    times: dict[str, float] = {}
    for s in segments:
        spk = s.get("speaker", "UNKNOWN")
        dur = s["end"] - s["start"]
        times[spk] = round(times.get(spk, 0) + dur, 2)
    return dict(sorted(times.items()))


def suspicious_speaker_switches(segments: list[dict], min_gap: float = 0.05) -> list[dict]:
    """
    Find suspiciously fast speaker switches — two adjacent segments with
    different speakers but a gap < min_gap seconds.
    These may indicate a diarization error.
    """
    issues = []
    for i in range(1, len(segments)):
        prev, curr = segments[i-1], segments[i]
        gap = curr["start"] - prev["end"]
        if prev.get("speaker") != curr.get("speaker") and gap < min_gap:
            issues.append({
                "idx": i,
                "prev_speaker": prev.get("speaker"),
                "curr_speaker": curr.get("speaker"),
                "gap_ms": round(gap * 1000),
                "prev_text": prev["text"].strip()[-40:],
                "curr_text": curr["text"].strip()[:40],
            })
    return issues


def low_confidence_word_rate(segments: list[dict], threshold: float = 0.75) -> float:
    words = [w for s in segments for w in s.get("words", [])]
    low = sum(1 for w in words if w.get("score", 1.0) < threshold)
    return round(low / max(len(words), 1), 3)


# ---------------------------------------------------------------------------
# 2. QualityReport
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    avg_confidence: float
    filler_rate: float
    punctuation_score: float
    capitalisation_score: float
    diarization_coverage: float
    duplicate_word_rate: float
    low_confidence_rate: float
    num_speakers: int
    avg_segment_duration: float
    speaking_time: dict
    suspicious_switches: list
    total_segments: int
    total_words: int

    @property
    def overall_score(self) -> float:
        """
        A composite quality score in [0, 1].
        Higher = better quality transcript.
        Weights reflect how much each dimension affects final readability.
        """
        return round(
            0.25 * self.avg_confidence
            + 0.20 * (1 - self.filler_rate * 5)      # filler > 20% is very bad
            + 0.15 * self.punctuation_score
            + 0.10 * self.capitalisation_score
            + 0.15 * self.diarization_coverage
            + 0.15 * (1 - self.duplicate_word_rate * 10)
            , 3
        )

    @property
    def grade(self) -> str:
        s = self.overall_score
        if s >= 0.85: return "A — Production ready"
        if s >= 0.70: return "B — Minor cleanup needed"
        if s >= 0.50: return "C — Significant improvement needed"
        return          "D — Major issues, full pipeline required"

    def issues(self) -> list[str]:
        """Return a list of human-readable issues detected."""
        found = []
        if self.avg_confidence < 0.85:
            found.append(f"Low average confidence ({self.avg_confidence:.2f}) — likely transcription errors")
        if self.filler_rate > 0.05:
            found.append(f"High filler rate ({self.filler_rate:.1%}) — needs cleanup")
        if self.punctuation_score < 0.3:
            found.append(f"Almost no punctuation ({self.punctuation_score:.1%}) — readability severely impacted")
        if self.capitalisation_score < 0.3:
            found.append(f"Almost no capitalisation ({self.capitalisation_score:.1%})")
        if self.diarization_coverage < 0.9:
            found.append(f"Incomplete diarization ({self.diarization_coverage:.1%} segments labelled)")
        if self.duplicate_word_rate > 0.01:
            found.append(f"Stutters/duplicates detected ({self.duplicate_word_rate:.1%} of word pairs)")
        if self.suspicious_switches:
            found.append(f"{len(self.suspicious_switches)} suspicious speaker switch(es) — may be diarization errors")
        if self.low_confidence_rate > 0.15:
            found.append(f"Many low-confidence words ({self.low_confidence_rate:.1%})")
        return found


# ---------------------------------------------------------------------------
# 3. Run the metrics
# ---------------------------------------------------------------------------

segs = raw["segments"]

report = QualityReport(
    avg_confidence=avg_confidence(segs),
    filler_rate=filler_rate(segs),
    punctuation_score=punctuation_score(segs),
    capitalisation_score=capitalisation_score(segs),
    diarization_coverage=diarization_coverage(segs),
    duplicate_word_rate=duplicate_word_rate(segs),
    low_confidence_rate=low_confidence_word_rate(segs),
    num_speakers=speaker_count(segs),
    avg_segment_duration=avg_segment_duration(segs),
    speaking_time=speaking_time_per_speaker(segs),
    suspicious_switches=suspicious_speaker_switches(segs),
    total_segments=len(segs),
    total_words=sum(len(s.get("words", [])) for s in segs),
)

print("=" * 60)
print("  TRANSCRIPT QUALITY REPORT")
print("=" * 60)
print(f"  Segments          : {report.total_segments}")
print(f"  Words             : {report.total_words}")
print(f"  Speakers          : {report.num_speakers}")
print()
print("  --- ASR Accuracy ---")
print(f"  Avg confidence    : {report.avg_confidence:.3f}  (target ≥ 0.85)")
print(f"  Low-conf words    : {report.low_confidence_rate:.1%}  (target < 15%)")
print(f"  Duplicate words   : {report.duplicate_word_rate:.1%}  (target < 1%)")
print()
print("  --- Readability ---")
print(f"  Filler rate       : {report.filler_rate:.1%}  (target < 5%)")
print(f"  Punctuation score : {report.punctuation_score:.1%}  (target ≥ 80%)")
print(f"  Capitalisation    : {report.capitalisation_score:.1%}  (target ≥ 80%)")
print()
print("  --- Diarization ---")
print(f"  Coverage          : {report.diarization_coverage:.1%}  (target = 100%)")
print(f"  Avg seg duration  : {report.avg_segment_duration:.1f}s")
print(f"  Suspicious swaps  : {len(report.suspicious_switches)}")
print()
print("  --- Speaking Time ---")
for spk, secs in report.speaking_time.items():
    pct = secs / raw["duration"] * 100
    bar = "█" * int(pct / 3)
    print(f"    {spk}: {secs:6.1f}s  ({pct:4.1f}%)  {bar}")
print()
print(f"  OVERALL SCORE : {report.overall_score:.3f}")
print(f"  GRADE         : {report.grade}")
print()
print("  Issues detected:")
for issue in report.issues():
    print(f"    • {issue}")
print()

if report.suspicious_switches:
    print("  Suspicious speaker switches:")
    for sw in report.suspicious_switches:
        print(f"    [seg {sw['idx']}] {sw['prev_speaker']} → {sw['curr_speaker']} "
              f"(gap={sw['gap_ms']}ms)")
        print(f"      '...{sw['prev_text']}' / '{sw['curr_text']}...'")
    print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Measure before you fix — you need a baseline to know if your agent helped
# ✅ overall_score gives a single number to track across pipeline iterations
# ✅ Issue list drives the agent's work order (worst issues first)
# ✅ suspicious_switches catches likely diarization errors quantitatively
# ✅ speaking_time distribution helps verify diarization makes sense
# ✅ In a real pipeline: run this report before AND after each stage
