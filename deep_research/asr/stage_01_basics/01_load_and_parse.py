"""
ASR Stage 1, File 1: Load and Parse WhisperX Output
=====================================================
CONCEPT: Understanding the WhisperX JSON format before touching any LLM.

WhisperX extends OpenAI Whisper with two extra capabilities:
  1. Word-level timestamps — each word has its own start/end time
  2. Speaker diarization  — each word/segment tagged with SPEAKER_XX

Before building an AI-powered improvement pipeline, you need to understand
the raw data: its structure, what each field means, and what can go wrong.

WhisperX JSON structure:
  {
    "language": "en",
    "duration": float,                     ← total audio duration in seconds
    "segments": [
      {
        "start": float,                    ← segment start time (seconds)
        "end": float,                      ← segment end time (seconds)
        "text": " raw transcribed text",  ← note the leading space
        "speaker": "SPEAKER_00",          ← diarization label (may be null)
        "words": [
          {
            "word": "hello",              ← individual word token
            "start": float,              ← word start time
            "end": float,                ← word end time
            "score": float,              ← ASR confidence 0.0–1.0
            "speaker": "SPEAKER_00"      ← word-level speaker (may differ from segment)
          }
        ]
      }
    ]
  }

Common problems you will fix in later stages:
  • Missing punctuation — whisper outputs lowercase, no commas/periods
  • Filler words — "uh", "um", "like", repeated words
  • Low-confidence words (score < 0.7) — likely transcription errors
  • Generic speaker labels — SPEAKER_00 instead of real names
  • Short isolated segments — backchannels like "yeah" in their own segment
  • Duplicate words — stutter captured as repeated word tokens

Run this file:
  uv run deep_research/asr/stage_01_basics/01_load_and_parse.py
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 1. Load the JSON
# ---------------------------------------------------------------------------

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"

print("=== 1. Load raw JSON ===")

with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

print(f"Language  : {raw['language']}")
print(f"Duration  : {raw['duration']:.1f} seconds ({raw['duration']/60:.1f} minutes)")
print(f"Segments  : {len(raw['segments'])}")
total_words = sum(len(s.get("words", [])) for s in raw["segments"])
print(f"Words     : {total_words}")
print()


# ---------------------------------------------------------------------------
# 2. Dataclass model — typed representation of the JSON
# ---------------------------------------------------------------------------
# Working with raw dicts is error-prone. Dataclasses give you type hints,
# dot-access, and auto-completion in your IDE.

@dataclass
class Word:
    word: str
    start: float
    end: float
    score: float
    speaker: str | None

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    @property
    def is_low_confidence(self) -> bool:
        return self.score < 0.75

    @property
    def is_filler(self) -> bool:
        return self.word.lower().strip(".,?!") in {"uh", "um", "like", "you know", "i mean", "right"}


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    @property
    def clean_text(self) -> str:
        return self.text.strip()

    @property
    def word_count(self) -> int:
        return len(self.words)

    @property
    def avg_confidence(self) -> float:
        if not self.words:
            return 0.0
        return round(sum(w.score for w in self.words) / len(self.words), 3)

    @property
    def low_confidence_words(self) -> list[Word]:
        return [w for w in self.words if w.is_low_confidence]

    @property
    def filler_words(self) -> list[Word]:
        return [w for w in self.words if w.is_filler]

    def timestamp(self) -> str:
        """Format as [MM:SS.s → MM:SS.s]"""
        def fmt(t: float) -> str:
            m, s = divmod(t, 60)
            return f"{int(m):02d}:{s:04.1f}"
        return f"[{fmt(self.start)} → {fmt(self.end)}]"


@dataclass
class Transcript:
    language: str
    duration: float
    segments: list[Segment]

    @property
    def speakers(self) -> set[str]:
        return {s.speaker for s in self.segments if s.speaker}

    @property
    def all_words(self) -> list[Word]:
        return [w for s in self.segments for w in s.words]


def parse_transcript(raw: dict) -> Transcript:
    """Convert raw JSON dict into typed Transcript dataclass."""
    segments = []
    for seg in raw["segments"]:
        words = [
            Word(
                word=w["word"],
                start=w["start"],
                end=w["end"],
                score=w.get("score", 1.0),
                speaker=w.get("speaker"),
            )
            for w in seg.get("words", [])
        ]
        segments.append(Segment(
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
            speaker=seg.get("speaker"),
            words=words,
        ))
    return Transcript(
        language=raw["language"],
        duration=raw["duration"],
        segments=segments,
    )


print("=== 2. Parse into typed dataclasses ===")
transcript = parse_transcript(raw)
print(f"Speakers found : {sorted(transcript.speakers)}")
print(f"Total words    : {len(transcript.all_words)}")
print()


# ---------------------------------------------------------------------------
# 3. Walk segments
# ---------------------------------------------------------------------------

print("=== 3. Inspect segments ===")
for i, seg in enumerate(transcript.segments[:5]):
    print(f"\nSegment {i:02d} {seg.timestamp()}")
    print(f"  Speaker   : {seg.speaker}")
    print(f"  Text      : {seg.clean_text[:80]}")
    print(f"  Words     : {seg.word_count}")
    print(f"  Avg conf  : {seg.avg_confidence}")
    if seg.filler_words:
        print(f"  Fillers   : {[w.word for w in seg.filler_words]}")
    if seg.low_confidence_words:
        print(f"  Low-conf  : {[(w.word, round(w.score,2)) for w in seg.low_confidence_words]}")
print()


# ---------------------------------------------------------------------------
# 4. Inspect word-level data
# ---------------------------------------------------------------------------

print("=== 4. Word-level detail (first segment) ===")
seg0 = transcript.segments[0]
print(f"Segment: '{seg0.clean_text}'")
print(f"{'WORD':<18} {'START':>6} {'END':>6} {'DUR':>5} {'SCORE':>6} {'FLAGS'}")
print("-" * 58)
for w in seg0.words:
    flags = []
    if w.is_filler:         flags.append("FILLER")
    if w.is_low_confidence: flags.append("LOW-CONF")
    print(f"  {w.word:<16} {w.start:>6.2f} {w.end:>6.2f} {w.duration:>5.2f} {w.score:>6.2f}  {' '.join(flags)}")
print()


# ---------------------------------------------------------------------------
# 5. Identify common issues at a glance
# ---------------------------------------------------------------------------

print("=== 5. Issue scan across all segments ===")

all_fillers = [w for w in transcript.all_words if w.is_filler]
all_low_conf = [w for w in transcript.all_words if w.is_low_confidence]

# Detect duplicate consecutive words (stutters)
duplicate_words = []
for seg in transcript.segments:
    for i in range(1, len(seg.words)):
        if seg.words[i].word.lower() == seg.words[i-1].word.lower():
            duplicate_words.append((seg.words[i-1], seg.words[i]))

# Detect very short segments (likely backchannels)
short_segments = [s for s in transcript.segments if s.word_count <= 2]

print(f"  Filler words          : {len(all_fillers)}")
print(f"  Low-confidence words  : {len(all_low_conf)}")
print(f"  Duplicate/stutter     : {len(duplicate_words)}")
print(f"  Very short segments   : {len(short_segments)}")
print()

print("  Sample low-confidence words:")
for w in sorted(all_low_conf, key=lambda x: x.score)[:8]:
    print(f"    '{w.word}' (score={w.score:.2f}, speaker={w.speaker})")
print()

print("  Short segments (backchannels):")
for s in short_segments:
    print(f"    {s.timestamp()} [{s.speaker}] '{s.clean_text}'")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ WhisperX JSON has segment-level and word-level data — both are useful
# ✅ score (0–1) is ASR confidence — below 0.75 is a likely transcription error
# ✅ speaker labels are generic (SPEAKER_00) — naming them is a key improvement
# ✅ Parse into dataclasses early — raw dicts are hard to work with at scale
# ✅ Three main issue categories: fillers, low-confidence words, short segments
# ✅ Understand your data before sending it to an LLM — garbage in, garbage out
