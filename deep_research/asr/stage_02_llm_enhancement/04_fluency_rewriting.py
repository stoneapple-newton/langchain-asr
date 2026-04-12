"""
ASR Stage 2, File 4: LLM Fluency Rewriting with Context
=========================================================
CONCEPT: LLM rewrites heavily disfluent speech using surrounding context
         to preserve meaning while producing readable output.

Files 1-3 handle punctuation, speaker naming, and word-level error correction.
Those are point fixes — they change individual words or add punctuation.

Some segments are too broken for point fixes:
  "the the uh we — we need to look at the auth — the jwt thing"
  "so basically itll be like uh three iterations maybe"
  "yeah no I mean — wait — I meant the other one"

Fixing these requires the LLM to:
  1. Understand what the speaker *meant* to say (requires context)
  2. Reconstruct the intended sentence without adding new meaning
  3. Track how aggressively it rewrote (minor/moderate/major)

New patterns:
  - Disfluency scoring: classify segments before sending to LLM
  - Context injection: show N turns before/after for meaning preservation
  - Structured rewrite tracking: what changed, was meaning preserved?
  - Two rewrite modes: conservative (punctuation/stutter only) vs. fluent (full reconstruction)
  - Batch rewrites with .batch() — respects max_concurrency

Run this file:
  uv run deep_research/asr/stage_02_llm_enhancement/04_fluency_rewriting.py
"""

import os
import json
import re
import copy
from pathlib import Path
import sys
from dataclasses import dataclass

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from pydantic import BaseModel, Field


TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
OUT_DIR = TRANSCRIPT_PATH.parent

with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)
segments = raw["segments"]

llm = create_chat_model(
    "asr",
    temperature=0.1,
    max_tokens=512,
)


# ---------------------------------------------------------------------------
# 1. Disfluency scoring — filter before sending to LLM
# ---------------------------------------------------------------------------

FILLERS = {"uh", "um", "like", "you know", "i mean", "basically", "literally"}
STUTTER_PATTERN = re.compile(r"\b(\w+)\s+\1\b", re.IGNORECASE)
DASH_PATTERN = re.compile(r"—|\s-\s")
INCOMPLETE_PATTERN = re.compile(r"\b(so|and|but|because|then|if)\s*$", re.IGNORECASE)


@dataclass
class DisfluencyScore:
    index: int
    text: str
    speaker: str
    filler_count: int
    stutter_count: int
    has_dash_break: bool
    is_incomplete: bool
    low_conf_words: list[str]
    score: float          # 0.0 (fluent) → 1.0 (very disfluent)
    rewrite_mode: str     # "skip" | "conservative" | "fluent"


def score_disfluency(index: int, seg: dict) -> DisfluencyScore:
    text = seg["text"].strip()
    words = text.lower().split()
    total = max(len(words), 1)

    filler_count = sum(1 for w in words if w.strip(".,?!") in FILLERS)
    stutter_count = len(STUTTER_PATTERN.findall(text))
    has_dash = bool(DASH_PATTERN.search(text))
    is_incomplete = bool(INCOMPLETE_PATTERN.search(text.rstrip()))
    low_conf = [w["word"] for w in seg.get("words", []) if w.get("score", 1.0) < 0.65]

    # Weighted score
    score = (
        (filler_count / total) * 0.35
        + (stutter_count * 0.15)
        + (0.20 if has_dash else 0)
        + (0.15 if is_incomplete else 0)
        + (min(len(low_conf), 3) / 3) * 0.15
    )
    score = min(score, 1.0)

    if score < 0.15:
        mode = "skip"
    elif score < 0.40:
        mode = "conservative"
    else:
        mode = "fluent"

    return DisfluencyScore(
        index=index, text=text, speaker=seg.get("speaker", "UNKNOWN"),
        filler_count=filler_count, stutter_count=stutter_count,
        has_dash_break=has_dash, is_incomplete=is_incomplete,
        low_conf_words=low_conf, score=round(score, 3), rewrite_mode=mode,
    )


print("=== 1. Disfluency Analysis ===\n")
scored = [score_disfluency(i, seg) for i, seg in enumerate(segments)]

mode_counts = {"skip": 0, "conservative": 0, "fluent": 0}
for s in scored:
    mode_counts[s.rewrite_mode] += 1

print(f"  {'Mode':<15} {'Count':>6}")
print("  " + "-" * 22)
for mode, count in mode_counts.items():
    print(f"  {mode:<15} {count:>6}")

top5 = sorted(scored, key=lambda x: x.score, reverse=True)[:5]
print(f"\n  Most disfluent segments:")
for s in top5:
    print(f"    [seg {s.index:>2}] score={s.score:.2f} mode={s.rewrite_mode} | {s.text[:60]}")
print()


# ---------------------------------------------------------------------------
# 2. Pydantic schema for rewrite tracking
# ---------------------------------------------------------------------------

class FluentRewrite(BaseModel):
    original: str
    rewritten: str
    rewrite_mode: str = Field(description="conservative or fluent")
    changes_made: list[str] = Field(description="Specific changes: 'removed stutter', 'completed sentence', etc.")
    meaning_preserved: bool = Field(description="True if no new information was added")
    confidence: float = Field(description="0.0-1.0: how confident the rewrite is correct")


rewrite_parser = JsonOutputParser(pydantic_object=FluentRewrite)


# ---------------------------------------------------------------------------
# 3. Rewrite chains — one per mode
# ---------------------------------------------------------------------------

CONSERVATIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a transcript editor. Fix ONLY: repeated words (stutters), "
     "filler words (uh, um, like, you know), and broken contractions.\n"
     "Do NOT restructure sentences or change vocabulary.\n"
     "Do NOT add any information that wasn't in the original.\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "Previous turn : {prev_turn}\n"
     "THIS SEGMENT   : {text}\n"
     "Next turn      : {next_turn}\n\n"
     "Speaker: {speaker}"),
]).partial(format_instructions=rewrite_parser.get_format_instructions())

FLUENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a professional transcript editor. Rewrite the segment so it reads naturally "
     "while preserving the speaker's exact meaning and vocabulary.\n\n"
     "Rules:\n"
     "  - Remove false starts, stutters, and excessive fillers\n"
     "  - Complete interrupted sentences using context from adjacent turns\n"
     "  - Keep technical terms and proper nouns exactly as-is\n"
     "  - Do NOT add opinions, facts, or information not in the original\n"
     "  - If the sentence cannot be completed confidently, leave it as a fragment\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "CONTEXT (2 turns before):\n{prev_turns}\n\n"
     "THIS SEGMENT [{speaker}]: {text}\n\n"
     "CONTEXT (1 turn after):\n{next_turn}"),
]).partial(format_instructions=rewrite_parser.get_format_instructions())

conservative_chain = CONSERVATIVE_PROMPT | llm | rewrite_parser
fluent_chain = FLUENT_PROMPT | llm | rewrite_parser


def get_context(index: int, before: int = 2, after: int = 1) -> tuple[str, str]:
    """Return (prev_turns_text, next_turn_text) for context injection."""
    prev_segs = segments[max(0, index - before):index]
    next_segs = segments[index + 1:index + 1 + after]

    prev = "\n".join(
        f"[{s.get('speaker', 'UNKNOWN')}] {s['text'].strip()}"
        for s in prev_segs
    ) or "(start of transcript)"

    nxt = "\n".join(
        f"[{s.get('speaker', 'UNKNOWN')}] {s['text'].strip()}"
        for s in next_segs
    ) or "(end of transcript)"

    return prev, nxt


# ---------------------------------------------------------------------------
# 4. Apply rewrites to disfluent segments
# ---------------------------------------------------------------------------

print("=== 2. Context-Aware Fluency Rewriting ===\n")

to_rewrite = [s for s in scored if s.rewrite_mode != "skip"]
print(f"  Rewriting {len(to_rewrite)} segments ({mode_counts['conservative']} conservative, "
      f"{mode_counts['fluent']} fluent)...\n")

rewrite_results: dict[int, dict] = {}

for ds in to_rewrite:
    prev_turns, next_turn = get_context(ds.index)
    try:
        if ds.rewrite_mode == "conservative":
            result = conservative_chain.invoke({
                "text": ds.text,
                "speaker": ds.speaker,
                "prev_turn": prev_turns.split("\n")[-1] if prev_turns else "",
                "next_turn": next_turn,
            })
        else:
            result = fluent_chain.invoke({
                "text": ds.text,
                "speaker": ds.speaker,
                "prev_turns": prev_turns,
                "next_turn": next_turn,
            })

        r = result if isinstance(result, dict) else {}
        rewrite_results[ds.index] = r

        changed = r.get("rewritten", ds.text) != ds.text
        preserved = r.get("meaning_preserved", True)
        conf = r.get("confidence", 0.0)
        mode = ds.rewrite_mode

        print(f"  [seg {ds.index:>2}] {mode:<15} conf={conf:.2f} "
              f"meaning={'✓' if preserved else '!'} changed={'yes' if changed else 'no'}")
        if changed:
            print(f"           before: {ds.text[:70]}")
            print(f"           after : {r.get('rewritten', '')[:70]}")
            changes = r.get("changes_made", [])
            if changes:
                print(f"           edits : {'; '.join(changes[:3])}")
        print()

    except Exception as e:
        print(f"  [seg {ds.index}] error: {e}\n")


# ---------------------------------------------------------------------------
# 5. Apply to transcript and save
# ---------------------------------------------------------------------------

print("=== 3. Quality Comparison ===\n")

improved = copy.deepcopy(segments)
applied = 0
meaning_warnings = 0

for idx, rw in rewrite_results.items():
    rewritten = rw.get("rewritten", "").strip()
    if rewritten and rewritten != improved[idx]["text"].strip():
        if not rw.get("meaning_preserved", True):
            meaning_warnings += 1
        improved[idx]["text"] = " " + rewritten
        applied += 1

print(f"  Segments rewritten     : {applied}")
print(f"  Meaning warnings       : {meaning_warnings} "
      f"(segments where LLM was uncertain)")
print()

# Simple before/after readability comparison
def count_fillers(segs: list[dict]) -> int:
    return sum(
        1 for s in segs for w in s["text"].lower().split()
        if w.strip(".,?!") in FILLERS
    )

def count_stutters(segs: list[dict]) -> int:
    return sum(
        len(STUTTER_PATTERN.findall(s["text"])) for s in segs
    )

print(f"  {'Metric':<25} {'Before':>8} {'After':>8}")
print("  " + "-" * 44)
fb, fa = count_fillers(segments), count_fillers(improved)
sb, sa = count_stutters(segments), count_stutters(improved)
print(f"  {'Filler words':<25} {fb:>8} {fa:>8}  ({fa - fb:+d})")
print(f"  {'Stutter repetitions':<25} {sb:>8} {sa:>8}  ({sa - sb:+d})")
print()

out = copy.deepcopy(raw)
out["segments"] = improved
out["rewrite_metadata"] = {
    "mode_counts": mode_counts,
    "applied": applied,
    "meaning_warnings": meaning_warnings,
}
out_path = OUT_DIR / "transcript_fluency_rewritten.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved: {out_path.name}\n")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Disfluency scoring filters segments before LLM — don't send clean text to LLM
# ✅ Context injection (prev/next turns) gives LLM what it needs to preserve meaning
# ✅ Two modes: conservative (safe, minimal edits) vs. fluent (full reconstruction)
# ✅ meaning_preserved flag is the safety check — flag warnings for human review
# ✅ Structured output tracks every change — auditable, not a black box
# ✅ temperature=0.1 for rewriting — slightly above 0 produces more natural output
# ✅ This layer runs AFTER word-level correction (Stage 2 File 3) — fixes structure last
