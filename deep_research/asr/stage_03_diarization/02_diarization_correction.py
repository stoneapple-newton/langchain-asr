"""
ASR Stage 3, File 2: LLM-Powered Diarization Correction
=========================================================
CONCEPT: Using conversational context to fix misattributed speaker labels.

Once anomalies are identified (Stage 3, File 1), the LLM can reason about
who actually said each suspicious segment by looking at:
  • Surrounding conversational context (who was speaking before/after)
  • The content of the segment (is it consistent with that speaker's role?)
  • Conversational coherence (does this response make sense from this speaker?)
  • Backchannel patterns (short "yeah"/"agreed" are usually the listener)

The LLM cannot hear audio — it relies entirely on linguistic evidence.
This is a best-effort improvement, not a perfect solution.

Strategies:
  A. Context-window correction — ask LLM to re-assign a specific suspicious segment
  B. Bulk re-diarization   — send a full turn block and ask LLM to re-assign all
  C. Merge tiny segments   — short backchannels folded into the correct speaker

Run this file:
  uv run deep_research/asr/stage_03_diarization/02_diarization_correction.py
"""

import os
import json
import copy
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

load_dotenv()

TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)
segments = raw["segments"]

# Load name map if available
name_map_path = TRANSCRIPT_PATH.parent / "speaker_names.json"
name_map: dict[str, str] = {}
if name_map_path.exists():
    with open(name_map_path) as f:
        name_map = json.load(f)

llm = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    num_predict=512,
)

KNOWN_SPEAKERS = sorted({s.get("speaker") for s in segments if s.get("speaker")})


# ---------------------------------------------------------------------------
# Helper: render a window of segments for LLM context
# ---------------------------------------------------------------------------

def render_context_window(
    segments: list[dict],
    center_idx: int,
    window: int = 4,
    name_map: dict[str, str] | None = None,
) -> str:
    nm = name_map or {}
    start = max(0, center_idx - window)
    end = min(len(segments), center_idx + window + 1)
    lines = []
    for i in range(start, end):
        seg = segments[i]
        spk = seg.get("speaker", "UNKNOWN")
        label = nm.get(spk, spk)
        marker = ">>>" if i == center_idx else "   "
        lines.append(f"{marker} [{i:02d}] {label}: {seg['text'].strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# STRATEGY A — Context-window correction for a single suspicious segment
# ---------------------------------------------------------------------------

class SegmentDecision(BaseModel):
    segment_idx: int = Field(description="The index of the segment being evaluated")
    original_speaker: str = Field(description="The original SPEAKER_XX label")
    correct_speaker: str = Field(description="The most likely correct SPEAKER_XX label")
    confidence: float = Field(description="Confidence in this correction (0–1)")
    reasoning: str = Field(description="Why this speaker assignment is correct")
    action: str = Field(description="One of: 'keep', 'reassign', 'merge_with_prev', 'merge_with_next'")


decision_parser = JsonOutputParser(pydantic_object=SegmentDecision)

context_correction_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a diarization correction expert. A segment has been flagged as "
     "potentially misattributed. Using the surrounding context, determine:\n"
     "  1. Whether the current speaker label is correct\n"
     "  2. If not, which speaker actually said it\n"
     "  3. Whether it should be merged with an adjacent segment\n\n"
     "Known speakers and their roles:\n{speaker_roles}\n\n"
     "Rules:\n"
     "  • Short affirmations ('yeah', 'agreed', 'okay') are usually the listener\n"
     "  • Questions are usually followed by the same speaker who asked them\n"
     "  • Technical content should be attributed to the technical speaker\n"
     "  • Only reassign if confident (> 0.7)\n\n"
     "Respond with JSON: {format_instructions}"),
    ("human",
     "Segment index: {seg_idx}\n"
     "Current speaker: {current_speaker}\n"
     "Available speakers: {available_speakers}\n\n"
     "Context window (>>> marks the target segment):\n{context}"),
]).partial(
    format_instructions=decision_parser.get_format_instructions(),
)

context_correction_chain = context_correction_prompt | llm | decision_parser


def build_speaker_roles_text(name_map: dict[str, str]) -> str:
    if not name_map:
        return "No role information available"
    return "\n".join(f"  {spk}: {name}" for spk, name in sorted(name_map.items()))


def find_suspicious_segments(segments: list[dict]) -> list[int]:
    """Find segment indices worth reviewing for diarization errors."""
    suspicious = []
    for i, seg in enumerate(segments):
        words = seg.get("words", [])
        # Short segments (possible backchannels misattributed)
        if len(words) <= 2 and i > 0 and i < len(segments) - 1:
            suspicious.append(i)
        # Fast switch from the previous segment
        if i > 0:
            gap = seg["start"] - segments[i-1]["end"]
            prev_spk = segments[i-1].get("speaker")
            curr_spk = seg.get("speaker")
            if prev_spk != curr_spk and gap < 0.15:
                suspicious.append(i)
    return list(set(suspicious))


print("=== STRATEGY A: Context-window correction ===")
suspicious = find_suspicious_segments(segments)
print(f"  Suspicious segments: {suspicious}")

corrections_a: list[SegmentDecision] = []
for idx in suspicious[:4]:   # limit to 4 for demo
    seg = segments[idx]
    context = render_context_window(segments, idx, window=3, name_map=name_map)
    print(f"\n  Evaluating segment {idx} [{seg.get('speaker')}]: '{seg['text'].strip()[:60]}'")

    result = context_correction_chain.invoke({
        "seg_idx": idx,
        "current_speaker": seg.get("speaker", "UNKNOWN"),
        "available_speakers": ", ".join(KNOWN_SPEAKERS),
        "speaker_roles": build_speaker_roles_text(name_map),
        "context": context,
    })

    action = result.get("action", "keep") if isinstance(result, dict) else result.action
    correct = result.get("correct_speaker", "") if isinstance(result, dict) else result.correct_speaker
    conf = result.get("confidence", 0) if isinstance(result, dict) else result.confidence
    reasoning = result.get("reasoning", "") if isinstance(result, dict) else result.reasoning

    print(f"    Action     : {action}")
    print(f"    Correct spk: {correct} (conf={conf:.0%})")
    print(f"    Reasoning  : {reasoning[:100]}")
    corrections_a.append(result)
print()


# ---------------------------------------------------------------------------
# STRATEGY B — Bulk re-diarization of a full conversational block
# ---------------------------------------------------------------------------

class BulkDiarizationResult(BaseModel):
    assignments: list[dict] = Field(
        description="List of {segment_idx, speaker} for each segment in the block"
    )
    changes_made: int = Field(description="Number of segments where speaker was changed")
    notes: str = Field(description="Any observations about the diarization quality")


bulk_parser = JsonOutputParser(pydantic_object=BulkDiarizationResult)

bulk_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a diarization expert. Re-assign each segment to the most likely "
     "speaker based on conversational context, content, and turn-taking patterns.\n\n"
     "Known speakers: {speakers}\n"
     "Speaker roles: {roles}\n\n"
     "Guidelines:\n"
     "  • Maintain conversational coherence — questions should be answered by others\n"
     "  • Short responses ('yeah', 'okay', 'right') usually come from the non-primary speaker\n"
     "  • Technical explanations come from the technical speaker\n"
     "  • The same speaker rarely switches label mid-sentence\n\n"
     "Respond with JSON: {format_instructions}"),
    ("human",
     "Re-assign speakers for these segments:\n{block}"),
]).partial(
    format_instructions=bulk_parser.get_format_instructions(),
)

bulk_chain = bulk_prompt | llm | bulk_parser


def format_block_for_bulk(segments: list[dict], start: int, size: int = 8) -> str:
    end = min(start + size, len(segments))
    lines = []
    for i in range(start, end):
        seg = segments[i]
        lines.append(
            f"Segment {i}: [{seg.get('speaker', 'UNKNOWN')}] {seg['text'].strip()}"
        )
    return "\n".join(lines)


print("=== STRATEGY B: Bulk re-diarization ===")
block_text = format_block_for_bulk(segments, 0, size=10)
bulk_result = bulk_chain.invoke({
    "speakers": ", ".join(KNOWN_SPEAKERS),
    "roles": build_speaker_roles_text(name_map),
    "block": block_text,
})

assignments = bulk_result.get("assignments", []) if isinstance(bulk_result, dict) else bulk_result.assignments
changes = bulk_result.get("changes_made", 0) if isinstance(bulk_result, dict) else bulk_result.changes_made
notes = bulk_result.get("notes", "") if isinstance(bulk_result, dict) else bulk_result.notes

print(f"  Changes suggested : {changes}")
print(f"  Notes             : {notes[:120]}")
print()
for a in assignments[:10]:
    idx = a.get("segment_idx", -1) if isinstance(a, dict) else -1
    new_spk = a.get("speaker", "") if isinstance(a, dict) else ""
    if idx >= 0 and idx < len(segments):
        orig_spk = segments[idx].get("speaker", "")
        changed = "  ←CHANGED" if orig_spk != new_spk else ""
        print(f"  Seg {idx:02d}: {orig_spk} → {new_spk}{changed}")
print()


# ---------------------------------------------------------------------------
# STRATEGY C — Merge tiny backchannel segments
# ---------------------------------------------------------------------------

def merge_backchannels(
    segments: list[dict],
    max_words: int = 2,
    max_duration: float = 2.0,
) -> list[dict]:
    """
    Merge very short segments into an adjacent segment from a different speaker.
    These are typically backchannels ("yeah", "okay", "right") that WhisperX
    over-segmented or misattributed.
    """
    merged = copy.deepcopy(segments)
    to_remove = set()

    for i, seg in enumerate(merged):
        if i in to_remove:
            continue
        words = seg.get("words", [])
        duration = seg["end"] - seg["start"]

        if len(words) <= max_words and duration <= max_duration and i > 0:
            # This is a backchannel — check if it fits better with prev or next speaker
            prev_spk = merged[i-1].get("speaker")
            curr_spk = seg.get("speaker")

            # If same speaker as previous: merge into previous
            if curr_spk == prev_spk:
                merged[i-1]["end"] = seg["end"]
                merged[i-1]["text"] += " " + seg["text"].strip()
                merged[i-1]["words"].extend(seg.get("words", []))
                to_remove.add(i)

    return [s for i, s in enumerate(merged) if i not in to_remove]


original_count = len(segments)
merged_segments = merge_backchannels(segments, max_words=2, max_duration=2.0)
print(f"=== STRATEGY C: Backchannel merging ===")
print(f"  Segments before: {original_count}")
print(f"  Segments after : {len(merged_segments)}")
print(f"  Merged         : {original_count - len(merged_segments)}")
print()


# ---------------------------------------------------------------------------
# Save corrected transcript
# ---------------------------------------------------------------------------

corrected = copy.deepcopy(raw)
corrected["segments"] = merged_segments
out_path = TRANSCRIPT_PATH.parent / "transcript_diarization_corrected.json"
with open(out_path, "w") as f:
    json.dump(corrected, f, indent=2)
print(f"  Saved: {out_path.name}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Strategy A (context window) is surgical — use it for specific anomalies
# ✅ Strategy B (bulk) is broad — use it for blocks where you trust the LLM
# ✅ Strategy C (merge backchannels) is rule-based — fast and reliable
# ✅ Never auto-apply LLM reassignments below 0.7 confidence
# ✅ The LLM uses linguistic evidence only — it cannot hear the audio
# ✅ Always save intermediate results so you can compare and roll back
