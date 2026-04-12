"""
ASR Stage 3, File 4: LLM Speaker Resolution with Chain-of-Thought
===================================================================
CONCEPT: LLM reasons through ambiguous speaker attribution using
         conversation context and topic segmentation.

Stage 3 Files 1-3 detect anomalies (SAME_SPEAKER_ADJACENT, FAST_SWITCH, etc.)
and apply context-window corrections. But they rely on rule-based thresholds —
they don't *understand* the conversation.

Problems rules can't solve:
  • Two speakers finishing each other's sentences (natural handoff)
  • A speaker quoting someone else (attributed speech)
  • A backchannel in the middle of a long turn ("yeah, exactly" mid-sentence)
  • Topic shifts that coincide with speaker changes

This file uses:
  1. LLM topic segmentation — identify where conversation topics shift
     (topic shifts often correlate with speaker changes)
  2. LLM chain-of-thought speaker attribution — for each ambiguous segment,
     the LLM writes out its reasoning before deciding
  3. Confidence-gated correction — only apply changes the LLM is confident about

New patterns:
  - Chain-of-thought (CoT) prompting: ask LLM to reason before deciding
  - Two-stage pipeline: topic_segmentation → speaker_resolution
  - Confidence gating: only apply high-confidence LLM decisions
  - Evidence-based attribution: LLM must cite which turn supports its decision

Run this file:
  uv run deep_research/asr/stage_03_diarization/04_llm_speaker_resolution.py
"""

import os
import json
import copy
from pathlib import Path
import sys

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
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
    temperature=0,
    max_tokens=1024,
)


# ---------------------------------------------------------------------------
# 1. Pydantic schemas
# ---------------------------------------------------------------------------

class TopicSegment(BaseModel):
    start_index: int = Field(description="First segment index of this topic")
    end_index: int = Field(description="Last segment index of this topic")
    topic_label: str = Field(description="Short label: what is being discussed")
    primary_speaker: str = Field(description="SPEAKER_XX who drives this topic, or 'multiple'")


class TopicMap(BaseModel):
    topics: list[TopicSegment]
    summary: str = Field(description="One sentence: what is this conversation about overall")


class SpeakerDecision(BaseModel):
    segment_index: int
    current_speaker: str
    reasoning: str = Field(description="Step-by-step reasoning about who is speaking")
    correct_speaker: str = Field(description="SPEAKER_XX — can be same as current if already correct")
    evidence: str = Field(description="Which surrounding segment supports this decision")
    confidence: float = Field(description="0.0-1.0: certainty of this attribution")
    change_recommended: bool


topic_parser = JsonOutputParser(pydantic_object=TopicMap)
decision_parser = JsonOutputParser(pydantic_object=SpeakerDecision)


# ---------------------------------------------------------------------------
# 2. Topic segmentation chain
# ---------------------------------------------------------------------------

topic_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a conversation analyst. Read the transcript and identify distinct topic "
     "segments — consecutive groups of segments discussing the same subject.\n\n"
     "A topic segment ends when:\n"
     "  • The subject clearly shifts to a new agenda item\n"
     "  • A speaker explicitly introduces a new topic ('OK moving on...')\n"
     "  • There is a question-and-answer boundary on a different subject\n\n"
     "Keep topics broad — aim for 3-6 topics in a typical meeting, not one per segment.\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human", "Full transcript ({n_segments} segments):\n\n{transcript}"),
]).partial(format_instructions=topic_parser.get_format_instructions())

topic_chain = topic_prompt | llm | topic_parser


def build_transcript_text(segs: list[dict], max_chars: int = 3000) -> str:
    lines = []
    total = 0
    for i, seg in enumerate(segs):
        line = f"[{i}] [{seg.get('speaker', 'UNKNOWN')}] {seg['text'].strip()}"
        lines.append(line)
        total += len(line)
        if total > max_chars:
            lines.append(f"... (truncated at segment {i}, {len(segs) - i - 1} more)")
            break
    return "\n".join(lines)


print("=" * 60)
print("  LLM SPEAKER RESOLUTION")
print("=" * 60)
print("\n=== 1. Topic Segmentation ===\n")

transcript_text = build_transcript_text(segments)

try:
    topic_result = topic_chain.invoke({
        "transcript": transcript_text,
        "n_segments": len(segments),
    })
    topic_dict = topic_result if isinstance(topic_result, dict) else {}
    topics = topic_dict.get("topics", [])
    summary = topic_dict.get("summary", "")

    print(f"  Conversation: {summary}\n")
    print(f"  {'Topic':<30} {'Segs':>8} {'Primary Speaker'}")
    print("  " + "-" * 60)
    for t in topics:
        label = t.get("topic_label", "?")[:28]
        start = t.get("start_index", 0)
        end = t.get("end_index", 0)
        spk = t.get("primary_speaker", "?")
        print(f"  {label:<30} {start:>3}–{end:<3}  {spk}")

except Exception as e:
    print(f"  Topic segmentation failed: {e}")
    topics = []

print()


# ---------------------------------------------------------------------------
# 3. Identify ambiguous segments from topic boundaries
# ---------------------------------------------------------------------------
# Segments at topic transition points are most likely to have wrong speaker labels.
# We also flag SAME_SPEAKER_ADJACENT anomalies from Stage 3 File 1's logic.

def find_ambiguous_segments(segs: list[dict], topic_list: list[dict]) -> list[int]:
    """Return indices of segments likely to have wrong speaker attribution."""
    ambiguous = set()

    # Topic boundary segments
    for t in topic_list:
        boundary = t.get("start_index", 0)
        for offset in range(-1, 2):  # segment before, at, and after boundary
            idx = boundary + offset
            if 0 <= idx < len(segs):
                ambiguous.add(idx)

    # Same-speaker-adjacent (possible missed speaker change)
    for i in range(1, len(segs)):
        if segs[i].get("speaker") == segs[i - 1].get("speaker"):
            # If combined turn is very short, might be two separate speakers
            combined_words = (
                len(segs[i - 1]["text"].split()) + len(segs[i]["text"].split())
            )
            if combined_words <= 8:
                ambiguous.add(i)

    # Very short single-word segments (likely backchannels mis-attributed)
    for i, seg in enumerate(segs):
        if len(seg["text"].strip().split()) <= 2:
            ambiguous.add(i)

    return sorted(ambiguous)


ambiguous_indices = find_ambiguous_segments(segments, topics)
print(f"=== 2. Ambiguous Segments: {len(ambiguous_indices)} candidates ===\n")
for idx in ambiguous_indices[:8]:
    seg = segments[idx]
    print(f"  [seg {idx:>2}] [{seg.get('speaker')}] {seg['text'].strip()[:60]}")
print()


# ---------------------------------------------------------------------------
# 4. Chain-of-thought speaker resolution
# ---------------------------------------------------------------------------

resolution_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert conversation analyst. Determine the correct speaker for "
     "the AMBIGUOUS segment using chain-of-thought reasoning.\n\n"
     "Step through:\n"
     "  1. What does the current speaker label say?\n"
     "  2. What did the previous speaker say? Does this segment continue their thought?\n"
     "  3. What did the next speaker say? Does this segment lead into it?\n"
     "  4. Is this a backchannel ('yeah', 'right', 'mm-hmm') that belongs to the listener?\n"
     "  5. Given all evidence, who most likely said this?\n\n"
     "Write your reasoning in the 'reasoning' field, then give your final decision.\n"
     "Only recommend a change if you are >0.75 confident.\n\n"
     "Known speakers: {known_speakers}\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "Context (2 turns before):\n{prev_context}\n\n"
     "AMBIGUOUS SEGMENT [index={index}] [current={current_speaker}]: {text}\n\n"
     "Context (2 turns after):\n{next_context}"),
]).partial(format_instructions=decision_parser.get_format_instructions())

resolution_chain = resolution_prompt | llm | decision_parser


def get_window(index: int, before: int = 2, after: int = 2) -> tuple[str, str]:
    prev = segments[max(0, index - before):index]
    nxt = segments[index + 1:index + 1 + after]
    fmt = lambda segs: "\n".join(
        f"  [{s.get('speaker', 'UNKNOWN')}] {s['text'].strip()}" for s in segs
    ) or "  (none)"
    return fmt(prev), fmt(nxt)


# Get unique speakers
known_speakers = list({s.get("speaker", "UNKNOWN") for s in segments})
known_speakers_str = ", ".join(known_speakers)

print(f"=== 3. Chain-of-Thought Speaker Resolution ===\n")
print(f"  Resolving {min(len(ambiguous_indices), 8)} segments "
      f"(capped at 8 for demo)...\n")

decisions: list[dict] = []
changes_applied = 0

for idx in ambiguous_indices[:8]:
    seg = segments[idx]
    prev_ctx, next_ctx = get_window(idx)

    try:
        result = resolution_chain.invoke({
            "index": idx,
            "current_speaker": seg.get("speaker", "UNKNOWN"),
            "text": seg["text"].strip(),
            "prev_context": prev_ctx,
            "next_context": next_ctx,
            "known_speakers": known_speakers_str,
        })
        d = result if isinstance(result, dict) else {}
        decisions.append(d)

        change = d.get("change_recommended", False)
        conf = d.get("confidence", 0.0)
        old_spk = d.get("current_speaker", "?")
        new_spk = d.get("correct_speaker", old_spk)
        reasoning_snippet = d.get("reasoning", "")[:80]

        print(f"  [seg {idx:>2}] conf={conf:.2f} change={str(change):<5} "
              f"{old_spk} → {new_spk if change else old_spk}")
        print(f"           reasoning: {reasoning_snippet}")
        print(f"           evidence : {d.get('evidence', '')[:70]}")
        print()

    except Exception as e:
        print(f"  [seg {idx}] resolution error: {e}\n")


# ---------------------------------------------------------------------------
# 5. Apply high-confidence decisions
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.75

improved = copy.deepcopy(segments)
for d in decisions:
    if (d.get("change_recommended", False)
            and d.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD):
        idx = d.get("segment_index")
        new_spk = d.get("correct_speaker")
        if idx is not None and new_spk and 0 <= idx < len(improved):
            improved[idx]["speaker"] = new_spk
            changes_applied += 1

print(f"=== 4. Results ===\n")
print(f"  Segments analysed  : {len(decisions)}")
print(f"  Changes recommended: {sum(1 for d in decisions if d.get('change_recommended'))}")
print(f"  Changes applied    : {changes_applied} "
      f"(confidence ≥ {CONFIDENCE_THRESHOLD})")
print()


# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------

out = copy.deepcopy(raw)
out["segments"] = improved
out["speaker_resolution"] = {
    "ambiguous_candidates": len(ambiguous_indices),
    "decisions": decisions,
    "changes_applied": changes_applied,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "topic_map": topics,
}
out_path = OUT_DIR / "transcript_speaker_resolved.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved: {out_path.name}\n")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Topic segmentation first — topic boundaries are speaker change candidates
# ✅ Chain-of-thought forces the LLM to reason step-by-step, not just guess
# ✅ evidence field makes the LLM cite which turn supports its decision
# ✅ Confidence gating (≥0.75) prevents low-confidence guesses from overwriting correct labels
# ✅ Two-pass: find candidates with heuristics, then apply LLM only where needed
# ✅ Topic map is a valuable output on its own — used by downstream stages for context
# ✅ CoT prompting is the single biggest quality upgrade for complex reasoning tasks
