"""
ASR Stage 2, File 2: Speaker Naming & Role Identification
===========================================================
CONCEPT: Replacing generic SPEAKER_00 labels with real names or roles.

WhisperX assigns anonymous labels: SPEAKER_00, SPEAKER_01, SPEAKER_02.
These are meaningless in a transcript. This file uses an LLM to:

  1. Analyse the speaking style and content of each speaker
  2. Infer the role of each speaker (meeting host, engineer, designer, etc.)
  3. Suggest a name if hints are available (someone says "thanks Alice")
  4. Apply a mapping throughout the transcript

Two approaches:
  A. Role-based naming — infer roles from conversation content
  B. Name extraction   — scan for first-person introductions and name mentions

Run this file:
  uv run deep_research/asr/stage_02_llm_enhancement/02_speaker_naming.py
"""

import os
import json
import re
from pathlib import Path
import sys

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import create_chat_model
from pydantic import BaseModel, Field


TRANSCRIPT_PATH = Path(__file__).parent.parent / "sample_data" / "sample_transcript.json"
with open(TRANSCRIPT_PATH) as f:
    raw = json.load(f)
segments = raw["segments"]

llm = create_chat_model(
    temperature=0,
    max_tokens=512,
)


# ---------------------------------------------------------------------------
# 1. Build per-speaker text samples
# ---------------------------------------------------------------------------
# Give the LLM enough text per speaker to infer role — not the whole transcript.

def build_speaker_samples(
    segments: list[dict],
    max_chars_per_speaker: int = 800,
) -> dict[str, str]:
    """Collect the first N characters of text per speaker."""
    samples: dict[str, str] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        current = samples.get(spk, "")
        if len(current) < max_chars_per_speaker:
            samples[spk] = (current + " " + seg["text"].strip()).strip()
    return {k: v[:max_chars_per_speaker] for k, v in samples.items()}


speaker_samples = build_speaker_samples(segments)
print("=== 1. Speaker samples ===")
for spk, text in speaker_samples.items():
    print(f"\n  {spk} ({len(text)} chars):")
    print(f"    '{text[:120]}...'")
print()


# ---------------------------------------------------------------------------
# 2. Role inference chain
# ---------------------------------------------------------------------------

class SpeakerProfile(BaseModel):
    speaker_id: str = Field(description="Original SPEAKER_XX label")
    inferred_role: str = Field(description="Inferred role, e.g. 'Meeting Facilitator', 'Engineer', 'Designer'")
    suggested_name: str = Field(description="Suggested display name — role-based if no real name found, e.g. 'Facilitator'")
    confidence: float = Field(description="Confidence in the role inference, 0.0–1.0")
    evidence: str = Field(description="Brief evidence for the inferred role")


class SpeakerMapping(BaseModel):
    speakers: list[SpeakerProfile]
    meeting_type: str = Field(description="Type of meeting inferred from content")
    summary: str = Field(description="One-sentence summary of the meeting")


mapping_parser = JsonOutputParser(pydantic_object=SpeakerMapping)

role_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at analysing meeting transcripts to identify speaker roles.\n"
     "Given text samples from each speaker, infer:\n"
     "  • Their role in the meeting (who facilitates, who is the technical expert, etc.)\n"
     "  • A display name to use in place of SPEAKER_XX\n"
     "  • Confidence in your inference\n\n"
     "Look for signals like: who opens/closes the meeting, who discusses technical details,\n"
     "who discusses design/UX, who sets deadlines, who asks clarifying questions.\n\n"
     "Respond with JSON: {format_instructions}"),
    ("human",
     "Meeting context: {context}\n\n"
     "Speaker samples:\n{samples}"),
]).partial(format_instructions=mapping_parser.get_format_instructions())

role_chain = role_prompt | llm | mapping_parser


def format_samples_for_prompt(samples: dict[str, str]) -> str:
    return "\n\n".join(
        f"[{spk}]:\n{text}"
        for spk, text in samples.items()
    )


print("=== 2. Role inference ===")
meta = raw.get("meeting_metadata", {})
result = role_chain.invoke({
    "context": f"Title: {meta.get('title', 'Unknown')}, Date: {meta.get('date', 'Unknown')}",
    "samples": format_samples_for_prompt(speaker_samples),
})

print(f"  Meeting type : {result.get('meeting_type', 'Unknown')}")
print(f"  Summary      : {result.get('summary', '')}")
print()
print("  Speaker profiles:")
name_map: dict[str, str] = {}
for profile in result.get("speakers", []):
    spk_id = profile.get("speaker_id", "")
    name = profile.get("suggested_name", spk_id)
    role = profile.get("inferred_role", "Unknown")
    conf = profile.get("confidence", 0.0)
    evidence = profile.get("evidence", "")
    name_map[spk_id] = name
    print(f"\n    {spk_id} → {name}")
    print(f"      Role       : {role}")
    print(f"      Confidence : {conf:.0%}")
    print(f"      Evidence   : {evidence[:100]}")
print()


# ---------------------------------------------------------------------------
# 3. Name extraction from dialogue
# ---------------------------------------------------------------------------
# Sometimes speakers introduce themselves or address each other by name.
# Scan the transcript for these patterns before using role inference.

NAME_PATTERNS = [
    re.compile(r"\bmy name is (\w+)\b", re.IGNORECASE),
    re.compile(r"\bI'?m (\w+)\b", re.IGNORECASE),
    re.compile(r"\bthanks? (\w+)\b", re.IGNORECASE),
    re.compile(r"\bhey (\w+)\b", re.IGNORECASE),
    re.compile(r"\bhi (\w+)\b", re.IGNORECASE),
]

COMMON_WORDS = {"you", "all", "everyone", "there", "again", "so", "um", "uh", "guys"}


def extract_name_mentions(segments: list[dict]) -> dict[str, list[str]]:
    """
    Find potential name mentions in the transcript.
    Returns {speaker_id: [candidate_names]} — names mentioned in that speaker's turns.
    """
    mentions: dict[str, list[str]] = {}
    for seg in segments:
        spk = seg.get("speaker")
        if not spk:
            continue
        text = seg["text"]
        for pattern in NAME_PATTERNS:
            for match in pattern.finditer(text):
                candidate = match.group(1).strip().capitalize()
                if candidate.lower() not in COMMON_WORDS and len(candidate) > 2:
                    mentions.setdefault(spk, []).append(candidate)
    return mentions


name_mentions = extract_name_mentions(segments)
print("=== 3. Name mentions extracted from dialogue ===")
if name_mentions:
    for spk, names in name_mentions.items():
        print(f"  {spk}: {names}")
else:
    print("  No direct name mentions found in this transcript")
print()


# ---------------------------------------------------------------------------
# 4. Merge inference + extraction → final name map
# ---------------------------------------------------------------------------

def build_final_name_map(
    role_map: dict[str, str],
    name_mentions: dict[str, list[str]],
    known_speakers: dict[str, str] | None = None,
) -> dict[str, str]:
    """
    Priority order:
      1. known_speakers from metadata (ground truth)
      2. extracted name mentions (direct evidence)
      3. role-inferred names (heuristic)
    """
    final = dict(role_map)   # start with role inferences

    # Override with extracted mentions (first unique name per speaker)
    for spk, names in name_mentions.items():
        # Use the first non-generic name found
        for name in names:
            if name.lower() not in COMMON_WORDS:
                final[spk] = name
                break

    # Override with known ground truth
    if known_speakers:
        final.update(known_speakers)

    return final


known = meta.get("known_speakers", {})
final_map = build_final_name_map(name_map, name_mentions, known)

print("=== 4. Final speaker name mapping ===")
for spk_id, display_name in sorted(final_map.items()):
    source = "ground truth" if spk_id in known else (
        "extracted" if spk_id in name_mentions else "inferred"
    )
    print(f"  {spk_id} → {display_name:<20} ({source})")
print()


# ---------------------------------------------------------------------------
# 5. Apply names and render
# ---------------------------------------------------------------------------

def apply_names(segments: list[dict], name_map: dict[str, str]) -> str:
    """Render the transcript with real names applied."""
    lines = []
    prev_spk = None
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        label = name_map.get(spk, spk)
        if spk != prev_spk:
            lines.append(f"\n{label}:")
            prev_spk = spk
        lines.append(f"  {seg['text'].strip()}")
    return "\n".join(lines)


print("=== 5. Transcript with real names ===")
named_transcript = apply_names(segments[:12], final_map)
print(named_transcript)

# Save the name map for use in later stages
name_map_path = TRANSCRIPT_PATH.parent / "speaker_names.json"
with open(name_map_path, "w") as f:
    json.dump(final_map, f, indent=2)
print(f"\n  Name map saved: {name_map_path.name}")
print()


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ Build per-speaker text samples before calling the LLM — don't send everything
# ✅ Role signals: who opens/closes meeting, who discusses tech vs. design vs. deadlines
# ✅ Pattern matching for name mentions is fast and precise — do it before LLM inference
# ✅ Priority stack: ground truth > extracted > inferred — never override known facts
# ✅ Save the name map as JSON so every downstream stage can reuse it
# ✅ Meeting type + summary are a bonus: useful for generating meeting notes
