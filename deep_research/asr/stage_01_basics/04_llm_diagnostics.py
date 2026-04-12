"""
ASR Stage 1, File 4: LLM-Powered Transcript Diagnostics
=========================================================
CONCEPT: Using the LLM as an intelligent first-pass quality assessor.

Rule-based metrics (Files 2 & 3) tell you *that* something is wrong:
  filler_rate=0.08, avg_confidence=0.71, punctuation_score=0.12

But they cannot tell you *what* is wrong or *why*:
  • "the the jwt" → probably "the JWT" — regex sees repeated word, LLM sees tech term
  • SPEAKER_01 says something SPEAKER_00 clearly said before → speaker confusion
  • Segment ends mid-clause → interrupted thought, needs different fix than missing period
  • "itll need to go through devops" → domain term + contraction, not just a "filler"

This file adds an LLM diagnostic layer that reads segments with understanding
and produces a structured, actionable remediation plan — telling downstream
stages exactly what needs fixing and in what order.

New patterns introduced:
  - JsonOutputParser + Pydantic for structured LLM output
  - Chunked transcript processing (handles any transcript length)
  - Two-step LLM pipeline: diagnose chunks → synthesise remediation plan
  - Severity ranking to let downstream stages prioritise work
  - Saving diagnostic_report.json so other stages can skip re-analysis

Run this file:
  uv run deep_research/asr/stage_01_basics/04_llm_diagnostics.py
"""

import os
import json
from pathlib import Path
import sys
from collections import Counter

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
    max_tokens=1024,
)


# ---------------------------------------------------------------------------
# 1. Pydantic schemas for structured LLM output
# ---------------------------------------------------------------------------

class SegmentIssue(BaseModel):
    index: int = Field(description="Segment index in the transcript")
    issue_type: str = Field(
        description=(
            "One of: transcription_error, speaker_confusion, incomplete_sentence, "
            "technical_term_error, incoherent, false_start, missing_context"
        )
    )
    description: str = Field(description="Specific description of the issue (max 80 chars)")
    severity: str = Field(description="low, medium, or high")
    suggested_fix: str = Field(description="Brief concrete fix suggestion")


class ChunkDiagnostic(BaseModel):
    issues: list[SegmentIssue] = Field(
        description="All issues found in this chunk. Empty list if no issues."
    )
    chunk_quality: str = Field(description="Overall quality of this chunk: poor, fair, or good")


class RemediationPlan(BaseModel):
    overall_quality: str = Field(description="poor, fair, good, or excellent")
    primary_issue_categories: list[str] = Field(
        description="Top 3-5 issue types found, ordered by frequency/impact"
    )
    remediation_steps: list[str] = Field(
        description="Ordered list of specific actions, most impactful first"
    )
    segments_needing_manual_review: list[int] = Field(
        description="Segment indices that are too complex for automated correction"
    )
    estimated_effort: str = Field(description="low, medium, or high")


chunk_parser = JsonOutputParser(pydantic_object=ChunkDiagnostic)
plan_parser = JsonOutputParser(pydantic_object=RemediationPlan)


# ---------------------------------------------------------------------------
# 2. Chunk diagnostic chain
# ---------------------------------------------------------------------------

chunk_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert transcript quality analyst. Analyze each segment and identify "
     "issues that would reduce accuracy or readability in a professional transcript.\n\n"
     "Issue types to detect:\n"
     "  transcription_error    — word clearly wrong (sounds like / OCR error)\n"
     "  speaker_confusion      — speaker label contradicts conversation flow\n"
     "  incomplete_sentence    — sentence cut off mid-clause, missing words\n"
     "  technical_term_error   — technical term wrong case/spelling (jwt→JWT, devops→DevOps)\n"
     "  incoherent             — segment grammatically impossible or makes no sense\n"
     "  false_start            — speaker restarts, creating redundancy\n"
     "  missing_context        — reference to something never introduced ('the project' — which?)\n\n"
     "Only flag genuine issues. Minor fillers (uh, um) are not issues unless excessive.\n"
     "Flag at most 2-3 issues per segment — focus on the most impactful.\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "Segments (starting at global index {offset}):\n{block}"),
]).partial(format_instructions=chunk_parser.get_format_instructions())

chunk_chain = chunk_prompt | llm | chunk_parser


# ---------------------------------------------------------------------------
# 3. Remediation plan chain
# ---------------------------------------------------------------------------

plan_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a transcript post-processing strategist. Given diagnosed issues from "
     "a transcript, create a practical, prioritised remediation plan.\n\n"
     "Be specific: name the correction type (e.g. 'capitalise JWT, DevOps, Node.js throughout') "
     "rather than vague goals ('fix technical terms').\n\n"
     "Respond with JSON only: {format_instructions}"),
    ("human",
     "Transcript stats:\n"
     "  Total segments : {total_segments}\n"
     "  High severity  : {high_count} issues\n"
     "  Medium severity: {med_count} issues\n"
     "  Low severity   : {low_count} issues\n\n"
     "Sample issues found:\n{issue_summary}"),
]).partial(format_instructions=plan_parser.get_format_instructions())

plan_chain = plan_prompt | llm | plan_parser


# ---------------------------------------------------------------------------
# 4. Run chunk diagnostics
# ---------------------------------------------------------------------------

print("=" * 60)
print("  LLM TRANSCRIPT DIAGNOSTICS")
print("=" * 60)
print(f"\n=== 1. Chunk-level issue detection ({len(segments)} segments, chunks of 6) ===\n")

CHUNK_SIZE = 6
all_issues: list[dict] = []

for chunk_start in range(0, len(segments), CHUNK_SIZE):
    chunk = segments[chunk_start:chunk_start + CHUNK_SIZE]
    block_lines = []
    for i, seg in enumerate(chunk):
        spk = seg.get("speaker", "UNKNOWN")
        text = seg["text"].strip()
        # Include word-level confidence hint for low-confidence words
        low_conf_words = [
            w["word"] for w in seg.get("words", []) if w.get("score", 1.0) < 0.72
        ]
        conf_hint = f" [low-conf: {', '.join(low_conf_words[:3])}]" if low_conf_words else ""
        block_lines.append(f"[{chunk_start + i}] [{spk}] {text}{conf_hint}")
    block = "\n".join(block_lines)

    try:
        result = chunk_chain.invoke({"block": block, "offset": chunk_start})
        issues = result.get("issues", []) if isinstance(result, dict) else []
        quality = result.get("chunk_quality", "?") if isinstance(result, dict) else "?"
        all_issues.extend(issues)
        print(f"  Chunk {chunk_start:>2}–{chunk_start + len(chunk) - 1:<2} | "
              f"quality={quality:<5} | issues={len(issues)}")
    except Exception as e:
        print(f"  Chunk {chunk_start:>2} | parse error: {e}")

print(f"\n  Total issues detected: {len(all_issues)}")


# ---------------------------------------------------------------------------
# 5. Severity and type breakdown
# ---------------------------------------------------------------------------

print("\n=== 2. Issue Breakdown ===\n")

type_counts: Counter = Counter(i.get("issue_type", "unknown") for i in all_issues)
severity_counts: Counter = Counter(i.get("severity", "unknown") for i in all_issues)

print("  By issue type:")
for itype, count in type_counts.most_common():
    bar = "█" * count
    print(f"    {itype:<28} {count:>2}  {bar}")

print("\n  By severity:")
for sev in ["high", "medium", "low"]:
    c = severity_counts.get(sev, 0)
    print(f"    {sev:<10} {c:>2}")

high_issues = [i for i in all_issues if i.get("severity") == "high"]
if high_issues:
    print(f"\n  High-severity issues (up to 5):")
    for issue in high_issues[:5]:
        idx = issue.get("index", "?")
        itype = issue.get("issue_type", "?")
        desc = issue.get("description", "")[:65]
        fix = issue.get("suggested_fix", "")[:50]
        print(f"    [seg {idx}] {itype}")
        print(f"             desc : {desc}")
        print(f"             fix  : {fix}")


# ---------------------------------------------------------------------------
# 6. Generate remediation plan
# ---------------------------------------------------------------------------

print("\n=== 3. LLM Remediation Plan ===\n")

# Build a concise issue summary for the plan prompt
issue_lines = []
for issue in sorted(all_issues, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("severity", "low"), 3)):
    sev = issue.get("severity", "?")
    idx = issue.get("index", "?")
    itype = issue.get("issue_type", "?")
    desc = issue.get("description", "")[:70]
    issue_lines.append(f"  [seg {idx}][{sev}] {itype}: {desc}")

issue_summary = "\n".join(issue_lines[:25])  # cap to avoid prompt overflow

try:
    plan = plan_chain.invoke({
        "issue_summary": issue_summary,
        "total_segments": len(segments),
        "high_count": severity_counts.get("high", 0),
        "med_count": severity_counts.get("medium", 0),
        "low_count": severity_counts.get("low", 0),
    })
    plan_dict = plan if isinstance(plan, dict) else {}

    print(f"  Overall quality   : {plan_dict.get('overall_quality', 'N/A')}")
    print(f"  Correction effort : {plan_dict.get('estimated_effort', 'N/A')}")

    print("\n  Primary issue categories:")
    for cat in plan_dict.get("primary_issue_categories", []):
        print(f"    • {cat}")

    print("\n  Remediation steps (priority order):")
    for i, step in enumerate(plan_dict.get("remediation_steps", []), 1):
        print(f"    {i}. {step}")

    manual = plan_dict.get("segments_needing_manual_review", [])
    if manual:
        print(f"\n  Manual review needed: segments {manual}")

except Exception as e:
    print(f"  Plan generation failed: {e}")
    plan_dict = {}


# ---------------------------------------------------------------------------
# 7. Save diagnostic report
# ---------------------------------------------------------------------------

report = {
    "source": str(TRANSCRIPT_PATH.name),
    "total_segments": len(segments),
    "issues": all_issues,
    "severity_counts": dict(severity_counts),
    "type_counts": dict(type_counts),
    "remediation_plan": plan_dict,
}
out_path = TRANSCRIPT_PATH.parent / "diagnostic_report.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"\n  Saved: {out_path.name}")
print("  → Downstream stages can load this to skip re-analysis\n")


# ---------------------------------------------------------------------------
# KEY TAKEAWAYS
# ---------------------------------------------------------------------------
# ✅ LLM reads transcript with understanding — not just counting patterns
# ✅ Structured Pydantic output makes LLM findings programmatically usable
# ✅ Chunked processing handles transcripts of any length
# ✅ Low-confidence word hints give LLM the same signal a human reviewer would
# ✅ Two-step pipeline: chunk diagnosis → synthesis — keeps prompts focused
# ✅ Severity ranking lets downstream stages prioritise high-impact fixes first
# ✅ diagnostic_report.json feeds Stage 2+ as a pre-computed issue map
