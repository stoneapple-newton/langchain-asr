"""
Stage 1, File 1: Basic One-Shot Summary
========================================
CONCEPT: Simple single-prompt summarization using LCEL chains.

This is the simplest approach to summarization: take the transcript,
put it in a prompt template, and ask the LLM for a summary.

Key concepts:
  - ChatPromptTemplate: reusable prompt with variables
  - LCEL chain composition with | operator
  - StrOutputParser: extract just the text content

Run this file:
  uv run deep_research/summary_agents/stage_01_one_shot/01_basic_summary.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

llm = create_chat_model(
    profile="asr_v2",  # Use the ASR-tuned model profile
    temperature=0,     # Deterministic output
    max_tokens=1024,
)

# Load the sample transcript
transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 1.1: BASIC ONE-SHOT SUMMARY")
print("=" * 70)
print(f"\nTranscript length: {len(transcript_text)} characters")
print(f"Speakers: {transcript.speakers}")
print(f"Duration: {transcript.duration:.1f} seconds")


# ---------------------------------------------------------------------------
# Basic Summary Chain
# ---------------------------------------------------------------------------

basic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a meeting summarization assistant. Create a concise summary "
        "of the provided meeting transcript. Focus on key points, decisions, "
        "and any action items mentioned."
    ),
    (
        "human",
        "Please summarize the following meeting transcript:\n\n{transcript}"
    ),
])

basic_chain = basic_prompt | llm | StrOutputParser()

print("\n" + "-" * 70)
print("  APPROACH 1: Basic Summary")
print("-" * 70)

basic_summary = basic_chain.invoke({"transcript": transcript_text})
print(f"\n{basic_summary}")


# ---------------------------------------------------------------------------
# Enhanced Prompt with Instructions
# ---------------------------------------------------------------------------

enhanced_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a professional meeting summarizer. Create a structured summary "
        "with the following sections:\n"
        "1. Overview (1-2 sentences)\n"
        "2. Key Discussion Points (bullet points)\n"
        "3. Decisions Made (if any)\n"
        "4. Action Items (if any)\n\n"
        "Be concise but comprehensive."
    ),
    (
        "human",
        "Meeting Transcript:\n{transcript}\n\n"
        "Provide a structured summary:"
    ),
])

enhanced_chain = enhanced_prompt | llm | StrOutputParser()

print("\n" + "-" * 70)
print("  APPROACH 2: Enhanced Structured Summary")
print("-" * 70)

enhanced_summary = enhanced_chain.invoke({"transcript": transcript_text})
print(f"\n{enhanced_summary}")


# ---------------------------------------------------------------------------
# Role-Based Summary
# ---------------------------------------------------------------------------

# Different stakeholders want different information from the same meeting
role_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are summarizing this meeting for a {role}. "
        "Tailor the summary to what would be most relevant for this audience."
    ),
    (
        "human",
        "Meeting Transcript:\n{transcript}\n\n"
        "Provide a summary for a {role}:"
    ),
])

role_chain = role_prompt | llm | StrOutputParser()

print("\n" + "-" * 70)
print("  APPROACH 3: Role-Based Summaries (Same Meeting)")
print("-" * 70)

for role in ["Executive", "Project Manager", "Engineer"]:
    print(f"\n>>> Summary for {role}:")
    role_summary = role_chain.invoke({"transcript": transcript_text, "role": role})
    print(role_summary[:300] + "..." if len(role_summary) > 300 else role_summary)


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ One-shot prompting is simple and fast — single LLM call
✅ Prompt engineering (structure, role, context) significantly impacts output
✅ LCEL chains compose easily: prompt | llm | parser
✅ Same transcript can yield different summaries for different audiences

⚠️ Limitations:
   - Output format can vary (not guaranteed structured)
   - May miss nuance in long transcripts
   - No way to iteratively improve the summary
   - Context window limits on very long meetings

Next: Stage 1.2 → Structured output with Pydantic for reliable parsing
""")
