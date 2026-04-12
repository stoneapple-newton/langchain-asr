"""
Stage 1, File 2: Structured Output Summary
============================================
CONCEPT: Using Pydantic models to enforce structured summary format.

Unstructured text summaries are hard to parse and use programmatically.
Structured output ensures we get consistent, typed data that we can
reliably extract fields from (action items, decisions, etc.).

Key concepts:
  - Pydantic BaseModel: define the schema
  - with_structured_output(): LLM returns structured data
  - Type safety and validation built-in

Run this file:
  uv run deep_research/summary_agents/stage_01_one_shot/02_structured_output.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Pydantic Models for Structured Summary
# ---------------------------------------------------------------------------

class ActionItem(BaseModel):
    """A task or action item from the meeting."""
    task: str = Field(description="Description of what needs to be done")
    assignee: str | None = Field(
        default=None,
        description="Who is responsible (null if not specified)"
    )
    deadline: str | None = Field(
        default=None,
        description="Deadline if mentioned (null if not specified)"
    )


class MeetingSummary(BaseModel):
    """Structured meeting summary output."""
    
    title: str = Field(description="A concise title for the meeting")
    
    overview: str = Field(
        description="1-2 sentence overview of what the meeting was about"
    )
    
    attendees: list[str] = Field(
        description="List of speakers/participants identified",
        default_factory=list
    )
    
    key_points: list[str] = Field(
        description="3-5 key discussion points from the meeting",
        min_length=1,
        max_length=10
    )
    
    decisions: list[str] = Field(
        description="Decisions made during the meeting (empty if none)",
        default_factory=list
    )
    
    action_items: list[ActionItem] = Field(
        description="Action items and tasks assigned (empty if none)",
        default_factory=list
    )
    
    sentiment: str = Field(
        description="Overall meeting sentiment: positive, neutral, or negative",
        pattern="^(positive|neutral|negative)$"
    )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

# Bind the Pydantic model to the LLM for structured output
llm = create_chat_model(
    profile="asr_v2",
    temperature=0,
    max_tokens=2048,
)
structured_llm = llm.with_structured_output(MeetingSummary)

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 1.2: STRUCTURED OUTPUT SUMMARY")
print("=" * 70)
print(f"\nTranscript: {len(transcript_text)} chars, {transcript.duration:.1f}s duration")


# ---------------------------------------------------------------------------
# Structured Summary Chain
# ---------------------------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise meeting summarizer. Analyze the transcript and "
        "extract all relevant information in the requested structured format. "
        "Be thorough but concise."
    ),
    (
        "human",
        "Analyze this meeting transcript and provide a structured summary:\n\n"
        "{transcript}"
    ),
])

chain = prompt | structured_llm

print("\n" + "-" * 70)
print("  GENERATING STRUCTURED SUMMARY")
print("-" * 70)

result: MeetingSummary = chain.invoke({"transcript": transcript_text})


# ---------------------------------------------------------------------------
# Display Results
# ---------------------------------------------------------------------------

print(f"\n📋 TITLE: {result.title}")
print(f"\n📝 OVERVIEW: {result.overview}")

print(f"\n👥 ATTENDEES ({len(result.attendees)}):")
for attendee in result.attendees:
    print(f"   • {attendee}")

print(f"\n💡 KEY POINTS ({len(result.key_points)}):")
for i, point in enumerate(result.key_points, 1):
    print(f"   {i}. {point}")

print(f"\n✅ DECISIONS ({len(result.decisions)}):")
if result.decisions:
    for decision in result.decisions:
        print(f"   • {decision}")
else:
    print("   (No explicit decisions recorded)")

print(f"\n🎯 ACTION ITEMS ({len(result.action_items)}):")
if result.action_items:
    for item in result.action_items:
        assignee = f" → {item.assignee}" if item.assignee else ""
        deadline = f" (by {item.deadline})" if item.deadline else ""
        print(f"   • {item.task}{assignee}{deadline}")
else:
    print("   (No action items recorded)")

print(f"\n😊 SENTIMENT: {result.sentiment}")


# ---------------------------------------------------------------------------
# Programmatic Access
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  PROGRAMMATIC ACCESS EXAMPLE")
print("-" * 70)

# Because we have structured data, we can use it programmatically
def has_action_for(result: MeetingSummary, keyword: str) -> bool:
    """Check if any action item contains a keyword."""
    return any(
        keyword.lower() in item.task.lower()
        for item in result.action_items
    )

print(f"\nHas JWT-related action? {has_action_for(result, 'JWT')}")
print(f"Has API-related action? {has_action_for(result, 'API')}")
print(f"Total action items to track: {len(result.action_items)}")

# Can easily serialize to JSON
import json
json_output = result.model_dump_json(indent=2)
print(f"\n📦 JSON Output Size: {len(json_output)} bytes")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ Pydantic models enforce consistent output structure
✅ Fields are typed and validated automatically
✅ Easy to extract specific information (action items, decisions)
✅ Output can be serialized to JSON for APIs/databases
✅ Reduces need for regex parsing of free text

⚠️ Limitations:
   - Still limited by context window for very long transcripts
   - Single pass: no chance to refine or improve
   - May hallucinate structure if transcript is unclear
   - Some models work better with structured output than others

Next: Stage 1.3 → Chunked summarization for long transcripts
""")
