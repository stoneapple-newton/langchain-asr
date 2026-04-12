"""
Stage 4, File 3: Persona-Aware Summaries
=========================================
CONCEPT: Different summary formats for different stakeholders.

An executive wants high-level outcomes. A PM wants detailed action items.
An engineer wants technical decisions. One transcript, multiple summaries.

Key concepts:
  - Persona-based prompt engineering
  - Content filtering by relevance
  - Audience-specific formatting

Run this file:
  uv run deep_research/summary_agents/stage_04_rag_enhanced/03_persona_aware_summary.py
"""

from pathlib import Path
import sys
from typing import Annotated
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Persona Definitions
# ---------------------------------------------------------------------------

PERSONAS = {
    "executive": {
        "name": "Executive",
        "description": "C-level leader focused on business outcomes",
        "interests": ["decisions", "risks", "timeline", "resources", "outcomes"],
        "format": "Brief bullet points, business language",
        "length": "1 paragraph or 3-5 bullets",
    },
    "product_manager": {
        "name": "Product Manager",
        "description": "PM tracking deliverables and dependencies",
        "interests": ["action items", "owners", "deadlines", "blockers", "requirements"],
        "format": "Structured with clear action items and owners",
        "length": "1 page",
    },
    "engineer": {
        "name": "Engineer",
        "description": "Developer implementing the technical work",
        "interests": ["technical decisions", "architecture", "implementation details", "API changes"],
        "format": "Technical, specific, with code references",
        "length": "Detailed, include technical context",
    },
    "designer": {
        "name": "Designer",
        "description": "UX/UI designer focused on user experience",
        "interests": ["user flow", "accessibility", "visual design", "usability", "WCAG"],
        "format": "Visual descriptions, user-centric language",
        "length": "Medium, focus on design implications",
    },
}


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class PersonaState(TypedDict):
    """State for persona-aware summarization."""
    
    transcript: str
    persona: str
    
    # Output
    summary: str
    
    # Audit
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

llm = create_chat_model(
    profile="asr_v2",
    temperature=0,
    max_tokens=1024,
)


# ---------------------------------------------------------------------------
# Persona-Based Summary Node
# ---------------------------------------------------------------------------

def generate_persona_summary(state: PersonaState) -> dict:
    """Generate summary tailored to specific persona."""
    persona_key = state["persona"]
    persona = PERSONAS.get(persona_key, PERSONAS["executive"])
    
    print(f"  [generate_persona_summary] Creating {persona['name']} summary...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are creating a meeting summary for a {persona_name}.\n\n"
            "Audience: {description}\n"
            "They care about: {interests}\n"
            "Format: {format}\n"
            "Length: {length}\n\n"
            "Filter and emphasize content relevant to this audience. "
            "Skip technical details if not relevant. "
            "Skip business context if not relevant."
        ),
        (
            "human",
            "Create a {persona_name}-focused summary of this meeting:\n\n{transcript}"
        ),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    summary = chain.invoke({
        "persona_name": persona["name"],
        "description": persona["description"],
        "interests": ", ".join(persona["interests"]),
        "format": persona["format"],
        "length": persona["length"],
        "transcript": state["transcript"][:2000],
    })
    
    return {
        "summary": summary,
        "messages": [HumanMessage(content=f"Generated {persona['name']} summary")],
    }


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(PersonaState)
workflow.add_node("generate", generate_persona_summary)
workflow.add_edge(START, "generate")
workflow.add_edge("generate", END)
app = workflow.compile()


# ---------------------------------------------------------------------------
# Run Comparison
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 4.3: PERSONA-AWARE SUMMARIES")
print("=" * 70)
print(f"\nTranscript: {len(transcript_text)} chars")
print(f"Available personas: {', '.join(PERSONAS.keys())}\n")

# Generate summary for each persona
for persona_key in PERSONAS.keys():
    print("-" * 70)
    print(f"  PERSONA: {PERSONAS[persona_key]['name'].upper()}")
    print("-" * 70)
    
    result = app.invoke({
        "transcript": transcript_text,
        "persona": persona_key,
        "summary": "",
        "messages": [],
    })
    
    print(result["summary"][:600])
    if len(result["summary"]) > 600:
        print("...")
    print()


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

print("=" * 70)
print("  COMPARISON ANALYSIS")
print("=" * 70)
print("""
Same meeting, different focus:

┌─────────────────────┬─────────────────────────────────────────────────┐
│ Persona             │ Focus Areas                                     │
├─────────────────────┼─────────────────────────────────────────────────┤
│ Executive           │ Business outcomes, timeline, resource decisions │
│ Product Manager     │ Action items, owners, blockers, requirements    │
│ Engineer            │ Technical decisions, APIs, architecture         │
│ Designer            │ UX implications, accessibility, user flows      │
└─────────────────────┴─────────────────────────────────────────────────┘

Benefits:
   - Each stakeholder gets relevant information
   - Reduces noise for non-technical audiences
   - Technical audiences get implementation details
   - Saves time for busy executives

Implementation Approaches:
   1. Single-pass with persona in prompt (this demo)
   2. Extract structured data → filter → format per persona
   3. Generate full summary → extract relevant sections

Next: Stage 5 → Memory and persistence
""")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ Different audiences need different information
✅ Persona-aware prompting filters and emphasizes appropriately
✅ Format and length should match audience expectations
✅ One transcript → multiple tailored summaries

Prompt Engineering for Personas:
   - Define audience clearly
   - Specify interests/concerns
   - Set format expectations
   - Provide length constraints

Production Considerations:
   - Cache structured extraction, regenerate per persona
   - Track which personas viewed which summaries
   - Allow custom persona definitions
   - Learn from feedback per persona
""")
