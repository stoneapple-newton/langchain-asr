"""
Stage 3, File 4: Conditional Routing by Transcript Type
========================================================
CONCEPT: Dynamic routing based on content classification.

Different meeting types need different summarization strategies:
- All-hands meetings → focus on announcements and Q&A
- 1-on-1s → focus on career development and feedback
- Standups → focus on blockers and progress
- Brainstorms → focus on ideas and decisions

Key concepts:
  - Classification node for routing decisions
  - Subgraphs for specialized processing
  - Dynamic workflow selection

Run this file:
  uv run deep_research/summary_agents/stage_03_langgraph/04_conditional_routing.py
"""

from pathlib import Path
import sys
from typing import Annotated
from typing_extensions import TypedDict, Literal

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
# State Schema
# ---------------------------------------------------------------------------

class RoutingState(TypedDict):
    """State with classification for routing."""
    
    transcript: str
    
    # Classification
    meeting_type: Literal["planning", "standup", "review", "general"]
    
    # Final output
    summary: str
    
    # Audit
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

llm = create_chat_model(
    profile="asr_v2",
    temperature=0,
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Classification Node
# ---------------------------------------------------------------------------

def classify_meeting(state: RoutingState) -> dict:
    """Classify the meeting type to determine routing."""
    print("  [classify_meeting] Analyzing content...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Classify this meeting into exactly one category: planning, standup, review, or general.\n"
            "- planning: roadmap, quarterly, strategy, goal-setting\n"
            "- standup: daily check-in, blockers, progress updates\n"
            "- review: retrospective, feedback, lessons learned\n"
            "- general: other meeting types\n\n"
            "Respond with just the category name."
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    classification = chain.invoke({"transcript": state["transcript"][:1000]}).strip().lower()
    
    # Normalize to valid types
    if "planning" in classification or "roadmap" in classification:
        meeting_type = "planning"
    elif "standup" in classification or "daily" in classification:
        meeting_type = "standup"
    elif "review" in classification or "retro" in classification:
        meeting_type = "review"
    else:
        meeting_type = "general"
    
    print(f"    Classified as: {meeting_type}")
    
    return {
        "meeting_type": meeting_type,
        "messages": [HumanMessage(content=f"Classified as {meeting_type}")],
    }


def route_by_type(state: RoutingState) -> str:
    """Return the node name to route to based on classification."""
    return state["meeting_type"]


# ---------------------------------------------------------------------------
# Specialized Summary Nodes
# ---------------------------------------------------------------------------

def summarize_planning(state: RoutingState) -> dict:
    """Specialized summary for planning meetings."""
    print("  [summarize_planning] Using planning template...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create a planning meeting summary focusing on:\n"
            "- Strategic goals and objectives\n"
            "- Timeline and milestones\n"
            "- Resource allocation\n"
            "- Risk factors\n"
            "- Decisions that impact roadmap"
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"transcript": state["transcript"][:2000]})
    
    return {
        "summary": f"# Planning Meeting Summary\n\n{summary}",
        "messages": [HumanMessage(content="Generated planning summary")],
    }


def summarize_standup(state: RoutingState) -> dict:
    """Specialized summary for standup meetings."""
    print("  [summarize_standup] Using standup template...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create a standup summary focusing on:\n"
            "- Blockers and impediments\n"
            "- Progress since last standup\n"
            "- Plans for today/next period\n"
            "- Who needs help"
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"transcript": state["transcript"][:2000]})
    
    return {
        "summary": f"# Standup Summary\n\n{summary}",
        "messages": [HumanMessage(content="Generated standup summary")],
    }


def summarize_review(state: RoutingState) -> dict:
    """Specialized summary for review/retrospective meetings."""
    print("  [summarize_review] Using review template...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create a review/retrospective summary focusing on:\n"
            "- What went well\n"
            "- What could be improved\n"
            "- Lessons learned\n"
            "- Action items for improvement"
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"transcript": state["transcript"][:2000]})
    
    return {
        "summary": f"# Review/Retrospective Summary\n\n{summary}",
        "messages": [HumanMessage(content="Generated review summary")],
    }


def summarize_general(state: RoutingState) -> dict:
    """Default summary for general meetings."""
    print("  [summarize_general] Using general template...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create a general meeting summary with:\n"
            "- Overview\n"
            "- Key discussion points\n"
            "- Decisions made\n"
            "- Action items"
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"transcript": state["transcript"][:2000]})
    
    return {
        "summary": f"# Meeting Summary\n\n{summary}",
        "messages": [HumanMessage(content="Generated general summary")],
    }


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(RoutingState)

# Add all nodes
workflow.add_node("classify", classify_meeting)
workflow.add_node("planning", summarize_planning)
workflow.add_node("standup", summarize_standup)
workflow.add_node("review", summarize_review)
workflow.add_node("general", summarize_general)

# Edges
workflow.add_edge(START, "classify")

# Conditional routing based on classification
workflow.add_conditional_edges(
    "classify",
    route_by_type,
    {
        "planning": "planning",
        "standup": "standup",
        "review": "review",
        "general": "general",
    },
)

# All specialized nodes end at END
for node in ["planning", "standup", "review", "general"]:
    workflow.add_edge(node, END)

app = workflow.compile()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 3.4: CONDITIONAL ROUTING BY TYPE")
print("=" * 70)
print(f"\nTranscript: {len(transcript_text)} chars\n")

result = app.invoke({
    "transcript": transcript_text,
    "meeting_type": "general",  # Will be updated by classification
    "summary": "",
    "messages": [],
})

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"Detected Type: {result['meeting_type']}")
print()
print(result["summary"])


# ---------------------------------------------------------------------------
# Demonstrate Different Types
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  TESTING DIFFERENT TRANSCRIPT TYPES")
print("-" * 70)

test_transcripts = {
    "standup": """
Alice: Yesterday I finished the API integration. Today I'll work on tests.
Bob: I'm blocked on the database issue, need help from Carol.
Carol: I'll pair with Bob after standup. Yesterday I reviewed PRs.
""",
    "review": """
Manager: Let's discuss the last sprint. What went well?
Alice: The deployment process was smooth.
Bob: Communication could be better between teams.
Manager: Let's document these lessons for next time.
""",
}

for test_type, test_text in test_transcripts.items():
    print(f"\n>>> Testing with {test_type} content:")
    test_result = app.invoke({
        "transcript": test_text,
        "meeting_type": "general",
        "summary": "",
        "messages": [],
    })
    print(f"    Detected: {test_result['meeting_type']}")
    print(f"    Summary type: {test_result['summary'].split(chr(10))[0]}")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ Classification enables dynamic routing
✅ Each meeting type gets specialized processing
✅ Easy to add new types without changing existing code
✅ Subgraphs can encapsulate complex type-specific logic

Graph Structure:
  START → classify ──┬──► planning ──► END
                     ├──► standup ───► END
                     ├──► review ────► END
                     └──► general ───► END

When to Use Conditional Routing:
   - Different content types need different processing
   - Want to optimize prompts/templates per use case
   - Need different output formats for different inputs
   - Quality improvement through specialization

Next: Stage 4 → RAG-enhanced summaries with historical context
""")
