"""
Stage 3, File 1: Summary StateGraph
====================================
CONCEPT: Basic LangGraph workflow for meeting summarization.

LangGraph provides explicit state management and graph-based workflows.
Unlike agent-based approaches, we define exactly which nodes run and when.

Key concepts:
  - StateGraph: define nodes and edges
  - TypedDict state: type-safe shared state
  - Explicit workflow vs autonomous agent

Run this file:
  uv run deep_research/summary_agents/stage_03_langgraph/01_summary_state_graph.py
"""

from pathlib import Path
import sys
from typing import Annotated
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Define State Schema
# ---------------------------------------------------------------------------

class SummaryState(TypedDict):
    """State that flows through the summarization graph."""
    
    # Input
    transcript: str
    
    # Intermediate results
    key_points: list[str]
    topics: list[str]
    action_items: list[str]
    
    # Final output
    summary: str
    
    # Audit trail
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Setup LLM
# ---------------------------------------------------------------------------

llm = create_chat_model(
    profile="asr_v2",
    temperature=0,
    max_tokens=1024,
)


# ---------------------------------------------------------------------------
# Define Nodes
# ---------------------------------------------------------------------------

def parse_transcript(state: SummaryState) -> dict:
    """Node 1: Parse and validate transcript input."""
    print("  [parse_transcript] Validating input...")
    
    transcript = state["transcript"]
    word_count = len(transcript.split())
    
    return {
        "messages": [HumanMessage(content=f"Processing transcript ({word_count} words)")]
    }


def extract_key_points(state: SummaryState) -> dict:
    """Node 2: Extract key discussion points from transcript."""
    print("  [extract_key_points] Analyzing content...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract the 3-5 most important discussion points from this meeting. "
            "Return as a bulleted list, one point per line."
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": state["transcript"][:2000]})
    
    # Parse into list
    points = [line.strip("- ").strip() for line in result.split("\n") if line.strip().startswith("-")]
    
    print(f"    Found {len(points)} key points")
    return {
        "key_points": points,
        "messages": [HumanMessage(content=f"Extracted {len(points)} key points")],
    }


def identify_topics(state: SummaryState) -> dict:
    """Node 3: Identify main topics/themes discussed."""
    print("  [identify_topics] Categorizing discussion...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Identify the main topics discussed in this meeting. "
            "Return a comma-separated list of 2-4 topics."
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": state["transcript"][:2000]})
    
    # Parse topics
    topics = [t.strip() for t in result.split(",") if t.strip()]
    
    print(f"    Identified topics: {', '.join(topics)}")
    return {
        "topics": topics,
        "messages": [HumanMessage(content=f"Identified topics: {', '.join(topics)}")],
    }


def find_action_items(state: SummaryState) -> dict:
    """Node 4: Find action items and assignments."""
    print("  [find_action_items] Looking for todos...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Find all action items, tasks, and todos mentioned in this meeting. "
            "Return as a bulleted list with assignees if mentioned. "
            "If no action items, say 'No action items identified.'"
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": state["transcript"][:2000]})
    
    # Parse action items
    items = [line.strip() for line in result.split("\n") if line.strip().startswith("-")]
    
    print(f"    Found {len(items)} action items")
    return {
        "action_items": items,
        "messages": [HumanMessage(content=f"Found {len(items)} action items")],
    }


def generate_summary(state: SummaryState) -> dict:
    """Node 5: Combine all extracted info into final summary."""
    print("  [generate_summary] Assembling final document...")
    
    # Build context from previous nodes
    key_points_text = "\n".join(f"- {p}" for p in state.get("key_points", []))
    topics_text = ", ".join(state.get("topics", []))
    action_items_text = "\n".join(f"- {i}" for i in state.get("action_items", []))
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a professional meeting summarizer. Create a polished summary "
            "using the provided extracted information."
        ),
        (
            "human",
            "Create a meeting summary using this information:\n\n"
            "Topics: {topics}\n\n"
            "Key Points:\n{key_points}\n\n"
            "Action Items:\n{action_items}\n\n"
            "Format with: Overview, Topics Discussed, Key Points, Action Items"
        ),
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({
        "topics": topics_text,
        "key_points": key_points_text,
        "action_items": action_items_text or "No action items identified.",
    })
    
    print("    Summary generated")
    return {
        "summary": summary,
        "messages": [HumanMessage(content="Summary generation complete")],
    }


# ---------------------------------------------------------------------------
# Build the Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(SummaryState)

# Add nodes
workflow.add_node("parse", parse_transcript)
workflow.add_node("extract_points", extract_key_points)
workflow.add_node("identify_topics", identify_topics)
workflow.add_node("find_actions", find_action_items)
workflow.add_node("generate", generate_summary)

# Add edges (workflow: parse → parallel extraction → generate)
workflow.add_edge(START, "parse")
workflow.add_edge("parse", "extract_points")
workflow.add_edge("parse", "identify_topics")
workflow.add_edge("parse", "find_actions")
workflow.add_edge("extract_points", "generate")
workflow.add_edge("identify_topics", "generate")
workflow.add_edge("find_actions", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()


# ---------------------------------------------------------------------------
# Run the Graph
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 3.1: SUMMARY STATE GRAPH")
print("=" * 70)
print(f"\nTranscript: {len(transcript_text)} chars\n")

# Run the workflow
result = app.invoke({
    "transcript": transcript_text,
    "key_points": [],
    "topics": [],
    "action_items": [],
    "summary": "",
    "messages": [],
})

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(result["summary"])


# ---------------------------------------------------------------------------
# Inspect State
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  INTERMEDIATE RESULTS")
print("-" * 70)
print(f"\nTopics ({len(result['topics'])}):")
for t in result["topics"]:
    print(f"  - {t}")

print(f"\nKey Points ({len(result['key_points'])}):")
for i, p in enumerate(result["key_points"][:3], 1):
    print(f"  {i}. {p[:60]}...")

print(f"\nAction Items ({len(result['action_items'])}):")
for item in result["action_items"][:2]:
    print(f"  - {item[:60]}...")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ StateGraph provides explicit workflow control
✅ TypedDict ensures type-safe state passing
✅ Nodes can run in parallel (multiple edges from one node)
✅ Intermediate state is inspectable for debugging

Graph Structure:
  START → parse → [extract_points, identify_topics, find_actions] → generate → END

Benefits over Agents:
   - Predictable execution order
   - No reasoning overhead
   - Full visibility into intermediate state
   - Easier to test individual nodes

Next: Stage 3.2 → Iterative refinement with conditional edges
""")
