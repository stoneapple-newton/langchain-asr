"""
Stage 3, File 3: Multi-Aspect Parallel Summary
===============================================
CONCEPT: Parallel extraction of different summary aspects.

Different aspects of a summary can be extracted in parallel:
- Topics discussed
- Action items
- Decisions made  
- Sentiment analysis
- Key participants

Key concepts:
  - Parallel node execution with Send
  - Aggregation of parallel results
  - Fan-out / fan-in pattern

Run this file:
  uv run deep_research/summary_agents/stage_03_langgraph/03_multi_aspect_summary.py
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
from langgraph.types import Send

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class ParallelSummaryState(TypedDict):
    """State for parallel multi-aspect summarization."""
    
    transcript: str
    
    # Parallel results (populated by different branches)
    topics_result: str
    action_items_result: str
    decisions_result: str
    sentiment_result: str
    participants_result: str
    
    # Final combined output
    final_summary: str
    
    # Internal routing
    aspect: str  # Which aspect a parallel node should process
    
    # Audit
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

llm = create_chat_model(
    profile="asr_v2",
    temperature=0,
    max_tokens=512,
)

# Define the aspects we want to extract in parallel
ASPECTS = ["topics", "action_items", "decisions", "sentiment", "participants"]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def distribute_work(state: ParallelSummaryState) -> list[Send]:
    """
    Fan-out: Create parallel tasks for each aspect.
    Returns list of Send objects to dispatch to parallel nodes.
    """
    print(f"  [distribute_work] Spawning {len(ASPECTS)} parallel extractions...")
    
    # Create a Send for each aspect - these run in parallel
    return [
        Send("extract_aspect", {"transcript": state["transcript"], "aspect": aspect})
        for aspect in ASPECTS
    ]


def extract_aspect(state: ParallelSummaryState) -> dict:
    """
    Parallel node: Extract one specific aspect of the summary.
    This node runs once for each aspect in parallel.
    """
    aspect = state["aspect"]
    transcript = state["transcript"]
    
    print(f"    [extract_aspect] Processing: {aspect}")
    
    # Aspect-specific prompts
    prompts = {
        "topics": "List the main topics discussed in this meeting (comma-separated):",
        "action_items": "List action items and tasks assigned (bullet points):",
        "decisions": "List decisions made during the meeting (bullet points):",
        "sentiment": "Describe the overall sentiment (positive/neutral/negative) and tone:",
        "participants": "List key participants and their roles if mentioned:",
    }
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are analyzing a meeting transcript. Be concise."),
        ("human", f"{prompts.get(aspect, 'Analyze:')}\n\n{{transcript}}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": transcript[:1500]})
    
    # Return result in the appropriate field
    return {
        f"{aspect}_result": result,
        "messages": [HumanMessage(content=f"Extracted {aspect}")],
    }


def combine_results(state: ParallelSummaryState) -> dict:
    """
    Fan-in: Combine all parallel extraction results into final summary.
    """
    print("  [combine_results] Merging parallel results...")
    
    # Check which results we have
    results = {
        "topics": state.get("topics_result", "N/A"),
        "action_items": state.get("action_items_result", "N/A"),
        "decisions": state.get("decisions_result", "N/A"),
        "sentiment": state.get("sentiment_result", "N/A"),
        "participants": state.get("participants_result", "N/A"),
    }
    
    print(f"    Collected: {', '.join(k for k, v in results.items() if v != 'N/A')}")
    
    # Combine into final summary
    combined = f"""# Meeting Summary (Multi-Aspect Analysis)

## Sentiment & Tone
{results['sentiment']}

## Key Participants
{results['participants']}

## Topics Discussed
{results['topics']}

## Decisions Made
{results['decisions']}

## Action Items
{results['action_items']}
"""
    
    return {
        "final_summary": combined,
        "messages": [HumanMessage(content="Combined all aspects")],
    }


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(ParallelSummaryState)

# Nodes
workflow.add_node("distribute", distribute_work)
workflow.add_node("extract_aspect", extract_aspect)
workflow.add_node("combine", combine_results)

# Edges
# START → distribute → [parallel extract_aspect nodes] → combine → END
workflow.add_conditional_edges(START, distribute_work, ["extract_aspect"])
workflow.add_edge("extract_aspect", "combine")
workflow.add_edge("combine", END)

app = workflow.compile()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 3.3: MULTI-ASPECT PARALLEL SUMMARY")
print("=" * 70)
print(f"\nTranscript: {len(transcript_text)} chars")
print(f"Parallel aspects: {', '.join(ASPECTS)}\n")

result = app.invoke({
    "transcript": transcript_text,
    "topics_result": "",
    "action_items_result": "",
    "decisions_result": "",
    "sentiment_result": "",
    "participants_result": "",
    "final_summary": "",
    "aspect": "",  # Will be set by Send
    "messages": [],
})

print("\n" + "=" * 70)
print("  FINAL COMBINED SUMMARY")
print("=" * 70)
print(result["final_summary"])


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  PARALLEL EXECUTION ANALYSIS")
print("-" * 70)
print("""
Execution flow:
  START
    │
    ▼
  distribute ──┬──► extract_aspect(topics) ──┐
               ├──► extract_aspect(actions) ──┤
               ├──► extract_aspect(decisions)─┼──► combine ──► END
               ├──► extract_aspect(sentiment)─┤
               └──► extract_aspect(people) ───┘

Benefits of Parallel Extraction:
   - Each aspect can use specialized prompting
   - Faster overall execution (parallel vs sequential)
   - Easier to debug individual aspects
   - Can retry single aspects on failure

Trade-offs:
   - Higher instantaneous resource usage
   - More complex state management
   - Results need careful merging
""")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ Send() dispatches parallel node executions
✅ Same node can run multiple times with different inputs
✅ All parallel branches must converge to a single node
✅ Pattern: Fan-out → parallel processing → Fan-in

When to Use Parallel Processing:
   - Independent extractions from same source
   - Multiple model calls with different prompts
   - Different aspects that don't depend on each other
   - Performance optimization

Next: Stage 3.4 → Conditional routing by transcript type
""")
