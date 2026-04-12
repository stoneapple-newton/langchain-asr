"""
Stage 3, File 2: Iterative Refinement Graph
============================================
CONCEPT: Quality-loop with conditional routing for iterative improvement.

Not all summaries are good on the first pass. This graph adds a quality
assessment step that can route back to revision if the summary isn't
good enough.

Key concepts:
  - Conditional edges based on state
  - Cycles in the graph (refinement loop)
  - Quality gating before completion

Run this file:
  uv run deep_research/summary_agents/stage_03_langgraph/02_iterative_refinement.py
"""

from pathlib import Path
import sys
from typing import Annotated
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Pydantic Schema for Quality Assessment
# ---------------------------------------------------------------------------

class QualityAssessment(BaseModel):
    """Quality score for a summary."""
    completeness: int = Field(description="Coverage of key points (1-10)", ge=1, le=10)
    clarity: int = Field(description="Clarity and readability (1-10)", ge=1, le=10)
    action_items_found: int = Field(description="Action items correctly identified (1-10)", ge=1, le=10)
    overall: int = Field(description="Overall quality (1-10)", ge=1, le=10)
    needs_rework: bool = Field(description="Whether summary needs revision")
    feedback: str = Field(description="Specific feedback for improvement")


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class RefinementState(TypedDict):
    """State with quality tracking for iterative refinement."""
    
    # Input
    transcript: str
    
    # Working summary
    summary: str
    revision_count: int
    
    # Quality tracking
    quality_score: int
    quality_feedback: str
    
    # Final output
    final_summary: str
    
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

QUALITY_THRESHOLD = 7  # Minimum acceptable quality (out of 10)
MAX_REVISIONS = 2      # Prevent infinite loops


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def generate_initial_summary(state: RefinementState) -> dict:
    """Generate first draft of summary."""
    revision = state.get("revision_count", 0)
    print(f"  [generate_summary] Draft #{revision + 1}...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create a meeting summary with: Overview, Key Points, Action Items. "
            "Be concise but comprehensive." if revision == 0 else
            "Revise the summary based on the feedback provided. Address all issues."
        ),
        (
            "human",
            "Transcript:\n{transcript}\n\n" +
            ("Feedback on previous version:\n{feedback}\n\n" if revision > 0 else "") +
            "Create summary:"
        ),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    kwargs = {"transcript": state["transcript"][:2000]}
    if revision > 0:
        kwargs["feedback"] = state.get("quality_feedback", "")
    
    summary = chain.invoke(kwargs)
    
    return {
        "summary": summary,
        "revision_count": revision + 1,
        "messages": [HumanMessage(content=f"Generated draft #{revision + 1}")],
    }


def assess_quality(state: RefinementState) -> dict:
    """Assess the quality of the current summary."""
    print(f"  [assess_quality] Checking quality...")
    
    parser = JsonOutputParser(pydantic_object=QualityAssessment)
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Assess the quality of this meeting summary against the original transcript. "
            "Be critical and thorough.\n\n{format_instructions}"
        ),
        (
            "human",
            "Original Transcript:\n{transcript}\n\n"
            "Summary to Assess:\n{summary}\n\n"
            "Provide quality assessment:"
        ),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser
    
    assessment = chain.invoke({
        "transcript": state["transcript"][:1500],
        "summary": state["summary"],
    })
    
    overall = assessment.get("overall", 5)
    needs_rework = assessment.get("needs_rework", overall < QUALITY_THRESHOLD)
    feedback = assessment.get("feedback", "")
    
    print(f"    Quality score: {overall}/10")
    print(f"    Needs rework: {needs_rework}")
    
    return {
        "quality_score": overall,
        "quality_feedback": feedback,
        "messages": [HumanMessage(content=f"Quality: {overall}/10, rework={needs_rework}")],
    }


def finalize_summary(state: RefinementState) -> dict:
    """Finalize the summary after quality approval."""
    print("  [finalize_summary] Quality approved, finalizing...")
    
    return {
        "final_summary": state["summary"],
        "messages": [HumanMessage(content=f"Finalized after {state['revision_count']} drafts")],
    }


# ---------------------------------------------------------------------------
# Conditional Routing
# ---------------------------------------------------------------------------

def route_based_on_quality(state: RefinementState) -> str:
    """Decide whether to revise or finalize based on quality."""
    score = state.get("quality_score", 0)
    revisions = state.get("revision_count", 0)
    
    if score >= QUALITY_THRESHOLD:
        print(f"    ✓ Quality {score} >= {QUALITY_THRESHOLD}, proceeding to finalize")
        return "finalize"
    elif revisions >= MAX_REVISIONS:
        print(f"    ⚠ Max revisions ({MAX_REVISIONS}) reached, finalizing anyway")
        return "finalize"
    else:
        print(f"    ↻ Quality {score} < {QUALITY_THRESHOLD}, revising...")
        return "revise"


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(RefinementState)

workflow.add_node("generate", generate_initial_summary)
workflow.add_node("assess", assess_quality)
workflow.add_node("finalize", finalize_summary)

workflow.add_edge(START, "generate")
workflow.add_edge("generate", "assess")

# Conditional edge: quality check determines next step
workflow.add_conditional_edges(
    "assess",
    route_based_on_quality,
    {
        "revise": "generate",   # Loop back for revision
        "finalize": "finalize",  # Proceed to finalization
    },
)

workflow.add_edge("finalize", END)

app = workflow.compile()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 3.2: ITERATIVE REFINEMENT GRAPH")
print("=" * 70)
print(f"\nQuality threshold: {QUALITY_THRESHOLD}/10")
print(f"Max revisions: {MAX_REVISIONS}")
print(f"\nTranscript: {len(transcript_text)} chars\n")

result = app.invoke({
    "transcript": transcript_text,
    "summary": "",
    "revision_count": 0,
    "quality_score": 0,
    "quality_feedback": "",
    "final_summary": "",
    "messages": [],
})

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(result["final_summary"])

print("\n" + "-" * 70)
print("  REFINEMENT STATS")
print("-" * 70)
print(f"  Drafts created: {result['revision_count']}")
print(f"  Final quality score: {result['quality_score']}/10")
if result['quality_feedback']:
    print(f"  Last feedback: {result['quality_feedback'][:100]}...")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ Conditional edges enable loops and branching
✅ Quality gating ensures minimum standards
✅ State tracks iteration count to prevent infinite loops
✅ Feedback from assessment drives improvements

Graph Flow:
  START → generate → assess → [finalize|generate] → END
                    ↓___________↑ (refinement loop)

Use Cases for Iteration:
   - Quality assurance before delivery
   - Format compliance checking
   - Length constraints (too short/long)
   - Missing required sections

Comparison to One-Shot:
   | Aspect        | One-Shot | Iterative |
   |---------------|----------|-----------|
   | Quality       | Variable | Gated     |
   | Latency       | Fast     | Slower    |
   | Cost          | Low      | Higher    |
   | Reliability   | Lower    | Higher    |

Next: Stage 3.3 → Parallel multi-aspect extraction
""")
