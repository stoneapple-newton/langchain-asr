"""
Stage 5, File 1: Thread Memory
===============================
CONCEPT: Checkpointing and resuming summary workflows.

MemorySaver checkpoints the graph state at each step, allowing:
- Resuming interrupted workflows
- Human-in-the-loop approval
- Time-travel debugging

Key concepts:
  - MemorySaver checkpointer
  - Thread IDs for conversation isolation
  - State persistence across invocations

Run this file:
  uv run deep_research/summary_agents/stage_05_memory_persistent/01_thread_memory.py
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
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class ThreadMemoryState(TypedDict):
    """State with thread memory tracking."""
    
    transcript: str
    
    # Processing stages
    extracted_points: str
    draft_summary: str
    final_summary: str
    
    # Human feedback
    feedback: str | None
    approved: bool
    
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
# Nodes
# ---------------------------------------------------------------------------

def extract_points(state: ThreadMemoryState) -> dict:
    """Extract key points from transcript."""
    print("  [extract_points] Analyzing transcript...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract 3-5 key discussion points from this meeting."),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    points = chain.invoke({"transcript": state["transcript"][:2000]})
    
    return {
        "extracted_points": points,
        "messages": [HumanMessage(content="Key points extracted")],
    }


def generate_draft(state: ThreadMemoryState) -> dict:
    """Generate draft summary."""
    print("  [generate_draft] Creating draft summary...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create a meeting summary draft using these key points.\n\n{points}"
        ),
        ("human", "Write a draft summary of this meeting."),
    ])
    
    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({"points": state["extracted_points"]})
    
    return {
        "draft_summary": draft,
        "messages": [HumanMessage(content="Draft summary created")],
    }


def apply_feedback(state: ThreadMemoryState) -> dict:
    """Revise summary based on human feedback."""
    print("  [apply_feedback] Incorporating feedback...")
    
    feedback = state.get("feedback", "")
    if not feedback:
        print("    (No feedback provided, keeping draft)")
        return {
            "final_summary": state["draft_summary"],
            "messages": [HumanMessage(content="No feedback to apply")],
        }
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Revise the summary based on the user's feedback."
        ),
        (
            "human",
            "Current draft:\n{draft}\n\n"
            "User feedback:\n{feedback}\n\n"
            "Please provide the revised summary:"
        ),
    ])
    
    chain = prompt | llm | StrOutputParser()
    revised = chain.invoke({
        "draft": state["draft_summary"],
        "feedback": feedback,
    })
    
    return {
        "final_summary": revised,
        "messages": [HumanMessage(content="Feedback incorporated")],
    }


def finalize(state: ThreadMemoryState) -> dict:
    """Finalize the summary."""
    print("  [finalize] Summary complete")
    
    # If we have a final summary, use it; otherwise use draft
    final = state.get("final_summary") or state["draft_summary"]
    
    return {
        "final_summary": final,
        "approved": True,
        "messages": [HumanMessage(content="Summary finalized")],
    }


# ---------------------------------------------------------------------------
# Conditional Routing
# ---------------------------------------------------------------------------

def check_approval(state: ThreadMemoryState) -> str:
    """Check if human has provided feedback or approval."""
    if state.get("feedback"):
        print("    → Feedback received, applying revisions")
        return "revise"
    elif state.get("approved"):
        print("    → Approved, finalizing")
        return "finalize"
    else:
        print("    → Awaiting human feedback")
        return "await_feedback"


# ---------------------------------------------------------------------------
# Build Graph with Memory
# ---------------------------------------------------------------------------

workflow = StateGraph(ThreadMemoryState)

workflow.add_node("extract", extract_points)
workflow.add_node("draft", generate_draft)
workflow.add_node("revise", apply_feedback)
workflow.add_node("await_feedback", lambda state: state)  # Human interrupt point
workflow.add_node("finalize", finalize)

workflow.add_edge(START, "extract")
workflow.add_edge("extract", "draft")
workflow.add_edge("draft", "await_feedback")

# Conditional: wait for human or proceed
workflow.add_conditional_edges(
    "await_feedback",
    check_approval,
    {
        "revise": "revise",
        "finalize": "finalize",
        "await_feedback": "await_feedback",  # Loop until human responds
    },
)

workflow.add_edge("revise", "finalize")
workflow.add_edge("finalize", END)

# Add checkpointer for memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Run with Thread Memory
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 5.1: THREAD MEMORY")
print("=" * 70)
print(f"\nTranscript: {len(transcript_text)} chars\n")

# Configuration with thread_id for persistence
thread_id = "summary_session_001"
config = {"configurable": {"thread_id": thread_id}}

print("-" * 70)
print("  STEP 1: Initial run (extract + draft)")
print("-" * 70)

# Initial invocation - runs up to await_feedback
result = app.invoke({
    "transcript": transcript_text,
    "extracted_points": "",
    "draft_summary": "",
    "final_summary": "",
    "feedback": None,
    "approved": False,
    "messages": [],
}, config=config)

print(f"\nDraft summary:\n{result['draft_summary'][:400]}...")

print("\n" + "-" * 70)
print("  STEP 2: Resume with human feedback")
print("-" * 70)

# Resume with feedback (simulating human-in-the-loop)
# In production, this would be triggered by a UI action
feedback_result = app.invoke({
    **result,
    "feedback": "Please emphasize the JWT authentication decisions and add specific action items with owners.",
}, config=config)

print(f"\nFinal summary:\n{feedback_result['final_summary'][:600]}...")

print("\n" + "-" * 70)
print("  STEP 3: Inspect thread state")
print("-" * 70)

# Get state history
state_history = list(app.get_state_history(config))
print(f"\nState checkpoints in thread: {len(state_history)}")
for i, state in enumerate(state_history[:3]):
    values = state.values
    print(f"  {i+1}. Step: extract→draft→await")
    print(f"     Draft length: {len(values.get('draft_summary', ''))} chars")
    print(f"     Feedback: {values.get('feedback', 'N/A')[:50] if values.get('feedback') else 'N/A'}")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ MemorySaver checkpoints state at each step
✅ Thread IDs isolate different conversations/sessions
✅ Can resume interrupted workflows
✅ State history enables time-travel debugging

Thread Memory Features:
   - Automatic checkpointing at each node
   - Resume from any checkpoint
   - Human-in-the-loop interrupts
   - Multi-turn conversations

Use Cases:
   - Long-running summary workflows
   - Human approval gates
   - Multi-step refinement with feedback
   - Debugging production issues

Next: Stage 5.2 → Persistent store for long-term memory
""")
