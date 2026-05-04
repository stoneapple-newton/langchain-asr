"""
Stage 5, File 3: Follow-Up Awareness
=====================================
CONCEPT: Track action items across meeting series.

Meetings don't exist in isolation. This graph tracks:
- Previous meeting action items
- What's been completed
- What's still open
- What's overdue

Key concepts:
  - Action item tracking
  - Cross-meeting continuity
  - Status updates

Run this file:
  uv run deep_research/summary_agents/shared/sample_data/meeting_sample.json"
"""

from pathlib import Path
import sys
from datetime import datetime, timedelta
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
from langgraph.store.memory import InMemoryStore

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Setup Store with Historical Action Items
# ---------------------------------------------------------------------------

store = InMemoryStore()

# Previous meeting action items
store.put(
    namespace=("action_items", "2024-11-05"),
    key="items",
    value=[
        {
            "id": "ai_001",
            "task": "Research JWT library options",
            "assignee": "Alice",
            "status": "completed",
            "due": "2024-11-10",
            "meeting": "2024-11-05"
        },
        {
            "id": "ai_002",
            "task": "Prepare WCAG compliance checklist",
            "assignee": "Bob",
            "status": "in_progress",
            "due": "2024-11-15",
            "meeting": "2024-11-05"
        },
        {
            "id": "ai_003",
            "task": "Schedule backend tech decision meeting",
            "assignee": "Carol",
            "status": "pending",
            "due": "2024-11-12",
            "meeting": "2024-11-05"
        },
    ]
)

print("=" * 70)
print("  STAGE 5.3: FOLLOW-UP AWARENESS")
print("=" * 70)
print("\nLoaded previous action items:")
prev_items = store.get(("action_items", "2024-11-05"), "items")
for item in prev_items["value"]:
    print(f"  [{item['status']}] {item['task']} ({item['assignee']}, due {item['due']})")


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class FollowUpState(TypedDict):
    """State for follow-up aware summarization."""
    
    transcript: str
    meeting_date: str
    
    # Previous action items
    previous_items: list[dict]
    
    # New items from this meeting
    new_items: list[dict]
    
    # Tracking
    completed_items: list[str]
    overdue_items: list[str]
    
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
    max_tokens=4096,
)


# ---------------------------------------------------------------------------
# Follow-Up Nodes
# ---------------------------------------------------------------------------

def load_previous_items(state: FollowUpState) -> dict:
    """Load action items from previous meetings."""
    print("  [load_previous_items] Loading historical action items...")
    
    # In production, would query by date range
    try:
        items_record = store.get(("action_items", "2024-11-05"), "items")
        items = items_record["value"]
        print(f"    Loaded {len(items)} items from previous meeting")
    except Exception:
        items = []
        print("    No previous items found")
    
    # Categorize by status
    pending = [i for i in items if i["status"] in ["pending", "in_progress"]]
    overdue = [i for i in pending if i["due"] < state["meeting_date"]]
    
    print(f"    Pending: {len(pending)}, Overdue: {len(overdue)}")
    
    return {
        "previous_items": items,
        "overdue_items": [i["id"] for i in overdue],
        "messages": [HumanMessage(content=f"Loaded {len(items)} previous items")],
    }


def extract_new_items(state: FollowUpState) -> dict:
    """Extract new action items from current transcript."""
    print("  [extract_new_items] Finding new action items...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract action items from this meeting. For each item, identify:\n"
            "- Task description\n"
            "- Assignee (if mentioned)\n"
            "- Due date (if mentioned)\n\n"
            "Also check if any previous action items were discussed as completed."
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"transcript": state["transcript"][:2000]})
    
    # Parse new items (simplified)
    new_items = []
    for line in result.split("\n"):
        if line.strip().startswith("-") and ("action" in line.lower() or "task" in line.lower()):
            new_items.append({
                "id": f"ai_new_{len(new_items)}",
                "task": line.strip("- "),
                "status": "pending",
                "meeting": state["meeting_date"]
            })
    
    print(f"    Found {len(new_items)} new action items")
    
    return {
        "new_items": new_items,
        "messages": [HumanMessage(content=f"Extracted {len(new_items)} new items")],
    }


def check_completions(state: FollowUpState) -> dict:
    """Check which previous items were marked as completed."""
    print("  [check_completions] Checking for completed items...")
    
    # In production, would use LLM to detect completions
    # For demo, simulate that JWT research was completed
    completed = ["ai_001"]  # JWT research completed
    
    print(f"    Detected {len(completed)} completed items")
    
    return {
        "completed_items": completed,
        "messages": [HumanMessage(content=f"{len(completed)} items completed")],
    }


def generate_follow_up_summary(state: FollowUpState) -> dict:
    """Generate summary with follow-up tracking."""
    print("  [generate_follow_up_summary] Creating follow-up aware summary...")
    
    # Build follow-up section
    sections = []
    
    # Completed items
    if state.get("completed_items"):
        sections.append("✅ Completed Since Last Meeting:")
        completed = [i for i in state["previous_items"] 
                     if i["id"] in state["completed_items"]]
        for item in completed:
            sections.append(f"  - {item['task']} ({item['assignee']})")
    
    # Overdue items
    if state.get("overdue_items"):
        sections.append("\n⚠️ Overdue Items:")
        overdue = [i for i in state["previous_items"]
                   if i["id"] in state["overdue_items"]]
        for item in overdue:
            sections.append(f"  - {item['task']} ({item['assignee']}, due {item['due']})")
    
    # Still pending
    pending = [i for i in state["previous_items"]
               if i["status"] in ["pending", "in_progress"]
               and i["id"] not in state.get("completed_items", [])]
    if pending:
        sections.append("\n📋 Still Pending:")
        for item in pending:
            sections.append(f"  - {item['task']} ({item['assignee']})")
    
    # New items
    if state.get("new_items"):
        sections.append("\n🆕 New Action Items:")
        for item in state["new_items"]:
            sections.append(f"  - {item['task']}")
    
    follow_up_text = "\n".join(sections)
    
    # Generate main summary
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create a meeting summary. Include this follow-up context:\n\n{follow_up}"
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    main_summary = chain.invoke({
        "follow_up": follow_up_text,
        "transcript": state["transcript"][:1500],
    })
    
    full_summary = f"""# Meeting Summary - {state['meeting_date']}

## Follow-Up Status
{follow_up_text}

## Meeting Content
{main_summary}
"""
    
    return {
        "summary": full_summary,
        "messages": [HumanMessage(content="Generated follow-up aware summary")],
    }


def persist_new_items(state: FollowUpState) -> dict:
    """Persist new action items for future tracking."""
    print("  [persist_new_items] Saving new action items...")
    
    # Combine with updated previous items
    all_items = []
    
    # Update previous items with completion status
    for item in state["previous_items"]:
        if item["id"] in state.get("completed_items", []):
            item["status"] = "completed"
        all_items.append(item)
    
    # Add new items
    all_items.extend(state.get("new_items", []))
    
    store.put(
        namespace=("action_items", state["meeting_date"]),
        key="items",
        value=all_items
    )
    
    print(f"    Persisted {len(all_items)} total items")
    
    return {
        "messages": [HumanMessage(content=f"Persisted {len(all_items)} items")],
    }


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(FollowUpState)

workflow.add_node("load_previous", load_previous_items)
workflow.add_node("extract_new", extract_new_items)
workflow.add_node("check_completions", check_completions)
workflow.add_node("generate", generate_follow_up_summary)
workflow.add_node("persist", persist_new_items)

workflow.add_edge(START, "load_previous")
workflow.add_edge("load_previous", "extract_new")
workflow.add_edge("extract_new", "check_completions")
workflow.add_edge("check_completions", "generate")
workflow.add_edge("generate", "persist")
workflow.add_edge("persist", END)

app = workflow.compile()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print(f"\nTranscript: {len(transcript_text)} chars\n")

result = app.invoke({
    "transcript": transcript_text,
    "meeting_date": "2024-11-12",
    "previous_items": [],
    "new_items": [],
    "completed_items": [],
    "overdue_items": [],
    "summary": "",
    "messages": [],
})

print("\n" + "=" * 70)
print("  FOLLOW-UP AWARE SUMMARY")
print("=" * 70)
print(result["summary"][:1200])


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ Track action items across meeting series
✅ Highlight completed, overdue, and pending items
✅ Show progress over time
✅ Create accountability through continuity

Follow-Up Pipeline:
  Load Previous → Extract New → Check Completions → Generate → Persist

Benefits:
   - No action items fall through cracks
   - Clear visibility on what's done vs pending
   - Meeting series feels like a continuous narrative
   - Accountability for assigned tasks

Production Features:
   - Automatic overdue detection
   - Reminder notifications
   - Completion confirmation workflow
   - Dashboard for all open action items

Next: Stage 6 → Deep Agents with subagent swarms
""")
