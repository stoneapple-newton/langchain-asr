"""
Stage 5, File 2: Persistent Store
==================================
CONCEPT: Long-term memory across meeting series with InMemoryStore.

Unlike thread memory (ephemeral session state), persistent store
retains knowledge across sessions - perfect for building up
institutional knowledge about ongoing projects and recurring topics.

Key concepts:
  - InMemoryStore for cross-thread persistence
  - Store API: put, get, search
  - Namespace organization

Run this file:
  uv run deep_research/summary_agents/stage_05_memory_persistent/02_persistent_store.py
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
from langgraph.store.memory import InMemoryStore

from config import create_chat_model
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Setup Persistent Store
# ---------------------------------------------------------------------------

# In production, this would be a real database (Postgres, Redis, etc.)
store = InMemoryStore()

# Pre-populate with some "institutional knowledge"
store.put(
    namespace=("projects", "nexus"),
    key="overview",
    value={
        "name": "Nexus Platform",
        "status": "in_progress",
        "start_date": "2024-01-15",
        "priority": "high",
        "description": "Core platform infrastructure upgrade"
    }
)

store.put(
    namespace=("topics", "authentication"),
    key="status",
    value={
        "current": "evaluating JWT",
        "decisions": ["OAuth ruled out due to complexity"],
        "open_questions": ["Token expiry strategy"]
    }
)

print("=" * 70)
print("  STAGE 5.2: PERSISTENT STORE")
print("=" * 70)
print("\nPre-populated store with:")
print("  - projects/nexus/overview")
print("  - topics/authentication/status")


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class PersistentState(TypedDict):
    """State with persistent memory integration."""
    
    transcript: str
    
    # Retrieved context
    project_context: dict | None
    topic_context: dict | None
    
    # Output
    summary: str
    extracted_facts: list[dict]
    
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
# Store Integration Nodes
# ---------------------------------------------------------------------------

def retrieve_context(state: PersistentState) -> dict:
    """Retrieve relevant context from persistent store."""
    print("  [retrieve_context] Loading institutional knowledge...")
    
    # Search for relevant project info
    project_context = None
    try:
        project_context = store.get(
            namespace=("projects", "nexus"),
            key="overview"
        )
        print(f"    Found project context: {project_context['value']['name']}")
    except Exception:
        print("    No project context found")
    
    # Search for relevant topic info
    topic_context = None
    try:
        topic_context = store.get(
            namespace=("topics", "authentication"),
            key="status"
        )
        print(f"    Found topic context: {topic_context['value']['current']}")
    except Exception:
        print("    No topic context found")
    
    return {
        "project_context": project_context["value"] if project_context else None,
        "topic_context": topic_context["value"] if topic_context else None,
        "messages": [HumanMessage(content="Retrieved persistent context")],
    }


def generate_summary(state: PersistentState) -> dict:
    """Generate summary enriched with persistent context."""
    print("  [generate_summary] Creating context-enriched summary...")
    
    # Build context section
    context_parts = []
    if state.get("project_context"):
        proj = state["project_context"]
        context_parts.append(f"Project: {proj['name']} ({proj['status']}, priority: {proj['priority']})")
    
    if state.get("topic_context"):
        topic = state["topic_context"]
        context_parts.append(f"Topic Context - Authentication: {topic['current']}")
    
    context_text = "\n".join(context_parts) if context_parts else "No prior context available."
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create a meeting summary. Use the institutional context to:\n"
            "- Highlight progress on ongoing initiatives\n"
            "- Connect decisions to previous context\n"
            "- Note any changes in direction\n\n"
            "Context:\n{context}"
        ),
        ("human", "Meeting transcript:\n{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({
        "context": context_text,
        "transcript": state["transcript"][:2000],
    })
    
    return {
        "summary": summary,
        "messages": [HumanMessage(content="Generated context-enriched summary")],
    }


def extract_facts(state: PersistentState) -> dict:
    """Extract facts to persist for future meetings."""
    print("  [extract_facts] Extracting facts for persistence...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract key facts from this meeting that should be remembered "
            "for future context. Focus on:\n"
            "- Decisions that change project direction\n"
            "- New priorities or timeline changes\n"
            "- Open questions that need follow-up"
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    facts_text = chain.invoke({"transcript": state["transcript"][:2000]})
    
    # Parse into structured facts (simplified)
    facts = [{"content": line.strip("- "), "type": "meeting_fact"} 
             for line in facts_text.split("\n") if line.strip().startswith("-")]
    
    print(f"    Extracted {len(facts)} facts to persist")
    
    return {
        "extracted_facts": facts,
        "messages": [HumanMessage(content=f"Extracted {len(facts)} facts")],
    }


def persist_facts(state: PersistentState) -> dict:
    """Persist extracted facts to the store."""
    print("  [persist_facts] Saving to persistent store...")
    
    facts = state.get("extracted_facts", [])
    for i, fact in enumerate(facts):
        store.put(
            namespace=("meetings", "2024-11-12"),
            key=f"fact_{i}",
            value=fact
        )
    
    print(f"    Persisted {len(facts)} facts to meetings/2024-11-12/")
    
    return {
        "messages": [HumanMessage(content=f"Persisted {len(facts)} facts")],
    }


# ---------------------------------------------------------------------------
# Build Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(PersistentState)

workflow.add_node("retrieve", retrieve_context)
workflow.add_node("generate", generate_summary)
workflow.add_node("extract", extract_facts)
workflow.add_node("persist", persist_facts)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "extract")
workflow.add_edge("extract", "persist")
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
    "project_context": None,
    "topic_context": None,
    "summary": "",
    "extracted_facts": [],
    "messages": [],
})

print("\n" + "=" * 70)
print("  SUMMARY WITH PERSISTENT CONTEXT")
print("=" * 70)
print(result["summary"][:800])

print("\n" + "-" * 70)
print("  EXTRACTED FACTS (NOW PERSISTED)")
print("-" * 70)
for fact in result["extracted_facts"]:
    print(f"  • {fact['content'][:80]}...")

print("\n" + "-" * 70)
print("  STORE CONTENTS")
print("-" * 70)

# Show what's in the store now
for ns in [("projects", "nexus"), ("topics", "authentication"), ("meetings", "2024-11-12")]:
    try:
        items = store.search(ns)
        print(f"\n  Namespace {ns}: {len(items)} items")
        for item in items[:2]:
            print(f"    - {item['key']}: {str(item['value'])[:60]}...")
    except Exception as e:
        print(f"  Namespace {ns}: empty or error")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ InMemoryStore persists across sessions
✅ Namespaces organize data hierarchically
✅ Store API: put(), get(), search()
✅ Cross-thread knowledge sharing

Store Organization:
   ("projects", "nexus") → project-specific data
   ("topics", "auth")    → topic-specific data
   ("meetings", date)    → meeting-specific data

Use Cases:
   - Accumulate knowledge across meeting series
   - Track project status over time
   - Build organizational memory
   - Share context between team members

Production Scaling:
   - InMemoryStore → PostgresStore, RedisStore
   - Async store operations
   - Access control per namespace
   - Backup and versioning

Next: Stage 5.3 → Follow-up awareness across meetings
""")
