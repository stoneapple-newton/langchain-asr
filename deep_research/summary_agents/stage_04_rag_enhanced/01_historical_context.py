"""
Stage 4, File 1: Historical Context RAG
========================================
CONCEPT: Enhancing summaries with retrieval from past meetings.

Meetings build on previous discussions. RAG lets us retrieve relevant
context from past meeting summaries to provide continuity and highlight
progress on ongoing topics.

Key concepts:
  - InMemoryVectorStore for document storage
  - Embedding-based similarity search
  - Context-augmented summarization

Run this file:
  uv run deep_research/summary_agents/stage_04_rag_enhanced/01_historical_context.py
"""

from pathlib import Path
import sys
from typing import Annotated
from typing_extensions import TypedDict

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from config import create_chat_model, create_embeddings
from deep_research.summary_agents.shared import load_transcript, format_transcript_for_llm


# ---------------------------------------------------------------------------
# Mock Historical Meeting Data
# ---------------------------------------------------------------------------

HISTORICAL_MEETINGS = [
    {
        "date": "2024-10-15",
        "title": "API Architecture Discussion",
        "summary": """
Discussed authentication approaches for the new API. JWT tokens selected 
over session-based auth for scalability. Concerns raised about token 
expiration handling. Action: Research refresh token patterns.
""",
        "topics": ["API", "authentication", "JWT", "architecture"],
    },
    {
        "date": "2024-10-22", 
        "title": "Accessibility Standards Review",
        "summary": """
WCAG 2.1 AA compliance mandated for all new features. Frontend team 
to integrate axe-core testing. Design system needs color contrast 
updates. Target completion: end of Q4.
""",
        "topics": ["WCAG", "accessibility", "testing", "design system"],
    },
    {
        "date": "2024-10-29",
        "title": "Backend Technology Evaluation",
        "summary": """
Evaluated Node.js vs Python for microservices. Node.js chosen for 
I/O-bound operations, Python for ML workloads. Team needs training 
on async/await patterns. Migration plan to be drafted.
""",
        "topics": ["Node.js", "Python", "microservices", "backend"],
    },
    {
        "date": "2024-11-05",
        "title": "Q4 Roadmap Planning",
        "summary": """
Prioritized Q4 deliverables: Beta release Nov 15, WCAG audit Dec 1, 
Q4 wrap-up Dec 15. Authentication service is on critical path. 
Resource reallocation approved from frontend to backend team.
""",
        "topics": ["roadmap", "Q4", "planning", "beta release"],
    },
]


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    """State with RAG context retrieval."""
    
    transcript: str
    
    # RAG results
    retrieved_context: list[str]
    
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

embeddings = create_embeddings()


def build_vector_store() -> InMemoryVectorStore:
    """Build vector store from historical meeting summaries."""
    documents = [
        Document(
            page_content=m["summary"],
            metadata={
                "date": m["date"],
                "title": m["title"],
                "topics": m["topics"],
            }
        )
        for m in HISTORICAL_MEETINGS
    ]
    return InMemoryVectorStore.from_documents(documents, embeddings)


vector_store = build_vector_store()


# ---------------------------------------------------------------------------
# RAG Nodes
# ---------------------------------------------------------------------------

def retrieve_context(state: RAGState) -> dict:
    """Retrieve relevant historical context for the transcript."""
    print("  [retrieve_context] Searching historical meetings...")
    
    transcript = state["transcript"]
    
    # Create a query from the transcript (use first 500 chars as query)
    query = transcript[:500]
    
    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)
    
    print(f"    Retrieved {len(docs)} relevant meetings:")
    contexts = []
    for doc in docs:
        date = doc.metadata.get("date", "unknown")
        title = doc.metadata.get("title", "Untitled")
        print(f"      - {date}: {title}")
        contexts.append(f"[{date}] {title}:\n{doc.page_content}")
    
    return {
        "retrieved_context": contexts,
        "messages": [HumanMessage(content=f"Retrieved {len(docs)} context documents")],
    }


def generate_with_context(state: RAGState) -> dict:
    """Generate summary augmented with historical context."""
    print("  [generate_with_context] Creating context-aware summary...")
    
    contexts = state.get("retrieved_context", [])
    context_text = "\n\n---\n\n".join(contexts) if contexts else "No relevant historical context."
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are summarizing a meeting with awareness of previous discussions.\n\n"
            "Historical context from past meetings:\n{context}\n\n"
            "Use this context to:\n"
            "- Highlight continuity (e.g., 'continuing discussion from Oct 15...')\n"
            "- Show progress on ongoing topics\n"
            "- Note any shifts in direction from previous decisions"
        ),
        (
            "human",
            "Current meeting transcript:\n{transcript}\n\n"
            "Create a summary that references relevant historical context:"
        ),
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({
        "context": context_text,
        "transcript": state["transcript"][:2000],
    })
    
    return {
        "summary": summary,
        "messages": [HumanMessage(content="Generated context-aware summary")],
    }


def generate_without_context(state: RAGState) -> dict:
    """Baseline: Generate summary without RAG context."""
    print("  [generate_without_context] Creating baseline summary...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Create a meeting summary with: Overview, Key Points, Action Items."
        ),
        ("human", "{transcript}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"transcript": state["transcript"][:2000]})
    
    return {
        "summary": summary,
        "messages": [HumanMessage(content="Generated baseline summary")],
    }


# ---------------------------------------------------------------------------
# Build Graphs
# ---------------------------------------------------------------------------

# RAG-enhanced workflow
rag_workflow = StateGraph(RAGState)
rag_workflow.add_node("retrieve", retrieve_context)
rag_workflow.add_node("generate", generate_with_context)
rag_workflow.add_edge(START, "retrieve")
rag_workflow.add_edge("retrieve", "generate")
rag_workflow.add_edge("generate", END)
rag_app = rag_workflow.compile()

# Baseline workflow (no RAG)
baseline_workflow = StateGraph(RAGState)
baseline_workflow.add_node("generate", generate_without_context)
baseline_workflow.add_edge(START, "generate")
baseline_workflow.add_edge("generate", END)
baseline_app = baseline_workflow.compile()


# ---------------------------------------------------------------------------
# Run Comparison
# ---------------------------------------------------------------------------

transcript = load_transcript("deep_research/summary_agents/shared/sample_data/meeting_sample.json")
transcript_text = format_transcript_for_llm(transcript, format_type="speaker_turns")

print("=" * 70)
print("  STAGE 4.1: HISTORICAL CONTEXT RAG")
print("=" * 70)
print(f"\nTranscript: {len(transcript_text)} chars")
print(f"Historical meetings in store: {len(HISTORICAL_MEETINGS)}\n")

print("-" * 70)
print("  BASELINE (NO CONTEXT)")
print("-" * 70)

baseline_result = baseline_app.invoke({
    "transcript": transcript_text,
    "retrieved_context": [],
    "summary": "",
    "messages": [],
})

print(baseline_result["summary"][:800])

print("\n" + "-" * 70)
print("  RAG-ENHANCED (WITH HISTORICAL CONTEXT)")
print("-" * 70)

rag_result = rag_app.invoke({
    "transcript": transcript_text,
    "retrieved_context": [],
    "summary": "",
    "messages": [],
})

print(rag_result["summary"][:1200])


# ---------------------------------------------------------------------------
# Show Retrieved Context
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("  RETRIEVED CONTEXT (USED BY RAG)")
print("-" * 70)
for ctx in rag_result["retrieved_context"]:
    print(f"\n{ctx[:300]}...")


# ---------------------------------------------------------------------------
# Key Takeaways
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("  KEY TAKEAWAYS")
print("=" * 70)
print("""
✅ RAG retrieves relevant historical context automatically
✅ Summaries can reference continuity and progress
✅ Context-aware summaries provide better institutional memory
✅ InMemoryVectorStore works great for prototyping

RAG Pipeline:
  Transcript → Embedding → Retrieve Similar → Augment Prompt → Generate

Benefits of Historical Context:
   - Highlights ongoing vs new topics
   - Shows decision evolution over time
   - Identifies when direction changed
   - Provides timeline continuity

Scaling Considerations:
   - InMemoryVectorStore → Chroma/Pinecone for production
   - Consider time-decay (recent meetings more relevant)
   - Filter by project/team for relevance
   - Store embeddings to avoid re-computation

Next: Stage 4.2 → Entity grounding with verified knowledge
""")
